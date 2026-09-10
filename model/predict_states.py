"""
Run a NORMA checkpoint over the test split with the query token set to EVERY
state, then derive leak-free forecasting variants.

Referee 3 (minor 1) pointed out that predict.py sets the query token to the
realized future state, which the baselines never see.  This script produces:

  predictions_states_{source}.csv     per row: the realized state, the last
                                      history state, and (mu, log_var[, q*]) for
                                      each queried state
  predictions_{source}_normal.csv     query fixed to "normal" (deployment setting)
  predictions_{source}_marginal.csv   mixture over states, p(s | s_last, analyte)
  predictions_{source}_marginal_freq.csv
                                      mixture over states, p(s | analyte)

The derived files use the same schema as predictions_{source}.csv, so
evaluate.py can score them unchanged.

Usage:
    python predict_states.py --run_id 334f7e21            # inference + derive
    python predict_states.py --run_id 334f7e21 --derive_only
    python predict_states.py --run_id 334f7e21 --max_sequences 20000   # smoke test
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(HERE)
import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path

from utils import load_checkpoint, create_model, to_device_batch, run_model
from model import is_quantile_mode
from data import TEST_VOCAB, CODE_TO_TEST_NAME, load_and_split_data, load_panel, TimeSeriesDataset, collate_fn
from state_prior import load_state_priors, prior_weights, DEFAULT_PATH as PRIOR_PATH, PUBLISHED_CHECKPOINT
from state_mixture import mix_gaussian_batched, mix_quantiles_batched, QUANTILE_COLS

STATE_NAMES = {0: 'low', 1: 'normal', 2: 'high'}


class LengthBucketSampler(Sampler):
    """Batches of similar history length: cuts padding from max_len to ~mean_len."""
    def __init__(self, lengths, batch_size):
        order = np.argsort(lengths, kind='stable')
        self.batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def run_all_states(run_id, split='test', source='combined', max_sequences=None,
                   batch_size=1024, device=None, seed=42, log_dir=None, checkpoint='latest'):
    log_dir = log_dir or os.path.join(HERE, 'logs')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the weights behind the published predictions file (see
    # state_prior.PUBLISHED_CHECKPOINT) so the oracle variant reproduces it.
    ckpt, hp = load_checkpoint(log_dir, run_id, best=(checkpoint == 'best'), device='cpu', quiet=True)
    print(f'{run_id}: checkpoint_{checkpoint} (epoch {ckpt.get("epoch")})')
    model = create_model(hp, ncodes=len(TEST_VOCAB), checkpoint=ckpt).to(device).eval()
    is_quantile = is_quantile_mode(getattr(hp, 'output_mode', 'gaussian')) and getattr(hp, 'model', '') == 'NORMA2'
    nstates = getattr(hp, 'nstates', 3)
    normalize = bool(getattr(hp, 'normalize', False))

    data_dir = os.path.join(ROOTDIR, '..', 'data', 'processed')
    version = getattr(hp, 'data_version', 'v2')
    train_seq, val_seq, test_seq = load_and_split_data(data_dir, source, print_info=False, nstates=nstates, version=version,
                                                       split_by=getattr(hp, 'split_by', 'sequence'))
    panel = load_panel(data_dir, source, version) if getattr(hp, 'use_coanalytes', False) else None
    seqs = {'train': train_seq, 'val': val_seq, 'test': test_seq}[split]
    if max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(seqs), size=min(max_sequences, len(seqs)), replace=False)
        seqs = [seqs[i] for i in idx]
    print(f'{run_id}: {"quantile" if is_quantile else "gaussian"} head, {len(seqs):,} {split} sequences, device={device}')

    ds = TimeSeriesDataset(seqs, nstates, normalize=normalize, panel=panel)
    lengths = np.array([len(s['x']) - 1 for s in seqs])
    loader = DataLoader(ds, batch_sampler=LengthBucketSampler(lengths, batch_size),
                        collate_fn=collate_fn, num_workers=0)

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = to_device_batch(batch, device)
            B = batch['x_next'].shape[0]
            rec = {
                'pid': list(batch['pids']),
                'cid': batch['cid'].view(-1).cpu().numpy(),
                'x_next': batch['x_next'].view(-1).cpu().numpy().astype(float),
                't_next': batch['t_next'].view(-1).cpu().numpy().astype(float),
                's_next': batch['s_next'].view(-1).cpu().numpy(),
                # last history state: pad_mask is False on real positions
                's_last': batch['s_h'].gather(1, ((~batch['pad_mask']).sum(1) - 1).clamp(min=0)[:, None]).view(-1).cpu().numpy(),
                'n_hist': (~batch['pad_mask']).sum(1).cpu().numpy(),
            }
            span = ref_low = None
            if normalize:
                ref_low = batch['ref_low'].view(-1).cpu().numpy()
                span = batch['ref_high'].view(-1).cpu().numpy() - ref_low
                rec['x_next'] = rec['x_next'] * span + ref_low
            for q in range(nstates):
                s_q = torch.full_like(batch['s_next'], q)
                out = run_model(model, batch, s_next=s_q)
                if is_quantile:
                    arr = out.cpu().numpy().astype(float)
                    if normalize:
                        arr = arr * span[:, None] + ref_low[:, None]
                    for k, col in enumerate(QUANTILE_COLS):
                        rec[f'{col}_{q}'] = arr[:, k]
                    rec[f'mu_{q}'] = arr[:, 2]
                    rec[f'log_var_{q}'] = 2.0 * np.log((arr[:, 4] - arr[:, 0]) / 3.92 + 1e-8)
                else:
                    mu, lv = out
                    mu = mu.view(-1).cpu().numpy().astype(float)
                    lv = lv.view(-1).cpu().numpy().astype(float)
                    if normalize:
                        mu = mu * span + ref_low
                        lv = lv + 2.0 * np.log(span + 1e-8)
                    rec[f'mu_{q}'] = mu
                    rec[f'log_var_{q}'] = lv
            rows.append(pd.DataFrame(rec))
            if bi % 50 == 0:
                done = sum(len(r) for r in rows)
                print(f'  {done:>8,}/{len(seqs):,}  {time.time() - t0:6.0f}s', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df['code'] = df['cid'].map(CODE_TO_TEST_NAME)
    df['split'] = split
    print(f'  done in {time.time() - t0:.0f}s')
    return df, is_quantile, nstates


def derive_variants(states_df, is_quantile, nstates=3, prior_path=PRIOR_PATH):
    """Build the normal-fixed and two marginalised prediction tables."""
    priors = load_state_priors(prior_path)
    base_cols = ['pid', 'cid', 'code', 'x_next', 't_next', 's_next', 's_last', 'n_hist', 'split']
    base = states_df[base_cols].copy()

    def from_state(q):
        d = base.copy()
        d['mu'] = states_df[f'mu_{q}'].values
        d['log_var'] = states_df[f'log_var_{q}'].values
        if is_quantile:
            for col in QUANTILE_COLS:
                d[col] = states_df[f'{col}_{q}'].values
        return d

    out = {'normal': from_state(1)}

    mu = np.stack([states_df[f'mu_{q}'].values for q in range(nstates)], 1)
    lv = np.stack([states_df[f'log_var_{q}'].values for q in range(nstates)], 1)
    if is_quantile:
        qarr = np.stack([np.stack([states_df[f'{c}_{q}'].values for c in QUANTILE_COLS], 1)
                         for q in range(nstates)], 1)                       # (N, S, 5)
    for name, kind in [('marginal', 'transition'), ('marginal_freq', 'marginal')]:
        w = prior_weights(priors, states_df['cid'].values, states_df['s_last'].values, kind=kind)
        t0 = time.time()
        mix = mix_quantiles_batched(qarr, w) if is_quantile else mix_gaussian_batched(mu, lv, w)
        d = base.copy()
        for k, v in mix.items():
            d[k] = v
        for q in range(nstates):
            d[f'p_{STATE_NAMES[q]}'] = w[:, q]
        out[name] = d
        print(f'  {name}: mixed {len(d):,} rows in {time.time() - t0:.0f}s')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_id', required=True)
    p.add_argument('--source', default='combined')
    p.add_argument('--split', default='test')
    p.add_argument('--max_sequences', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--derive_only', action='store_true')
    p.add_argument('--suffix', default='', help='tag appended to output filenames (smoke tests)')
    p.add_argument('--checkpoint', default='auto', choices=['auto', 'latest', 'best'],
                   help='auto = the weights behind the published predictions_combined.csv')
    args = p.parse_args()

    if args.checkpoint == 'auto':
        args.checkpoint = PUBLISHED_CHECKPOINT.get(args.run_id, 'latest')
    out_dir = os.path.join(HERE, 'logs', args.run_id)
    tag = f'{args.source}{args.suffix}'
    states_path = os.path.join(out_dir, f'predictions_states_{tag}.csv')

    if args.derive_only:
        states_df = pd.read_csv(states_path)
        is_quantile = 'q50_0' in states_df.columns
        nstates = sum(c.startswith('mu_') for c in states_df.columns)
    else:
        states_df, is_quantile, nstates = run_all_states(
            args.run_id, split=args.split, source=args.source,
            max_sequences=args.max_sequences, batch_size=args.batch_size,
            checkpoint=args.checkpoint)
        states_df.to_csv(states_path, index=False)
        print(f'wrote {states_path}')

    variants = derive_variants(states_df, is_quantile, nstates)
    for name, d in variants.items():
        path = os.path.join(out_dir, f'predictions_{tag}_{name}.csv')
        d.to_csv(path, index=False)
        print(f'wrote {path}')

    # quick look: test-split MAE by variant and by realized state
    pc = 'q50' if is_quantile else 'mu'
    oracle = states_df.copy()
    oracle[pc] = np.select([oracle['s_next'] == q for q in range(nstates)],
                           [oracle[f'{pc}_{q}'] for q in range(nstates)])
    variants = {'oracle': oracle, **variants}
    print('\nMAE (median across analytes) by variant and realized state:')
    lines = []
    for name, d in variants.items():
        d = d[~d['code'].isin({'CRP', 'GGT', 'LDH', 'PT'})]
        err = (d['x_next'] - d[pc]).abs()
        by = d.assign(err=err).groupby(['code', 's_next'])['err'].mean().unstack()
        row = {'variant': name, 'all': d.assign(err=err).groupby('code')['err'].mean().median()}
        for q in range(nstates):
            row[STATE_NAMES[q]] = by[q].median() if q in by else np.nan
        lines.append(row)
    print(pd.DataFrame(lines).round(3).to_string(index=False))
