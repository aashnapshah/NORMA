"""Run a NORMA checkpoint over a dataloader and build a prediction frame.

Two entry points, one pass over the batch:

  get_predictions()     query token = the realized future state, one row per
                        (patient, analyte, draw). What train.py writes as
                        predictions_{source}.csv after the last epoch.
  predict_all_states()  query token set to EVERY state, one column block per
                        state. Referee 3 (minor 1) observed that the oracle
                        query above uses a state the baselines never see;
                        derive_variants() turns this into the leak-free
                        variants (normal-fixed and two marginalisations).

Was predict.py and predict_states.py, which carried two implementations of the
same batch-to-frame loop -- the first row by row through a per-scalar
`float(x[i].item() if hasattr(x[i], "item") else x[i])` idiom repeated thirteen
times, the second vectorised.

Usage:
    python inference.py --run_id q_age_set                  # all states + variants
    python inference.py --run_id q_age_set --derive_only
    python inference.py --run_id q_age_set --max_sequences 20000   # smoke test
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from data import (TEST_VOCAB, CODE_TO_TEST_NAME, TimeSeriesDataset, collate_fn,
                  load_and_split_data, load_panel)
from model import is_quantile_mode
from states import (QUANTILE_COLS, STATE_NAMES, PUBLISHED_CHECKPOINT,
                    DEFAULT_PATH as PRIOR_PATH, load_state_priors, prior_weights,
                    mix_gaussian_batched, mix_quantiles_batched)
from utils import create_model, load_checkpoint, run_model, to_device_batch

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def head_outputs(model, batch, is_quantile, s_next=None, span=None, ref_low=None,
                 denorm_quantiles=True):
    """Model outputs for one queried state, as flat numpy arrays.

    s_next=None queries the realized state carried in the batch. span/ref_low
    denormalise back to the analyte's units when the model was trained with
    --normalize.

    denorm_quantiles exists only to preserve a divergence between the two
    callers this module merged: get_predictions() denormalised the Gaussian
    head but not the quantile head, while the all-states path denormalised
    both. Every run in model/logs/ has normalize=False, so the two agree on
    every published number; kept explicit rather than silently picking one.
    """
    out = run_model(model, batch, s_next=s_next)
    if is_quantile:
        arr = out.cpu().numpy().astype(float)                    # (B, 5)
        if span is not None and denorm_quantiles:
            arr = arr * span[:, None] + ref_low[:, None]
        # sigma ~ (q97.5 - q2.5) / 3.92, the normal-quantile width.
        # Computed in float64 (arr is widened above). get_predictions used to do
        # this in the tensor's float32, so a regenerated predictions_*.csv has
        # log_var differing by <5e-7 -- a 2e-7 relative change in sigma. mu, the
        # five quantiles and x_next are bit-identical: they involve no arithmetic.
        res = {'mu': arr[:, 2],
               'log_var': 2.0 * np.log((arr[:, 4] - arr[:, 0]) / 3.92 + 1e-8)}
        res.update({col: arr[:, k] for k, col in enumerate(QUANTILE_COLS)})
        return res
    mu, log_var = out
    mu = mu.view(-1).cpu().numpy().astype(float)
    log_var = log_var.view(-1).cpu().numpy().astype(float)
    if span is not None:
        mu = mu * span + ref_low
        log_var = log_var + 2.0 * np.log(span + 1e-8)
    return {'mu': mu, 'log_var': log_var}


def _denorm_factors(batch, normalize):
    if not normalize:
        return None, None
    ref_low = batch['ref_low'].view(-1).cpu().numpy()
    return batch['ref_high'].view(-1).cpu().numpy() - ref_low, ref_low


# ─────────────────────────────────────────────────────────────────────────────
# Oracle query: the realized future state (training-time predictions)
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, device, loader, split_name, normalize=False):
    """One row per query, with the query token at the realized future state."""
    model.eval()
    model.to(device)
    is_quantile = hasattr(model, 'output_mode') and is_quantile_mode(model.output_mode)

    frames = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split_name} (predict)", leave=False):
            batch = to_device_batch(batch, device)
            span, ref_low = _denorm_factors(batch, normalize)
            # the Gaussian head is denormalised here, the quantile head is not
            out = head_outputs(model, batch, is_quantile, span=span, ref_low=ref_low,
                               denorm_quantiles=False)

            cid = batch['cid'].view(-1).cpu().numpy()
            x_next = batch['x_next'].view(-1).cpu().numpy().astype(float)
            if span is not None:
                x_next = x_next * span + ref_low
            rec = {
                'pid': list(batch['pids']),
                'cid': cid.astype(int),
                'code': [CODE_TO_TEST_NAME[int(c)] for c in cid],
                'x_next': x_next,
                't_next': batch['t_next'].view(-1).cpu().numpy().astype(float),
                's_next': batch['s_next'].view(-1).cpu().numpy().astype(int),
                'mu': out['mu'],
                'log_var': out['log_var'],
            }
            if 'n_hist' in batch:
                rec['n_hist'] = batch['n_hist'].view(-1).cpu().numpy().astype(int)
            lp = getattr(model, 'last_params', None) if is_quantile else None
            for key in ('gate', 'n_eff'):
                if lp and key in lp:
                    rec[key] = lp[key].detach().view(-1).cpu().numpy().astype(float)
            if is_quantile:
                rec.update({col: out[col] for col in QUANTILE_COLS})
            frames.append(pd.DataFrame(rec))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predict(model, device, train_loader, val_loader, test_loader, normalize=False):
    print('Generating predictions and computing metrics...')
    all_predictions = []
    for split_name, loader in [('train', train_loader), ('val', val_loader),
                               ('test', test_loader)]:
        print(f"Evaluating {split_name} set...")
        predictions_df = get_predictions(model, device, loader, split_name, normalize=normalize)
        predictions_df['split'] = split_name
        all_predictions.append(predictions_df)
    return pd.concat(all_predictions)


def load_predictions(run_ids, base, source):
    """(path, target col, prediction col) per run_id, for interactive comparison.

    `base` is a cohort key for the forecasting baselines: they live in
    results/raw/<cohort>/05_forecast_baselines.parquet, one column per model
    (model/baselines/forecast.py)."""
    outputs = {}
    for run_id in run_ids:
        if run_id in ['Mean', 'ARIMA', 'last']:
            path = f'../results/raw/{base}/05_forecast_baselines.parquet'
            outputs[run_id] = (path, 'x_next', run_id.lower())
        elif run_id == '58ba1f1c':
            path = f'../model/logs/{run_id}/predictions_ehrshot.csv'
            outputs[run_id] = (path, 'x_next', 'q50')
        else:
            log_path = f'../model/logs/{run_id}/predictions_{source}.csv'
            pred_path = f'../model/predictions/{run_id}/predictions_{run_id}.csv'
            path = log_path if os.path.exists(log_path) else pred_path
            outputs[run_id] = (path, 'x_next', 'mu')
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Every-state query: the leak-free variants
# ─────────────────────────────────────────────────────────────────────────────

class LengthBucketSampler(Sampler):
    """Batch sequences of similar length together so padding stays cheap."""

    def __init__(self, lengths, batch_size):
        order = np.argsort(lengths, kind='stable')
        self.batches = [order[i:i + batch_size].tolist()
                        for i in range(0, len(order), batch_size)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def predict_all_states(run_id, split='test', source='combined', max_sequences=None,
                       batch_size=1024, device=None, seed=42, log_dir=None,
                       checkpoint='latest'):
    """Query the model at every state; returns (frame, is_quantile, nstates)."""
    log_dir = log_dir or os.path.join(HERE, 'logs')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the weights behind the published predictions file (see
    # states.PUBLISHED_CHECKPOINT) so the oracle variant reproduces it.
    ckpt, hp = load_checkpoint(log_dir, run_id, best=(checkpoint == 'best'),
                               device='cpu', quiet=True)
    print(f'{run_id}: checkpoint_{checkpoint} (epoch {ckpt.get("epoch")})')
    model = create_model(hp, ncodes=len(TEST_VOCAB), checkpoint=ckpt).to(device).eval()
    is_quantile = (is_quantile_mode(getattr(hp, 'output_mode', 'gaussian'))
                   and getattr(hp, 'model', '') == 'NORMA2')
    nstates = getattr(hp, 'nstates', 3)
    normalize = bool(getattr(hp, 'normalize', False))

    data_dir = os.path.join(ROOTDIR, '..', 'data', 'processed')
    version = getattr(hp, 'data_version', 'v2')
    train_seq, val_seq, test_seq = load_and_split_data(
        data_dir, source, print_info=False, nstates=nstates, version=version,
        split_by=getattr(hp, 'split_by', 'sequence'))
    panel = load_panel(data_dir, source, version) if getattr(hp, 'use_coanalytes', False) else None
    seqs = {'train': train_seq, 'val': val_seq, 'test': test_seq}[split]
    if max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(seqs), size=min(max_sequences, len(seqs)), replace=False)
        seqs = [seqs[i] for i in idx]
    print(f'{run_id}: {"quantile" if is_quantile else "gaussian"} head, '
          f'{len(seqs):,} {split} sequences, device={device}')

    ds = TimeSeriesDataset(seqs, nstates, normalize=normalize, panel=panel)
    lengths = np.array([len(s['x']) - 1 for s in seqs])
    loader = DataLoader(ds, batch_sampler=LengthBucketSampler(lengths, batch_size),
                        collate_fn=collate_fn, num_workers=0)

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = to_device_batch(batch, device)
            rec = {
                'pid': list(batch['pids']),
                'cid': batch['cid'].view(-1).cpu().numpy(),
                'x_next': batch['x_next'].view(-1).cpu().numpy().astype(float),
                't_next': batch['t_next'].view(-1).cpu().numpy().astype(float),
                's_next': batch['s_next'].view(-1).cpu().numpy(),
                # last history state: pad_mask is False on real positions
                's_last': batch['s_h'].gather(
                    1, ((~batch['pad_mask']).sum(1) - 1).clamp(min=0)[:, None]
                ).view(-1).cpu().numpy(),
                'n_hist': (~batch['pad_mask']).sum(1).cpu().numpy(),
            }
            span, ref_low = _denorm_factors(batch, normalize)
            if span is not None:
                rec['x_next'] = rec['x_next'] * span + ref_low
            for q in range(nstates):
                s_q = torch.full_like(batch['s_next'], q)
                out = head_outputs(model, batch, is_quantile, s_next=s_q,
                                   span=span, ref_low=ref_low)
                for key, val in out.items():
                    rec[f'{key}_{q}'] = val
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
        states_df, is_quantile, nstates = predict_all_states(
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
