#!/usr/bin/env python
"""Post-hoc Bayesian shrinkage of NORMA's interval toward the population prior,
and the real-data diagnostic behind it: coverage and width by history length.

For every prediction row the blended quantiles are

    q_blend = w * q_norma + (1 - w) * q_prior,   w = n / (n + k)

with n the number of target-analyte observations in the history and q_prior the
state-conditional population quantiles (priors.py: the sex-specific Pop_RI read as
a normal distribution for normal-state queries, the dev-cohort empirical
distribution for low/high). k is chosen on the VAL split so the 95% interval of
normal-state queries covers 95%; the test split is then reported by history-length
bin for the raw model, the blend, and a conformal comparator that multiplies every
interval by one constant fitted the same way (what a constant widening can and
cannot fix).

Runs without n_hist in predictions_combined.csv (arms trained before 2026-09-08)
get it from the sequence pickle by (pid, cid, t_next, x_next); sex comes from the
same lookup.

Outputs (logs/prior_ablation/<run>/):
    shrinkage_grid.csv     val coverage / width for every k
    by_nhist.csv           test metrics x history bin x method x state
    summary.csv            one row per method (test, normal-state and all)
    by_nhist.pdf           coverage and relative width vs history length

    python shrinkage.py --runs q_age_set 334f7e21
    python shrinkage.py --runs pa_k5 --k 0          # diagnostic only, no blend
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from evaluate import EXCLUDE_CODES  # noqa: E402
from data import TEST_VOCAB  # noqa: E402
from process.config import REFERENCE_INTERVALS  # noqa: E402

QCOLS = ['q025', 'q25', 'q50', 'q75', 'q975']
LEVELS = np.array([0.025, 0.25, 0.50, 0.75, 0.975])
BINS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]
BIN_LABELS = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50+']
K_GRID = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 50, 100]
CODE_OF = {i: n for n, i in TEST_VOCAB.items()}


def load_sequences(data_dir, version):
    with open(os.path.join(data_dir, f'combined_sequences_{version}.pkl'), 'rb') as f:
        return pickle.load(f)


def attach_history(df, sequences):
    """n_hist and sex for prediction rows, keyed on (pid, cid, t_next, x_next)."""
    key = {}
    for s in sequences:
        x, t = np.asarray(s['x']), np.asarray(s['t'])
        sex = 1 if (s['sex'] == 'F' or s['sex'] == 1) else 0
        key[(str(s['pid']), int(s['cid']), round(float(t[-1]), 2), round(float(x[-1]), 3))] = (len(x) - 1, sex)
    k = list(zip(df['pid'].astype(str), df['cid'].astype(int), df['t_next'].round(2), df['x_next'].round(3)))
    hit = [key.get(kk, (np.nan, np.nan)) for kk in k]
    df = df.copy()
    df['n_hist'] = [h[0] for h in hit]
    df['sex'] = [h[1] for h in hit]
    miss = df['n_hist'].isna().mean()
    if miss > 0.01:
        print(f'  warning: {miss:.1%} of rows did not match a sequence')
    return df.dropna(subset=['n_hist'])


def prior_quantiles(df, table=None):
    """(N, 5) state-conditional prior quantiles per row."""
    out = np.zeros((len(df), 5))
    z = norm.ppf(LEVELS)
    if table is not None:
        q = table['q'].numpy()
        out[:] = q[df['cid'].astype(int).values, df['s_next'].astype(int).values, df['sex'].astype(int).values]
        return out
    for i, (cid, sex) in enumerate(zip(df['cid'].astype(int), df['sex'].astype(int))):
        lo, hi, _ = REFERENCE_INTERVALS[CODE_OF[cid]]['F' if sex == 1 else 'M']
        m, sd = 0.5 * (lo + hi), (hi - lo) / 3.92
        out[i] = m + z * sd
    return out


def pop_width(df):
    w = np.zeros(len(df))
    for i, (cid, sex) in enumerate(zip(df['cid'].astype(int), df['sex'].astype(int))):
        lo, hi, _ = REFERENCE_INTERVALS[CODE_OF[cid]]['F' if sex == 1 else 'M']
        w[i] = hi - lo
    return w


def blend(q, q_prior, n, k):
    if k == 0:
        return q
    w = (n / (n + k))[:, None]
    return w * q + (1 - w) * q_prior


def conformal(q, c):
    lo = q[:, 2] - c * (q[:, 2] - q[:, 0])
    hi = q[:, 2] + c * (q[:, 4] - q[:, 2])
    out = q.copy(); out[:, 0], out[:, 4] = lo, hi
    return out


def metrics(q, y, popw):
    inside = (y >= q[:, 0]) & (y <= q[:, 4])
    width = q[:, 4] - q[:, 0]
    return {'coverage95': float(inside.mean()), 'width_rel': float(np.median(width / popw)),
            'mae': float(np.abs(y - q[:, 2]).mean()), 'n': int(len(y))}


def fit_k(qv, qpv, nv, yv, popv, normal_mask, target=0.95):
    rows = []
    for k in K_GRID:
        m = metrics(blend(qv, qpv, nv, k)[normal_mask], yv[normal_mask], popv[normal_mask])
        rows.append({'k': k, **m})
    grid = pd.DataFrame(rows)
    best = grid.iloc[(grid['coverage95'] - target).abs().argmin()]
    return float(best['k']), grid


def fit_conformal(qv, yv, normal_mask, target=0.95):
    cs = np.linspace(1, 6, 101)
    cov = [metrics(conformal(qv, c)[normal_mask], yv[normal_mask], np.ones(normal_mask.sum()))['coverage95'] for c in cs]
    return float(cs[int(np.argmin(np.abs(np.array(cov) - target)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--log_dir', default=os.path.join(HERE, 'logs'))
    ap.add_argument('--data_dir', default=os.path.join(HERE, '..', '..', 'data', 'processed'))
    ap.add_argument('--data_version', default='v3')
    ap.add_argument('--k', type=float, default=None, help='fixed k instead of fitting on val (0 = raw only)')
    ap.add_argument('--state_prior', action='store_true',
                    help='abnormal-state prior from the dev train split (priors.build_prior_table); '
                         'default: Pop_RI prior for every state')
    ap.add_argument('--out', default=os.path.join(HERE, 'logs', 'prior_ablation'))
    args = ap.parse_args()

    sequences = None
    table = None
    for run in args.runs:
        p = os.path.join(args.log_dir, run, 'predictions_combined.csv')
        if not os.path.exists(p):
            print(f'{run}: no predictions_combined.csv, skipped'); continue
        df = pd.read_csv(p, keep_default_na=False, na_values=[''])
        df = df[df['split'].isin(['val', 'test']) & ~df['code'].isin(EXCLUDE_CODES)]
        if 'n_hist' not in df.columns or 'sex' not in df.columns:
            if sequences is None:
                print('loading sequences for n_hist / sex lookup ...')
                sequences = load_sequences(args.data_dir, args.data_version)
            df = attach_history(df, sequences)
        if args.state_prior and table is None:
            from data import load_and_split_data
            from priors import build_prior_table
            tr, _, _ = load_and_split_data(args.data_dir, 'combined', nstates=3, version=args.data_version, print_info=False)
            table = build_prior_table(tr, nstates=3)
        out_dir = os.path.join(args.out, run); os.makedirs(out_dir, exist_ok=True)

        q_all = df[QCOLS].values.astype(float)
        qp_all = prior_quantiles(df, table)
        n_all = df['n_hist'].values.astype(float)
        y_all = df['x_next'].values.astype(float)
        pop_all = pop_width(df)
        normal_all = (df['s_next'].astype(int) == 1).values
        val = (df['split'] == 'val').values
        test = (df['split'] == 'test').values

        if args.k is None:
            k_star, grid = fit_k(q_all[val], qp_all[val], n_all[val], y_all[val], pop_all[val], normal_all[val])
            grid.to_csv(os.path.join(out_dir, 'shrinkage_grid.csv'), index=False)
        else:
            k_star = args.k
        c_star = fit_conformal(q_all[val], y_all[val], normal_all[val])
        print(f'{run}: k* = {k_star} (val), conformal factor = {c_star:.2f}')

        methods = {'raw': q_all, f'blend_k{k_star:g}': blend(q_all, qp_all, n_all, k_star),
                   f'conformal_x{c_star:.2f}': conformal(q_all, c_star)}
        rows, summ = [], []
        for name, q in methods.items():
            for state_name, smask in (('normal', normal_all), ('all', np.ones(len(df), bool))):
                base = test & smask
                summ.append({'method': name, 'state': state_name, **metrics(q[base], y_all[base], pop_all[base])})
                for (lo, hi), lab in zip(BINS, BIN_LABELS):
                    m = base & (n_all >= lo) & (n_all <= hi)
                    if m.sum() < 50:
                        continue
                    rows.append({'method': name, 'state': state_name, 'n_hist': lab,
                                 **metrics(q[m], y_all[m], pop_all[m])})
        by = pd.DataFrame(rows); by.to_csv(os.path.join(out_dir, 'by_nhist.csv'), index=False)
        pd.DataFrame(summ).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
        print(pd.DataFrame(summ).to_string(index=False))
        plot(by, run, os.path.join(out_dir, 'by_nhist.pdf'))


def plot(by, run, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = by[by['state'] == 'normal']
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for name, g in d.groupby('method', sort=False):
        g = g.set_index('n_hist').reindex(BIN_LABELS)
        axes[0].plot(BIN_LABELS, g['coverage95'], marker='o', label=name)
        axes[1].plot(BIN_LABELS, g['width_rel'], marker='o', label=name)
    axes[0].axhline(0.95, color='grey', lw=0.8, ls='--')
    axes[1].axhline(1.0, color='grey', lw=0.8, ls='--')
    axes[0].set_ylabel('Coverage of 95% interval'); axes[1].set_ylabel('Width / Pop RI width')
    for ax in axes:
        ax.set_xlabel('History length (observations)')
    fig.suptitle(f'{run}, test split, normal-state queries', fontsize=10)
    axes[0].legend(loc='lower center', bbox_to_anchor=(1.1, 1.02), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    main()
