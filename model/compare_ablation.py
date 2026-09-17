#!/usr/bin/env python
"""Compare the covariate-ablation arms against the quantile baseline.

Usage:
    python compare_ablation.py                       # all arms present
    python compare_ablation.py --runs q_age q_age_set
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from evaluate import calibration_by_analyte, EXCLUDE_CODES  # noqa: E402
from run_names import arm_label  # noqa: E402

BASELINE = '334f7e21'
ARMS = {  # display order; label; covariates
    '334f7e21': (arm_label('334f7e21'), 'none'),
    'q_age': (arm_label('q_age'), 'age at draw'),
    'q_set': (arm_label('q_set'), 'care setting'),
    'q_co': (arm_label('q_co'), 'co-analytes'),
    'q_age_set': (arm_label('q_age_set'), 'age, setting'),
    'q_age_co': (arm_label('q_age_co'), 'age, co-analytes'),
    'q_set_co': (arm_label('q_set_co'), 'setting, co-analytes'),
    'q_age_set_co': (arm_label('q_age_set_co'), 'age, setting, co-analytes'),
    'q_co_q': (arm_label('q_co_q'), 'co-analytes (query-aware)'),
    # prior-anchored arms (run_prior_ablation.sh); compare with q_age_set
    'pa_k5': (arm_label('pa_k5'), 'age, setting + prior anchor k=5'),
    'pa_k20': (arm_label('pa_k20'), 'age, setting + prior anchor k=20'),
    'pa_tau': (arm_label('pa_tau'), 'age, setting + prior anchor k=5 tau=365'),
    'pf_k5': (arm_label('pf_k5'), 'age, setting + width floor k=5'),
    'pg_k5': (arm_label('pg_k5'), 'age, setting + learned gate k=5'),
    'pn_k5': (arm_label('pn_k5'), 'age, setting + NIG head'),
    'gk_k5': (arm_label('gk_k5'), 'age, setting + Gaussian KL k=5'),
}


def load_run(run_id, log_dir):
    p = os.path.join(log_dir, run_id, 'predictions_combined.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, keep_default_na=False, na_values=[''])
    return df[df['split'] == 'test']


def point_metrics(df, exclude):
    rows = []
    for code, g in df[~df['code'].isin(exclude)].groupby('code'):
        err = g['x_next'] - g['q50']
        ss_res = float((err ** 2).sum())
        ss_tot = float(((g['x_next'] - g['x_next'].mean()) ** 2).sum())
        rows.append({'code': code, 'n': len(g),
                     'MAE': float(err.abs().mean()),
                     'R2': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


def arm_table(df, exclude):
    pm = point_metrics(df, exclude)
    cal = calibration_by_analyte(df, is_quantile=True, exclude=exclude, split='test')
    cal_all = (cal.assign(w=cal['n'])
               .groupby('code').apply(lambda g: pd.Series({
                   'coverage95': np.average(g['coverage95'], weights=g['n']),
                   'width_rel': np.average(g['width_rel'], weights=g['n'])})).reset_index())
    cal_normal = (cal[cal['state'] == 'normal'][['code', 'coverage95', 'width_rel', 'inside_pop']]
                  .rename(columns={'coverage95': 'coverage95_normal', 'width_rel': 'width_rel_normal',
                                   'inside_pop': 'inside_pop_normal'}))
    return pm.merge(cal_all, on='code', how='left').merge(cal_normal, on='code', how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=list(ARMS.keys()))
    ap.add_argument('--log_dir', default=os.path.join(HERE, 'logs'))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'logs', 'ablation'))
    ap.add_argument('--no_wandb', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    exclude = set(EXCLUDE_CODES)

    tables = {}
    for rid in args.runs:
        df = load_run(rid, args.log_dir)
        if df is None:
            print(f'  {rid}: no predictions yet, skipped')
            continue
        t = arm_table(df, exclude)
        t.insert(0, 'run', rid)
        t.insert(1, 'arm', ARMS.get(rid, (rid, ''))[0])
        tables[rid] = t
        print(f'  {rid:14s} {ARMS.get(rid, (rid,))[0]:32s} n_test={len(df):,}')
    if BASELINE not in tables:
        sys.exit(f'baseline {BASELINE} predictions missing')

    by_analyte = pd.concat(tables.values(), ignore_index=True)
    base = tables[BASELINE].set_index('code')
    metric_cols = ['MAE', 'R2', 'coverage95', 'width_rel', 'coverage95_normal', 'width_rel_normal', 'inside_pop_normal']
    for m in metric_cols:
        by_analyte[f'd_{m}'] = by_analyte[m].values - by_analyte['code'].map(base[m]).values
    by_analyte['d_MAE_pct'] = 100 * by_analyte['d_MAE'] / by_analyte['code'].map(base['MAE']).values
    by_analyte.to_csv(os.path.join(args.out_dir, 'by_analyte.csv'), index=False)

    summ = []
    for rid, t in tables.items():
        r = {'run': rid, 'arm': ARMS.get(rid, (rid, ''))[0], 'covariates': ARMS.get(rid, ('', ''))[1],
             'n_analytes': len(t)}
        for m in metric_cols:
            r[f'{m}_median'] = float(t[m].median())
        sub = by_analyte[by_analyte['run'] == rid]
        r['MAE_median_delta_pct'] = float(sub['d_MAE_pct'].median())
        r['n_analytes_MAE_improved'] = int((sub['d_MAE'] < 0).sum())
        r['coverage95_normal_abs_err'] = float((t['coverage95_normal'] - 0.95).abs().median())
        summ.append(r)
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n' + summary[['arm', 'MAE_median', 'MAE_median_delta_pct', 'n_analytes_MAE_improved', 'R2_median',
                          'coverage95_normal_median', 'width_rel_normal_median', 'inside_pop_normal_median']]
          .round(3).to_string(index=False))

    # ---- figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = {}
    order = [r for r in ARMS if r in tables]
    labels = [ARMS[r][0] for r in order]
    for m, title in [('MAE_median', 'median MAE across analytes (test)'),
                     ('coverage95_normal_median', '95% interval coverage, normal-conditioned (nominal 0.95)'),
                     ('width_rel_normal_median', 'interval width / population RI width, normal-conditioned')]:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        vals = summary.set_index('run').loc[order, m]
        ax.bar(range(len(order)), vals, color=['#888'] + ['#3b6fb6'] * (len(order) - 1))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        if 'coverage' in m:
            ax.axhline(0.95, ls='--', c='k', lw=0.8)
        fig.tight_layout()
        p = os.path.join(args.out_dir, f'{m}.png')
        fig.savefig(p, dpi=150)
        figs[m] = p
        plt.close(fig)

    piv = by_analyte[by_analyte['run'] != BASELINE].pivot(index='code', columns='run', values='d_MAE_pct')
    piv = piv[[r for r in order if r != BASELINE]]
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * piv.shape[1], 0.28 * len(piv) + 1))
    lim = np.nanmax(np.abs(piv.values)) if piv.size else 1
    im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([ARMS[r][0] for r in piv.columns], rotation=25, ha='right', fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label='ΔMAE vs baseline (%)  (blue = better)')
    ax.set_title('per-analyte change in test MAE', fontsize=9)
    fig.tight_layout()
    p = os.path.join(args.out_dir, 'delta_mae_heatmap.png')
    fig.savefig(p, dpi=150)
    figs['delta_mae_heatmap'] = p
    plt.close(fig)
    print(f'\nsaved summary.csv, by_analyte.csv and {len(figs)} figures to {args.out_dir}')

    # ---- W&B ----
    if not args.no_wandb:
        import wandb
        run = wandb.init(project='NORMA', group='covariate-ablation', job_type='summary',
                         name='ablation-summary', id='ablation-summary', resume='allow',
                         config={'runs': list(tables.keys()), 'baseline': BASELINE})
        run.log({'ablation/summary': wandb.Table(dataframe=summary),
                 'ablation/by_analyte': wandb.Table(dataframe=by_analyte),
                 **{f'ablation/{k}': wandb.Image(v) for k, v in figs.items()}})
        for _, r in summary.iterrows():
            for m in ['MAE_median', 'R2_median', 'coverage95_normal_median', 'width_rel_normal_median',
                      'MAE_median_delta_pct', 'n_analytes_MAE_improved']:
                run.summary[f"{r['run']}/{m}"] = r[m]
        run.finish()
        print('logged to W&B run ablation-summary (group covariate-ablation)')


if __name__ == '__main__':
    main()
