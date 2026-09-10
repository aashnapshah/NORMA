#!/usr/bin/env python
"""Index every NORMA training run and put them side by side.

The experiments live in three places, on three different schemas:

  model/wandb/run-*/          every run that ever logged, including the sweep
                              era. config.yaml carries the hyperparameters and
                              wandb-summary.json the last epoch's losses, both
                              readable without the wandb package or the network.
  model/logs/<run>/           checkpoint_{latest,best}.json for the runs whose
                              weights were saved: hyperparameters plus the
                              train/val loss, r2, mae, coverage and width at the
                              best epoch.
  model/logs/<run>/bootstrap_metrics[_by_code].csv
                              held-out test metrics with bootstrap CIs, overall
                              and per analyte, for the runs that were evaluated.

Two config schemas coexist: the sweep era wrote model_type / loss_type /
num_layers and no run_id, later runs write model / loss / nlayers / run_id.
Both are normalised here.

Outputs (--out_dir, default model/logs/experiments/):

  experiments.csv             one row per run: identity, architecture, loss,
                              features, val loss, and every test metric
  experiments_by_analyte.csv  one row per (run, analyte, metric)
  val_loss.pdf                val loss per run, grouped by loss function
  test_metrics.pdf            one panel per test metric, runs ranked, with CIs
  by_analyte.pdf              run x analyte heatmap, one panel per metric

    python experiments.py                    # build the tables and figures
    python experiments.py --wandb            # also push them to W&B
    python experiments.py --min_epochs 5     # drop runs that died early
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from data import CODE_TO_TEST_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, 'logs')
WANDB_DIR = os.path.join(HERE, 'wandb')

# One name per concept, whatever the run called it.
ALIASES = {
    'model': ('model', 'model_type'),
    'loss': ('loss', 'loss_type'),
    'nlayers': ('nlayers', 'num_layers'),
    'nhead': ('nhead',),
    'd_model': ('d_model',),
    'nstates': ('nstates', 'num_states'),
    'output_mode': ('output_mode',),
    'lr': ('lr', 'learning_rate'),
    'batch_size': ('batch_size',),
    'epochs': ('epochs',),
    'normalize': ('normalize',),
    'split_by': ('split_by',),
    'data_version': ('data_version',),
    'train_source': ('train',),
    'lambda_align': ('lambda_align',),
    'prior_k': ('prior_k',),
    'prior_lambda': ('prior_lambda',),
    'prior_mode': ('prior_mode',),
    'prior_tau': ('prior_tau',),
    'nig_nu0': ('nig_nu0',),
    'run_id': ('run_id',),
    'wandb_name': ('wandb_name',),
    'description': ('description',),
}

# The per-measurement covariates and attention variants, in the order the
# ablation scripts add them.
FEATURE_FLAGS = ['use_age_t', 'use_setting', 'use_coanalytes', 'query_coanalytes',
                 'causal_memory', 'use_full_panel']
FEATURE_SHORT = {'use_age_t': 'age', 'use_setting': 'setting', 'use_coanalytes': 'co',
                 'query_coanalytes': 'co-query', 'causal_memory': 'causal',
                 'use_full_panel': 'full-panel'}


def _norm_config(raw):
    """Flatten a wandb config (values may be wrapped in {'value': ...}) and map
    both schema versions onto the ALIASES names."""
    flat = {}
    for k, v in (raw or {}).items():
        if k.startswith('_'):
            continue
        flat[k] = v.get('value') if isinstance(v, dict) and 'value' in v else v
    out = {}
    for name, keys in ALIASES.items():
        for k in keys:
            if k in flat and flat[k] is not None:
                out[name] = flat[k]
                break
    for f in FEATURE_FLAGS:
        out[f] = bool(flat.get(f, False))
    return out


def _features(row):
    on = [FEATURE_SHORT[f] for f in FEATURE_FLAGS if row.get(f)]
    return ', '.join(on) if on else 'none'


def from_wandb(wandb_dir=WANDB_DIR):
    """One row per local W&B trace: config plus the last epoch's losses."""
    import yaml
    rows = []
    # offline-run-* too: a run started with WANDB_MODE=offline never synced,
    # and globbing 'run-*' alone silently drops it. The 2026-09-08
    # prior-anchored smoke tests are all offline, and they are the only
    # runs that exercise NORMALoss and StudentTNLLLoss.
    traces = (glob.glob(os.path.join(wandb_dir, 'run-*'))
              + glob.glob(os.path.join(wandb_dir, 'offline-run-*')))
    skipped = []
    for d in sorted(traces):
        files = os.path.join(d, 'files')
        cp, sp = os.path.join(files, 'config.yaml'), os.path.join(files, 'wandb-summary.json')
        if not os.path.exists(cp):
            # A run started with WANDB_MODE=offline keeps everything in its
            # binary .wandb file and only writes config.yaml on sync, so it
            # cannot be read here. Reported rather than dropped in silence:
            #   wandb sync model/wandb/offline-run-*
            skipped.append(os.path.basename(d))
            continue
        try:
            cfg = _norm_config(yaml.safe_load(open(cp)))
        except Exception:
            continue
        summary = {}
        if os.path.exists(sp):
            try:
                summary = json.load(open(sp))
            except Exception:
                summary = {}
        trace = os.path.basename(d)
        rows.append({
            'run': cfg.get('run_id') or cfg.get('wandb_name') or trace.rsplit('-', 1)[-1],
            'wandb_trace': trace,
            'wandb_synced': not trace.startswith('offline-run-'),
            'started': trace.split('-')[1] if '-' in trace else None,
            'epochs_ran': summary.get('epoch'),
            'val_loss_last': summary.get('val/loss', summary.get('val_loss')),
            'train_loss_last': summary.get('train/loss', summary.get('train_loss')),
            'val_r2_last': summary.get('val/r2', summary.get('val_r2')),
            'runtime_s': summary.get('_runtime'),
            **{k: v for k, v in cfg.items() if k not in ('run_id', 'wandb_name')},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A resumed run logs a fresh trace each time, so one run_id can have several.
    # Keep the trace that got furthest, and record how many there were.
    if skipped:
        print(f'  {len(skipped)} traces unreadable offline (no config.yaml, never synced):')
        for s in skipped:
            print(f'      {s}')
        print('      sync them with: wandb sync model/wandb/offline-run-*')
    df['n_traces'] = df.groupby('run')['run'].transform('size')
    df = (df.sort_values(['run', 'epochs_ran', 'started'], na_position='first')
            .groupby('run', as_index=False).last())
    return df


def from_checkpoints(log_dir=LOG_DIR):
    """One row per saved checkpoint: config plus the best epoch's metrics."""
    rows = []
    for p in sorted(glob.glob(os.path.join(log_dir, '*', 'checkpoint_best.json'))):
        run = os.path.basename(os.path.dirname(p))
        try:
            d = json.load(open(p))
        except Exception:
            continue
        cfg = _norm_config(d.get('hyperparameters', {}))
        met = d.get('metrics', {}) or {}
        val, train = met.get('val', {}) or {}, met.get('train', {}) or {}
        rows.append({
            'run': run,
            'has_checkpoint': True,
            'best_epoch': d.get('epoch'),
            'val_loss_best': met.get('best_val_loss', val.get('loss')),
            'val_r2_best': val.get('r2'), 'val_mae_best': val.get('mae'),
            'val_coverage95': val.get('coverage95'), 'val_width': val.get('width'),
            'train_loss_best': train.get('loss'),
            **{k: v for k, v in cfg.items() if k not in ('run_id', 'wandb_name')},
        })
    return pd.DataFrame(rows)


# The bootstrap files drifted over time: three header variants for the overall
# table (model/split/metric lower case with CIs; Model/Metric/Split title case
# with only std; and the same three title-case columns in a different order) and
# two for the per-analyte one (analyte vs Code, n vs n_samples).
_BOOT_RENAME = {'model': 'model', 'split': 'split', 'metric': 'metric',
                'code': 'analyte', 'analyte': 'analyte',
                'n_samples': 'n', 'n': 'n', 'mean': 'mean', 'std': 'std',
                'point_estimate': 'point_estimate',
                'ci_lower': 'ci_lower', 'ci_upper': 'ci_upper'}


def _read_boot(path, run):
    """Read a bootstrap file onto one schema, keeping only `run`'s own rows.

    Three things have to be handled:

      * header drift, mapped through _BOOT_RENAME;
      * the older files score the run *and* the forecasting baselines (ARIMA,
        Mean, last) in the same file, so the model column has to be filtered --
        taking the last row per metric silently reports a baseline's number as
        the model's. Newer files label the run's own rows "NORMA";
      * the older files store R2 as a percentage (71.5) where the newer ones
        store a fraction (0.715). Anything above 1.5 is rescaled, since a
        forecasting R2 that high is not otherwise achievable here.

    Returns None if the file has no schema this understands, or if its rows
    cannot be attributed to this run.
    """
    try:
        # keep_default_na=False: the sodium analyte's code is "NA", which the
        # default parser turns into a null, silently dropping that analyte from
        # every per-analyte table and leaving a blank row in the heatmaps.
        d = pd.read_csv(path, keep_default_na=False, na_values=[''])
    except Exception:
        return None
    d = d.rename(columns={c: _BOOT_RENAME.get(c.strip().lower(), c.strip().lower())
                          for c in d.columns})
    for c in ('mean', 'std', 'point_estimate', 'ci_lower', 'ci_upper', 'n'):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    if 'split' not in d.columns or 'metric' not in d.columns:
        return None
    d = d[d['split'].astype(str).str.lower() == 'test']
    if 'model' in d.columns:
        models = set(d['model'].astype(str))
        if run in models:
            pick = run                      # the file names the run explicitly
        elif 'NORMA' in models:
            pick = 'NORMA'                  # newer files label the run this way
        elif len(models) == 1:
            pick = next(iter(models))       # only one model scored; take it
        else:
            return None                     # ambiguous: baselines only, or a
                                            # file written for another run
        d = d[d['model'].astype(str) == pick].copy()
        d['scored_as'] = pick
    if 'ci_lower' not in d.columns and 'std' in d.columns:
        d['ci_lower'] = d['mean'] - 1.96 * d['std']
        d['ci_upper'] = d['mean'] + 1.96 * d['std']
        d['ci_approx'] = True
    r2 = d['metric'].astype(str).str.upper() == 'R2'
    if r2.any():
        scale = d.loc[r2, 'mean'].abs() > 1.5
        for col in ('mean', 'ci_lower', 'ci_upper', 'point_estimate', 'std'):
            if col in d.columns:
                d.loc[r2 & scale, col] = d.loc[r2 & scale, col] / 100.0
    return d


def from_bootstrap(log_dir=LOG_DIR):
    """Test metrics: (overall wide frame, per-analyte long frame)."""
    overall, by_analyte = [], []
    for p in sorted(glob.glob(os.path.join(log_dir, '*', 'bootstrap_metrics.csv'))):
        run = os.path.basename(os.path.dirname(p))
        d = _read_boot(p, run)
        if d is None or d.empty:
            continue
        rec = {'run': run, 'evaluated': True,
               'scored_as': d['scored_as'].iloc[0] if 'scored_as' in d.columns else None}
        for _, r in d.iterrows():
            rec[f"test_{r['metric']}"] = r['mean']
            rec[f"test_{r['metric']}_lo"] = r.get('ci_lower')
            rec[f"test_{r['metric']}_hi"] = r.get('ci_upper')
        overall.append(rec)
    for p in sorted(glob.glob(os.path.join(log_dir, '*', 'bootstrap_metrics_by_code.csv'))):
        run = os.path.basename(os.path.dirname(p))
        d = _read_boot(p, run)
        if d is None or d.empty or 'analyte' not in d.columns:
            continue
        keep = [c for c in ('analyte', 'metric', 'n', 'point_estimate', 'mean',
                            'ci_lower', 'ci_upper') if c in d.columns]
        d = d[keep].copy()
        # Older runs stored the analyte as the integer cid, newer ones as the
        # code; without this the two land on separate rows of every heatmap.
        as_str = d['analyte'].astype(str)
        numeric = as_str.str.fullmatch(r'\d+')
        if numeric.any():
            names = as_str.astype(object)
            names[numeric] = as_str[numeric].astype(int).map(CODE_TO_TEST_NAME)
            d['analyte'] = names
            d = d[d['analyte'].notna() & (d['analyte'].astype(str) != 'nan')]
        d.insert(0, 'run', run)
        by_analyte.append(d)
    return (pd.DataFrame(overall) if overall else pd.DataFrame(columns=['run']),
            pd.concat(by_analyte, ignore_index=True) if by_analyte else pd.DataFrame())


def build(log_dir=LOG_DIR, wandb_dir=WANDB_DIR, min_epochs=0):
    wb, ck = from_wandb(wandb_dir), from_checkpoints(log_dir)
    boot, by_analyte = from_bootstrap(log_dir)

    # A run's own checkpoint config wins over the W&B copy; W&B contributes the
    # runs that never saved weights.
    df = wb.copy()
    if len(ck):
        shared = [c for c in ck.columns if c in df.columns and c != 'run']
        df = df.merge(ck, on='run', how='outer', suffixes=('_wb', ''))
        for c in shared:
            if f'{c}_wb' in df.columns:
                df[c] = df[c].where(df[c].notna(), df[f'{c}_wb'])
                df = df.drop(columns=[f'{c}_wb'])
    if len(boot):
        df = df.merge(boot, on='run', how='left')

    # `== True` rather than fillna(False).astype(bool): these columns arrive as
    # object dtype with NaN for the runs that have no checkpoint or evaluation,
    # and bool(nan) is True.
    for col in ('has_checkpoint', 'evaluated'):
        df[col] = (df[col] == True) if col in df.columns else False   # noqa: E712
    for f in FEATURE_FLAGS:
        df[f] = (df[f] == True) if f in df.columns else False         # noqa: E712
    df['features'] = df.apply(_features, axis=1)
    df['val_loss'] = df['val_loss_best'].where(
        df.get('val_loss_best').notna(), df.get('val_loss_last')) \
        if 'val_loss_best' in df.columns else df.get('val_loss_last')
    if min_epochs:
        ran = df['epochs_ran'].fillna(df.get('best_epoch', 0)).fillna(0)
        df = df[ran >= min_epochs]

    front = ['run', 'model', 'loss', 'output_mode', 'features', 'val_loss',
             'val_loss_best', 'val_loss_last', 'best_epoch', 'epochs_ran',
             'has_checkpoint', 'evaluated']
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols].sort_values('val_loss', na_position='last').reset_index(drop=True), by_analyte


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _label(row):
    bits = [str(row['run'])[:12]]
    if row.get('features') and row['features'] != 'none':
        bits.append(row['features'])
    return '  '.join(bits)


def fig_val_loss(df, path, top=40):
    """Val loss per run, one row of bars per loss function."""
    plt = _style()
    d = df[df['val_loss'].notna()].copy()
    if d.empty:
        return None
    d['loss'] = d['loss'].fillna('unknown')
    groups = [g for g, _ in sorted(d.groupby('loss'), key=lambda kv: len(kv[1]), reverse=True)]
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.1 + 2.0 * len(groups)),
                             squeeze=False)
    for ax, g in zip(axes[:, 0], groups):
        sub = d[d['loss'] == g].nsmallest(top, 'val_loss')
        colors = ['#3b6fb6' if e else '#b0b0b0' for e in sub['evaluated']]
        ax.bar(range(len(sub)), sub['val_loss'], color=colors)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([_label(r) for _, r in sub.iterrows()], rotation=90, fontsize=5)
        ax.set_ylabel('Val loss')
        ax.set_title(f'{g}  ({len(d[d["loss"] == g])} runs, best {top} shown)', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#3b6fb6', '#b0b0b0')]
    fig.legend(handles, ['Evaluated on test', 'Training curve only'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_test_metrics(df, path):
    """One panel per test metric: runs ranked, with bootstrap CIs."""
    plt = _style()
    metrics = [c[5:] for c in df.columns
               if c.startswith('test_') and not c.endswith(('_lo', '_hi'))]
    d = df[df['evaluated']].copy()
    if d.empty or not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics) + 1.5,
                                                       0.28 * len(d) + 1.8), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        col = f'test_{m}'
        sub = d.sort_values(col, ascending=(m != 'R2'))
        y = np.arange(len(sub))
        x = sub[col].to_numpy(dtype=float)
        lo, hi = sub.get(f'{col}_lo'), sub.get(f'{col}_hi')
        if lo is not None and hi is not None:
            lo = lo.to_numpy(dtype=float)
            hi = hi.to_numpy(dtype=float)
            err = np.vstack([np.abs(x - lo), np.abs(hi - x)])
            err = np.nan_to_num(err, nan=0.0)
            ax.errorbar(x, y, xerr=err, fmt='o', ms=3, lw=0.8,
                        color='#3b6fb6', ecolor='#9bb8dd')
        else:
            ax.plot(x, y, 'o', ms=3, color='#3b6fb6')
        ax.set_yticks(y)
        ax.set_yticklabels([_label(r) for _, r in sub.iterrows()], fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel(m)
        ax.set_title(f'Test {m}', fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_by_analyte(by_analyte, path, metrics=('MAE', 'R2', 'MAPE')):
    """Run x analyte heatmap, one panel per metric."""
    plt = _style()
    if by_analyte is None or by_analyte.empty:
        return None
    have = [m for m in metrics if m in set(by_analyte['metric'])]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have) + 1.5, 0.30 * 34 + 2),
                             squeeze=False)
    for ax, m in zip(axes[0], have):
        piv = (by_analyte[by_analyte['metric'] == m]
               .pivot_table(index='analyte', columns='run', values='mean'))
        # normalise each analyte to its own best run so one big analyte cannot
        # dominate the colour scale
        best = piv.min(axis=1) if m != 'R2' else piv.max(axis=1)
        rel = piv.div(best, axis=0) if m != 'R2' else piv.sub(best, axis=0)
        im = ax.imshow(rel.values, aspect='auto', cmap='RdBu_r' if m == 'R2' else 'RdBu')
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([str(c)[:12] for c in piv.columns], rotation=90, fontsize=5)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index, fontsize=5)
        ax.set_title(f'Test {m} per analyte', fontsize=8)
        fig.colorbar(im, ax=ax, label=('R2 - best' if m == 'R2' else f'{m} / best'),
                     fraction=0.035)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--log_dir', default=LOG_DIR)
    p.add_argument('--wandb_dir', default=WANDB_DIR)
    p.add_argument('--out_dir', default=os.path.join(LOG_DIR, 'experiments'))
    p.add_argument('--min_epochs', type=int, default=0,
                   help='drop runs that stopped before this epoch')
    p.add_argument('--wandb', action='store_true',
                   help='also push the tables and figures to W&B as run experiment-index')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, by_analyte = build(args.log_dir, args.wandb_dir, args.min_epochs)

    idx = os.path.join(args.out_dir, 'experiments.csv')
    df.to_csv(idx, index=False)
    print(f'{len(df):,} runs indexed -> {idx}')
    print(f'  with a saved checkpoint: {int(df["has_checkpoint"].sum())}')
    print(f'  evaluated on test:       {int(df["evaluated"].sum())}')
    print(f'  with a val loss:         {int(df["val_loss"].notna().sum())}'
          f'  (the rest logged a config but no epoch metrics)')
    if not by_analyte.empty:
        ba = os.path.join(args.out_dir, 'experiments_by_analyte.csv')
        by_analyte.to_csv(ba, index=False)
        print(f'  per-analyte rows:        {len(by_analyte):,} '
              f'({by_analyte["run"].nunique()} runs) -> {ba}')

    print('\nruns by architecture x loss:')
    grid = (df.assign(model=df['model'].fillna('unknown'), loss=df['loss'].fillna('unknown'))
              .pivot_table(index='model', columns='loss', values='run', aggfunc='count')
              .fillna(0).astype(int))
    print('  ' + grid.to_string().replace('\n', '\n  '))

    ev = df[df['evaluated']]
    if len(ev):
        cols = ['run', 'model', 'loss', 'output_mode', 'features', 'val_loss'] + \
               [c for c in ev.columns if c.startswith('test_') and not c.endswith(('_lo', '_hi'))]
        print('\nevaluated runs:')
        print('  ' + ev[cols].round(3).to_string(index=False).replace('\n', '\n  '))

    figs = {}
    for name, fn in [('val_loss', lambda pth: fig_val_loss(df, pth)),
                     ('test_metrics', lambda pth: fig_test_metrics(df, pth)),
                     ('by_analyte', lambda pth: fig_by_analyte(by_analyte, pth))]:
        pth = os.path.join(args.out_dir, f'{name}.pdf')
        if fn(pth):
            figs[name] = pth
            print(f'  wrote {pth}')

    if args.wandb:
        import wandb
        run = wandb.init(project='NORMA', group='experiment-index', job_type='summary',
                         name='experiment-index', id='experiment-index', resume='allow',
                         config={'n_runs': len(df),
                                 'n_checkpoints': int(df['has_checkpoint'].sum()),
                                 'n_evaluated': int(df['evaluated'].sum())})
        payload = {'experiments/index': wandb.Table(dataframe=df.astype(object).where(df.notna(), None))}
        if not by_analyte.empty:
            payload['experiments/by_analyte'] = wandb.Table(dataframe=by_analyte)
        payload.update({f'experiments/{k}': wandb.Image(v) for k, v in figs.items()})
        run.log(payload)
        for _, r in df[df['evaluated']].iterrows():
            for m in [c for c in df.columns if c.startswith('test_')
                      and not c.endswith(('_lo', '_hi'))]:
                if pd.notna(r[m]):
                    run.summary[f"{r['run']}/{m}"] = float(r[m])
        run.finish()
        print('logged to W&B run experiment-index (group experiment-index)')


if __name__ == '__main__':
    main()
