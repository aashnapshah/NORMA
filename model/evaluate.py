"""
Evaluate NORMA forecasting performance.

Two modes:
  1. Called from train.py during training → saves per-run metrics to logs/
  2. Standalone → compares NORMA runs + baselines → validation/results/dev/raw/prediction/raw/

Usage:
    python evaluate.py
    python evaluate.py --runs 334f7e21 167f05e8 --n_bootstrap 1000
"""

import os
import sys
import argparse
from run_names import arm_label
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# process/ moved under scripts/ in the 2026-09-08 layout change; scripts/ carries
# `process.config`, scripts/process/ the bare `from config import ...` model/ does
sys.path.insert(0, os.path.join(ROOTDIR, 'scripts'))
sys.path.insert(0, os.path.join(ROOTDIR, 'scripts', 'process'))
sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process.config import REFERENCE_INTERVALS
from data import TEST_VOCAB, CODE_TO_TEST_NAME
from model import is_quantile_mode

METRIC_FUNCTIONS = {
    'MAE':   lambda y, p: mean_absolute_error(y, p),
    'RMSE':  lambda y, p: np.sqrt(mean_squared_error(y, p)),
    'MAPE':  lambda y, p: np.mean(np.abs((y[y != 0] - p[y != 0]) / y[y != 0])) * 100,   # zero targets are undefined
    'R2':    lambda y, p: r2_score(y, p),
    'MSE':   lambda y, p: mean_squared_error(y, p),
    'NMAE':  lambda y, p: mean_absolute_error(y, p) / np.mean(y),
    'NRMSE': lambda y, p: np.sqrt(mean_squared_error(y, p)) / np.mean(y),
    'NMSE':  lambda y, p: mean_squared_error(y, p) / np.mean(y) ** 2,
}

EXCLUDE_CODES = {'CRP', 'GGT', 'LDH', 'PT'}

NORMA_RUNS = {
    '334f7e21': (arm_label('334f7e21'), 'q50'),
    '167f05e8': ('NORMA-Gaussian', 'mu'),
    # covariate ablation (run_covariate_ablation.sh; run_id == run name; baseline = 334f7e21)
    'q_age': (arm_label('q_age'), 'q50'),
    'q_set': (arm_label('q_set'), 'q50'),
    'q_co': (arm_label('q_co'), 'q50'),
    'q_age_set': (arm_label('q_age_set'), 'q50'),
    'q_age_set_co': (arm_label('q_age_set_co'), 'q50'),
    'q_age_co': (arm_label('q_age_co'), 'q50'),
    'q_set_co': (arm_label('q_set_co'), 'q50'),
    'q_co_q': (arm_label('q_co_q'), 'q50'),
}

# Leak-free forecasting variants produced by predict_states.py (Referee 3, minor 1).
# 'oracle' is the original predictions_combined.csv, query = realized future state.
STATE_VARIANTS = {
    'oracle':        ('oracle',        'predictions_combined.csv'),
    'normal':        ('normal-fixed',  'predictions_combined_normal.csv'),
    'marginal':      ('marginal',      'predictions_combined_marginal.csv'),
    'marginal_freq': ('marginal-freq', 'predictions_combined_marginal_freq.csv'),
}

# Baselines built on NORMA's three-state split by baselines/forecast.py, which
# writes results/raw/<cohort>/forecast_baselines.parquet -- one file per cohort,
# one column per model.  The archived Nov-2025 files
# (_archive/baselines_legacy_2026-09-08/) used a different split and a
# per-analyte subsample, so their test rows were not NORMA's test rows.
from baselines.forecast import read_baselines, DEV_COHORTS   # noqa: E402

BASELINE_FILES = {          # display name -> the model's column
    'ARIMA': 'arima',
    'Mean':  'mean',
    'Last':  'last',
}
ARIMA_MIN_HIST = 3   # ARIMA(1,1,1) needs p+d+q points; shorter histories are undefined, not LVCF

# Baselines given the realized future state (matched inputs with the oracle NORMA run).
STATE_BASELINE_FILES = {
    'ARIMAX (state)': 'arimax',
    'Mean (state)':   'mean_state',
    'Last (state)':   'last_state',
}

CID_TO_CODE = {i: name for i, name in enumerate(REFERENCE_INTERVALS.keys())}

# The development split pools two sources (EHRSHOT and MIMIC-IV); the prediction
# files only carry pid, so the source is recovered from the sequences pickle and
# cached here (pids do not overlap between the two sources).
PID_SOURCE_CACHE = os.path.join(ROOTDIR, 'model', 'predictions', 'pid_source.csv')
SEQUENCES_PKL = os.path.join(ROOTDIR, '..', 'data', 'processed', 'combined_sequences_v2.pkl')
SOURCE_SPLIT = {'ehrshot': 'ehrshot', 'mimiciv': 'mimiciv'}   # source in pickle -> split label


def load_pid_source():
    """pid -> source ('ehrshot' | 'mimiciv') for the development cohort."""
    if not os.path.exists(PID_SOURCE_CACHE):
        import pickle
        with open(SEQUENCES_PKL, 'rb') as f:
            seqs = pickle.load(f)
        m = pd.DataFrame({'pid': [s['pid'] for s in seqs], 'source': [s['source'] for s in seqs]}).drop_duplicates()
        assert not m['pid'].duplicated().any(), 'a pid maps to more than one source'
        os.makedirs(os.path.dirname(PID_SOURCE_CACHE), exist_ok=True)
        m.to_csv(PID_SOURCE_CACHE, index=False)
        del seqs
    m = pd.read_csv(PID_SOURCE_CACHE)
    return m.set_index('pid')['source'].to_dict()


def bootstrap_metrics_df(df, y_col, pred_col, exclude=None,
                         metrics_to_agg=None, n_bootstrap=1000,
                         seed=42, code_level=True):
    if metrics_to_agg is None:
        metrics_to_agg = ['MAE', 'MAPE', 'R2', 'MSE']
    if exclude is None:
        exclude = EXCLUDE_CODES

    rng = np.random.RandomState(seed)
    df = df.copy()

    if 'analyte' not in df.columns:
        if 'code' in df.columns:
            df['analyte'] = df['code']
        elif 'cid' in df.columns:
            df['analyte'] = df['cid'].map(CID_TO_CODE)

    df['split'] = df['split'].astype(str).str.lower()
    df = df[~df['analyte'].isin(exclude)]

    groups = df.groupby(['split', 'analyte']) if code_level else df.groupby('split')

    records = []
    for group_key, group in groups:
        if code_level:
            split, analyte = group_key
        else:
            split, analyte = group_key, None

        y_true = group[y_col].to_numpy()
        y_pred = group[pred_col].to_numpy()
        n = len(group)
        if n < 10:
            continue

        for metric_name in metrics_to_agg:
            metric_fn = METRIC_FUNCTIONS[metric_name]
            point = metric_fn(y_true, y_pred)
            boot_vals = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot_vals[b] = metric_fn(y_true[idx], y_pred[idx])

            record = {
                'split': split,
                'metric': metric_name,
                'n': int(n),
                'point_estimate': point,
                'mean': float(np.nanmean(boot_vals)),
                'std': float(np.nanstd(boot_vals, ddof=1)) if n > 1 else 0.0,
                'ci_lower': float(np.nanpercentile(boot_vals, 2.5)),
                'ci_upper': float(np.nanpercentile(boot_vals, 97.5)),
            }
            if code_level:
                record['analyte'] = analyte
            records.append(record)

    return pd.DataFrame.from_records(records)


def compute_overall_metrics(by_analyte_df, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    records = []
    for (model, split, metric), grp in by_analyte_df.groupby(['model', 'split', 'metric']):
        vals = grp['point_estimate'].values
        n = len(vals)
        point = float(np.mean(vals))
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_means[b] = np.mean(vals[idx])
        records.append({
            'model': model,
            'split': split,
            'metric': metric,
            'mean': point,
            'ci_lower': float(np.percentile(boot_means, 2.5)),
            'ci_upper': float(np.percentile(boot_means, 97.5)),
        })
    return pd.DataFrame.from_records(records)


# --- train.py interface ---

def evaluate_and_save_metrics(predictions_df, run_id, exclude=None,
                              metrics_to_agg=None, save_dir=None,
                              n_bootstrap=100):
    if metrics_to_agg is None:
        metrics_to_agg = ['MAE', 'MAPE', 'R2', 'MSE']
    if exclude is None:
        exclude = EXCLUDE_CODES
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'logs', run_id)

    pred_col = 'q50' if 'q50' in predictions_df.columns else 'mu'

    by_code = bootstrap_metrics_df(
        predictions_df, y_col='x_next', pred_col=pred_col,
        exclude=exclude, metrics_to_agg=metrics_to_agg,
        n_bootstrap=n_bootstrap, code_level=True
    )
    by_code.insert(0, 'model', 'NORMA')
    by_code.to_csv(os.path.join(save_dir, 'bootstrap_metrics_by_code.csv'), index=False)

    overall = compute_overall_metrics(by_code, n_bootstrap=n_bootstrap)
    overall.to_csv(os.path.join(save_dir, 'bootstrap_metrics.csv'), index=False)

    print(f"  Evaluation saved to {save_dir}")
    return overall


# --- Calibration of the predicted conditional distributions ---
#
# Referee 1 asked whether the 95% interval really covers 95% of realized values,
# whether the predicted quantiles are calibrated, how wide the intervals are, and
# whether the normal-conditioned interval agrees with the population interval.
# Everything below is computed per analyte and per queried future state.

STATE_NAMES = {0: 'low', 1: 'normal', 2: 'high'}
NOMINAL_QUANTILES = {'q025': 0.025, 'q25': 0.25, 'q50': 0.50, 'q75': 0.75, 'q975': 0.975}


def pop_ri_bounds(code):
    """Population reference interval for an analyte.

    predictions_combined.csv carries no sex column, so where male and female
    bounds differ we use their midpoint. Affects HGB, HCT, RBC and CRE.
    """
    ref = REFERENCE_INTERVALS.get(CODE_TO_TEST_NAME.get(code, code))
    if ref is None:
        return None, None
    lo = np.mean([ref['M'][0], ref['F'][0]])
    hi = np.mean([ref['M'][1], ref['F'][1]])
    return float(lo), float(hi)


def interval_bounds(df, is_quantile):
    """95% prediction interval as (low, high)."""
    if is_quantile:
        return df['q025'].values, df['q975'].values
    sigma = np.exp(0.5 * df['log_var'].values)
    return df['mu'].values - 1.96 * sigma, df['mu'].values + 1.96 * sigma


def calibration_by_analyte(df, is_quantile, exclude=None, split='test'):
    """One row per (analyte, queried state).

    coverage95  fraction of realized values inside the 95% interval (nominal 0.95)
    width       mean interval width
    width_rel   mean width divided by the population reference interval width
    inside_pop  fraction of the interval that falls within the population interval
    q*_emp      fraction of realized values below each predicted quantile
    """
    exclude = exclude or EXCLUDE_CODES
    d = df[(df['split'] == split) & (~df['code'].isin(exclude))].copy()
    lo, hi = interval_bounds(d, is_quantile)
    d['_lo'], d['_hi'] = lo, hi

    rows = []
    for (code, state), g in d.groupby(['code', 's_next']):
        p_lo, p_hi = pop_ri_bounds(code)
        if p_lo is None:
            continue
        pop_width = p_hi - p_lo
        width = g['_hi'] - g['_lo']
        # how much of the predicted interval sits inside the population interval
        overlap = np.clip(np.minimum(g['_hi'], p_hi) - np.maximum(g['_lo'], p_lo), 0, None)
        row = {
            'code': code,
            'state': STATE_NAMES.get(int(state), int(state)),
            'n': len(g),
            'coverage95': float(((g['x_next'] >= g['_lo']) & (g['x_next'] <= g['_hi'])).mean()),
            'width': float(width.mean()),
            'width_rel': float((width / pop_width).mean()),
            'inside_pop': float((overlap / width.replace(0, np.nan)).mean()),
        }
        if is_quantile:
            for col, nominal in NOMINAL_QUANTILES.items():
                row[f'{col}_emp'] = float((g['x_next'] <= g[col]).mean())
                row[f'{col}_nominal'] = nominal
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['code', 'state'])


def calibration_summary(by_analyte):
    """Median across analytes, per queried state."""
    cols = [c for c in by_analyte.columns
            if c not in ('model', 'code', 'state', 'n') and not c.endswith('_nominal')]
    out = by_analyte.groupby('state')[cols].median().reset_index()
    out['n_analytes'] = by_analyte.groupby('state')['code'].nunique().values
    return out


def cross_state_coverage(run_id, max_sequences=None, split='test', exclude=None, seed=42):
    """Coverage of the s-conditioned interval for values whose realized state is s'.

    In deployment the future state is unknown, so NORMA_RI always queries "normal".
    The diagonal of this table is ordinary calibration; the off-diagonal says whether
    a normal-conditioned interval correctly excludes values that turn out abnormal.

    Returns a tidy frame with one row per (queried state, realized state).
    """
    import torch
    from utils import load_checkpoint, create_model, run_model
    from data import load_and_split_data, load_panel, TimeSeriesDataset, collate_fn
    from torch.utils.data import DataLoader

    exclude = exclude or EXCLUDE_CODES
    device = torch.device('cpu')
    ckpt, hparams = load_checkpoint(os.path.join(ROOTDIR, 'model', 'logs'), run_id,
                                    best=True, device=device, quiet=True)
    model = create_model(hparams, ncodes=len(TEST_VOCAB), checkpoint=ckpt).to(device).eval()
    is_quantile = is_quantile_mode(getattr(hparams, 'output_mode', 'gaussian')) and \
                  getattr(hparams, 'model', '') == 'NORMA2'
    nstates = getattr(hparams, 'nstates', 3)

    data_dir = os.path.join(ROOTDIR, '..', 'data', 'processed')
    version = getattr(hparams, 'data_version', 'v2')
    _, _, test_seq = load_and_split_data(data_dir, 'combined', print_info=False, nstates=nstates, version=version,
                                         split_by=getattr(hparams, 'split_by', 'sequence'))
    panel = load_panel(data_dir, 'combined', version) if getattr(hparams, 'use_coanalytes', False) else None
    if max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(test_seq), size=min(max_sequences, len(test_seq)), replace=False)
        test_seq = [test_seq[i] for i in idx]

    normalize = bool(getattr(hparams, 'normalize', False))
    ds = TimeSeriesDataset(test_seq, nstates, normalize=normalize, panel=panel)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        for batch in loader:
            realized = batch['s_next'].view(-1).numpy()
            x_next = batch['x_next'].view(-1).numpy()
            codes = [CODE_TO_TEST_NAME.get(int(c), int(c)) for c in batch['cid'].view(-1).numpy()]
            for q in range(nstates):
                s_q = torch.full_like(batch['s_next'], q)
                out = run_model(model, batch, s_next=s_q)
                if is_quantile:
                    arr = out.cpu().numpy()
                    lo, hi = arr[:, 0], arr[:, 4]
                else:
                    mu, lv = out
                    mu = mu.cpu().numpy().ravel()
                    sd = np.exp(0.5 * lv.cpu().numpy().ravel())
                    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
                inside = (x_next >= lo) & (x_next <= hi)
                for c, r, ins, w in zip(codes, realized, inside, hi - lo):
                    if c in exclude:
                        continue
                    rows.append({'code': c, 'queried': STATE_NAMES.get(q, q),
                                 'realized': STATE_NAMES.get(int(r), int(r)),
                                 'inside': bool(ins), 'width': float(w)})
    df = pd.DataFrame(rows)
    table = (df.groupby(['queried', 'realized'])
               .agg(coverage=('inside', 'mean'), n=('inside', 'size'), width=('width', 'mean'))
               .reset_index())
    return table, df


def out_path(output_dir, name, suffix=None):
    """output_dir/<name>[_<suffix>].csv — results are one flat folder per cohort,
    so a run's qualifier (state_variants, common) is a filename suffix, not a
    subfolder."""
    stem, ext = os.path.splitext(name)
    return os.path.join(output_dir, f"{stem}_{suffix}{ext}" if suffix else name)


def evaluate_calibration(run_ids, output_dir, exclude=None, split='test'):
    """Write calibration tables for each run."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    for run_id in run_ids:
        label, _, df = load_norma_predictions(run_id)
        is_quantile = 'q025' in df.columns
        by_analyte = calibration_by_analyte(df, is_quantile, exclude=exclude, split=split)
        by_analyte.insert(0, 'model', label)
        summary = calibration_summary(by_analyte)
        summary.insert(0, 'model', label)
        by_analyte.to_csv(os.path.join(output_dir, f'calibration_by_analyte_{run_id}.csv'), index=False)
        all_rows.append(summary)
        print(f'{label} ({run_id}), {split} split')
        print(summary.round(3).to_string(index=False))
        print()
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, 'calibration_summary.csv'), index=False)
    return combined


# --- Standalone: multiple runs + baselines ---

def load_norma_predictions(run_id, variant='oracle'):
    label, pred_col = NORMA_RUNS[run_id]
    vlabel, filename = STATE_VARIANTS[variant]
    if variant != 'oracle':
        label = f'{label} ({vlabel})'
    path = os.path.join(ROOTDIR, 'model', 'logs', run_id, filename)
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if 'code' not in df.columns:
        df['code'] = df['cid'].map(CID_TO_CODE)
    return label, pred_col, df


_BASELINES = None


def load_baseline_predictions(name):
    """One model's column out of the per-cohort baseline parquets, pooled over the
    development cohorts (they are read once and cached)."""
    global _BASELINES
    pred_col = {**BASELINE_FILES, **STATE_BASELINE_FILES}[name]
    if _BASELINES is None:
        _BASELINES = read_baselines(DEV_COHORTS)
        print(f"  Baselines: {len(_BASELINES):,} rows from "
              f"{', '.join(sorted(_BASELINES['source'].unique()))}")
    if pred_col not in _BASELINES.columns:
        raise FileNotFoundError(f"no '{pred_col}' column in the baseline parquets "
                                f"(re-run forecast.py --with_state for the state variants)")
    df = _BASELINES[[c for c in ('pid', 'cid', 'code', 'source', 'split', 'x_next',
                                 's_next', 's_last', 'n_hist') if c in _BASELINES.columns]
                    + [pred_col]].copy()
    if name.startswith('ARIMA') and 'n_hist' in df.columns:
        df.loc[df['n_hist'] < ARIMA_MIN_HIST, pred_col] = np.nan   # undefined, not a forecast
    df = df.dropna(subset=[pred_col])
    if 'code' not in df.columns:
        df['code'] = df['cid'].map(CID_TO_CODE)
    return 'x_next', pred_col, df



def evaluate_all(run_ids, output_dir, n_bootstrap=1000, exclude=None, suffix=None,
                 metrics_to_agg=None, skip_baselines=False, variants=('oracle',),
                 by_state=False, state_baselines=False, common_rows=False, by_source=False):
    if metrics_to_agg is None:
        metrics_to_agg = ['MAE', 'MAPE', 'R2', 'MSE']
    if exclude is None:
        exclude = EXCLUDE_CODES

    os.makedirs(output_dir, exist_ok=True)
    pid_source = load_pid_source() if by_source else None
    all_by_analyte = []
    all_by_state = []
    loaded = []      # (label, df, y_col, pred_col)

    def _score(df, label, y_col, pred_col):
        if common_rows:
            loaded.append((label, df, y_col, pred_col))
            return
        _score_now(df, label, y_col, pred_col)

    def _score_now(df, label, y_col, pred_col):
        by_analyte = bootstrap_metrics_df(
            df, y_col=y_col, pred_col=pred_col,
            exclude=exclude, metrics_to_agg=metrics_to_agg,
            n_bootstrap=n_bootstrap, code_level=True
        )
        by_analyte.insert(0, 'model', label)
        all_by_analyte.append(by_analyte)
        if by_source:
            # test rows scored separately per source cohort (split = 'ehrshot' | 'mimiciv')
            t = df[df['split'].astype(str).str.lower() == 'test']
            src = t['pid'].map(pid_source)
            for source, split_name in SOURCE_SPLIT.items():
                g = t[src == source].copy()
                if len(g) == 0:
                    continue
                g['split'] = split_name
                b = bootstrap_metrics_df(
                    g, y_col=y_col, pred_col=pred_col, exclude=exclude,
                    metrics_to_agg=metrics_to_agg, n_bootstrap=n_bootstrap, code_level=True)
                b.insert(0, 'model', label)
                all_by_analyte.append(b)
        if by_state and 's_next' in df.columns:
            # split the analysis by the realized future state so the reader can see
            # where a normal-fixed query is *meant* to miss
            for state, g in df.groupby('s_next'):
                b = bootstrap_metrics_df(
                    g, y_col=y_col, pred_col=pred_col, exclude=exclude,
                    metrics_to_agg=metrics_to_agg, n_bootstrap=n_bootstrap, code_level=True)
                b.insert(0, 'realized_state', STATE_NAMES.get(int(state), int(state)))
                b.insert(0, 'model', label)
                all_by_state.append(b)

    for run_id in run_ids:
        for variant in variants:
            try:
                label, pred_col, df = load_norma_predictions(run_id, variant)
            except FileNotFoundError as e:
                print(f"\n  Skipping {run_id}/{variant}: {e}")
                continue
            print(f"\nEvaluating {label} ({run_id}): {len(df):,} predictions")
            _score(df, label, 'x_next', pred_col)

    if not skip_baselines:
        names = list(BASELINE_FILES) + (list(STATE_BASELINE_FILES) if state_baselines else [])
        for name in names:
            try:
                y_col, pred_col, df = load_baseline_predictions(name)
                print(f"\nEvaluating {name}: {len(df):,} predictions")
                _score(df, name, y_col, pred_col)
            except FileNotFoundError as e:
                print(f"\n  Skipping {name}: {e}")

    if common_rows:
        # restrict every model to the test rows where all models have a prediction
        keys = None
        for label, df, y_col, pred_col in loaded:
            k = df.loc[df[pred_col].notna(), ['pid', 'cid', 'x_next']].drop_duplicates()
            keys = k if keys is None else keys.merge(k, on=['pid', 'cid', 'x_next'])
        print(f"\nCommon rows across {len(loaded)} models: {len(keys):,}")
        for label, df, y_col, pred_col in loaded:
            d = df.merge(keys, on=['pid', 'cid', 'x_next'])
            _score_now(d, label, y_col, pred_col)

    by_analyte_df = pd.concat(all_by_analyte, ignore_index=True)
    overall_df = compute_overall_metrics(by_analyte_df, n_bootstrap=n_bootstrap)

    by_analyte_path = out_path(output_dir, 'forecasting_by_analyte.csv', suffix)
    overall_path = out_path(output_dir, 'forecasting_overall.csv', suffix)
    by_analyte_df.to_csv(by_analyte_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print(f"\nSaved:")
    print(f"  {by_analyte_path}  ({len(by_analyte_df)} rows)")
    print(f"  {overall_path}  ({len(overall_df)} rows)")

    if all_by_state:
        by_state_df = pd.concat(all_by_state, ignore_index=True)
        by_state_df.to_csv(out_path(output_dir, 'forecasting_by_state.csv', suffix), index=False)
        # median across analytes, per model x realized state, on the test split
        t = by_state_df[by_state_df['split'] == 'test']
        summ = (t.groupby(['metric', 'model', 'realized_state'])['point_estimate']
                 .median().unstack('realized_state'))
        summ.to_csv(out_path(output_dir, 'forecasting_by_state_summary.csv', suffix))
        print(f"  {out_path(output_dir, 'forecasting_by_state.csv', suffix)}  ({len(by_state_df)} rows)")
        print("\nTest set, median across analytes, by realized future state:")
        print(summ.round(3).to_string())

    test = overall_df[overall_df['split'] == 'test']
    print(f"\nTest set summary (mean [95% CI] across analytes):\n")
    for metric in metrics_to_agg:
        print(f"  {metric}:")
        for _, row in test[test['metric'] == metric].iterrows():
            print(f"    {row['model']:<20s}  {row['mean']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
        print()

    return by_analyte_df, overall_df


# --- External cohorts (validation/17_forecasting.py output) ---

COHORT_MODEL_LABELS = {
    'oracle': '', 'normal': ' (normal-fixed)', 'marginal': ' (marginal)', 'marginal_freq': ' (marginal-freq)',
    'mean': 'Mean', 'last': 'Last', 'arima': 'ARIMA',
    'mean_state': 'Mean (state)', 'last_state': 'Last (state)', 'arimax': 'ARIMAX (state)',
}


def cohort_prediction_columns(df):
    """Map prediction columns of a forecasting_predictions.csv to display labels."""
    baselines = ('mean', 'last', 'arima', 'mean_state', 'last_state', 'arimax')
    labels = {}
    for c in df.columns:
        if c in baselines:
            labels[c] = COHORT_MODEL_LABELS[c]
            continue
        for run_id, (name, _) in NORMA_RUNS.items():
            for variant in ('oracle', 'normal', 'marginal', 'marginal_freq'):
                if c == f'{run_id}_{variant}':
                    labels[c] = f'{name}{COHORT_MODEL_LABELS[variant]}'
    return labels


def evaluate_cohorts(cohorts, output_dir, n_bootstrap=1000, exclude=None, metrics_to_agg=None, suffix=None,
                     by_state=True, common_rows=False):
    """Score every model column in results/raw/{cohort}/forecasting_predictions.csv
    with the same bootstrap machinery as the development test split."""
    if metrics_to_agg is None:
        metrics_to_agg = ['MAE', 'MAPE', 'R2', 'MSE']
    if exclude is None:
        exclude = EXCLUDE_CODES
    os.makedirs(output_dir, exist_ok=True)
    for cohort in cohorts:
        path = os.path.join(ROOTDIR, 'results', 'raw', cohort, 'forecasting_predictions.csv')
        if not os.path.exists(path):
            print(f'\n  Skipping {cohort}: {path} not found (run scripts/05_forecasting.py --dataset {cohort})')
            continue
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
        df['code'] = df['analyte']
        df['split'] = cohort
        labels = cohort_prediction_columns(df)
        for col in labels:
            if col.startswith('arima'):
                df.loc[df['n_hist'] < ARIMA_MIN_HIST, col] = np.nan
        if common_rows:
            df = df.dropna(subset=list(labels))
        print(f'\n{cohort}: {len(df):,} targets{" (common rows)" if common_rows else ""}, '
              f'models: {list(labels.values())}')
        by_analyte, by_state_rows = [], []
        for col, label in labels.items():
            d = df.dropna(subset=[col])
            b = bootstrap_metrics_df(d, y_col='x_next', pred_col=col, exclude=exclude,
                                     metrics_to_agg=metrics_to_agg, n_bootstrap=n_bootstrap, code_level=True)
            b.insert(0, 'model', label)
            by_analyte.append(b)
            if by_state:
                for state, g in d.groupby('s_next'):
                    bs = bootstrap_metrics_df(g, y_col='x_next', pred_col=col, exclude=exclude,
                                              metrics_to_agg=metrics_to_agg, n_bootstrap=n_bootstrap, code_level=True)
                    bs.insert(0, 'realized_state', STATE_NAMES.get(int(state), int(state)))
                    bs.insert(0, 'model', label)
                    by_state_rows.append(bs)
        by_analyte_df = pd.concat(by_analyte, ignore_index=True)
        overall_df = compute_overall_metrics(by_analyte_df, n_bootstrap=n_bootstrap)
        by_analyte_df.to_csv(out_path(output_dir, f'forecasting_{cohort}_by_analyte.csv', suffix), index=False)
        overall_df.to_csv(out_path(output_dir, f'forecasting_{cohort}_overall.csv', suffix), index=False)
        print(f'  saved forecasting_{cohort}_by_analyte.csv, forecasting_{cohort}_overall.csv')
        for metric in metrics_to_agg:
            print(f'  {metric} (mean [95% CI] across analytes):')
            for _, row in overall_df[overall_df['metric'] == metric].iterrows():
                print(f"    {row['model']:<36s} {row['mean']:8.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        if by_state_rows:
            bsd = pd.concat(by_state_rows, ignore_index=True)
            bsd.to_csv(out_path(output_dir, f'forecasting_{cohort}_by_state.csv', suffix), index=False)
            summ = bsd.groupby(['metric', 'model', 'realized_state'])['point_estimate'].median().unstack('realized_state')
            summ.to_csv(out_path(output_dir, f'forecasting_{cohort}_by_state_summary.csv', suffix))
            print(f'\n  median across analytes by realized future state:')
            print(summ.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=list(NORMA_RUNS.keys()))
    parser.add_argument('--suffix', type=str, default=None,
                        help="append _SUFFIX to every output filename "
                             "(results are one flat folder, so runs are suffixed not nested)")
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(ROOTDIR, 'results', 'raw', 'dev'))
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--metrics', nargs='+', default=['MAE', 'MAPE', 'R2', 'MSE'])
    parser.add_argument('--exclude', nargs='+', default=['CRP', 'GGT', 'LDH', 'PT'])
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--cross_state', action='store_true',
                        help='coverage of the s-conditioned interval by realized state')
    parser.add_argument('--max_sequences', type=int, default=None)
    parser.add_argument('--calibration_only', action='store_true',
                        help='only compute calibration of the predicted distributions')
    parser.add_argument('--variants', nargs='+', default=['oracle'],
                        choices=list(STATE_VARIANTS.keys()),
                        help='query-token settings to score (see predict_states.py)')
    parser.add_argument('--by_state', action='store_true',
                        help='also report metrics split by the realized future state')
    parser.add_argument('--state_baselines', action='store_true',
                        help='also score baselines that were given the realized future state')
    parser.add_argument('--common_rows', action='store_true',
                        help='score every model on the rows where all models have a prediction')
    parser.add_argument('--by_source', action='store_true',
                        help='also score the dev test split separately for EHRSHOT and MIMIC-IV')
    parser.add_argument('--cohorts', nargs='+', default=None, choices=['eicu', 'inspire', 'chs'],
                        help='score external-cohort forecasts from validation/17_forecasting.py instead')
    args = parser.parse_args()

    if args.cohorts:
        evaluate_cohorts(args.cohorts, args.output_dir, n_bootstrap=args.n_bootstrap, suffix=args.suffix,
                         exclude=set(args.exclude), metrics_to_agg=args.metrics, by_state=True,
                         common_rows=args.common_rows)
        sys.exit(0)

    if args.cross_state:
        for r in args.runs:
            table, _ = cross_state_coverage(r, max_sequences=args.max_sequences,
                                            exclude=set(args.exclude))
            os.makedirs(args.output_dir, exist_ok=True)
            table.to_csv(os.path.join(args.output_dir, f'cross_state_coverage_{r}.csv'), index=False)
            print(f'\n{NORMA_RUNS[r][0]} ({r}) - coverage by queried x realized state')
            print(table.pivot(index='queried', columns='realized', values='coverage').round(3).to_string())
        sys.exit(0)

    if args.calibration_only:
        evaluate_calibration(args.runs, args.output_dir, exclude=set(args.exclude))
        sys.exit(0)

    evaluate_all(
        run_ids=args.runs,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        exclude=set(args.exclude),
        metrics_to_agg=args.metrics,
        skip_baselines=args.skip_baselines,
        variants=args.variants,
        by_state=args.by_state,
        state_baselines=args.state_baselines,
        common_rows=args.common_rows,
        by_source=args.by_source,
        suffix=args.suffix,
    )
