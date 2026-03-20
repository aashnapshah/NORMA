"""
Evaluate NORMA forecasting performance.

Two modes:
  1. Called from train.py during training → saves per-run metrics to logs/
  2. Standalone → compares NORMA runs + baselines → results/prediction/raw/

Usage:
    python evaluate.py
    python evaluate.py --runs 334f7e21 167f05e8 --n_bootstrap 1000
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process.config import REFERENCE_INTERVALS
from data import TEST_VOCAB, CODE_TO_TEST_NAME

METRIC_FUNCTIONS = {
    'MAE':   lambda y, p: mean_absolute_error(y, p),
    'RMSE':  lambda y, p: np.sqrt(mean_squared_error(y, p)),
    'MAPE':  lambda y, p: np.mean(np.abs((y - p) / y)) * 100,
    'R2':    lambda y, p: r2_score(y, p),
    'MSE':   lambda y, p: mean_squared_error(y, p),
    'NMAE':  lambda y, p: mean_absolute_error(y, p) / np.mean(y),
    'NRMSE': lambda y, p: np.sqrt(mean_squared_error(y, p)) / np.mean(y),
    'NMSE':  lambda y, p: mean_squared_error(y, p) / np.mean(y) ** 2,
}

EXCLUDE_CODES = {'CRP', 'GGT', 'LDH', 'PT'}

NORMA_RUNS = {
    '334f7e21': ('NORMA-Quantile', 'q50'),
    '167f05e8': ('NORMA-Gaussian', 'mu'),
}

BASELINE_FILES = {
    'ARIMA': ('arima_baseline_combined.csv', 'x_next', 'x_pred'),
    'Mean':  ('mean_baseline_combined.csv',  'x_next', 'x_pred'),
    'Last':  ('last_baseline_combined.csv',  'x_next', 'x_pred'),
}

CID_TO_CODE = {i: name for i, name in enumerate(REFERENCE_INTERVALS.keys())}


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


# --- Standalone: multiple runs + baselines ---

def load_norma_predictions(run_id):
    label, pred_col = NORMA_RUNS[run_id]
    path = os.path.join(ROOTDIR, 'model', 'logs', run_id, 'predictions_combined.csv')
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if 'code' not in df.columns:
        df['code'] = df['cid'].map(CID_TO_CODE)
    return label, pred_col, df


def load_baseline_predictions(name):
    filename, y_col, pred_col = BASELINE_FILES[name]
    path = os.path.join(ROOTDIR, 'ARIMA', 'predictions', filename)
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if 'code' not in df.columns:
        df['code'] = df['cid'].map(CID_TO_CODE)
    return y_col, pred_col, df



def evaluate_all(run_ids, output_dir, n_bootstrap=1000, exclude=None,
                 metrics_to_agg=None, skip_baselines=False):
    if metrics_to_agg is None:
        metrics_to_agg = ['MAE', 'MAPE', 'R2', 'MSE']
    if exclude is None:
        exclude = EXCLUDE_CODES

    os.makedirs(output_dir, exist_ok=True)
    all_by_analyte = []

    for run_id in run_ids:
        label, pred_col, df = load_norma_predictions(run_id)
        print(f"\nEvaluating {label} ({run_id}): {len(df):,} predictions")
        by_analyte = bootstrap_metrics_df(
            df, y_col='x_next', pred_col=pred_col,
            exclude=exclude, metrics_to_agg=metrics_to_agg,
            n_bootstrap=n_bootstrap, code_level=True
        )
        by_analyte.insert(0, 'model', label)
        all_by_analyte.append(by_analyte)

    if not skip_baselines:
        for name in BASELINE_FILES:
            try:
                y_col, pred_col, df = load_baseline_predictions(name)
                print(f"\nEvaluating {name}: {len(df):,} predictions")
                by_analyte = bootstrap_metrics_df(
                    df, y_col=y_col, pred_col=pred_col,
                    exclude=exclude, metrics_to_agg=metrics_to_agg,
                    n_bootstrap=n_bootstrap, code_level=True
                )
                by_analyte.insert(0, 'model', name)
                all_by_analyte.append(by_analyte)
            except FileNotFoundError as e:
                print(f"\n  Skipping {name}: {e}")

    by_analyte_df = pd.concat(all_by_analyte, ignore_index=True)
    overall_df = compute_overall_metrics(by_analyte_df, n_bootstrap=n_bootstrap)

    by_analyte_path = os.path.join(output_dir, 'forecasting_by_analyte.csv')
    overall_path = os.path.join(output_dir, 'forecasting_overall.csv')
    by_analyte_df.to_csv(by_analyte_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print(f"\nSaved:")
    print(f"  {by_analyte_path}  ({len(by_analyte_df)} rows)")
    print(f"  {overall_path}  ({len(overall_df)} rows)")

    test = overall_df[overall_df['split'] == 'test']
    print(f"\nTest set summary (mean [95% CI] across analytes):\n")
    for metric in metrics_to_agg:
        print(f"  {metric}:")
        for _, row in test[test['metric'] == metric].iterrows():
            print(f"    {row['model']:<20s}  {row['mean']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
        print()

    return by_analyte_df, overall_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=list(NORMA_RUNS.keys()))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(ROOTDIR, 'results', 'prediction', 'raw'))
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--metrics', nargs='+', default=['MAE', 'MAPE', 'R2', 'MSE'])
    parser.add_argument('--exclude', nargs='+', default=['CRP', 'GGT', 'LDH', 'PT'])
    parser.add_argument('--skip_baselines', action='store_true')
    args = parser.parse_args()

    evaluate_all(
        run_ids=args.runs,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        exclude=set(args.exclude),
        metrics_to_agg=args.metrics,
        skip_baselines=args.skip_baselines,
    )
