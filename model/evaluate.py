import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

ROOTDIR = '/n/data1/hms/dbmi/manrai/aashna/NORMA/'

sys.path.append(ROOTDIR)
sys.path.append('../model/')

from process.config import REFERENCE_INTERVALS 
from data import TEST_VOCAB, INVERSE_TEST_VOCAB, CODE_TO_TEST_NAME
from utils import *
from predict import *
from edit import *
from model import *
from helpers.plots import *

METRIC_FUNCTIONS = {
        'MAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R2': lambda y_true, y_pred: r2_score(y_true, y_pred) ,
        'MSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        'NMAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred) / np.mean(y_true),
        'NRMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true),
        'NMSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred) / np.mean(y_true)**2,
 }

def bootstrap_metrics(
    models,
    exclude_codes=None,
    metrics_to_agg=None,
    bootstrap_samples=1000,
    bootstrap_seed=42,
    code_level=True
):
    rng = np.random.RandomState(bootstrap_seed)
    records = []
    for model_name, model_fields in models.items():
        path, y_col, pred_col = model_fields
        
        df = pd.read_csv(path)
        df = df.query('cid not in @exclude_codes')
        df['Split'] = df['split'].astype(str).str.capitalize()

        if code_level:
            group_iter = df.groupby(['Split', 'cid'])
        else:
            group_iter = ((split, split_df) for split, split_df in df.groupby('Split'))

        for group_key, group in group_iter:
            if code_level:
                if isinstance(group_key, tuple) and len(group_key) == 2:
                    split, code = group_key
                else:
                    split, code = group_key, None
            else:
                split = group_key
                code = None
            y_true = group[y_col].to_numpy()
            y_pred = group[pred_col].to_numpy()
            n = len(group)
            for metric in metrics_to_agg:
                metric_fn = METRIC_FUNCTIONS[metric]
                point_estimate = metric_fn(y_true, y_pred)
                boot_vals = []
                for _ in range(bootstrap_samples):
                    idx = rng.choice(n, size=n, replace=True)
                    boot_vals.append(metric_fn(y_true[idx], y_pred[idx]))
                boot_vals = np.asarray(boot_vals, dtype=float)
                mean_val = float(np.nanmean(boot_vals))
                std_val = float(np.nanstd(boot_vals, ddof=1)) if np.sum(~np.isnan(boot_vals)) > 1 else 0.0
                upper_val = float(np.nanpercentile(boot_vals, 97.5))
                lower_val = float(np.nanpercentile(boot_vals, 2.5))

                record = {
                    'Model': model_name,
                    'Split': split,
                    'Metric': metric,
                    'n_samples': int(n),
                    'point_estimate': point_estimate,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': lower_val,
                    'ci_upper': upper_val,
                }
                if code_level:
                    record['Code'] = code
                records.append(record)
    return pd.DataFrame.from_records(records)

def calculate_all_metrics(bootstrap_metrics, exclude_codes):
    all_code_metrics = (
        bootstrap_metrics
        .query('Code not in @exclude_codes')
        .groupby(['Split', 'Model', 'Train | Test', 'Metric'])['point_estimate']
        .agg(['mean', 'std'])
        .sort_index()
    ).reset_index()
    
    return all_code_metrics

def print_formatted_metrics(all_metrics, metrics_to_agg):
    splits = ['Train', 'Val', 'Test']

    for split in splits:
        split_metrics = all_metrics[all_metrics["Split"] == split]

        rows = {}
        models = split_metrics["Model"].unique()
        for model in models:
            subset = split_metrics[(split_metrics["Model"] == model)]
    
            entry = {}
            for metric in metrics_to_agg:
                metric_row = subset[subset["Metric"] == metric]
                if not metric_row.empty:
                    mean_val = metric_row["mean"].iloc[0]
                    std_val = metric_row["std"].iloc[0]
                    entry[metric] = f"{mean_val:.1f} ± {std_val:.1f}"
                else:
                    entry[metric] = "N/A"
            rows[model] = entry

        formatted = (
            pd.DataFrame(rows)
            .reindex(index=metrics_to_agg)
        )
        formatted.columns = pd.MultiIndex.from_tuples(formatted.columns, names=['Model', 'Train | Test'])
        formatted.index.name = 'Metric'
        print(f"\nMetrics for {split} split:")
        print(formatted)

def evaluate_and_save_metrics(preds, model, exclude_codes=None, metrics_to_agg=['MAE', 'MAPE', 'R2', 'MSE'], log_dir='../model/logs/'):
    gm = bootstrap_metrics(
        preds,
        code_level=True,
        exclude_codes=exclude,
        metrics_to_agg=metrics,
        bootstrap_samples=2,
    )
    gm.to_csv(f'{log_dir}/{model}/bootstrap_metrics_by_code.csv', index=False)
    print('bootstrap_metrics_by_code.csv saved')
    gm = (
        gm.query('Metric == "R2"')
        .groupby(['Model', 'Metric', 'Split'])['mean']
        .agg(['mean', 'std'])
        .reset_index()
    )
    om = bootstrap_metrics(
        preds,
        code_level=False,
        exclude_codes=exclude,
        metrics_to_agg=metrics,
        bootstrap_samples=2,
    )
    om = om.query('Metric != "R2"')[['Split', 'Model', 'Metric', 'mean', 'std']]
    metrics_df = pd.concat([gm, om]).replace(MODEL, 'NORMA')
    save_path = f'{log_dir}/{model}/bootstrap_metrics.csv'
    metrics_df.to_csv(save_path, index=False)
    print(f"\nMetrics saved to: {save_path}")
    return metrics_df

def main(args):
    preds = load_predictions(args.run_ids, args.base, args.source)
    metrics_df = evaluate_and_save_metrics(preds, args.exclude, args.metrics, args.model, args.log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/n/data1/hms/dbmi/manrai/aashna/NORMA/model/logs/', help='Directory for log files')
    parser.add_argument('--model', type=str, default='e4fdacb7', help='Model identifier')
    parser.add_argument('--run_ids', nargs='+', default=['ARIMA', 'Mean', 'last', 'e4fdacb7'], help='List of run IDs')
    parser.add_argument('--base', type=str, default='combined', help='Base dataset name')
    parser.add_argument('--source', type=str, default='combined', help='Source dataset name')
    parser.add_argument('--metrics', nargs='+', default=['MAE', 'MAPE', 'R2', 'MSE'], help='Metrics to use')
    parser.add_argument('--exclude', nargs='+', default=['CRP', 'GGT', 'LDH'], help='Codes to exclude')
    args = parser.parse_args()

    # assign arguments to variables, maintaining backwards compatibility
    LOG_DIR = args.log_dir
    MODEL = args.model
    RUN_IDS = args.run_ids
    BASE = args.base
    SOURCE = args.source
    METRICS = args.metrics
    EXCLUDE = args.exclude

    main(args)