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
    exclude=None,
    metrics_to_agg=None,
    bootstrap_samples=1000,
    bootstrap_seed=42,
    code_level=True
):
    rng = np.random.RandomState(bootstrap_seed)
    records = []
    print(f"Bootstrapping metrics for {models}...")
    for model_name, model_fields in models.items():
        path, y_col, pred_col = model_fields
        print(f"Reading {path}...")
        df = pd.read_csv(path)
        df = df.query('cid not in @exclude')
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

def calculate_all_metrics(bootstrap_metrics, exclude):
    all_code_metrics = (
        bootstrap_metrics
        .query('Code not in @exclude')
        .groupby(['Split', 'Model', 'Train | Test', 'Metric'])['point_estimate']
        .agg(['mean', 'std'])
        .sort_index()
    ).reset_index()
    
    return all_code_metrics

def evaluate_and_save_metrics(preds, model, exclude=None, 
                              metrics_to_agg=['MAE', 'MAPE', 'R2', 'MSE'], 
                              log_dir='../model/logs/', n_bootstraps=100):
    gm = bootstrap_metrics(
        preds,
        code_level=True,
        exclude=exclude,
        metrics_to_agg=metrics,
        bootstrap_samples=n_bootstraps,
    )
    path = f'{log_dir}/{model}/bootstrap_metrics_by_code.csv'
    print("Saving bootstrap_metrics_by_code.csv...", path)
    gm.to_csv(path, index=False)
    
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
        exclude=exclude,
        metrics_to_agg=metrics,
        bootstrap_samples=n_bootstraps,
    )
    om = om.query('Metric != "R2"')[['Split', 'Model', 'Metric', 'mean', 'std']]
    metrics_df = pd.concat([gm, om]).replace(model, 'NORMA')
    save_path = f'{log_dir}/{model}/bootstrap_metrics.csv'
    metrics_df.to_csv(save_path, index=False)
    print(f"\nMetrics saved to: {save_path}")
    return metrics_df

def main(args):
    preds = load_predictions(args.run_ids, args.base, args.source)
    metrics_df = evaluate_and_save_metrics(preds, args.model, args.exclude, args.metrics, args.log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/n/data1/hms/dbmi/manrai/aashna/NORMA/model/logs/', help='Directory for log files')
    parser.add_argument('--model', type=str, default='167f05e8', help='Model identifier')
    parser.add_argument('--run_ids', nargs='+', default=['ARIMA', 'Mean', 'last', '167f05e8'], help='List of run IDs')
    parser.add_argument('--base', type=str, default='combined', help='Base dataset name')
    parser.add_argument('--source', type=str, default='combined', help='Source dataset name')
    parser.add_argument('--metrics', nargs='+', default=['MAE', 'MAPE', 'R2', 'MSE'], help='Metrics to use')
    parser.add_argument('--exclude', nargs='+', default=['CRP', 'GGT', 'LDH'], help='Codes to exclude')
    args = parser.parse_args()

    log_dir = args.log_dir
    model = args.model
    run_ids = args.run_ids + [model]
    base = args.base
    source = args.source
    metrics = args.metrics
    exclude = args.exclude

    main(args)