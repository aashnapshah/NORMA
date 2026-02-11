import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import TEST_VOCAB, INVERSE_TEST_VOCAB
from utils import *

CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

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

def predict(model, device, train_loader, val_loader, test_loader):
    print('Generating predictions and computing metrics...')
    all_predictions = []
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"Evaluating {split_name} set...")
        predictions_df = get_predictions(model, device, loader, split_name)
        predictions_df['split'] = split_name
        all_predictions.append(predictions_df)
    return pd.concat(all_predictions)
        
def get_predictions(model, device, loader, split_name):
    """Get predictions from model for a given dataloader."""
    model.eval()
    model.to(device)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split_name} (predict)", leave=False):
            batch = to_device_batch(batch, device)
            
            mu, log_var = model(
                batch['x_h'], batch['s_h'], batch['t_h'], batch['sex'], 
                batch['cid'], batch['s_next'], batch['t_next'], batch['pad_mask']
            )
            
            cid = batch['cid'].cpu().numpy()
            pids = batch['pids'] 
            mu = mu.cpu().numpy()
            log_var = log_var.cpu().numpy()
            x_next = batch['x_next'].cpu().numpy()
            t_next = batch['t_next'].cpu().numpy()
            s_next = batch['s_next'].cpu().numpy()
            
            for i in range(len(mu)):
                cid_val = int(cid[i].item() if hasattr(cid[i], 'item') else cid[i])
                x_next_val = float(x_next[i].item() if hasattr(x_next[i], 'item') else x_next[i])
                t_next_val = float(t_next[i].item() if hasattr(t_next[i], 'item') else t_next[i])
                s_next_val = int(s_next[i].item() if hasattr(s_next[i], 'item') else s_next[i])
                mu_val = float(mu[i].item() if hasattr(mu[i], 'item') else mu[i])
                log_var_val = float(log_var[i].item() if hasattr(log_var[i], 'item') else log_var[i])
                predictions.append({
                    'pid': pids[i],
                    'cid': cid_val,
                    'code': CODE_TO_TEST_NAME[cid_val],
                    'x_next': x_next_val,
                    't_next': t_next_val,
                    's_next': s_next_val,
                    'mu': mu_val,
                    'log_var': log_var_val,
                })
    
    return pd.DataFrame(predictions)

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
        # Filter the rows for the current split
        split_metrics = all_metrics[all_metrics["Split"] == split]

        # Use a dict-of-dicts to collect mean±std for each model & metric
        rows = {}
        # Use sorted unique values to ensure ordering is consistent
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

def get_features(records):
    def extract(record):
        x = np.asarray(record['x'][:-1], float)
        t = np.asarray(record['t'][:-1], float)
        s = np.asarray(record['s'][:-1], int)
        
        if len(x) < 2:
            return None
        
        # Features
        n_normal = (s == 1).sum()
        mu_true = np.mean(x)
        mu_true_normal = np.mean(x[s == 1])
        var_true = np.var(x)
        var_true_normal = np.var(x[s == 1])
        var_true_normal = var_true_normal if n_normal > 0 else 0
        
        n_obs = len(x)
        t_delta_last = t[-1] - t[-2]
        t_duration = t.max() - t.min()
        p_normal = n_normal / n_obs

        eps = 1e-8
        true_cv = np.sqrt(max(var_true, 0.0)) / (abs(mu_true) + eps)
        log_true_cv = np.log(true_cv + eps)
        true_cv_normal = np.sqrt(max(var_true_normal, 0.0)) / (abs(mu_true_normal) + eps)
        log_true_cv_normal = np.log(true_cv_normal + eps)
        
        return {
            'pid': record['pid'],
            'code': INVERSE_TEST_VOCAB.get(int(record['cid']), record['cid']),
            'mu_true': mu_true,
            'var_true': var_true,
            'var_true_normal': var_true_normal,
            'n_obs': n_obs,
            't_delta_last': t_delta_last,
            't_duration': t_duration,
            'n_normal': n_normal,
            'p_normal': p_normal,
            'true_cv': true_cv,
            'log_true_cv': log_true_cv,
            'true_cv_normal': true_cv_normal,
            'log_true_cv_normal': log_true_cv_normal,
        }

    rows = [r for r in (extract(rec) for rec in records) if r is not None]
    return pd.DataFrame(rows)

def load_counterfactual_predictions(run_ids, log_dir, split='test'):
    results = []
    for id in run_ids:
        counterfactual_path = os.path.join(log_dir, id, "counterfactual_predictions_combined.csv") 
        predictions_path = os.path.join(log_dir, id, "predictions_combined.csv") 
        path = counterfactual_path if os.path.exists(counterfactual_path) else predictions_path
        state_col = 'state' if 'state' in pd.read_csv(path).columns else 's_next'

        subresults = pd.read_csv(path).query(f"split == '{split}'")
        if 'log_var' not in subresults.columns:
            subresults['std_pred'] = np.exp(subresults['std'])
            subresults['var_pred'] = subresults['std']**2
            subresults['log_var'] = np.log(subresults['var_pred'])
            
        subresults['var_pred'] = np.exp(subresults['log_var']).fillna(0)
        subresults['std_pred'] = np.sqrt(subresults['var_pred'])
        subresults['code'] = subresults['code'].fillna('NA') 
        subresults['Run ID'] = id
        results.append(subresults)
    return pd.concat(results), state_col
    