import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import TEST_VOCAB, INVERSE_TEST_VOCAB
from utils import *

CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}


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
            pids = batch['pids'] #.cpu().numpy()
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

def get_metrics(predictions_df, save_dir=None):
    """Calculate metrics from predictions DataFrame."""
    metrics_list = []
    
    # Overall metrics
    y_true = predictions_df['x_next'].values
    y_pred = predictions_df['mu'].values
    
    overall_metrics = {
        'overall_mae': mean_absolute_error(y_true, y_pred),
        'overall_mse': mean_squared_error(y_true, y_pred),
        'overall_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'overall_r2': r2_score(y_true, y_pred),
        'overall_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'n_samples': len(predictions_df)
    }
    
    # Per-test metrics
    for test_name in predictions_df['test_name'].unique():
        test_data = predictions_df[predictions_df['test_name'] == test_name]
        
        if len(test_data) < 2:  # Need at least 2 samples for R²
            continue
            
        y_true_test = test_data['target'].values
        y_pred_test = test_data['predicted_mu'].values
        
        metrics_list.append({
            'test_name': test_name,
            'n_samples': len(test_data),
            'mae': mean_absolute_error(y_true_test, y_pred_test),
            'mse': mean_squared_error(y_true_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_true_test, y_pred_test)),
            'r2': r2_score(y_true_test, y_pred_test),
            'mape': np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
        })
    
    per_test_metrics = pd.DataFrame(metrics_list)
    overall_metrics = pd.DataFrame(overall_metrics)
    
    if save_dir:
        per_test_metrics.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
        overall_metrics.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False)
    return overall_metrics, per_test_metrics

def calculate_model_metrics(models, exclude_codes=None, metrics_to_agg=None, bootstrap_samples=1000, bootstrap_seed=42):
    
    metric_functions = {
        'MAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R2': lambda y_true, y_pred: r2_score(y_true, y_pred) * 100,
        'MSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        'NMAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred) / np.mean(y_true),
        'NRMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true),
        'NMSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred) / np.mean(y_true)**2,
    }

    
    selected_metric_functions = {
        name: fn for name, fn in metric_functions.items()
        if name in metrics_to_agg
    }
    
    rng = np.random.RandomState(bootstrap_seed)
    code_level_records = []
    global_level_records = []

    for model_name, (path, train_test, y_col, pred_col) in models.items():
        df = pd.read_csv(path)
        if exclude_codes is not None:
            df = df.query('cid not in @exclude_codes')

        df['Split'] = df['split'].astype(str).str.capitalize()
        
        for (split, code), group in df.groupby(['Split', 'cid']):
            y_true = group[y_col].to_numpy()
            y_pred = group[pred_col].to_numpy()
            
            for metric_name, metric_fn in selected_metric_functions.items():
                point_estimate = metric_fn(y_true, y_pred)
                
                boot_vals = []
                for _ in range(bootstrap_samples):
                    idx = rng.choice(len(group), size=len(group), replace=True) #, random_state = 42)
                    boot_vals.append(metric_fn(y_true[idx], y_pred[idx]))
                    
                boot_vals = np.asarray(boot_vals, dtype=float)
                
                mean_val = float(np.mean(boot_vals))
                std_val = float(np.std(boot_vals, ddof=1)) if boot_vals.size > 1 else 0.0
                upper_val = float(np.percentile(boot_vals, 97.5))
                lower_val = float(np.percentile(boot_vals, 2.5))
                
                code_level_records.append({
                    'Model': model_name,
                    'Train | Test': train_test,
                    'Code': code,
                    'Split': split,
                    'Metric': metric_name,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': lower_val,
                    'ci_upper': upper_val,
                    'point_estimate': float(point_estimate) if point_estimate == point_estimate else np.nan,  # keep NaN if needed
                    'n_samples': int(len(group)),
                })

    bootstrap_metrics = pd.DataFrame.from_records(code_level_records)
    return bootstrap_metrics

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
        train_tests = split_metrics["Train | Test"].unique()
        for model in models:
            for train_test in train_tests:
                subset = split_metrics[(split_metrics["Model"] == model) & (split_metrics["Train | Test"] == train_test)]
                if subset.empty:
                    continue
                entry = {}
                for metric in metrics_to_agg:
                    metric_row = subset[subset["Metric"] == metric]
                    if not metric_row.empty:
                        mean_val = metric_row["mean"].iloc[0]
                        std_val = metric_row["std"].iloc[0]
                        entry[metric] = f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        entry[metric] = "N/A"
                rows[(model, train_test)] = entry

        if not rows:
            print(f"\nNo data for {split} split.")
            continue

        # Create dataframe: metrics (rows), (Model, Train | Test) (columns)
        formatted = (
            pd.DataFrame(rows)
            .reindex(index=metrics_to_agg)
        )
        formatted.columns = pd.MultiIndex.from_tuples(formatted.columns, names=['Model', 'Train | Test'])
        formatted.index.name = 'Metric'
        print(f"\nMetrics for {split} split:")
        display(formatted)
