import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import TEST_VOCAB
from utils import *

CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

def predict(model, device, train_loader, val_loader, test_loader):
    print('Generating predictions and computing metrics...')
    all_predictions = []
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"Evaluating {split_name} set...")
        predictions_df = get_predictions(model, device, loader)
        predictions_df['split'] = split_name
        all_predictions.append(predictions_df)
    return pd.concat(all_predictions)
        
def get_predictions(model, device, loader):
    """Get predictions from model for a given dataloader."""
    model.eval()
    model.to(device)
    
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
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

def get_metrics(predictions_df):
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
    return overall_metrics, per_test_metrics

def save_predictions_and_metrics(predictions_df, overall_metrics, per_test_metrics, save_dir, split_name, run_id):
    """Save predictions and metrics to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_path = save_dir / f'predictions_{split_name}_{run_id}.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save per-test metrics
    metrics_path = save_dir / f'metrics_{split_name}_{run_id}.csv'
    per_test_metrics.to_csv(metrics_path, index=False)
    
    print(f"Saved {split_name} predictions to {predictions_path}")
    print(f"Saved {split_name} metrics to {metrics_path}")
    
    return predictions_path, metrics_path