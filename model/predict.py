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

def load_predictions(run_ids, base, source):
    """
    Build dictionary of model configs for evaluation.
    Returns dict: {run_id: (path, true_col, pred_col)}
    """
    outputs = {}
    for run_id in run_ids:
        if run_id in ['Mean', 'ARIMA', 'last']:
            path = f'../ARIMA/predictions/{run_id.lower()}_baseline_{base}.csv'
            outputs[run_id] = (path, 'x_next', 'x_pred')
        else:
            log_path = f'../model/logs/{run_id}/predictions_{source}.csv'
            pred_path = f'../model/predictions/{run_id}/predictions_{run_id}.csv'
            path = log_path if os.path.exists(log_path) else pred_path
            outputs[run_id] = (path, 'x_next', 'mu')
    return outputs