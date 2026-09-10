import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import TEST_VOCAB, INVERSE_TEST_VOCAB
from model import is_quantile_mode
from utils import run_model, to_device_batch

CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

def predict(model, device, train_loader, val_loader, test_loader, normalize=False):
    print('Generating predictions and computing metrics...')
    all_predictions = []
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"Evaluating {split_name} set...")
        predictions_df = get_predictions(model, device, loader, split_name, normalize=normalize)
        predictions_df['split'] = split_name
        all_predictions.append(predictions_df)
    return pd.concat(all_predictions)

def get_predictions(model, device, loader, split_name, normalize=False):
    """Get predictions from model for a given dataloader."""
    model.eval()
    model.to(device)

    # Detect quantile mode
    is_quantile = hasattr(model, 'output_mode') and is_quantile_mode(model.output_mode)

    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split_name} (predict)", leave=False):
            batch = to_device_batch(batch, device)

            output = run_model(model, batch)

            if is_quantile:
                # output: (B, 5) — [q2.5, q25, q50, q75, q97.5]
                q_np = output.cpu().numpy()  # (B, 5)
                mu_np = q_np[:, 2:3]         # median as point estimate
                # Approximate log_var from IQR: sigma ≈ (q97.5 - q2.5) / 3.92
                sigma_np = (q_np[:, 4:5] - q_np[:, 0:1]) / 3.92
                log_var_np = 2.0 * np.log(sigma_np + 1e-8)
            else:
                mu, log_var = output
                mu_np = mu.cpu().numpy()
                log_var_np = log_var.cpu().numpy()

            cid = batch['cid'].cpu().numpy()
            pids = batch['pids']
            x_next = batch['x_next'].cpu().numpy()
            t_next = batch['t_next'].cpu().numpy()
            s_next = batch['s_next'].cpu().numpy()
            n_hist = batch['n_hist'].cpu().numpy() if 'n_hist' in batch else None
            lp = getattr(model, 'last_params', None) if is_quantile else None
            gate_np = lp['gate'].detach().cpu().numpy() if lp and 'gate' in lp else None
            neff_np = lp['n_eff'].detach().cpu().numpy() if lp and 'n_eff' in lp else None

            if normalize and not is_quantile:
                ref_low = batch['ref_low'].cpu().numpy()
                ref_high = batch['ref_high'].cpu().numpy()
                span = ref_high - ref_low
                mu_np = mu_np * span + ref_low
                x_next = x_next * span + ref_low
                log_var_np = log_var_np + 2.0 * np.log(span + 1e-8)

            for i in range(len(mu_np)):
                cid_val = int(cid[i].item() if hasattr(cid[i], 'item') else cid[i])
                x_next_val = float(x_next[i].item() if hasattr(x_next[i], 'item') else x_next[i])
                t_next_val = float(t_next[i].item() if hasattr(t_next[i], 'item') else t_next[i])
                s_next_val = int(s_next[i].item() if hasattr(s_next[i], 'item') else s_next[i])
                mu_val = float(mu_np[i].item() if hasattr(mu_np[i], 'item') else mu_np[i])
                log_var_val = float(log_var_np[i].item() if hasattr(log_var_np[i], 'item') else log_var_np[i])
                row = {
                    'pid': pids[i],
                    'cid': cid_val,
                    'code': CODE_TO_TEST_NAME[cid_val],
                    'x_next': x_next_val,
                    't_next': t_next_val,
                    's_next': s_next_val,
                    'mu': mu_val,
                    'log_var': log_var_val,
                }
                if n_hist is not None:
                    row['n_hist'] = int(n_hist[i].item() if hasattr(n_hist[i], 'item') else n_hist[i])
                if gate_np is not None:
                    row['gate'] = float(gate_np[i].item())
                if neff_np is not None:
                    row['n_eff'] = float(neff_np[i].item())
                if is_quantile:
                    row['q025'] = float(q_np[i, 0])
                    row['q25']  = float(q_np[i, 1])
                    row['q50']  = float(q_np[i, 2])
                    row['q75']  = float(q_np[i, 3])
                    row['q975'] = float(q_np[i, 4])
                predictions.append(row)

    return pd.DataFrame(predictions)

def load_predictions(run_ids, base, source):
    """(path, target col, prediction col) per run_id, for interactive comparison.

    `base` is a cohort key for the forecasting baselines: they live in
    results/raw/<cohort>/05_forecast_baselines.parquet, one column per model
    (model/baselines/forecast.py)."""
    outputs = {}
    for run_id in run_ids:
        if run_id in ['Mean', 'ARIMA', 'last']:
            path = f'../results/raw/{base}/05_forecast_baselines.parquet'
            outputs[run_id] = (path, 'x_next', run_id.lower())
        elif run_id == '58ba1f1c':
            path = f'../model/logs/{run_id}/predictions_ehrshot.csv'
            outputs[run_id] = (path, 'x_next', 'q50')
        else:
            log_path = f'../model/logs/{run_id}/predictions_{source}.csv'
            pred_path = f'../model/predictions/{run_id}/predictions_{run_id}.csv'
            path = log_path if os.path.exists(log_path) else pred_path
            outputs[run_id] = (path, 'x_next', 'mu')
    return outputs