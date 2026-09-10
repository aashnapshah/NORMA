import gc
import os
import sys
import time
import uuid
import argparse
from pathlib import Path

# Third-party imports
import torch
# Co-analyte batches carry extra tensors per item; with the default "file_descriptor" sharing
# strategy the 4 workers exhausted fds ("received 0 items of ancdata", q_co / q_age_set_co
# 2026-08-28). file_system sharing uses shm names instead.
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import tempfile
import base64
from sklearn.metrics import r2_score
import wandb

# Project/module imports
from model import *
from loss import *
from model import is_quantile_mode
from data import *
from predict import *
from evaluate import *
try:
    from edit import *  # legacy counterfactual helpers; needs scripts/plots.py which is not always present
except ImportError:
    pass
from utils import *

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
    
class NORMATrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print('=' * 90)
        
    def _run_epoch(self, loader, is_training=True):
        self.model.train() if is_training else self.model.eval()
        desc = 'Training' if is_training else 'Validation'
        is_quantile = is_quantile_mode(getattr(self.args, 'output_mode', 'gaussian')) and self.args.model == 'NORMA2'

        total_loss = n_batches = 0
        y_list, mu_list, lo_list, hi_list = [], [], [], []

        for step, batch in tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False):
            batch = to_device_batch(batch, self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                output = run_model(self.model, batch)

                if is_quantile:
                    q_pred = output  # (B, n_quantiles)
                    loss = compute_loss(q_pred, None, batch['x_next'], self.criterion, loss_extras(batch, self.model))
                    # Use median (index 2) for R2
                    mu_for_r2 = q_pred[:, 2:3]
                    lo, hi = q_pred[:, 0:1], q_pred[:, 4:5]
                else:
                    mu, log_var = output
                    loss = compute_loss(mu, log_var, batch['x_next'], self.criterion, loss_extras(batch, self.model))
                    mu_for_r2 = mu
                    sd = torch.exp(0.5 * log_var)
                    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd

                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            y_list.append(batch['x_next'].detach().cpu())
            mu_list.append(mu_for_r2.detach().cpu())
            lo_list.append(lo.detach().cpu())
            hi_list.append(hi.detach().cpu())
            n_batches += 1

        y_all = torch.cat(y_list, dim=0).view(-1).numpy()
        mu_all = torch.cat(mu_list, dim=0).view(-1).numpy()
        lo_all = torch.cat(lo_list, dim=0).view(-1).numpy()
        hi_all = torch.cat(hi_list, dim=0).view(-1).numpy()
        var = 'train' if is_training else 'val'
        self.metrics[var] = {
            'loss': total_loss / n_batches,
            'r2': r2_score(y_all, mu_all),
            'mae': float(np.mean(np.abs(y_all - mu_all))),
            'coverage95': float(np.mean((y_all >= lo_all) & (y_all <= hi_all))),
            'width': float(np.mean(hi_all - lo_all)),
        }

    def _data_version(self):
        """Per-measurement covariates only exist in the v3 sequences (process/covariates.py)."""
        version = getattr(self.args, 'data_version', 'v2')
        needs_v3 = any(getattr(self.args, f, False) for f in ('use_age_t', 'use_setting', 'use_coanalytes'))
        if needs_v3 and version != 'v3':
            print(f"Covariate flags set: switching --data_version {version} -> v3")
            version = 'v3'
            self.args.data_version = version
        return version

    def _load_data(self, source):
        version = self._data_version()
        train_seq, val_seq, test_seq = load_and_split_data(
            self.args.data_dir, source, self.args.sample, nstates=self.args.nstates, version=version,
            split_by=getattr(self.args, 'split_by', 'sequence'))
        self._set_prior_table(train_seq)
        full = bool(getattr(self.args, 'use_full_panel', False))
        panel = load_panel(self.args.data_dir, source, version) if getattr(self.args, 'use_coanalytes', False) else None
        drawmeta = load_drawmeta(self.args.data_dir, source, version) if full else None
        if full and drawmeta is None:
            raise FileNotFoundError(
                f"--use_full_panel needs {source}_drawmeta_{version}.npz in {self.args.data_dir} "
                f"(build it with python process/draw_meta.py)")
        if getattr(self.args, 'use_coanalytes', False) and panel is None:
            raise FileNotFoundError(f"--use_coanalytes needs {source}_panel_{version}.npy in {self.args.data_dir}")
        loaders = create_dataloaders(train_seq, val_seq, test_seq, self.args.nstates, batch_size=self.args.batch_size,
                                     random_state=self.args.seed, normalize=self.args.normalize, panel=panel,
                                     drawmeta=drawmeta, max_draws=getattr(self.args, 'max_draws', 128))
        return loaders

    def _set_prior_table(self, train_seq):
        """Fill NORMA2's state-conditional prior buffers from the training split (gate /
        NIG heads). Restored from the checkpoint on resume / eval, so only built once."""
        model = getattr(self, 'model', None)
        if model is None or not hasattr(model, 'prior_ready') or float(model.prior_ready) > 0:
            return
        if self.args.output_mode not in ('gate', 'nig'):
            return
        from priors import build_prior_table, save_prior_table
        table = build_prior_table(train_seq, nstates=self.args.nstates, ncodes=model.prior_q.shape[0])
        model.set_prior_table(table)
        model.nig_nu0 = float(getattr(self.args, 'nig_nu0', 5.0))
        save_prior_table(table, os.path.join(self.args.log_dir, self.run_id, 'prior_table.pt'))
        print(f"Prior table built from {len(train_seq)} training sequences "
              f"(rho median {float(table['rho'].median()):.2f})")

    def _predict(self):
        train_loader, val_loader, test_loader = self._load_data(self.args.test)
        print(f'Performing Prediction and Evaluation on {self.args.test.title()}...')

        predictions_df = predict(self.model, self.device, train_loader, val_loader, test_loader, normalize=self.args.normalize)
        self.predictions_df = predictions_df
        predictions_df.to_csv(os.path.join(self.args.log_dir, self.run_id, f"predictions_{self.args.test.lower()}.csv"), index=False)
        
        print(f"Predictions saved to {os.path.join(self.args.log_dir, self.run_id, f'predictions_{self.args.test.lower()}.csv')}")
        print('=' * 90)
        
        # print(f'Performing Counterfactual Prediction on {self.args.test.title()}...')
        # counterfactual_predictions_df = predict_cf(self.model, self.device, train_loader, val_loader, test_loader, normalize=self.args.normalize)
        # counterfactual_predictions_df.to_csv(os.path.join(self.args.log_dir, self.run_id, f"counterfactual_predictions_{self.args.test.lower()}.csv"), index=False)
        # print(f"Counterfactual predictions saved to {os.path.join(self.args.log_dir, self.run_id, f'counterfactual_predictions_{self.args.test.lower()}.csv')}")
        # print('=' * 90)
  
    # def _edit(self):
    #     train_seq, val_seq, test_seq = load_and_split_data(self.args.data_dir, self.args.test, self.args.sample, print_info=False, nstates=getattr(self.args, 'nstates', 2))
    #     train_loader, val_loader, test_loader = create_dataloaders(train_seq, val_seq, test_seq, getattr(self.args, 'nstates', 2), batch_size=self.args.batch_size, random_state=self.args.seed)
        
    #     print(f'Performing Counterfactual Prediction on {self.args.test.title()}...')
    #     counterfactual_predictions_df = predict_cf(self.model, self.device, train_loader, val_loader, test_loader)
    #     counterfactual_predictions_df.to_csv(os.path.join(self.args.log_dir, self.run_id, f"counterfactual_predictions_{self.args.test.lower()}.csv"), index=False)
    #     print(f"Counterfactual predictions saved to {os.path.join(self.args.log_dir, self.run_id, f'counterfactual_predictions_{self.args.test.lower()}.csv')}")
    #     print('=' * 90)
        
    def _evaluate(self):
        print(f'Performing Evaluation...')
        save_dir = os.path.join(self.args.log_dir, self.run_id)
        exclude = {'CRP', 'GGT', 'LDH', 'PT'}
        metrics_df = evaluate_and_save_metrics(
            self.predictions_df, self.run_id,
            exclude=exclude,
            metrics_to_agg=['MAE', 'MAPE', 'R2', 'MSE'],
            save_dir=save_dir
        )
        # Calibration of the 95% interval per analyte / queried state (test split)
        is_quantile = 'q50' in self.predictions_df.columns
        calib = calibration_by_analyte(self.predictions_df, is_quantile, exclude=exclude, split='test')
        calib.to_csv(os.path.join(save_dir, 'calibration_test.csv'), index=False)
        self._wandb_log_eval(metrics_df, calib)
        print('=' * 90)

    def _wandb_log_eval(self, overall, calib):
        """Summary metrics + tables so ablation arms can be compared in W&B."""
        if wandb.run is None:
            return
        try:
            for _, r in overall.iterrows():
                wandb.run.summary[f"{r['split']}/{r['metric']}"] = float(r['mean'])
                wandb.run.summary[f"{r['split']}/{r['metric']}_ci_lower"] = float(r['ci_lower'])
                wandb.run.summary[f"{r['split']}/{r['metric']}_ci_upper"] = float(r['ci_upper'])
            normal = calib[calib['state'] == 'normal']
            wandb.run.summary['test/coverage95_normal'] = float(normal['coverage95'].mean())
            wandb.run.summary['test/width_rel_normal'] = float(normal['width_rel'].mean())
            wandb.run.summary['test/inside_pop_normal'] = float(normal['inside_pop'].mean())
            wandb.run.summary['test/coverage95_all'] = float(calib['coverage95'].mean())
            by_code = pd.read_csv(os.path.join(self.args.log_dir, self.run_id, 'bootstrap_metrics_by_code.csv'))
            wandb.log({'eval/by_code': wandb.Table(dataframe=by_code),
                       'eval/calibration': wandb.Table(dataframe=calib)})
        except Exception as exc:  # never let logging kill a finished run
            print(f'  (W&B eval logging skipped: {exc})')

    def _sensitivity(self):
        print(f'Running Sensitivity Analysis...')
        from sensitivity_analysis import init_model, run_sweeps

        init_model(model=self.model, device=self.device, hparams=self.args)
        results_df = run_sweeps()

        save_dir = os.path.join(self.args.log_dir, self.run_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'sensitivity_results.csv')
        results_df.to_csv(save_path, index=False)
        print(f'  Sensitivity results ({len(results_df)} rows) saved to {save_path}')
        if wandb.run is not None:
            try:
                wandb.log({'sensitivity/results': wandb.Table(dataframe=results_df)})
            except Exception as exc:
                print(f'  (W&B sensitivity logging skipped: {exc})')
        print('=' * 90)
        
    def _load_model(self, best=False):
        checkpoint, self.args = load_checkpoint(self.args.log_dir, self.args.run_id, self.args, best=best, device=self.device)
        self.run_id = self.args.run_id
        self.model = create_model(self.args, ncodes=len(TEST_VOCAB), checkpoint=checkpoint).to(self.device)
        self.epoch = checkpoint.get('epoch') + 1
        self.metrics = checkpoint.get('metrics', {})

        if self.args.resume:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.epoch = checkpoint.get('epoch') + 1
            self.metrics = checkpoint.get('metrics', {})
            self.best_val_loss = self.metrics['best_val_loss'] if 'best_val_loss' in self.metrics else self.metrics['val']['loss']
    
    def _set_up_model(self):
        if self.args.run_id and self.args.resume:
            self._load_model(best=False)
        elif self.args.run_id and not self.args.resume:
            # Eval-only rerun: default to checkpoint_latest — a completed run evaluates the
            # in-memory model after the last trained epoch, which IS checkpoint_latest
            # (and the published 334f7e21 predictions are latest too).
            self._load_model(best=(getattr(self.args, 'eval_checkpoint', 'latest') == 'best'))
            self.args.resume = False
        else:
            self.run_id = getattr(self.args, 'run_name', None) or str(uuid.uuid4())[:8]
            if os.path.exists(os.path.join(self.args.log_dir, self.run_id, 'checkpoint_latest.pth')):
                raise FileExistsError(f"run '{self.run_id}' already exists in {self.args.log_dir}; "
                                      f"use --run_id {self.run_id} --resume to continue it")
            self.args.resume = True
            self.best_val_loss = float('inf')
            self.metrics = {}
            self.epoch = 0
            self.model = create_model(self.args, len(TEST_VOCAB)).to(self.device)
            self.model.apply(initialize_weights_small)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=self.args.patience)
        self.criterion = create_loss(self.args.loss, args=self.args)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Summary:")
        print(f"  Run ID          : {self.run_id}")
        print(f"  Model Type      : {self.args.model}")
        print(f"  Num Parameters  : {num_params:,}")
        print(f"  Embedding Dim   : {self.args.d_model}")
        print(f"  Num Heads       : {self.args.nhead}")
        print(f"  Num Layers      : {self.args.nlayers}")
        print(f"  Loss Function   : {self.args.loss}")
        print(f"  Learning Rate   : {self.args.lr}")
        print(f"  Lab Codes       : {len(TEST_VOCAB)}")
        cov = [f for f in ('use_age_t', 'use_setting', 'use_coanalytes', 'query_coanalytes') if getattr(self.args, f, False)]
        print(f"  Covariates      : {', '.join(cov) if cov else 'none'} (data {getattr(self.args, 'data_version', 'v2')})")
        print("=" * 90)
    
    def _save_model(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmpfile:
            save_file(self.model.state_dict(), tmpfile.name)
            tmpfile.seek(0)
            b64_data = base64.b64encode(tmpfile.read()).decode('utf-8')

        output_path = os.path.join(self.args.log_dir, self.run_id, 'model.safetensors.b64')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(b64_data)
            print(f"Model saved to {output_path}")
            print('=' * 90)
            
    def train(self):
        self._set_up_model()
        if self.args.resume:
            # Loaders live only while training: _predict() reloads the data itself, and two
            # copies of the v3 sequences + two sets of persistent workers exceed 128G
            # (q_age 51580118 / q_co 51683437 OOM-killed in the final eval, MaxRSS 133.5G).
            train_loader, val_loader, test_loader = self._load_data(self.args.train)
            setup_logging(self.args, self.run_id)
            print('=' * 90)
            for epoch in range(self.epoch, self.args.epochs):
                epoch_start_time = time.time()
                
                self.current_epoch = epoch
                self._run_epoch(train_loader, is_training=True)
                self._run_epoch(val_loader, is_training=False)
                self.scheduler.step(self.metrics['val']['loss'])
                
                epoch_time = time.time() - epoch_start_time
                lr = self.optimizer.param_groups[0]['lr']

                is_best = self.metrics['val']['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = self.metrics['val']['loss']
                    self.metrics['best_val_loss'] = self.best_val_loss
                    
                log_epoch(self.metrics, epoch, self.args.epochs, lr, epoch_time, is_best)
                save_checkpoint(self.model, self.optimizer, self.scheduler, self.args, self.run_id, epoch, self.metrics, is_best)

                if self.early_stopping(self.metrics['val']['loss']):
                    print(f"\nEarly Stopping Triggered at Epoch {epoch + 1}")
                    break

                print('-' * 90)
                
            print(f"\nTraining Completed! Best Validation Loss: {self.best_val_loss:.4f}")
            print('\n' + '=' * 90)

            del train_loader, val_loader, test_loader
            gc.collect()


        self._save_model()
        self._predict()
        self._evaluate()
        self._sensitivity()

        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description='Train NORMA model (clean version)')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--num_patients', type=int, default=None)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed'),
                        help='sequence pickles; default = <repo>/../data/processed (same layout evaluate.py assumes)')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--train', type=str, default='combined', choices=['EHRSHOT', 'MIMIC-IV', 'combined'])
    parser.add_argument('--test', type=str, default='combined', choices=['EHRSHOT', 'MIMIC-IV', 'combined'])
    parser.add_argument('--model', type=str, default='NormaLight', choices=['NormaLight', 'NORMA', 'NORMA2'])
    parser.add_argument('--loss', type=str, default='GaussianNLLLoss', choices=['NORMALoss', 'GaussianNLLLoss', 'MSELoss', 'QuantileLoss', 'QuantilePriorLoss', 'StudentTNLLLoss'])
    # Prior-anchored losses (loss.py QuantilePriorLoss / NORMALoss); see run_prior_ablation.sh
    parser.add_argument('--prior_lambda', type=float, default=1.0,
                        help='QuantilePriorLoss: weight of the population-prior term relative to pinball')
    parser.add_argument('--prior_k', type=float, default=5.0,
                        help='QuantilePriorLoss: prior strength in observations, w(n) = k / (n + k)')
    parser.add_argument('--prior_mode', type=str, default='anchor', choices=['anchor', 'floor', 'gate'],
                        help='QuantilePriorLoss: anchor = expected pinball under the population prior '
                             '(normal-state queries); floor = soft width floor, every state; '
                             'gate = Bernoulli KL of the learned gate toward n/(n+k) (needs --output_mode gate)')
    parser.add_argument('--prior_tau', type=float, default=None,
                        help='time constant (units of t) for a time-decayed history count in the prior weight')
    parser.add_argument('--align_by_n', action='store_true', default=False,
                        help='NORMALoss: KL weight k/(n+k) with --prior_k instead of the variance ratio')
    parser.add_argument('--nig_nu0', type=float, default=5.0,
                        help='NIG head: prior pseudo-observations for the within-person variance')
    parser.add_argument('--lambda_align', type=float, default=0.01,
                        help='NORMALoss (Gaussian head): weight of the KL alignment to the population interval')
    parser.add_argument('--output_mode', type=str, default='quantile', choices=['quantile', 'gaussian', 'gate', 'nig'],
                        help='Output mode for NORMA2: quantile (pinball loss), gaussian (NLL loss), '
                             'gate (quantiles gated toward the state prior; QuantilePriorLoss --prior_mode gate), '
                             'nig (conjugate normal-inverse-gamma head; StudentTNLLLoss)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4) # 4 for original, 2 for smaller model
    parser.add_argument('--nlayers', type=int, default=8) # 8 for original, 2 for smaller model
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--eval_checkpoint', type=str, default='latest', choices=['latest', 'best'],
                        help='checkpoint for an eval-only run (--run_id without --resume); '
                             'latest reproduces what a completed training run evaluates')
    parser.add_argument('--nstates', type=int, default=3)
    parser.add_argument('--edit', dest='edit', action='store_true', default=False)
    parser.add_argument('--predict', dest='predict', action='store_true', default=True)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=False,
                        help='Normalize x values by per-test reference range before training. '
                             'Predictions are denormalized back to original scale at inference.')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    # Per-measurement covariates (revision ablation; require v3 sequences from process/covariates.py)
    parser.add_argument('--data_version', type=str, default='v2', choices=['v2', 'v3'],
                        help='sequence pickle version; v3 adds age_t/setting/draw_idx per measurement')
    parser.add_argument('--use_age_t', action='store_true', default=False,
                        help='age at each draw on history tokens + age at query on the query token')
    parser.add_argument('--use_setting', action='store_true', default=False,
                        help='care-setting embedding on history tokens and the query token')
    parser.add_argument('--use_full_panel', action='store_true', default=False,
                        help="condition on every draw in the patient's past, not just the "
                             'timestamps where the target analyte was drawn; implies '
                             '--use_coanalytes and needs {source}_drawmeta_v3.npz')
    parser.add_argument('--max_draws', type=int, default=128,
                        help='cap on history tokens in --use_full_panel mode (most recent kept)')
    parser.add_argument('--causal_memory', action='store_true', default=False,
                        help='mask the cross-attention block as well as self-attention; '
                             'changes predictions, so leave off to reproduce published runs')
    parser.add_argument('--split_by', choices=['sequence', 'patient'], default='sequence',
                        help="'sequence' = published patient-analyte split; 'patient' = whole patients "
                             "held out, required when conditioning on a patient's other analytes")
    parser.add_argument('--use_coanalytes', action='store_true', default=False,
                        help='same-draw co-analyte panel (PopRI-normalised + mask) on history tokens')
    parser.add_argument('--query_coanalytes', action='store_true', default=False,
                        help='also give the query token the most recent prior panel, through the '
                             'same co_proj (no new parameters). Requires --use_coanalytes.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='human-readable id for a NEW run (log dir + W&B name), e.g. q_age_set')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B group (e.g. covariate-ablation)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)
    args = parser.parse_args()
    if getattr(args, 'use_full_panel', False):
        # full-panel tokens carry the co-analyte block, so co_proj must exist. This has
        # to happen before _set_up_model() -> create_model(), which train() calls ahead
        # of _load_data(); setting it later builds a model with no co_proj and silently
        # drops co_h in NORMA2.forward.
        args.use_coanalytes = True
    if getattr(args, 'query_coanalytes', False) and not getattr(args, 'use_coanalytes', False):
        raise SystemExit('--query_coanalytes requires --use_coanalytes (there is no co_proj without it)')
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    trainer = NORMATrainer(args)
    trainer.train()

def sweep_main():
    """Entry point for wandb sweep agent. Reads hyperparams from wandb.config."""
    wandb.init()
    args = parse_args()
    # Override args with sweep config values
    for key, val in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, val)
    set_seed(args.seed)
    trainer = NORMATrainer(args)
    trainer.train()

if __name__ == '__main__':
    import sys
    if '--sweep' in sys.argv:
        sys.argv.remove('--sweep')
        sweep_main()
    else:
        main()

