import os
import sys
import time
import uuid
import argparse
from pathlib import Path

# Third-party imports
import torch
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
from data import *
from predict import *
from evaluate import *
from edit import *
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
        is_quantile = getattr(self.args, 'output_mode', 'gaussian') == 'quantile' and self.args.model == 'NORMA2'

        total_loss = n_batches = 0
        y_list, mu_list = [], []

        for step, batch in tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False):
            batch = to_device_batch(batch, self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                output = self.model(
                    batch['x_h'], batch['s_h'], batch['t_h'], batch['sex'],
                    batch['age'], batch['cid'], batch['s_next'], batch['t_next'], batch['pad_mask']
                )

                if is_quantile:
                    q_pred = output  # (B, n_quantiles)
                    loss = compute_loss(q_pred, None, batch['x_next'], self.criterion)
                    # Use median (index 2) for R2
                    mu_for_r2 = q_pred[:, 2:3]
                else:
                    mu, log_var = output
                    loss = compute_loss(mu, log_var, batch['x_next'], self.criterion)
                    mu_for_r2 = mu

                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            y_list.append(batch['x_next'].detach().cpu())
            mu_list.append(mu_for_r2.detach().cpu())
            n_batches += 1

        y_all = torch.cat(y_list, dim=0).view(-1).numpy()
        mu_all = torch.cat(mu_list, dim=0).view(-1).numpy()
        var = 'train' if is_training else 'val'
        self.metrics[var] = {
            'loss': total_loss / n_batches,
            'r2': r2_score(y_all, mu_all)
        }

    def _predict(self):
        train_seq, val_seq, test_seq = load_and_split_data(self.args.data_dir, self.args.test, self.args.sample, print_info=False, nstates=self.args.nstates)
        train_loader, val_loader, test_loader = create_dataloaders(train_seq, val_seq, test_seq, self.args.nstates, batch_size=self.args.batch_size, random_state=self.args.seed, normalize=self.args.normalize)
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
        metrics_df = evaluate_and_save_metrics(
            self.predictions_df, self.run_id,
            exclude={'CRP', 'GGT', 'LDH', 'PT'},
            metrics_to_agg=['MAE', 'MAPE', 'R2', 'MSE'],
            save_dir=save_dir
        )
        print('=' * 90)

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
            self._load_model(best=True)
            self.args.resume = False
        else:
            self.run_id = str(uuid.uuid4())[:8]
            self.args.resume = True
            self.best_val_loss = float('inf')
            self.metrics = {}
            self.epoch = 0
            self.model = create_model(self.args, len(TEST_VOCAB)).to(self.device)
            self.model.apply(initialize_weights_small)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=self.args.patience)
        self.criterion = create_loss(self.args.loss) 
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
        train_seq, val_seq, test_seq = load_and_split_data(self.args.data_dir, self.args.train, self.args.sample, nstates=self.args.nstates)
        train_loader, val_loader, test_loader = create_dataloaders(train_seq, val_seq, test_seq, self.args.nstates, batch_size=self.args.batch_size, random_state=self.args.seed, normalize=self.args.normalize)

        self._set_up_model()
        if self.args.resume: 
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
            
        self._save_model()
        self._predict()
        self._evaluate()
        self._sensitivity()

        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description='Train NORMA model (clean version)')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--num_patients', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='../data/processed/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--train', type=str, default='combined', choices=['EHRSHOT', 'MIMIC-IV', 'combined'])
    parser.add_argument('--test', type=str, default='combined', choices=['EHRSHOT', 'MIMIC-IV', 'combined'])
    parser.add_argument('--model', type=str, default='NormaLight', choices=['NormaLight', 'NORMA', 'NORMA2'])
    parser.add_argument('--loss', type=str, default='GaussianNLLLoss', choices=['NORMALoss', 'GaussianNLLLoss', 'MSELoss', 'QuantileLoss'])
    parser.add_argument('--output_mode', type=str, default='quantile', choices=['quantile', 'gaussian'],
                        help='Output mode for NORMA2: quantile (pinball loss) or gaussian (NLL loss)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4) # 4 for original, 2 for smaller model
    parser.add_argument('--nlayers', type=int, default=8) # 8 for original, 2 for smaller model
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--run_id', type=str, default=None) 
    parser.add_argument('--nstates', type=int, default=3)
    parser.add_argument('--edit', dest='edit', action='store_true', default=False)
    parser.add_argument('--predict', dest='predict', action='store_true', default=True)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=False,
                        help='Normalize x values by per-test reference range before training. '
                             'Predictions are denormalized back to original scale at inference.')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

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

