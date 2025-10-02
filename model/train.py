import argparse
import os
import time
import json
import uuid
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import pandas as pd

from model import NORMADecoder, NormaLight
from loss import NORMALoss, GaussianNLLLoss
from utils import TEST_VOCAB
from data import create_dataloaders, load_data
from evaluate import get_predictions, get_metrics, save_predictions_and_metrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def create_model(model_type, d_model, nhead, num_layers, num_lab_codes):
    """Create model based on type."""
    if model_type == 'NormaLight':
        return NormaLight(d_model=d_model, nhead=nhead, num_layers=num_layers, num_lab_codes=num_lab_codes)
    elif model_type == 'NORMADecoder':
        return NORMADecoder(d_model=d_model, nhead=nhead, num_layers=num_layers, num_lab_codes=num_lab_codes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_loss(loss_type, lambda_align=0.01):
    """Create loss function based on type."""
    if loss_type == 'NORMALoss':
        return NORMALoss(lambda_align=lambda_align)
    elif loss_type == 'GaussianNLLLoss':
        return GaussianNLLLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class NORMATrainer:
    """Trainer class for NORMA model with wandb integration."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            
        # Only setup logging components that don't require model
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging components that don't require model."""
        # Setup wandb and run ID
        self.run_id = str(uuid.uuid4())[:8]
        run_name = self.args.wandb_name or self.run_id
        
        wandb.init(
            project=self.args.wandb_project,
            name=run_name,
            config=vars(self.args)
        )
        wandb.watch_called = False
        
        # Setup checkpointing
        self.log_dir = Path(self.args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=self.args.patience)
        
    def _setup_model(self, num_lab_codes):
        """Setup model and training components after data loading."""
        # Setup model using factory function
        self.model = create_model(
            model_type=self.args.model_type,
            d_model=self.args.d_model,
            nhead=self.args.nhead,
            num_layers=self.args.num_layers,
            num_lab_codes=num_lab_codes
        ).to(self.device)
        print(f"Model ({self.args.model_type}) initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Number of lab codes: {num_lab_codes}")
        
        # Setup training components
        self.criterion = create_loss(self.args.loss_type, lambda_align=self.args.lambda_align)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def _extract_metrics(self, total_loss, total_forecast_loss, total_align_loss, num_batches):
        """Extract and format metrics from training/validation."""
        return {
            'loss': total_loss / num_batches,
            'forecast_loss': total_forecast_loss / num_batches,
            'align_loss': total_align_loss / num_batches
        }
        
    def _create_metrics_dict(self, train_metrics, val_metrics, epoch=None, learning_rate=None):
        """Create metrics dictionary for logging and checkpointing."""
        metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_forecast_loss': train_metrics['forecast_loss'],
            'train_align_loss': train_metrics['align_loss'],
            'val_forecast_loss': val_metrics['forecast_loss'],
            'val_align_loss': val_metrics['align_loss']
        }
        
        if epoch is not None:
            metrics.update({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/forecast_loss': train_metrics['forecast_loss'],
                'train/align_loss': train_metrics['align_loss'],
                'val/loss': val_metrics['loss'],
                'val/forecast_loss': val_metrics['forecast_loss'],
                'val/align_loss': val_metrics['align_loss'],
                'learning_rate': learning_rate
            })
        
        return metrics
        
    def _process_batch(self, batch, is_training=True):
        """Process a single batch (training or validation)."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Ensure categorical variables are long tensors for embeddings
        batch['sex'] = batch['sex'].long()
        batch['cid'] = batch['cid'].long()
        batch['s_next'] = batch['s_next'].long()
        batch['s_h'] = batch['s_h'].long()
        
        if is_training:
            self.optimizer.zero_grad()
        
        # Use the batch processor to get the correct inputs for the model
        mu, log_var = self.model(
                batch['x_h'], batch['s_h'], batch['t_h'], batch['sex'], 
                batch['cid'], batch['s_next'], batch['t_next'], batch['pad_mask']
            )
        
        # Handle different loss function interfaces
        y_true = batch['x_next']  # Target values
        
        if isinstance(self.criterion, NORMALoss):
            # NORMALoss expects additional parameters
            condition = batch['s_next']
            ref_mu = batch['ref_mu']
            ref_var = batch['ref_var']
            loss_val, forecast_loss, align_loss = self.criterion(
                mu, log_var, y_true, condition, ref_mu, torch.sqrt(ref_var)
            )
        else:
            # Simple Gaussian NLL loss
            loss_val = self.criterion(mu, y_true, torch.exp(log_var))
            forecast_loss = align_loss = loss_val
        
        if is_training:
            loss_val.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        
        return loss_val.item(), forecast_loss.item(), align_loss.item()
    
    def _run_epoch(self, loader, is_training=True):
        """Run one epoch (training or validation)."""
        if is_training:
            self.model.train()
            desc = "Training"
        else:
            self.model.eval()
            desc = "Validation"
        
        total_loss = total_forecast_loss = total_align_loss = num_batches = 0
        
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            if is_training:
                loss_val, forecast_val, align_val = self._process_batch(batch, is_training=True)
                pbar.set_postfix({
                    'Loss': f'{loss_val:.4f}',
                    'Forecast': f'{forecast_val:.4f}',
                    'Align': f'{align_val:.4f}'
                })
            else:
                with torch.no_grad():
                    loss_val, forecast_val, align_val = self._process_batch(batch, is_training=False)
            
            total_loss += loss_val
            total_forecast_loss += forecast_val
            total_align_loss += align_val
            num_batches += 1
        
        return self._extract_metrics(total_loss, total_forecast_loss, total_align_loss, num_batches)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        return self._run_epoch(train_loader, is_training=True)
        
    def validate(self, val_loader):
        """Validate the model."""
        return self._run_epoch(val_loader, is_training=False)
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint with run ID."""
        checkpoint = {
            'epoch': epoch,
            'run_id': self.run_id,
            'hyperparameters': vars(self.args),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.log_dir / f'checkpoint_latest_{self.run_id}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / f'checkpoint_best_{self.run_id}.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch} with ID: {self.run_id}")
            
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract run ID and hyperparameters
        self.run_id = checkpoint.get('run_id', 'unknown')
        hyperparams = checkpoint.get('hyperparameters', {})
        
        print(f"Loading checkpoint for run ID: {self.run_id}")
        print(f"Hyperparameters: {hyperparams}")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
        
    def train(self):
        print("Loading data...")
        df = load_data(self.args.data_path, num_patients=self.args.num_patients, min_points=self.args.min_points)
        cbcs = ['HBC', 'HCT'] #, 'HGB', 'MCV', 'MCH', 'MCHC', 'RDW']
        df_cbc = df.query("test_name in @cbcs")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            df_cbc, batch_size=self.args.batch_size, random_state=self.args.seed
        )
        num_lab_codes = len(TEST_VOCAB)  # Use TEST_VOCAB size instead of unique test names
        
        self._setup_model(num_lab_codes)
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        start_epoch = 0
        best_val_loss = float('inf')
        
        if self.args.resume:
            # Find the most recent checkpoint
            checkpoint_files = list(self.log_dir.glob('checkpoint_latest_*.pth'))
            if checkpoint_files:
                checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                print(f"Resuming from {checkpoint_path}")
                start_epoch, metrics = self.load_checkpoint(checkpoint_path)
                best_val_loss = metrics['val_loss']
                
        print(f"\nStarting training from epoch {start_epoch + 1}")
        print("=" * 60)
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step(val_metrics['loss'])
            
            epoch_time = time.time() - epoch_start_time
            learning_rate = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch + 1}/{self.args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Forecast: {train_metrics['forecast_loss']:.4f}, "
                  f"Align: {train_metrics['align_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(Forecast: {val_metrics['forecast_loss']:.4f}, "
                  f"Align: {val_metrics['align_loss']:.4f})")
            print(f"  LR: {learning_rate:.6f}")
            
            # Create metrics once
            all_metrics = self._create_metrics_dict(train_metrics, val_metrics, epoch, learning_rate)
            
            # Log to wandb
            wandb.log(all_metrics)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                
            self.save_checkpoint(epoch, all_metrics, is_best)
            
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
                
            print("-" * 60)
            
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        
        # Run evaluation on all splits
        print("\n" + "="*60)
        print("RUNNING EVALUATION")
        print("="*60)
        
        self.evaluate(train_loader, val_loader, test_loader)
        
        wandb.finish()

    def evaluate(self, train_loader, val_loader, test_loader):
        """Evaluate model on all splits and save results."""
        print("Generating predictions and computing metrics...")
        
        # Evaluate each split
        for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            print(f"\nEvaluating {split_name} set...")
            predictions_df = get_predictions(self.model, loader, self.device, self.args.model_type)
            print(predictions_df.head())
            
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NORMA model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default="../../SETPOINT/data/processed/lab_measurements.csv")
    parser.add_argument('--min_points', type=int, default=5)
    parser.add_argument('--num_patients', type=int, default=None)
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='NormaLight',
                        choices=['NormaLight', 'NORMADecoder'])
    parser.add_argument('--loss_type', type=str, default='GaussianNLLLoss',
                        choices=['NORMALoss', 'GaussianNLLLoss'])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lambda_align', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='norma-training')
    parser.add_argument('--wandb_name', type=str, default=None)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    trainer = NORMATrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()