import torch
import wandb
import numpy as np
import os
from pathlib import Path
import json
import time
import uuid
from model import NORMADecoder, NormaLight, NormaLightDecoder
from loss import NORMALoss, GaussianNLLLoss, MSELoss
from data import TEST_VOCAB
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def setup_logging(args, run_id):
    wandb.init(
        project='NORMA',
        name=run_id, 
        config=vars(args), 
        resume = True
    )
    log_dir = Path(args.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    hyperparams = {
        **vars(args),
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    hyperparams_path = Path(args.log_dir) / 'hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    pass 

def save_checkpoint(model, optimizer, scheduler, args, run_id, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'run_id': run_id,
            'hyperparameters': vars(args),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, Path(args.log_dir)/ run_id / 'checkpoint_latest.pth')
        if is_best:
            torch.save(checkpoint, Path(args.log_dir)/ run_id / 'checkpoint_best.pth')

def load_checkpoint(args, best=False, device='cpu'):
    run_dir = Path(args.log_dir) / args.run_id
    latest_path = run_dir / 'checkpoint_latest.pth'
    best_path = run_dir / 'checkpoint_best.pth'
    ckp_path = best_path if best else latest_path
    
    checkpoint = torch.load(ckp_path, map_location=device)
    hparams = checkpoint['hyperparameters']
    
    print(f"Loading Model with Run ID: {args.run_id} ({ckp_path})")
    print(f"Epoch: {checkpoint['epoch']}, Validation Loss: {checkpoint['metrics']['val']['loss']:.4f}")
    print("=" * 90)
    return checkpoint, hparams
    
# def load_checkpoint(args, best=False, device='cpu'):
#     run_dir = Path(args.log_dir) / args.run_id
#     latest_path = run_dir / 'checkpoint_latest.pth'
#     best_path = run_dir / 'checkpoint_best.pth'
#     ckp_path = best_path if best else latest_path

#     checkpoint = torch.load(ckp_path, map_location=device)
#     model = create_model(args, len(TEST_VOCAB)).to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer = torch.optim.AdamW(model.parameters(), lr=checkpoint['hyperparameters']['lr'], weight_decay=checkpoint['hyperparameters']['weight_decay'])
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     epoch = checkpoint.get('epoch', -1)
#     metrics = checkpoint.get('metrics', {})
#     best_val_loss = metrics.get('val', {}).get('loss', float('inf'))
    
    # return model, optimizer, scheduler, epoch, metrics, best_val_loss

def to_device_batch(batch, device):
    """Move tensors to device and fix dtype for embedding indices if present."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    for key in ('sex', 'cid', 's_next', 's_h'):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].long()
    return batch

def create_model(args, num_lab_codes):
    if args.model == 'NormaLight':
        return NormaLight(d_model=args.d_model, nhead=args.nhead, num_layers=args.nlayers, num_lab_codes=num_lab_codes)
    if args.model == 'NORMADecoder':
        return NORMADecoder(d_model=args.d_model, nhead=args.nhead, num_layers=args.nlayers, num_lab_codes=num_lab_codes)
    if args.model == 'NormaLightDecoder':
        return NormaLightDecoder(d_model=args.d_model, nhead=args.nhead, num_layers=args.nlayers, num_lab_codes=num_lab_codes)
    raise ValueError(f"Unknown model type: {args.model}")
        
def initialize_weights_small(module):
    """Initialize weights to small values: Linear/Embedding ~ N(0, 0.02), biases zero, LN to 1/0."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
            
def create_loss(loss, lambda_align=None):
    if loss == 'NORMALoss':
        return NORMALoss(lambda_align=lambda_align)
    if loss == 'GaussianNLLLoss':
        return GaussianNLLLoss()
    if loss == 'MSELoss':
        return MSELoss()
    raise ValueError(f"Unknown loss type: {loss}")

def compute_loss(mu, log_var, y_true, criterion, extra: dict = None):
    if isinstance(criterion, NORMALoss):
        if extra is None or not all(k in extra for k in ('s_next', 'ref_mu', 'ref_var')):
            raise ValueError('NORMALoss requires extra keys: s_next, ref_mu, ref_var')
        return criterion(mu, log_var, y_true, extra['s_next'], extra['ref_mu'], torch.sqrt(extra['ref_var']))
    if isinstance(criterion, GaussianNLLLoss):
        return criterion(mu, log_var, y_true)
    if isinstance(criterion, MSELoss):
        return criterion(mu, y_true)
    raise ValueError('Unknown loss type')

def log_epoch(metrics, epoch_index: int, total_epochs: int, lr: float, epoch_time: float, is_best: bool):
    """Log metrics to wandb and print a concise console summary."""
    metrics['epoch'] = epoch_index + 1
    metrics['lr'] = lr
    wandb.log(metrics)
    s = f"Epoch {epoch_index + 1}/{total_epochs} ({epoch_time:.1f}s), LR: {lr:.0e}, Train Loss: {metrics['train']['loss']:.2f}, Val Loss: {metrics['val']['loss']:.2f}, Train R2: {metrics['train']['r2']:.2f}, Val R2: {metrics['val']['r2']:.2f}"
    if is_best:
        s += " (Best Model)"
    print(s)
    