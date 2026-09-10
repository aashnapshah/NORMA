import torch
import numpy as np
import os
from pathlib import Path
import json
import time
import uuid
import argparse

from model import NORMA2, is_quantile_mode, QUANTILE_OUTPUT_MODES
from legacy import NORMA, NormaLight, NormaLightV1   # old checkpoints only
from loss import NORMALoss, GaussianNLLLoss, MSELoss, QuantileLoss, QuantilePriorLoss, StudentTNLLLoss
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def setup_logging(args, run_id):
    import wandb
    if wandb.run is None:
        wandb.init(
            project='NORMA',
            name=run_id,
            config=vars(args),
            id = run_id,
            resume = 'allow',
            group=getattr(args, 'wandb_group', None),
            tags=getattr(args, 'wandb_tags', None) or None,
        )
    else:
        # Already initialized (e.g. by sweep agent) — just update config
        wandb.config.update(vars(args), allow_val_change=True)
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
    save_dir = Path(args.log_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_dir / 'checkpoint_latest.pth')
    
    checkpoint_json = {
        k: v for k, v in checkpoint.items()
        if not k.endswith('_state_dict')  # skip tensors for json
    }
    with open(save_dir / 'checkpoint_latest.json', 'w') as f:
        json.dump(checkpoint_json, f, indent=2)

    # Save model parameters as a separate file for easier inspection/sharing
    # This will save just the model "named parameters"
    model_params = {name: param.detach().cpu().numpy().tolist() for name, param in model.named_parameters()}
    with open(save_dir / 'model_parameters.json', 'w') as f:
        json.dump(model_params, f)

    if is_best:
        torch.save(checkpoint, save_dir / 'checkpoint_best.pth')
        with open(save_dir / 'checkpoint_best.json', 'w') as f:
            json.dump(checkpoint_json, f, indent=2)
        # Save best model parameters as well
        with open(save_dir / 'model_parameters_best.json', 'w') as f:
            json.dump(model_params, f)

def load_checkpoint(log_dir, run_id, args=None, best=False, device='cpu', quiet=False):
    run_dir = Path(log_dir) / run_id
    latest_path = run_dir / 'checkpoint_latest.pth'
    best_path = run_dir / 'checkpoint_best.pth'
    ckp_path = best_path if best else latest_path
    
    checkpoint = torch.load(ckp_path, map_location=device)
    hparams = checkpoint['hyperparameters']
    
    hparams['run_id'] = run_id
    if not hasattr(hparams, 'model'): # or not hasattr(hparams, 'best_val_loss'):
        for old_key, new_key in {'model_type': 'model', 
                                 'nlayers': 'nlayers', 
                                 'source': 'train', 
                                 'learning_rate': 'lr', 
                                 'loss_type': 'loss'}.items():
            if old_key in hparams:
                hparams[new_key] = hparams.pop(old_key)
    if args is not None:
        hparams['resume'] = args.resume 
        hparams['test'] = args.test
        vars(args).update(**hparams)
        hparams = argparse.Namespace(**vars(args))
    else:
        hparams = argparse.Namespace(**hparams)
    if not quiet:
        print(f"Loading Model with Run ID: {run_id}")
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
    for key in ('sex', 'age', 'cid', 's_next', 's_h', 'setting_h', 'setting_next'):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].long()
    return batch


# Number of care-setting levels (process/covariates.SETTING_VOCAB): unknown/pad, outpatient, ed, inpatient, icu
N_SETTINGS = 5
COVARIATE_KEYS = ('age_h', 'age_next', 'setting_h', 'setting_next', 'co_h', 'co_mask', 'obs_h')


def uses_covariates(model):
    return any(getattr(model, f, False) for f in ('use_age_t', 'use_setting', 'use_coanalytes', 'use_full_panel'))


def model_extras(model, batch):
    """Optional covariate kwargs for NORMA2.forward, only for models that use them."""
    if not uses_covariates(model):
        return {}
    return {k: batch[k] for k in COVARIATE_KEYS if k in batch}


def run_model(model, batch, s_next=None):
    """Single entry point for the forward pass from a collated batch (optionally overriding s_next)."""
    s = batch['s_next'] if s_next is None else s_next
    return model(batch['x_h'], batch['s_h'], batch['t_h'], batch['sex'], batch['age'], batch['cid'],
                 s, batch['t_next'], batch['pad_mask'], **model_extras(model, batch))

def create_model(args, ncodes, checkpoint=None):
    nstates = getattr(args, 'nstates', getattr(args, 'num_states', 2))
    if args.model == 'NORMA':
        model = NORMA(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, nstates=nstates, ncodes=ncodes)
    elif args.model == 'NormaLight':
        # Detect legacy checkpoint (decoder-named layers, no age_emb)
        is_legacy = (checkpoint is not None
                     and any(k.startswith('decoder.') for k in checkpoint['model_state_dict']))
        if is_legacy:
            model = NormaLightV1(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, nstates=nstates, ncodes=ncodes)
        else:
            model = NormaLight(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, nstates=nstates, ncodes=ncodes)
    elif args.model == 'NORMA2':
        output_mode = getattr(args, 'output_mode', 'quantile')
        model = NORMA2(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
                       nstates=nstates, ncodes=ncodes, output_mode=output_mode,
                       # per-measurement covariates (revision ablation); absent in old checkpoints
                       use_age_t=bool(getattr(args, 'use_age_t', False)),
                       use_setting=bool(getattr(args, 'use_setting', False)),
                       use_coanalytes=bool(getattr(args, 'use_coanalytes', False)),
                       query_coanalytes=bool(getattr(args, 'query_coanalytes', False)),
                       n_settings=N_SETTINGS, n_panel=ncodes,
                       causal_memory=bool(getattr(args, 'causal_memory', False)),
                       use_full_panel=bool(getattr(args, 'use_full_panel', False)))
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model
        
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
            
def create_loss(loss, lambda_align=None, args=None):
    if loss == 'NORMALoss':
        la = getattr(args, 'lambda_align', None) if lambda_align is None else lambda_align
        return NORMALoss(lambda_align=0.01 if la is None else la,
                         k=getattr(args, 'prior_k', None) if getattr(args, 'align_by_n', False) else None)
    if loss == 'QuantilePriorLoss':
        if getattr(args, 'nstates', 3) != 3:
            raise ValueError('QuantilePriorLoss assumes the 3-state coding (normal = 1)')
        if getattr(args, 'normalize', False):
            raise ValueError('QuantilePriorLoss expects raw units (no --normalize)')
        return QuantilePriorLoss(lambda_prior=getattr(args, 'prior_lambda', 1.0),
                                 k=getattr(args, 'prior_k', 5.0),
                                 mode=getattr(args, 'prior_mode', 'anchor'),
                                 normal_state=1, tau=getattr(args, 'prior_tau', None))
    if loss == 'StudentTNLLLoss':
        return StudentTNLLLoss()
    if loss == 'GaussianNLLLoss':
        return GaussianNLLLoss()
    if loss == 'MSELoss':
        return MSELoss()
    if loss == 'QuantileLoss':
        return QuantileLoss()
    raise ValueError(f"Unknown loss type: {loss}")

def loss_extras(batch, model=None):
    """Side inputs for the prior-aware losses, from a collated batch (+ the model's
    last_params / last_gate for the gate and NIG heads).

    ref_mu / ref_var are the population interval read as a normal distribution
    (midpoint, (width/3.92)^2), in the model's units."""
    if 'pop_low' not in batch:
        return None
    lo, hi = batch['pop_low'], batch['pop_high']
    out = {'s_next': batch['s_next'], 'n_hist': batch['n_hist'], 'pop_low': lo, 'pop_high': hi,
           'ref_mu': 0.5 * (lo + hi), 'ref_var': ((hi - lo) / 3.92) ** 2,
           't_h': batch['t_h'], 't_next': batch['t_next'], 'pad_mask': batch['pad_mask']}
    if model is not None:
        out['gate'] = getattr(model, 'last_gate', None)
        out['params'] = getattr(model, 'last_params', None)
    return out


def compute_loss(mu, log_var, y_true, criterion, extra: dict = None):
    if isinstance(criterion, QuantileLoss):
        # mu is actually q_pred (B, n_quantiles) for quantile models
        return criterion(mu, y_true)
    if isinstance(criterion, QuantilePriorLoss):
        need = ('s_next', 'n_hist', 'pop_low', 'pop_high')
        if extra is None or not all(k in extra for k in need):
            raise ValueError(f'QuantilePriorLoss requires extra keys: {need}')
        if criterion.mode == 'gate' and extra.get('gate') is None:
            raise ValueError("QuantilePriorLoss mode 'gate' needs a NORMA2 --output_mode gate model")
        return criterion(mu, y_true, extra['s_next'], extra['n_hist'], extra['pop_low'], extra['pop_high'],
                         t_h=extra.get('t_h'), t_next=extra.get('t_next'), pad_mask=extra.get('pad_mask'),
                         gate=extra.get('gate'))
    if isinstance(criterion, StudentTNLLLoss):
        params = extra.get('params') if extra else None
        if params is None or 'df' not in params:
            raise ValueError('StudentTNLLLoss needs a NORMA2 --output_mode nig model (model.last_params)')
        return criterion(params['df'], params['loc'], params['scale'], y_true)
    if isinstance(criterion, NORMALoss):
        if extra is None or not all(k in extra for k in ('s_next', 'ref_mu', 'ref_var')):
            raise ValueError('NORMALoss requires extra keys: s_next, ref_mu, ref_var')
        return criterion(mu, log_var, y_true, extra['s_next'], extra['ref_mu'], torch.sqrt(extra['ref_var']),
                         n_hist=extra.get('n_hist'))
    if isinstance(criterion, GaussianNLLLoss):
        return criterion(mu, log_var, y_true)
    if isinstance(criterion, MSELoss):
        return criterion(mu, y_true)
    raise ValueError('Unknown loss type')

def log_epoch(metrics, epoch_index: int, total_epochs: int, lr: float, epoch_time: float, is_best: bool):
    """Log metrics to wandb and print a concise console summary."""
    import wandb
    metrics['epoch'] = epoch_index + 1
    metrics['lr'] = lr
    wandb.log(metrics)
    s = f"Epoch {epoch_index + 1}/{total_epochs} ({epoch_time:.1f}s), LR: {lr:.0e}, Train Loss: {metrics['train']['loss']:.2f}, Val Loss: {metrics['val']['loss']:.2f}, Train R2: {metrics['train']['r2']:.2f}, Val R2: {metrics['val']['r2']:.2f}"
    if is_best:
        s += " (Best Model)"
    print(s)
    