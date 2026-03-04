# Flexible NORMA Training Guide

This guide explains how to use the flexible training system that supports both **NORMA Light** and **NORMADecoder** models with different loss functions and metrics.

## 🎯 Overview

The training system now supports:

### Models
- **`norma_light`**: Lightweight transformer for conditional prediction
- **`norma_decoder`**: Full decoder-based architecture for conditional prediction

### Loss Functions
- **`norma_loss`**: Comprehensive loss with forecasting + alignment components (requires reference statistics)
- **`gaussian_nll`**: Simple Gaussian Negative Log Likelihood loss (faster training)

## 🚀 Quick Start

### Train NORMA Light with NORMA Loss
```bash
python train.py \
    --model_type norma_light \
    --loss_type norma_loss \
    --epochs 10 \
    --batch_size 32 \
    --lambda_align 0.01 \
    --adaptive_weight
```

### Train NORMADecoder with Gaussian NLL Loss
```bash
python train.py \
    --model_type norma_decoder \
    --loss_type gaussian_nll \
    --epochs 10 \
    --batch_size 32
```

### Train NORMA Light with Simple Gaussian Loss (Fast Training)
```bash
python train.py \
    --model_type norma_light \
    --loss_type gaussian_nll \
    --epochs 5 \
    --batch_size 64 \
    --learning_rate 0.001
```

## 📊 Model Capabilities

### NORMA Light (`norma_light`)
- **Input**: Historical measurements, condition states, timestamps, demographics, lab codes
- **Output**: Mean and log variance predictions
- **Use case**: Fast training, good for prototyping and simple forecasting

### NORMADecoder (`norma_decoder`) 
- **Input**: Full sequence, query timestamp and condition, demographics, lab codes
- **Output**: Mean and log variance predictions
- **Use case**: More sophisticated conditional prediction

## 🎛️ Loss Functions

### NORMA Loss (`norma_loss`)
```python
total_loss = forecast_loss + λ * align_loss
```
- **forecast_loss**: Gaussian NLL loss (μ, σ², y_true)
- **align_loss**: KL divergence with reference distributions
- **Requires**: Reference statistics (`ref_mu`, `ref_var`)
- **Best for**: When you have reference normal ranges

### Gaussian NLL Loss (`gaussian_nll`)
```python
loss = -log N(y_true | μ, σ²)
```
- **Simple**: Just likelihood between prediction and truth
- **Fast**: No additional alignment computation
- **Best for**: Quick training, baseline comparisons

## 🔧 Command Line Arguments

### Model Selection
- `--model_type`: Choose `norma_light` or `norma_decoder`
- `--loss_type`: Choose `norma_loss` or `gaussian_nll`

### Model Architecture
- `--d_model`: Hidden dimension (default: 64)
- `--nhead`: Attention heads (default: 2)  
- `--num_layers`: Transformer layers (default: 2)

### Training
- `--epochs`: Training epochs (default: 2)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)

### Loss Parameters (for `norma_loss`)
- `--lambda_align`: Alignment loss weight (default: 0.01)
- `--adaptive_weight`: Use adaptive weighting (default: True)

## 📈 Example Training Results

### NORMA Light + NORMA Loss
- Probabilistic forecasts with uncertainty
- Alignment to reference distributions
- Best calibration for clinical predictions

### NORMADecoder + Gaussian NLL
- Pure probabilistic forecasting
- Faster convergence
- Good baseline for comparison

## 🎯 Probabilistic Predictions

Both models now output:
- **Mean (μ)**: Expected value prediction
- **Log Variance (log σ²)**: Uncertainty quantification

You can convert log variance to standard deviation:
```python
sigma = torch.exp(0.5 * log_var)
```

Generate predictions with confidence intervals:
```python
with torch.no_grad():
    mu, log_var = model(batch_inputs)
    sigma = torch.exp(0.5 * log_var)
    
    # 95% confidence interval
    ci_lower = mu - 1.96 * sigma
    ci_upper = mu + 1.96 * sigma
```

## 🔍 Monitoring Training

The system logs to Weights & Biases with metrics:
- `train_loss`, `val_loss`: Total loss
- `train_forecast_loss`, `val_forecast_loss`: Forecasting component  
- `train_align_loss`, `val_align_loss`: Alignment component (if using NORMA loss)
- `learning_rate`: Current learning rate

## 🚨 Troubleshooting

### Compatible Model/Loss Combinations
- ✅ `norma_light` + `norma_loss`
- ✅ `norma_light` + `gaussian_nll` 
- ✅ `norma_decoder` + `norma_loss`
- ✅ `norma_decoder` + `gaussian_nll`

### Data Requirements
- NORMA Loss requires reference statistics (`ref_mu`, `ref_var`) in your dataset
- Gaussian NLL Loss works with any dataset that has target values

### Performance Tips
- Start with `gaussian_nll` for faster prototyping
- Use `norma_loss` for production models with reference data
- Increase `lambda_align` for stronger reference alignment
- Use larger `batch_size` for stable variance estimation
