import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class MSELoss(nn.Module):
    """Mean Squared Error Loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, y_true):
        """Compute Mean Squared Error Loss."""
        return nn.MSELoss()(mu, y_true)
    
class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log Likelihood Loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, log_var, y_true):
        """Compute Gaussian Negative Log Likelihood Loss."""
        return nn.GaussianNLLLoss()(mu, y_true, torch.exp(log_var))
    
class NORMALoss(nn.Module):
    """Loss function for NORMA conditional transformer."""
    
    def __init__(self, lambda_align=0.01, adaptive_weight=True):
        super().__init__()
        self.lambda_align = lambda_align
        self.adaptive_weight = adaptive_weight
        self.forecast_fn = nn.GaussianNLLLoss()
        
    def forward(self, mu, log_var, y_true, condition, ref_mu, ref_sigma):
        """Compute NORMA loss."""
        # Forecasting loss
        forecast_loss = self.forecast_fn(mu, y_true, torch.exp(log_var))
        align_loss = self._align_loss(mu, log_var, condition, ref_mu, ref_sigma)
        total_loss = forecast_loss + self.lambda_align * align_loss
        
        return total_loss, forecast_loss, align_loss
    
    def _align_loss(self, mu, log_var, condition, ref_mu, ref_sigma):
        """Compute alignment loss component."""
        # Create distributions
        pred_dist = Normal(mu, torch.exp(0.5 * log_var))
        ref_dist = Normal(ref_mu, ref_sigma)
        
        # KL divergence
        kl_div = torch.distributions.kl_divergence(pred_dist, ref_dist)
        kl_weight = self._get_weight(log_var, ref_sigma)
                # DEBUG: Print tensor shapes and dimensions
        
        # Ensure kl_div has the same shape as mu for proper indexing
        if kl_div.dim() == 0:
            kl_div = kl_div.expand_as(mu)
        
        kl_weight = torch.exp(log_var) / (torch.exp(log_var) + ref_sigma**2 + 1e-6)
        if not self.adaptive_weight:
            kl_weight = torch.ones_like(kl_weight)

        # Split predictions based on condition
        normal_mask = (condition == 1)
        abnormal_mask = (condition == 0)
        
        normal_loss = self._normal_loss(kl_div, kl_weight, normal_mask, mu.device)
        abnormal_loss = self._abnormal_loss(kl_div, kl_weight, abnormal_mask, mu.device)
        
        return normal_loss + abnormal_loss
    
    def _get_weight(self, log_var, ref_sigma):
        """Get adaptive KL weighting."""
        if not self.adaptive_weight:
            return torch.ones_like(log_var)
        
        var = torch.exp(log_var)
        return var / (var + ref_sigma**2 + 1e-6)
    
    def _normal_loss(self, kl_div, kl_weight, normal_mask, device):
        """Compute loss for normal patients."""
        if not normal_mask.any():
            return torch.tensor(0.0, device=device)
        
        return (kl_div[normal_mask] * kl_weight[normal_mask]).mean()
    
    def _abnormal_loss(self, kl_div, kl_weight, abnormal_mask, device):
        """Compute loss for abnormal patients."""
        if not abnormal_mask.any():
            return torch.tensor(0.0, device=device)
        
        # Encourage divergence from reference
        divergence = torch.clamp(1.0 - kl_div[abnormal_mask], min=0.0).pow(2)
        return (divergence * kl_weight[abnormal_mask]).mean()

