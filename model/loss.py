import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal

class HuberLoss(nn.Module):
    """Huber Loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, y_true):
        """Compute Huber Loss."""
        return nn.HuberLoss()(mu, y_true)

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

class studentNLLLoss(nn.Module):
    """Student's t-distribution Negative Log Likelihood Loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, logsig, lnu, y_true):
        """Compute Student's t-distribution Negative Log Likelihood Loss."""
        sigma = F.softplus(logsig) + 1e-6
        nu = F.softplus(lnu) + 2.0   # ensures nu > 2 for finite variance
        z = (y_true - mu) / sigma
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * (torch.log(nu * torch.pi) + 2 * torch.log(sigma))
        term3 = - ((nu + 1) / 2) * torch.log1p((z * z) / nu)
        nll = -(term1 + term2 + term3)     # shape (B,1)
        loss = nll.mean()
        return loss
    
class QuantileLoss(nn.Module):
    """Pinball (quantile) loss for multiple quantile levels."""

    def __init__(self, quantiles=(0.025, 0.25, 0.50, 0.75, 0.975)):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, q_pred, y_true):
        """
        Args:
            q_pred: (B, n_quantiles) predicted quantile values
            y_true: (B, 1) ground-truth values
        Returns:
            scalar loss (mean pinball loss across quantiles and batch)
        """
        tau = self.quantiles.to(q_pred.device)
        y = y_true.expand_as(q_pred)  # (B, n_quantiles)
        errors = y - q_pred            # (B, n_quantiles)
        loss = torch.max(tau * errors, (tau - 1) * errors)
        return loss.mean()


class NORMALoss(nn.Module):
    """Loss function for NORMA conditional transformer."""
    
    def __init__(self, lambda_align=0.01, adaptive_weight=True, k=None):
        """k: if set, the KL weight is the conjugate prior weight k / (n_hist + k)
        instead of the variance ratio var / (var + ref_var) (which has no
        history dependence: a confident model on two draws is barely aligned)."""
        super().__init__()
        self.lambda_align = lambda_align
        self.adaptive_weight = adaptive_weight
        self.k = k
        self.forecast_fn = nn.GaussianNLLLoss()
        
    def forward(self, mu, log_var, y_true, condition, ref_mu, ref_sigma, n_hist=None):
        """Compute NORMA loss (scalar)."""
        # Forecasting loss
        forecast_loss = self.forecast_fn(mu, y_true, torch.exp(log_var))
        align_loss = self._align_loss(mu, log_var, condition, ref_mu, ref_sigma, n_hist)
        return forecast_loss + self.lambda_align * align_loss
    
    def _align_loss(self, mu, log_var, condition, ref_mu, ref_sigma, n_hist=None):
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
        if self.k is not None and n_hist is not None:
            kl_weight = (self.k / (n_hist.float().view(-1, 1) + self.k)).expand_as(kl_weight)
        elif not self.adaptive_weight:
            kl_weight = torch.ones_like(kl_weight)

        # Split predictions based on condition
        condition = condition.view(-1, 1)
        normal_mask = (condition == 1)
        abnormal_mask = (condition != 1)
        
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



def effective_n(n_hist, t_h=None, t_next=None, pad_mask=None, tau=None):
    """History length, optionally decayed with elapsed time.

    tau (same units as t) gives n_eff = sum_i exp(-(t_query - t_i) / tau): the
    information an observation still carries under a setpoint that drifts as an
    Ornstein-Uhlenbeck process with time constant tau. tau=None -> the plain count."""
    n = n_hist.float().view(-1, 1)
    if tau is None or t_h is None:
        return n
    dt = (t_next.view(-1, 1, 1) - t_h.view(t_h.shape[0], -1, 1)).clamp_min(0.0)
    decay = torch.exp(-dt.squeeze(-1) / float(tau))
    if pad_mask is not None:
        decay = decay * (~pad_mask).float()
    return decay.sum(dim=1, keepdim=True)


class StudentTNLLLoss(nn.Module):
    """Negative log likelihood of the NIG head's Student-t posterior predictive."""

    def forward(self, df, loc, scale, y_true):
        dist = torch.distributions.StudentT(df.view(-1), loc.view(-1), scale.view(-1).clamp_min(1e-6))
        return -dist.log_prob(y_true.view(-1)).mean()


class QuantilePriorLoss(nn.Module):
    """Pinball loss plus a population-prior term whose weight decays with history length.

    The pinball loss alone learns the dev-cohort conditional quantiles and nothing
    pulls the interval toward the population reference when the history is short or
    flat (model/logs/ablation: 95% intervals cover ~77% on test, and the synthetic
    sweep gives ~half the Pop_RI width from two draws and zero width from a flat
    history). This loss adds prior pseudo-observations:

        L = pinball(q, y) + lambda * w(n) * prior(q)        w(n) = k / (n + k)

    n = number of target-analyte observations in the history, k = prior strength in
    "observations" (k = 5: a 5-draw history is weighted 50/50 with the prior; two
    draws ~70% prior; 50 draws ~10%).

    mode='anchor'  prior(q) = E_{Y~N(m, s)}[pinball(q, Y)], closed form, with m the
                   Pop_RI midpoint and s = width / 3.92 (Pop_RI = central 95%).
                   Minimised when q equals the prior quantiles, so it is exactly
                   "train on a k/(n+k) mixture of real targets and draws from the
                   population prior" without sampling. Applied to queries conditioned
                   on the normal state only (the reference-interval use case); the
                   prior for an abnormal query is not the population interval.
    mode='floor'   prior(q) = relu(width_pop - (q975 - q025)): a soft floor on the
                   interval width that decays with n, every state. Weaker: it does
                   not pull the centre, only stops collapse.

    Both terms are in the analyte's units, like the pinball loss, so lambda is a
    ratio of the two (lambda = 1: a prior pseudo-draw counts as much as a real one).
    """

    def __init__(self, quantiles=(0.025, 0.25, 0.50, 0.75, 0.975), lambda_prior=1.0,
                 k=5.0, mode='anchor', normal_state=1, tau=None):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))
        self.lambda_prior = float(lambda_prior)
        self.k = float(k)
        self.mode = mode
        self.normal_state = int(normal_state)
        self.tau = tau
        self.pinball = QuantileLoss(quantiles)

    @staticmethod
    def _expected_pinball(q, tau, m, s):
        """E[rho_tau(Y - q)] for Y ~ N(m, s): s * (phi(z) + z * (Phi(z) - tau)), z = (q - m)/s."""
        z = (q - m) / s
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        return s * (phi + z * (Phi - tau))

    def forward(self, q_pred, y_true, s_next, n_hist, pop_low, pop_high,
                t_h=None, t_next=None, pad_mask=None, gate=None):
        """q_pred (B, Q); y_true, s_next, n_hist, pop_low, pop_high (B, 1).
        t_h/t_next/pad_mask only for tau (time-decayed n); gate (B, 1) for mode 'gate'."""
        data_term = self.pinball(q_pred, y_true)
        tau = self.quantiles.to(q_pred.device).unsqueeze(0)          # (1, Q)
        n_eff = effective_n(n_hist, t_h, t_next, pad_mask, self.tau)
        w = self.k / (n_eff + self.k)                                 # (B, 1)
        if self.mode == 'gate':
            # q_pred is already g * own + (1 - g) * prior; the pinball scores that
            # combination. Pull the learned gate toward the conjugate weight
            # n / (n + k) with a Bernoulli KL, so it departs from the Bayesian
            # answer only where the data say the patient is more (or less)
            # trustworthy than its history length implies.
            g0 = (1 - w).clamp(1e-4, 1 - 1e-4)
            g = gate.clamp(1e-4, 1 - 1e-4)
            kl = g * torch.log(g / g0) + (1 - g) * torch.log((1 - g) / (1 - g0))
            return data_term + self.lambda_prior * kl.mean()
        width = (pop_high - pop_low).clamp_min(1e-6)
        if self.mode == 'anchor':
            m = 0.5 * (pop_low + pop_high)
            s = width / 3.92
            per_q = self._expected_pinball(q_pred, tau, m, s)         # (B, Q)
            prior = per_q.mean(dim=1, keepdim=True)
            mask = (s_next.view(-1, 1) == self.normal_state).float()
            prior_term = (w * mask * prior).sum() / mask.sum().clamp_min(1.0)
        elif self.mode == 'floor':
            pred_width = q_pred[:, -1:] - q_pred[:, :1]
            prior = F.relu(width - pred_width)
            prior_term = (w * prior).mean()
        else:
            raise ValueError(f'unknown prior mode {self.mode!r}')
        return data_term + self.lambda_prior * prior_term
