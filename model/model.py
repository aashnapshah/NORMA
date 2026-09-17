import torch
import torch.nn as nn
import math
from scipy.stats import t as scipy_t
import torch.nn.functional as F

# Output modes whose forward returns five quantiles (B, 5) at NORMA2.QUANTILES. 'gate' and 'nig'
# also keep their distribution parameters on model.last_params for the loss (see loss.py...
QUANTILE_OUTPUT_MODES = ('quantile', 'gate', 'nig')


def is_quantile_mode(output_mode):
    return output_mode in QUANTILE_OUTPUT_MODES


class TimeEmbedding(nn.Module):
    """Improved time embedding: log-delta-t + Time2Vec on deltas."""

    def __init__(self, d_model):
        super().__init__()
        self.log_proj = nn.Linear(1, d_model // 4)
        self.periodic = nn.Linear(1, d_model // 2)
        self.linear = nn.Linear(1, d_model - d_model // 4 - d_model // 2)

    def forward(self, t):
        """Args: t (B, T, 1) absolute times. Returns: (B, T, D)"""
        # Compute inter-measurement deltas (first delta is 0)
        delta = torch.zeros_like(t)
        delta[:, 1:, :] = t[:, 1:, :] - t[:, :-1, :]
        delta = delta.clamp(min=0)

        log_dt = torch.log1p(delta)           # monotonic, compresses large gaps
        v_log = self.log_proj(log_dt)          # (B, T, d_model//4)
        v_periodic = torch.sin(self.periodic(delta))  # (B, T, d_model//2)
        v_linear = self.linear(delta)          # (B, T, remainder)
        return torch.cat([v_log, v_periodic, v_linear], dim=-1)


class TimeEmbeddingQuery(nn.Module):
    """Time embedding for the query token: encodes the horizon (gap from last obs)."""

    def __init__(self, d_model):
        super().__init__()
        self.log_proj = nn.Linear(1, d_model // 4)
        self.periodic = nn.Linear(1, d_model // 2)
        self.linear = nn.Linear(1, d_model - d_model // 4 - d_model // 2)

    def forward(self, horizon):
        """Args: horizon (B, 1, 1) time gap from last observation. Returns: (B, 1, D)"""
        horizon = horizon.clamp(min=0)
        log_h = torch.log1p(horizon)
        v_log = self.log_proj(log_h)
        v_periodic = torch.sin(self.periodic(horizon))
        v_linear = self.linear(horizon)
        return torch.cat([v_log, v_periodic, v_linear], dim=-1)


class NORMA2(nn.Module):
    """NORMA v2: decoder-only transformer with improved time encoding,
    within-sequence normalization, and quantile output heads.
    """

    QUANTILES = [0.025, 0.25, 0.50, 0.75, 0.975]

    def __init__(self, d_model, nhead, nlayers, nstates, ncodes,
                 output_mode='quantile', mlp_dropout=0.1, age_bins=7,
                 use_age_t=False, use_setting=False, use_coanalytes=False,
                 n_settings=5, n_panel=None, causal_memory=False, use_full_panel=False,
                 query_coanalytes=False):
        super().__init__()
        self.output_mode = output_mode
        self.d_model = d_model
        self.causal_memory = causal_memory
        self.use_full_panel = use_full_panel
        self.use_age_t = use_age_t
        self.use_setting = use_setting
        self.use_coanalytes = use_coanalytes
        self.query_coanalytes = query_coanalytes
        self.n_panel = n_panel if n_panel is not None else ncodes

        # History token embeddings
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(nstates, d_model)

        # Context token embeddings (sex, age, lab)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(ncodes, d_model)
        self.age_bins = age_bins
        self.age_emb = nn.Embedding(age_bins, d_model)

        # Time embeddings
        self.time_emb = TimeEmbedding(d_model)
        self.time_emb_query = TimeEmbeddingQuery(d_model)

        # Within-sequence normalization (learnable scale/shift)
        self.seq_norm_scale = nn.Parameter(torch.ones(1))
        self.seq_norm_bias = nn.Parameter(torch.zeros(1))

        # Optional covariate embeddings (only instantiated when used)
        if use_age_t:
            self.age_t_emb = nn.Embedding(age_bins, d_model)
        if use_setting:
            self.setting_emb = nn.Embedding(n_settings, d_model, padding_idx=0)  # 0 = unknown/pad
        if use_coanalytes:
            # [value*mask, mask, is_low, is_normal, is_high] per co-analyte
            self.co_proj = nn.Linear(5 * self.n_panel, d_model)
        if use_full_panel:
            # marks whether the target analyte was itself drawn at this timestamp
            self.obs_emb = nn.Embedding(2, d_model)

        # Decoder-only transformer (causal self-attention)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                           dropout=mlp_dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, nlayers)

        # Output heads
        if output_mode == 'quantile':
            self.quantile_head = nn.Linear(d_model, len(self.QUANTILES))
        elif output_mode == 'gate':
            # Own quantiles + a trust gate g in (0, 1); the returned quantiles are g * own + (1 -
            # g) * state-conditional population quantiles (prior_q).
            self.quantile_head = nn.Linear(d_model, len(self.QUANTILES))
            self.gate_head = nn.Linear(d_model, 1)
        elif output_mode == 'nig':
            # Conjugate normal-inverse-gamma head: the network emits sufficient statistics
            # (setpoint estimate, observed within-person variance, an effective sample size <= n)
            # and the interval is the closed-form...
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)
            self.neff_head = nn.Linear(d_model, 1)
        else:
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)

        # State-conditional population prior, indexed [code, state, sex].
        Q = len(self.QUANTILES)
        if output_mode in ('gate', 'nig'):
            self.register_buffer('prior_q', torch.zeros(ncodes, nstates, 2, Q))
            self.register_buffer('prior_mu', torch.zeros(ncodes, nstates, 2))
            self.register_buffer('prior_var', torch.ones(ncodes, nstates, 2))
            self.register_buffer('prior_rho', torch.full((ncodes,), 0.5))   # within / total variance
            self.register_buffer('prior_ready', torch.zeros(1))
        self.nig_nu0 = 5.0          # prior pseudo-observations for the within-person variance
        self.nig_kappa0_scale = 1.0 # multiplies rho/(1-rho); 1 = the conjugate value
        if output_mode == 'nig':
            # Student-t quantile table for the NIG head: t.ppf(level, nu) on a log-nu grid
            nu_grid = torch.logspace(math.log10(2.05), math.log10(500.0), 400)
            self.register_buffer('t_nu_grid', nu_grid)
            self.register_buffer('t_ppf_table', torch.tensor(
                [[float(scipy_t.ppf(q, nu)) for q in self.QUANTILES] for nu in nu_grid.tolist()]))
        self.last_params = None
        self.last_gate = None

    def set_prior_table(self, table):
        """table: dict from priors.build_prior_table (q, mu, var, rho as tensors)."""
        self.prior_q.copy_(table['q'].to(self.prior_q.device))
        self.prior_mu.copy_(table['mu'].to(self.prior_mu.device))
        self.prior_var.copy_(table['var'].to(self.prior_var.device))
        self.prior_rho.copy_(table['rho'].to(self.prior_rho.device))
        self.prior_ready.fill_(1.0)

    def _t_ppf(self, nu):
        """(B,) degrees of freedom -> (B, Q) Student-t quantiles by table interpolation."""
        nu = nu.clamp(self.t_nu_grid[0], self.t_nu_grid[-1])
        pos = torch.searchsorted(self.t_nu_grid, nu.contiguous()).clamp(1, len(self.t_nu_grid) - 1)
        lo, hi = self.t_nu_grid[pos - 1], self.t_nu_grid[pos]
        w = ((nu - lo) / (hi - lo)).unsqueeze(-1)
        return (1 - w) * self.t_ppf_table[pos - 1] + w * self.t_ppf_table[pos]

    def _co_features(self, co_h, co_mask):
        """[value*mask, mask, is_low, is_normal, is_high] per co-analyte -> (..., 5K)."""
        co_low = (co_h < 0).float() * co_mask
        co_high = (co_h > 1).float() * co_mask
        co_norm = co_mask - co_low - co_high
        return torch.cat([co_h * co_mask, co_mask, co_low, co_norm, co_high], dim=-1)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def _bin_age(self, age):
        age = age.float()
        return torch.clamp((age - 20) / 10, min=0, max=self.age_bins - 1).long()

    def _seq_normalize(self, x_h, pad_mask, obs_mask=None):
        """(x - mean) / std over valid positions. Returns normalized x, mean, std."""
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).float()
        else:
            valid = torch.ones_like(x_h)
        if obs_mask is not None:
            valid = valid * obs_mask.unsqueeze(-1).float()
        n_obs = valid.sum(dim=1, keepdim=True)

        n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x_h * valid).sum(dim=1, keepdim=True) / n_valid
        var = ((x_h - mean) ** 2 * valid).sum(dim=1, keepdim=True) / n_valid
        std = (var + 1e-6).sqrt()

        if obs_mask is not None:
            # a patient with draws but no target-analyte value would give std=0
            std = torch.where(n_obs > 0, std, torch.ones_like(std))
        x_norm = (x_h - mean) / std
        x_norm = x_norm * self.seq_norm_scale + self.seq_norm_bias
        if obs_mask is not None:
            x_norm = x_norm * obs_mask.unsqueeze(-1).float()
        return x_norm, mean, std

    def forward(self, x_h, s_h, t_h, sex, age, lab, s_next, t_next, pad_mask=None,
                age_h=None, age_next=None, setting_h=None, setting_next=None,
                co_h=None, co_mask=None, obs_h=None):
        """Optional covariates (ignored unless the matching use_* flag is set):
        age_h (B,T) age at each history draw; age_next (B,) age at the query
        (default: age at last draw + horizon in years); setting_h (B,T) long;
        setting_next (B,) long; co_h (B,T,K) PopRI-normalised co-analyte values
        (0 where missing); co_mask (B,T,K) 1 where drawn."""
        B, T = x_h.shape[:2]
        sex = sex.view(-1).long()
        lab = lab.view(-1).long()
        s_h = s_h.long()
        s_next = s_next.view(-1).long()

        # use_full_panel: the history spans every draw of the patient, so some positions carry no
        # target-analyte value.
        obs = obs_h.float() if (self.use_full_panel and obs_h is not None) else None

        # Within-sequence normalize values
        x_norm, seq_mean, seq_std = self._seq_normalize(x_h, pad_mask, obs_mask=obs)

        # Context token: sex + age + lab (single token, position 0)
        ctx = (self.sex_emb(sex) + self.age_emb(self._bin_age(age.view(-1))) + self.lab_emb(lab)).unsqueeze(1)  # (B, 1, D)

        # History tokens: value + state + time only
        if obs is None:
            hist = (
                self.value_emb(x_norm)
                + self.state_emb(s_h)
                + self.time_emb(t_h)
            )  # (B, T, D)
        else:
            g = obs.unsqueeze(-1)
            hist = (
                g * (self.value_emb(x_norm) + self.state_emb(s_h))
                + self.time_emb(t_h)
                + self.obs_emb(obs.long())
            )  # (B, N, D)

        # Query token: state + horizon only
        if pad_mask is not None:
            lengths = (~pad_mask).sum(dim=1)
            last = lengths - 1
            t_last = t_h[torch.arange(B, device=t_h.device), last, 0]
        else:
            last = torch.full((B,), T - 1, device=t_h.device, dtype=torch.long)
            t_last = t_h[:, -1, 0]

        horizon = (t_next.view(B) - t_last).clamp(min=0).view(B, 1, 1)
        query = (
            self.state_emb(s_next)
            + self.time_emb_query(horizon).squeeze(1)
        )  # (B, D)

        # ---- optional per-measurement covariates ----
        if self.use_age_t:
            if age_h is None:
                age_h = age.view(B, 1).float().expand(B, T)
            hist = hist + self.age_t_emb(self._bin_age(age_h))
            if age_next is None:
                age_last = age_h[torch.arange(B, device=age_h.device), last]
                age_next = age_last + horizon.view(B) / 365.25
            query = query + self.age_t_emb(self._bin_age(age_next.view(-1)))
        if self.use_setting:
            if setting_h is not None:
                hist = hist + self.setting_emb(setting_h.long())
            if setting_next is not None:
                query = query + self.setting_emb(setting_next.view(-1).long())
        if self.use_coanalytes and co_h is not None:
            if co_mask is None:
                co_mask = torch.isfinite(co_h).float()
            co_mask = co_mask.float()
            co_h = torch.nan_to_num(co_h.float(), nan=0.0)
            co_feat = self._co_features(co_h, co_mask)                 # (B, T, 5K)
            hist = hist + self.co_proj(co_feat)
            if self.query_coanalytes:
                # The most recent panel the patient already has.
                co_q = co_feat[torch.arange(B, device=co_feat.device), last]   # (B, 5K)
                query = query + self.co_proj(co_q)

        query = query.unsqueeze(1)  # (B, 1, D)

        # Sequence: [ctx, hist_1, ..., hist_T, query]
        tokens = torch.cat([ctx, hist, query], dim=1)  # (B, 1+T+1, D)
        L = T + 2
        attn_mask = self._causal_mask(L, tokens.device)

        # Padding mask: context and query are never padded
        if pad_mask is not None:
            pad_mask_ext = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device),  # ctx
                pad_mask,                                                       # hist
                torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device),  # query
            ], dim=1)
        else:
            pad_mask_ext = None

        H = self.transformer(tokens, tokens, tgt_mask=attn_mask,
                             memory_mask=attn_mask if self.causal_memory else None,
                             tgt_key_padding_mask=pad_mask_ext,
                             memory_key_padding_mask=pad_mask_ext)
        query_features = H[:, -1]  # (B, D)

        if self.output_mode == 'quantile':
            q_norm = self.quantile_head(query_features)
            q_denorm = q_norm * seq_std.squeeze(-1) + seq_mean.squeeze(-1)
            return q_denorm
        elif self.output_mode == 'gate':
            q_norm = self.quantile_head(query_features)
            q_own = q_norm * seq_std.squeeze(-1) + seq_mean.squeeze(-1)
            g = torch.sigmoid(self.gate_head(query_features))                  # (B, 1)
            prior_q = self.prior_q[lab, s_next, sex]                             # (B, Q)
            self.last_gate = g
            self.last_params = {'q_own': q_own, 'gate': g, 'prior_q': prior_q}
            return g * q_own + (1 - g) * prior_q
        elif self.output_mode == 'nig':
            # sufficient statistics from the network, in raw units
            sd_seq = seq_std.squeeze(-1)
            mu_hat = self.mean_head(query_features) * sd_seq + seq_mean.squeeze(-1)
            s2_obs = torch.exp(torch.clamp(self.logvar_head(query_features), min=-10.0)) * sd_seq ** 2
            if obs is not None:
                n = obs.sum(dim=1, keepdim=True)
            elif pad_mask is not None:
                n = (~pad_mask).float().sum(dim=1, keepdim=True)
            else:
                n = torch.full_like(mu_hat, float(T))
            n_eff = n * torch.sigmoid(self.neff_head(query_features))
            # population prior: setpoint ~ N(m0, sigma^2 / kappa0), sigma^2 ~ InvGamma(nu0/2, nu0
            # sigma0^2/2) with sigma0^2 = rho * var_pop (within-person share) and kappa0 =
            # rho/(1-rho), so that at n_eff = 0...
            m0 = self.prior_mu[lab, s_next, sex].unsqueeze(-1)
            var_pop = self.prior_var[lab, s_next, sex].unsqueeze(-1)
            rho = self.prior_rho[lab].unsqueeze(-1).clamp(0.05, 0.95)
            sigma0_2 = rho * var_pop
            kappa0 = self.nig_kappa0_scale * rho / (1 - rho)
            nu0 = torch.full_like(kappa0, self.nig_nu0)
            kappa_n = kappa0 + n_eff
            mu_n = (kappa0 * m0 + n_eff * mu_hat) / kappa_n
            nu_n = nu0 + n_eff
            ss = nu0 * sigma0_2 + n_eff * s2_obs + (kappa0 * n_eff / kappa_n) * (mu_hat - m0) ** 2
            sigma_n2 = ss / nu_n
            scale = torch.sqrt(sigma_n2 * (1 + 1 / kappa_n))
            self.last_params = {'df': nu_n, 'loc': mu_n, 'scale': scale, 'n_eff': n_eff, 'n': n,
                                'mu_hat': mu_hat, 's2_obs': s2_obs}
            return mu_n + scale * self._t_ppf(nu_n.view(-1))
        else:
            mu_norm = self.mean_head(query_features)
            raw_log_var = self.logvar_head(query_features)
            log_var = torch.clamp(raw_log_var, min=-10.0)
            mu = mu_norm * seq_std.squeeze(-1) + seq_mean.squeeze(-1)
            log_var = log_var + 2.0 * torch.log(seq_std.squeeze(-1) + 1e-8)
            return mu, log_var
