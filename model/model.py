import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    """Time2Vec embedding module combining linear and periodic components."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, t):
        """Args: t (B, T, 1), Returns: (B, T, D)"""
        v_linear = self.linear(t)
        v_periodic = torch.sin(self.periodic(t))
        return torch.cat([v_linear, v_periodic], dim=-1)


class TimeEmbedding(nn.Module):
    """Improved time embedding: log-delta-t + Time2Vec on deltas.

    Uses inter-measurement gaps (delta_t) instead of absolute time,
    plus log(delta_t + 1) for a monotonic signal that encodes "how far apart."
    """

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
    """Time embedding for the query token: encodes the horizon (gap from last obs).

    Uses log(horizon + 1) + periodic + linear on the raw horizon value.
    """

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

class NormaLightV1(nn.Module):
    """Legacy NormaLight (pre-age-embedding, decoder-named encoder layers).

    Compatible with older checkpoints (e.g. 87345aff) whose state_dict has
    'decoder.layers.*' keys and no 'age_emb'.
    """

    def __init__(self, d_model, nhead, nlayers, ncodes, nstates=2):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(nstates, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(ncodes, d_model)
        self.time_emb = Time2Vec(d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, nlayers)

        self.mean_head = nn.Linear(d_model, 1)
        self.logvar_head = nn.Linear(d_model, 1)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def forward(self, x_h, s_h, t_h, sex, age, lab, s_next, t_next, pad_mask=None):
        B, T = x_h.shape[:2]
        sex = sex.view(-1).long()
        lab = lab.view(-1).long()
        s_h = s_h.long()
        s_next = s_next.view(-1).long()

        sex_e = self.sex_emb(sex)
        lab_e = self.lab_emb(lab)

        hist = (
            self.value_emb(x_h)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
            + sex_e.unsqueeze(1).expand(B, T, -1)
            + lab_e.unsqueeze(1).expand(B, T, -1)
        )

        t_next_reshaped = t_next.view(B, 1, 1)
        q = (
            self.state_emb(s_next)
            + self.time_emb(t_next_reshaped).squeeze(1)
            + sex_e
            + lab_e
        ).unsqueeze(1)

        tokens = torch.cat([hist, q], dim=1)
        attn_mask = self._causal_mask(T + 1, tokens.device)

        pad_mask_ext = None
        if pad_mask is not None:
            pad_mask_ext = torch.cat(
                [pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)],
                dim=1,
            )

        H = self.decoder(tokens, mask=attn_mask, src_key_padding_mask=pad_mask_ext)

        query_features = H[:, -1]
        mu = self.mean_head(query_features)
        raw_log_var = self.logvar_head(query_features)
        log_var = torch.clamp(raw_log_var, min=-10.0)
        return mu, log_var


class NormaLight(nn.Module):
    def __init__(self, d_model, nhead, nlayers, ncodes, nstates=2, shared_mlp=False, mlp_dropout=0.1):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(nstates, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(ncodes, d_model)
        self.age_emb = nn.Linear(1, d_model)
        self.time_emb = Time2Vec(d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        self.shared_mlp = shared_mlp
        if shared_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
            )
            self.mean_head = nn.Linear(128, 1)
            self.logvar_head = nn.Linear(128, 1)
        else:
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def forward(self, x_h, s_h, t_h, sex, age, lab, s_next, t_next, pad_mask=None):
        B, T = x_h.shape[:2]
        sex = sex.view(-1).long()
        age = age.view(-1, 1).float()  # Fix: ensure age is (B, 1) for passing to nn.Linear(1, d_model)
        lab = lab.view(-1).long()
        s_h = s_h.long()
        s_next = s_next.view(-1).long()

        sex_e = self.sex_emb(sex)
        lab_e = self.lab_emb(lab)
        age_e = self.age_emb(age)  # compute age embedding with correct shape

        hist = (
            self.value_emb(x_h)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
            + sex_e.unsqueeze(1).expand(B, T, -1)
            + age_e.unsqueeze(1).expand(B, T, -1)
            + lab_e.unsqueeze(1).expand(B, T, -1)
        )

        t_next_reshaped = t_next.view(B, 1, 1)
        q = (
            self.state_emb(s_next)
            + self.time_emb(t_next_reshaped).squeeze(1)
            + sex_e
            + age_e
            + lab_e
        ).unsqueeze(1)

        tokens = torch.cat([hist, q], dim=1)
        attn_mask = self._causal_mask(T + 1, tokens.device)

        pad_mask_ext = None
        if pad_mask is not None:
            pad_mask_ext = torch.cat(
                [pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)],
                dim=1,
            )

        H = self.encoder(tokens, mask=attn_mask, src_key_padding_mask=pad_mask_ext)

        query_features = H[:, -1]
        if self.shared_mlp:
            query_features = self.output_mlp(query_features)

        mu = self.mean_head(query_features)
        raw_log_var = self.logvar_head(query_features)
        log_var = torch.clamp(raw_log_var, min=-10.0)
        return mu, log_var


class NORMA(nn.Module):
    def __init__(self, d_model, nhead, nlayers, nstates, ncodes, shared_mlp=False, mlp_dropout=0.1):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(nstates, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(ncodes, d_model)
        self.age_emb = nn.Linear(1, d_model)
        self.time_emb = Time2Vec(d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        self.shared_mlp = shared_mlp
        if shared_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
            )
            self.mean_head = nn.Linear(128, 1)
            self.logvar_head = nn.Linear(128, 1)
        else:
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def forward(self, x_h, s_h, t_h, sex, age, lab, s_next, t_next, pad_mask=None):
        B, T = x_h.shape[:2]
        sex = sex.view(-1).long()
        age = age.view(-1).float()
        lab = lab.view(-1).long()
        s_h = s_h.long()
        s_next = s_next.view(-1).long()

        sex_e = self.sex_emb(sex)
        lab_e = self.lab_emb(lab)
        age_e = self.age_emb(age)
        
        hist = (
            self.value_emb(x_h)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
            + age_e.unsqueeze(1).expand(B, T, -1)
            + sex_e.unsqueeze(1).expand(B, T, -1)
            + lab_e.unsqueeze(1).expand(B, T, -1)
        )

        t_next = t_next.view(B, 1, 1)
        q = (
            self.state_emb(s_next)
            + self.time_emb(t_next).squeeze(1)
            + sex_e
            + age_e
            + lab_e
        ).unsqueeze(1)

        tokens = torch.cat([hist, q], dim=1)
        attn_mask = self._causal_mask(T + 1, tokens.device)

        pad_mask_ext = None
        if pad_mask is not None:
            pad_mask_ext = torch.cat(
                [pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)],
                dim=1,
            )

        H = self.encoder(tokens, mask=attn_mask, src_key_padding_mask=pad_mask_ext)

        query_features = H[:, -1]
        if self.shared_mlp:
            query_features = self.output_mlp(query_features)

        mu = self.mean_head(query_features)
        raw_log_var = self.logvar_head(query_features)
        log_var = torch.clamp(raw_log_var, min=-10.0)
        return mu, log_var


class NORMA2(nn.Module):
    """NORMA v2: decoder-only transformer with improved time encoding,
    within-sequence normalization, and quantile output heads.

    Sequence layout: [context] [hist_1 ... hist_T] [query]
    - context: single token encoding patient demographics (sex, age, lab code)
    - hist: value + state + time per measurement
    - query: target state + prediction horizon

    Key changes from NormaLight:
    1. Context token for demographics (no repeated sex/age/lab on every token)
    2. Log-delta-t time encoding (monotonic) instead of Time2Vec
    3. Within-sequence normalization (subtract mean, divide by std)
    4. Quantile output heads (or Gaussian via output_mode='gaussian')
    5. Binned age embedding
    """

    QUANTILES = [0.025, 0.25, 0.50, 0.75, 0.975]

    def __init__(self, d_model, nhead, nlayers, nstates, ncodes,
                 output_mode='quantile', mlp_dropout=0.1, age_bins=7):
        super().__init__()
        self.output_mode = output_mode
        self.d_model = d_model

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

        # Decoder-only transformer (causal self-attention)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                           dropout=mlp_dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, nlayers)

        # Output heads
        if output_mode == 'quantile':
            self.quantile_head = nn.Linear(d_model, len(self.QUANTILES))
        else:
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def _bin_age(self, age):
        age = age.view(-1).float()
        return torch.clamp((age - 20) / 10, min=0, max=self.age_bins - 1).long()

    def _seq_normalize(self, x_h, pad_mask):
        """(x - mean) / std over valid positions. Returns normalized x, mean, std."""
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).float()
        else:
            valid = torch.ones_like(x_h)

        n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x_h * valid).sum(dim=1, keepdim=True) / n_valid
        var = ((x_h - mean) ** 2 * valid).sum(dim=1, keepdim=True) / n_valid
        std = (var + 1e-6).sqrt()

        x_norm = (x_h - mean) / std
        x_norm = x_norm * self.seq_norm_scale + self.seq_norm_bias
        return x_norm, mean, std

    def forward(self, x_h, s_h, t_h, sex, age, lab, s_next, t_next, pad_mask=None):
        B, T = x_h.shape[:2]
        sex = sex.view(-1).long()
        lab = lab.view(-1).long()
        s_h = s_h.long()
        s_next = s_next.view(-1).long()

        # Within-sequence normalize values
        x_norm, seq_mean, seq_std = self._seq_normalize(x_h, pad_mask)

        # Context token: sex + age + lab (single token, position 0)
        ctx = (self.sex_emb(sex) + self.age_emb(self._bin_age(age)) + self.lab_emb(lab)).unsqueeze(1)  # (B, 1, D)

        # History tokens: value + state + time only
        hist = (
            self.value_emb(x_norm)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
        )  # (B, T, D)

        # Query token: state + horizon only
        if pad_mask is not None:
            lengths = (~pad_mask).sum(dim=1)
            t_last = t_h[torch.arange(B, device=t_h.device), lengths - 1, 0]
        else:
            t_last = t_h[:, -1, 0]

        horizon = (t_next.view(B) - t_last).clamp(min=0).view(B, 1, 1)
        query = (
            self.state_emb(s_next)
            + self.time_emb_query(horizon).squeeze(1)
        ).unsqueeze(1)  # (B, 1, D)

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
                             tgt_key_padding_mask=pad_mask_ext,
                             memory_key_padding_mask=pad_mask_ext)
        query_features = H[:, -1]  # (B, D)

        if self.output_mode == 'quantile':
            q_norm = self.quantile_head(query_features)
            q_denorm = q_norm * seq_std.squeeze(-1) + seq_mean.squeeze(-1)
            return q_denorm
        else:
            mu_norm = self.mean_head(query_features)
            raw_log_var = self.logvar_head(query_features)
            log_var = torch.clamp(raw_log_var, min=-10.0)
            mu = mu_norm * seq_std.squeeze(-1) + seq_mean.squeeze(-1)
            log_var = log_var + 2.0 * torch.log(seq_std.squeeze(-1) + 1e-8)
            return mu, log_var


# class NormaLightDecoder(nn.Module):
#     """NORMA light model with decoder-only architecture for conditional prediction."""
    
#     def __init__(self, d_model, nhead, nlayers, num_lab_codes):
#         super().__init__()
#         self.value_emb = nn.Linear(1, d_model)
#         self.state_emb = nn.Embedding(2, d_model)
#         self.sex_emb = nn.Embedding(2, d_model)
#         self.lab_emb = nn.Embedding(num_lab_codes, d_model)
#         self.time_emb = Time2Vec(d_model)
        
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model, nhead, batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        
#         # Single output head for both mean and log variance
#         self.output_head = nn.Sequential(
#             nn.Linear(d_model, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)  # Outputs [mean, log_var]
#         )

#     def _causal_mask(self, L, device):
#         """Generate causal attention mask for decoder."""
#         mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
#         return mask.masked_fill(mask == 1, float("-inf"))

#     def forward(
#         self,
#         x_h,      # (B,T,1) - Historical values
#         s_h,      # (B,T) - Historical states
#         t_h,      # (B,T,1) - Historical times
#         sex,      # (B,) - Patient sex
#         lab,      # (B,) - Lab code
#         s_next,   # (B,) - Next state (condition)
#         t_next,   # (B,1) - Next time (condition)
#         pad_mask=None  # (B,T) - Padding mask
#     ):
#         B, T = x_h.shape[:2]
        
#         # Embed patient-level features
#         sex_e = self.sex_emb(sex.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)
#         lab_e = self.lab_emb(lab.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)

#         # Create historical sequence embeddings
#         hist_emb = (
#             self.value_emb(x_h)
#             + self.state_emb(s_h)
#             + self.time_emb(t_h)
#             + sex_e.expand(B, T, -1)
#             + lab_e.expand(B, T, -1)
#         )

#         # Create query token embedding (what we want to predict)
#         query_emb = (
#             self.state_emb(s_next.squeeze(-1))
#             + self.time_emb(t_next)
#             + sex_e.squeeze(1)
#             + lab_e.squeeze(1)
#         ).unsqueeze(1)  # (B, 1, d_model)

#         # For decoder-only: query is the target, history is the memory
#         # The decoder will attend to historical context while generating the query
#         memory = hist_emb  # (B, T, d_model) - Historical context
#         tgt = query_emb    # (B, 1, d_model) - What we're predicting

#         # Create causal mask for target sequence (only 1 token, so no mask needed)
#         tgt_mask = None  # Single token doesn't need causal masking
        
#         # Create padding mask for memory (historical sequence)
#         memory_key_padding_mask = pad_mask

#         # Decoder forward pass
#         # tgt: what we're generating (query token)
#         # memory: what we're attending to (historical sequence)
#         decoder_output = self.decoder(
#             tgt=tgt,
#             memory=memory,
#             tgt_mask=tgt_mask,
#             memory_key_padding_mask=memory_key_padding_mask
#         )  # (B, 1, d_model)
        
#         # Extract features from the generated token
#         query_features = decoder_output.squeeze(1)  # (B, d_model)
        
#         # Predict mean and stabilized log variance
#         output = self.output_head(query_features)  # (B, 2)
#         mu = output[:, 0:1]  # (B, 1) - mean
#         raw_log_var = output[:, 1:2]  # (B, 1) - raw log variance logits
#         #log_var = torch.log(F.softplus(raw_log_var) + 1e-6)
#         log_var = torch.clamp(raw_log_var, min=-10.0) #, max=5.0)
#         return mu, log_var
    
# class NORMAEncoder(nn.Module):
#     """Shared base logic for NORMA transformer models."""
    
#     def __init__(self, d_model, nhead, nlayers, num_lab_codes):
#         super().__init__()
#         self.value_emb = nn.Linear(1, d_model)
#         self.sex_emb = nn.Embedding(2, d_model)
#         self.lab_emb = nn.Embedding(num_lab_codes, d_model)
#         self.time_emb = Time2Vec(d_model)

#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

#     def _causal_mask(self, seq_len, device):
#         """Generate causal attention mask (boolean: True indicates masked)."""
#         return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

#     def encode(self, x, t, sex, lab, pad_mask=None, causal=True):
#         """
#         Encode sequence of measurements.
#         Args: x(B,T,1), t(B,T,1), sex(B,1), lab(B,1)
#         Returns: (B, d_model)
#         """
#         B, T = x.shape[:2]
#         sex_emb = self.sex_emb(sex).squeeze(1).unsqueeze(1).expand(B, T, -1)
#         lab_emb = self.lab_emb(lab).squeeze(1).unsqueeze(1).expand(B, T, -1)
#         enc_input = self.value_emb(x) + self.time_emb(t) + sex_emb + lab_emb
        
#         attn_mask = self._causal_mask(T, x.device) if causal else None
#         H = self.encoder(enc_input, mask=attn_mask, src_key_padding_mask=pad_mask)
#         if pad_mask is None:
#             return H[:, -1]
#         idx = (~pad_mask).sum(dim=1).clamp(min=1) - 1  # (B,)
#         batch_indices = torch.arange(B, device=H.device)  # Ensure indices are on same device
#         return H[batch_indices, idx] # last non-padded timestep instead of [:, -1]
#         #return H[:, -1]  # Last timestep

# class NORMADecoder(NORMAEncoder):
#     """NORMA decoder for conditional prediction."""
    
#     def __init__(self, d_model=128, nhead=4, nlayers=4, num_lab_codes=2):
#         super().__init__(d_model, nhead, nlayers, num_lab_codes)

#         self.q_time_emb = Time2Vec(d_model)
#         self.q_cond_emb = nn.Embedding(2, d_model)
#         self.q_proj = nn.Linear(d_model * 2, d_model)

#         self.output_head = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, 2)  # mu, log_var
#         )

#     def _process_query(self, q_t, q_c):
#         """Process query time and condition into embedding."""
#         t_emb = self.q_time_emb(q_t).squeeze(1)
#         c_emb = self.q_cond_emb(q_c)
#         return self.q_proj(torch.cat([t_emb, c_emb], dim=-1))

#     def forward(self, x, t, sex, lab, q_t, q_c, pad_mask=None, causal=True):
#         """l
#         Forward pass for conditional prediction.
#         Returns: (mu, log_var)
#         """
#         # Encode sequence and process query
#         Z = self.encode(x, t, sex, lab, pad_mask, causal)
#         q = self._process_query(q_t, q_c)
        
#         # Combine and predict   
#         combined = Z + q
#         output = self.output_head(combined)
        
#         # Predict mean and stabilized log variance
#         mu = output[:, 0:1]
#         raw_log_var = output[:, 1:2]
#         log_var = torch.clamp(raw_log_var, min=-10.0) #, max=5.0)
#         return mu, log_var