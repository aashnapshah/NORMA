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

class NormaLight(nn.Module):
    """NORMA light model for conditional prediction, with shared MLP for output heads."""
    def __init__(self, d_model, nhead, num_layers, num_lab_codes, mlp_dropout=0.1):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(2, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(num_lab_codes, d_model)
        self.time_emb = Time2Vec(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(mlp_dropout)
        )
        self.mean_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)
        
    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), 1).bool()

    def forward(
        self,
        x_h,      # (B,T,1)
        s_h,      # (B,T) int {0,1}
        t_h,      # (B,T,1)
        sex,         # (B,) int {0,1}
        lab,         # (B,) int
        s_next,      # (B,) int {0,1}
        t_next,      # (B,1)
        pad_mask=None # (B,T) True for pad
    ):
        B, T = x_h.shape[:2]
        sex_e = self.sex_emb(sex.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)
        lab_e = self.lab_emb(lab.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)

        hist = (
            self.value_emb(x_h)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
            + sex_e.expand(B, T, -1)
            + lab_e.expand(B, T, -1)
        )

        q = (
            self.state_emb(s_next.squeeze(-1))
            + self.time_emb(t_next)
            + sex_e.squeeze(1)
            + lab_e.squeeze(1)
        ).unsqueeze(1) 

        tokens = torch.cat([hist, q], dim=1)  
        attn_mask = self._causal_mask(T + 1, tokens.device)

        if pad_mask is not None:
            pad_mask_ext = torch.cat([pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)], dim=1)
        else:
            pad_mask_ext = None

        H = self.decoder(tokens, mask=attn_mask, src_key_padding_mask=pad_mask_ext)
        if pad_mask_ext is not None:
            idx = (~pad_mask_ext).sum(dim=1).clamp(min=1) - 1  # (B,)
            batch_indices = torch.arange(B, device=H.device)  # Ensure indices are on same device
            query_features = H[batch_indices, idx]
        else:
            query_features = H[:, -1]  # (B, d_model)

        shared = self.output_mlp(query_features)  # (B, 128)
        mu = self.mean_head(shared)               # (B, 1)
        raw_log_var = self.logvar_head(shared)    # (B, 1)
        log_var = torch.clamp(raw_log_var, min=-10.0) #, max=5.0)
        return mu, log_var


class NormaLightDecoder(nn.Module):
    """NORMA light model with decoder-only architecture for conditional prediction."""
    
    def __init__(self, d_model, nhead, num_layers, num_lab_codes):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.state_emb = nn.Embedding(2, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(num_lab_codes, d_model)
        self.time_emb = Time2Vec(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Single output head for both mean and log variance
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Outputs [mean, log_var]
        )

    def _causal_mask(self, L, device):
        """Generate causal attention mask for decoder."""
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self,
        x_h,      # (B,T,1) - Historical values
        s_h,      # (B,T) - Historical states
        t_h,      # (B,T,1) - Historical times
        sex,      # (B,) - Patient sex
        lab,      # (B,) - Lab code
        s_next,   # (B,) - Next state (condition)
        t_next,   # (B,1) - Next time (condition)
        pad_mask=None  # (B,T) - Padding mask
    ):
        B, T = x_h.shape[:2]
        
        # Embed patient-level features
        sex_e = self.sex_emb(sex.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)
        lab_e = self.lab_emb(lab.squeeze(-1)).unsqueeze(1)  # (B, 1, d_model)

        # Create historical sequence embeddings
        hist_emb = (
            self.value_emb(x_h)
            + self.state_emb(s_h)
            + self.time_emb(t_h)
            + sex_e.expand(B, T, -1)
            + lab_e.expand(B, T, -1)
        )

        # Create query token embedding (what we want to predict)
        query_emb = (
            self.state_emb(s_next.squeeze(-1))
            + self.time_emb(t_next)
            + sex_e.squeeze(1)
            + lab_e.squeeze(1)
        ).unsqueeze(1)  # (B, 1, d_model)

        # For decoder-only: query is the target, history is the memory
        # The decoder will attend to historical context while generating the query
        memory = hist_emb  # (B, T, d_model) - Historical context
        tgt = query_emb    # (B, 1, d_model) - What we're predicting

        # Create causal mask for target sequence (only 1 token, so no mask needed)
        tgt_mask = None  # Single token doesn't need causal masking
        
        # Create padding mask for memory (historical sequence)
        memory_key_padding_mask = pad_mask

        # Decoder forward pass
        # tgt: what we're generating (query token)
        # memory: what we're attending to (historical sequence)
        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (B, 1, d_model)
        
        # Extract features from the generated token
        query_features = decoder_output.squeeze(1)  # (B, d_model)
        
        # Predict mean and stabilized log variance
        output = self.output_head(query_features)  # (B, 2)
        mu = output[:, 0:1]  # (B, 1) - mean
        raw_log_var = output[:, 1:2]  # (B, 1) - raw log variance logits
        #log_var = torch.log(F.softplus(raw_log_var) + 1e-6)
        log_var = torch.clamp(raw_log_var, min=-10.0) #, max=5.0)
        return mu, log_var
    
class NORMAEncoder(nn.Module):
    """Shared base logic for NORMA transformer models."""
    
    def __init__(self, d_model, nhead, num_layers, num_lab_codes):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model)
        self.sex_emb = nn.Embedding(2, d_model)
        self.lab_emb = nn.Embedding(num_lab_codes, d_model)
        self.time_emb = Time2Vec(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def _causal_mask(self, seq_len, device):
        """Generate causal attention mask (boolean: True indicates masked)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def encode(self, x, t, sex, lab, pad_mask=None, causal=True):
        """
        Encode sequence of measurements.
        Args: x(B,T,1), t(B,T,1), sex(B,1), lab(B,1)
        Returns: (B, d_model)
        """
        B, T = x.shape[:2]
        sex_emb = self.sex_emb(sex).squeeze(1).unsqueeze(1).expand(B, T, -1)
        lab_emb = self.lab_emb(lab).squeeze(1).unsqueeze(1).expand(B, T, -1)
        enc_input = self.value_emb(x) + self.time_emb(t) + sex_emb + lab_emb
        
        attn_mask = self._causal_mask(T, x.device) if causal else None
        H = self.encoder(enc_input, mask=attn_mask, src_key_padding_mask=pad_mask)
        if pad_mask is None:
            return H[:, -1]
        idx = (~pad_mask).sum(dim=1).clamp(min=1) - 1  # (B,)
        batch_indices = torch.arange(B, device=H.device)  # Ensure indices are on same device
        return H[batch_indices, idx] # last non-padded timestep instead of [:, -1]
        #return H[:, -1]  # Last timestep

class NORMADecoder(NORMAEncoder):
    """NORMA decoder for conditional prediction."""
    
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_lab_codes=2):
        super().__init__(d_model, nhead, num_layers, num_lab_codes)

        self.q_time_emb = Time2Vec(d_model)
        self.q_cond_emb = nn.Embedding(2, d_model)
        self.q_proj = nn.Linear(d_model * 2, d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)  # mu, log_var
        )

    def _process_query(self, q_t, q_c):
        """Process query time and condition into embedding."""
        t_emb = self.q_time_emb(q_t).squeeze(1)
        c_emb = self.q_cond_emb(q_c)
        return self.q_proj(torch.cat([t_emb, c_emb], dim=-1))

    def forward(self, x, t, sex, lab, q_t, q_c, pad_mask=None, causal=True):
        """l
        Forward pass for conditional prediction.
        Returns: (mu, log_var)
        """
        # Encode sequence and process query
        Z = self.encode(x, t, sex, lab, pad_mask, causal)
        q = self._process_query(q_t, q_c)
        
        # Combine and predict   
        combined = Z + q
        output = self.output_head(combined)
        
        # Predict mean and stabilized log variance
        mu = output[:, 0:1]
        raw_log_var = output[:, 1:2]
        log_var = torch.clamp(raw_log_var, min=-10.0) #, max=5.0)
        return mu, log_var
