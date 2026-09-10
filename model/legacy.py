"""Architectures that predate the published model, kept only so the
checkpoints in model/logs/ stay loadable.

Of the 48 runs with saved checkpoints, 21 are NormaLight and 4 are NORMA --
all Gaussian-head runs from before the quantile head. No published result uses
them: the paper's model is NORMA2 with the quantile head (run q_age_set, see
scripts/lib/datasets.py NORMA_RUN_ID), and the Gaussian head was dropped in the
revision. utils.create_model is the only importer.

The 21 NormaLight checkpoints load. The 4 NORMA ones do not, and did not
before this file existed: they predate the age embedding, and create_model's
legacy detection only covers the NormaLight/NormaLightV1 pair. Left as is --
no result depends on them.
"""
import torch
import torch.nn as nn
import math


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
