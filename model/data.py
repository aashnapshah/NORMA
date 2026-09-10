import os
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import random
import sys
import warnings

if hasattr(pd.errors, 'SettingWithCopyWarning'):
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

sys.path.append('../../NORMA/process/')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'process'))  # repo-relative
from config import REFERENCE_INTERVALS

TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}
CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

# must match process/draw_meta.py: subject_id is only unique within a source
SRC_MULT = 10 ** 12
SRC_ID = {'mimiciv': 0, 'ehrshot': 1}

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""

    def __init__(self, sequences, nstates, normalize=False, panel=None,
                 drawmeta=None, max_draws=128):
        self.seq = sequences
        self.nstates = nstates
        self.normalize = normalize
        # Optional co-analyte draw table (n_draws, K) from process/covariates.py;
        # sequences index it via seq['draw_idx'].
        self.panel = panel
        self.drawmeta = drawmeta      # process/draw_meta.py index; enables the full past
        self.max_draws = max_draws
        self.collapse_icu = True

    def __len__(self):
        return len(self.seq)

    def _get_ref_bounds(self, test_name, sex01):
        sex_str = 'F' if sex01 == 1 else 'M'
        low, high, _ = REFERENCE_INTERVALS[test_name][sex_str]
        return float(low), float(high)

    def __getitem__(self, idx):
        seq = self.seq[idx]

        x = torch.from_numpy(seq["x"]).float().unsqueeze(-1)
        t = torch.from_numpy(seq["t"]).float().unsqueeze(-1)
        s_raw = np.asarray(seq["s"] if self.nstates == 2 else seq["s3"], dtype=np.int64)
        s = torch.from_numpy(s_raw).long()
        if self.nstates == 3:
            s = s + 1  # -1,0,1 -> 0,1,2

        sex_val = 1 if (seq['sex'] == 'F' or seq['sex'] == 1) else 0

        if self.normalize:
            ref_low, ref_high = self._get_ref_bounds(seq['test_name'], sex_val)
            span = ref_high - ref_low
            x = (x - ref_low) / span
        else:
            ref_low, ref_high = 0.0, 1.0

        x_h = x[:-1]
        t_h = t[:-1]
        s_h = s[:-1]

        t_next = t[-1]
        s_next = s[-1].unsqueeze(0).clone()
        x_next = x[-1]

        sex = torch.tensor([sex_val], dtype=torch.long)
        age = torch.tensor([seq["age"]], dtype=torch.float)
        cid = torch.tensor([seq["cid"]], dtype=torch.long)
        pids = seq["pid"]
        ref_low_t = torch.tensor([ref_low], dtype=torch.float)
        ref_high_t = torch.tensor([ref_high], dtype=torch.float)

        # Optional per-measurement covariates (v3 sequences). History = [:-1],
        # the last element belongs to the query/target measurement.
        extras = {}
        # Always carried (prior-anchored losses and post-hoc shrinkage): the
        # sex-specific population interval in the model's units and the number of
        # target-analyte observations in the history.
        pop_low, pop_high = self._get_ref_bounds(seq['test_name'], sex_val)
        if self.normalize:
            pop_low, pop_high = 0.0, 1.0
        extras['pop_low'] = torch.tensor([pop_low], dtype=torch.float)
        extras['pop_high'] = torch.tensor([pop_high], dtype=torch.float)
        extras['n_hist'] = torch.tensor([x_h.shape[0]], dtype=torch.float)
        if 'age_t' in seq:
            age_t = torch.from_numpy(np.asarray(seq['age_t'], dtype=np.float32))
            extras['age_h'] = age_t[:-1]
            extras['age_next'] = age_t[-1:].clone()
        if 'setting' in seq:
            st = torch.from_numpy(np.asarray(seq['setting'], dtype=np.int64))
            if self.collapse_icu:
                st[st == 4] = 3  # 3-level vocabulary: icu -> inpatient (dev cohorts carry no ICU signal)
            extras['setting_h'] = st[:-1]
            extras['setting_next'] = st[-1:].clone()
        if self.panel is not None and self.drawmeta is not None and 'draw_idx' in seq:
            # Full irregular past: every draw this patient had before the query,
            # not just the timestamps where the target analyte happened to be drawn.
            dm = self.drawmeta
            di = np.asarray(seq['draw_idx'], dtype=np.int64)
            q_time = float(dm['row_time'][di[-1]])          # absolute time of the query draw
            key = SRC_ID[seq['source']] * SRC_MULT + int(seq['pid'])
            j = int(np.searchsorted(dm['pat_key'], key))
            rows = dm['order'][dm['pat_ptr'][j]:dm['pat_ptr'][j + 1]]   # ascending in time
            rt = dm['row_time'][rows]
            rows = rows[rt < q_time]        # strict: co-analytes drawn at the query
            rows = rows[-self.max_draws:]   # timestamp are not available at prediction

            if len(rows) == 0:              # no prior draw; keep one empty token
                rows = di[-1:][:0]
                rt_sel = np.zeros(1, dtype=np.float32)
                obs = np.zeros(1, dtype=np.int64)
                xv = np.zeros(1, dtype=np.float32)
                sv = np.zeros(1, dtype=np.int64)
                co = np.full((1, self.panel.shape[1]), np.nan, dtype=np.float32)
            else:
                rt_sel = dm['row_time'][rows]
                obs = np.zeros(len(rows), dtype=np.int64)
                xv = np.zeros(len(rows), dtype=np.float32)
                sv = np.zeros(len(rows), dtype=np.int64)
                hist_di = di[:-1]
                if len(hist_di):
                    # map each selected panel row back to its position in the
                    # target analyte's own history, where it has one
                    srt = np.argsort(hist_di, kind='stable')
                    sd = hist_di[srt]
                    pos = np.clip(np.searchsorted(sd, rows), 0, len(sd) - 1)
                    hit = sd[pos] == rows
                    src_pos = srt[pos[hit]]
                    obs[hit] = 1
                    xv[hit] = x_h.squeeze(-1).numpy()[src_pos]
                    sv[hit] = s_h.numpy()[src_pos]
                co = self.panel[rows].astype(np.float32)
                co[:, seq['cid']] = np.nan  # target analyte is carried by the token value

            x_h = torch.from_numpy(xv).float().unsqueeze(-1)
            s_h = torch.from_numpy(sv).long()
            t_h = torch.from_numpy(rt_sel.astype(np.float32)).float().unsqueeze(-1)
            t_next = torch.tensor([q_time], dtype=torch.float)  # same absolute clock
            extras['obs_h'] = torch.from_numpy(obs).long()
            extras['co_h'] = torch.from_numpy(np.nan_to_num(co, nan=0.0))
            extras['co_mask'] = torch.from_numpy(np.isfinite(co).astype(np.float32))

        elif self.panel is not None and 'draw_idx' in seq:
            rows = self.panel[np.asarray(seq['draw_idx'][:-1])].astype(np.float32)  # (T, K)
            rows[:, seq['cid']] = np.nan  # the target analyte is already the token value
            mask = np.isfinite(rows)
            extras['co_h'] = torch.from_numpy(np.nan_to_num(rows, nan=0.0))
            extras['co_mask'] = torch.from_numpy(mask.astype(np.float32))

        return x_h, s_h, t_h, sex, age, cid, s_next, t_next, x_next, pids, ref_low_t, ref_high_t, extras

def sample_by_key(seq_list, n, key="cid", seed=0, replace=False):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in seq_list:
        buckets[s[key]].append(s)
    out = []
    for cid, items in buckets.items():
        k = n if replace else min(n, len(items))
        if replace and len(items) > 0:
            out.extend(rng.choices(items, k=k))
        else:
            out.extend(rng.sample(items, k))
    return out

def get_stratify_labels(sequences, nstates=2):
    """
    For each code (cid), get counts per state. If any state has < 2 samples per cid,
    stratify only by source and code; else include s_next.
    nstates: 2 uses seq['s'], 3 uses seq['s3'].
    """
    from collections import defaultdict

    state_arr_key = 's' if nstates == 2 else 's3'
    state_keys = list(range(nstates))
    code_snext_counts = defaultdict(lambda: {k: 0 for k in state_keys})
    for seq in sequences:
        cid = seq['cid']
        s_next = seq[state_arr_key][-1].astype(int)
        if s_next in code_snext_counts[cid]:
            code_snext_counts[cid][s_next] += 1

    stratify_labels = []
    for seq in sequences:
        cid = seq['cid']
        source = seq['source']
        s_next = seq[state_arr_key][-1]
        counts = code_snext_counts[cid]
        if any(counts[k] < 2 for k in state_keys):
            label = f"{cid}_{source}"
        else:
            label = f"{cid}_{source}_{s_next}"
        stratify_labels.append(label)
    return stratify_labels

def create_weighted_sampler(sequences):
    cid_counts = {}
    for seq in sequences:
        cid = seq['cid']
        cid_counts[cid] = cid_counts.get(cid, 0) + 1
    
    weights = []
    for seq in sequences:
        cid = seq['cid']
        weights.append(1.0 / cid_counts[cid])
        
    return WeightedRandomSampler(weights, len(weights))

def collate_fn(batch):
    """Collate function for DataLoader."""
    cols = list(zip(*batch))
    x_h, s_h, t_h, sex, age, cid, s_next, t_next, x_next, pids, ref_low, ref_high = cols[:12]
    extras = cols[12] if len(cols) > 12 else tuple({} for _ in batch)

    lengths = [xh.shape[0] for xh in x_h]
    x_h = pad_sequence(x_h, batch_first=True)
    t_h = pad_sequence(t_h, batch_first=True)
    s_h = pad_sequence(s_h, batch_first=True)

    sex = torch.stack(sex)
    age = torch.stack(age)
    cid = torch.stack(cid)
    s_next = torch.stack(s_next)
    t_next = torch.stack(t_next)
    x_next = torch.stack(x_next)
    ref_low = torch.stack(ref_low)
    ref_high = torch.stack(ref_high)

    max_len = x_h.shape[1]
    pad_mask = torch.ones(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        pad_mask[i, :l] = False

    out = {
        'x_h': x_h,
        't_h': t_h,
        's_h': s_h,
        'sex': sex,
        'age': age,
        'cid': cid,
        's_next': s_next,
        't_next': t_next,
        'x_next': x_next,
        'ref_low': ref_low,
        'ref_high': ref_high,
        'pad_mask': pad_mask,
        'pids': list(pids)
    }
    # Covariates present in every item of the batch
    keys = set.intersection(*[set(e.keys()) for e in extras]) if len(extras) else set()
    if 'age_h' in keys:
        out['age_h'] = pad_sequence([e['age_h'] for e in extras], batch_first=True)
        out['age_next'] = torch.stack([e['age_next'] for e in extras])
    if 'setting_h' in keys:
        out['setting_h'] = pad_sequence([e['setting_h'] for e in extras], batch_first=True, padding_value=0)
        out['setting_next'] = torch.stack([e['setting_next'] for e in extras])
    if 'co_h' in keys:
        out['co_h'] = pad_sequence([e['co_h'] for e in extras], batch_first=True)
        out['co_mask'] = pad_sequence([e['co_mask'] for e in extras], batch_first=True)
    if 'obs_h' in keys:
        out['obs_h'] = pad_sequence([e['obs_h'] for e in extras], batch_first=True, padding_value=0)
    for k in ('pop_low', 'pop_high', 'n_hist'):
        if k in keys:
            out[k] = torch.stack([e[k] for e in extras])
    return out

def partial_stratified_split(X, y, **kwargs):
    counts = Counter(y)
    y = np.array(y)

    stratifiable = np.array([counts[label] >= 2 for label in y])
    X_strat = [x for x, s in zip(X, stratifiable) if s]
    y_strat = y[stratifiable]

    X_rare = [x for x, s in zip(X, stratifiable) if not s]

    if len(X_strat) > 0:
        X1, X2 = train_test_split(
            X_strat,
            stratify=y_strat,
            **kwargs
        )
    else:
        X1, X2 = [], []

    return X1, X2

def load_drawmeta(sequences_path, source, version='v3'):
    """Patient index over the draw table (process/draw_meta.py), or None if absent."""
    p = os.path.join(sequences_path, f'{source}_drawmeta_{version}.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in ('row_time', 'order', 'pat_key', 'pat_ptr')}


def load_panel(sequences_path, source, version='v3'):
    """Co-analyte draw table written by process/covariates.py, or None if absent."""
    p = os.path.join(sequences_path, f'{source}_panel_{version}.npy')
    if not os.path.exists(p):
        return None
    return np.load(p)


def patient_key(seq):
    """Patient identity. subject_id is only unique *within* a source, so keying on
    seq['pid'] alone would merge distinct MIMIC-IV and EHRSHOT patients."""
    return (seq['source'], seq['pid'])


def patient_split(sequences, test_size=0.2, val_size=0.125, random_state=42):
    """Split whole patients into train/val/test so no patient appears in two splits.

    Unlike the sequence-level split, every analyte sequence belonging to a patient
    lands in the same partition. Required before conditioning a target sequence on
    the patient's other analytes, which would otherwise carry training targets into
    the test set (R3 comment 11).

    Patients are stratified by source, so the MIMIC-IV / EHRSHOT ratio of the full
    set is preserved in each split. cid / next-state balance is left to the size of
    the patient pool and to create_weighted_sampler, since a patient contributes
    many analytes and has no single sequence-level label to stratify on.
    """
    keys = [patient_key(s) for s in sequences]
    patients = sorted(set(keys))
    src = [k[0] for k in patients]
    idx = np.arange(len(patients))

    trainval_i, test_i = train_test_split(
        idx, test_size=test_size, stratify=src, random_state=random_state)
    train_i, val_i = train_test_split(
        trainval_i, test_size=val_size,
        stratify=[src[i] for i in trainval_i], random_state=random_state)

    assign = {}
    for split, part in enumerate((train_i, val_i, test_i)):
        for i in part:
            assign[patients[i]] = split

    out = ([], [], [])
    for seq, k in zip(sequences, keys):
        out[assign[k]].append(seq)
    return out


def load_and_split_data(sequences_path, source, num_patients=None, random_state=42, print_info=True, nstates=2,
                        version='v2', split_by='sequence'):
    """split_by='sequence' reproduces the published split (patient-analyte level,
    stratified by cid/source/next-state). split_by='patient' keeps every sequence of
    a patient in one partition."""
    sequences_path = os.path.join(sequences_path, f'{source}_sequences_{version}.pkl')
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)

    if num_patients is not None:
        sequences = sample_by_key(sequences, num_patients, key="cid", seed=random_state, replace=False)

    if split_by == 'patient':
        train_seq, val_seq, test_seq = patient_split(
            sequences, test_size=0.2, val_size=0.125, random_state=random_state)
    elif split_by == 'sequence':
        stratify_labels = get_stratify_labels(sequences, nstates=nstates)

        train_val_seq, test_seq = partial_stratified_split(
            sequences, stratify_labels, test_size=0.2, random_state=random_state
        )

        train_seq, val_seq = partial_stratified_split(
            train_val_seq, get_stratify_labels(train_val_seq, nstates=nstates), test_size=0.125, random_state=random_state
        )
    else:
        raise ValueError(f"split_by must be 'sequence' or 'patient', got {split_by!r}")

    sequences_ids = set(patient_key(seq) for seq in sequences)
    train_ids = set(patient_key(seq) for seq in train_seq)
    val_ids = set(patient_key(seq) for seq in val_seq)
    test_ids = set(patient_key(seq) for seq in test_seq)
    if split_by == 'patient':
        assert not (train_ids & val_ids), 'patient leak train/val'
        assert not (train_ids & test_ids), 'patient leak train/test'
        assert not (val_ids & test_ids), 'patient leak val/test'
    
    if print_info:
        print(f"{source} Dataset Split Summary (split_by={split_by})")
        if num_patients:
            print(f"{'Sampled Set':<18}: {len(sequences):>4} sequences, {len(sequences_ids):>4} patients")
        else:
            print(f"{'Total Set':<18}: {len(sequences):>4} sequences, {len(sequences_ids):>4} patients")
        print(f"{'Training Set':<18}: {len(train_seq):>4} sequences, {len(train_ids):>4} patients")
        print(f"{'Validation Set':<18}: {len(val_seq):>4} sequences, {len(val_ids):>4} patients")
        print(f"{'Test Set':<18}: {len(test_seq):>4} sequences, {len(test_ids):>4} patients")
        print('=' * 90)
    return train_seq, val_seq, test_seq


def create_dataloaders(train_seq, val_seq, test_seq, nstates, batch_size=16, random_state=42, normalize=False,
                       panel=None, drawmeta=None, max_draws=128):
    """Create train/val/test dataloaders."""

    train_loader = DataLoader(
        TimeSeriesDataset(train_seq, nstates, normalize=normalize, panel=panel,
                          drawmeta=drawmeta, max_draws=max_draws),
        batch_size=batch_size,
        sampler=create_weighted_sampler(train_seq),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        TimeSeriesDataset(val_seq, nstates, normalize=normalize, panel=panel,
                          drawmeta=drawmeta, max_draws=max_draws),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        TimeSeriesDataset(test_seq, nstates, normalize=normalize, panel=panel,
                          drawmeta=drawmeta, max_draws=max_draws),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader