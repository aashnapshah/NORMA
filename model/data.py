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

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

sys.path.append('../../NORMA/process/')
from config import REFERENCE_INTERVALS

TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}
CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, sequences, nstates):
        self.seq = sequences
        self.nstates = nstates
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        
        x = torch.from_numpy(seq["x"]).float().unsqueeze(-1)
        t = torch.from_numpy(seq["t"]).float().unsqueeze(-1)
        s_raw = np.asarray(seq["s"] if self.nstates == 2 else seq["s3"], dtype=np.int64)
        s = torch.from_numpy(s_raw).long()
        if self.nstates == 3:
            s = s + 1  # -1,0,1 -> 0,1,2

        x_h = x[:-1]
        t_h = t[:-1]
        s_h = s[:-1]

        t_next = t[-1]
        s_next = s[-1].unsqueeze(0).clone()   
        x_next = x[-1]   
        
        sex_val = 1 if (seq['sex'] == 'F' or seq['sex'] == 1) else 0
        sex = torch.tensor([sex_val], dtype=torch.long)
        age = torch.tensor([seq["age"]], dtype=torch.float)
        cid = torch.tensor([seq["cid"]], dtype=torch.long)
        pids = seq["pid"] 
        
        return x_h, s_h, t_h, sex, age, cid, s_next, t_next, x_next, pids

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
    x_h, s_h, t_h, sex, age, cid, s_next, t_next, x_next, pids = zip(*batch)

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

    max_len = x_h.shape[1]
    pad_mask = torch.ones(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        pad_mask[i, :l] = False

    return {
        'x_h': x_h,
        't_h': t_h, 
        's_h': s_h,
        'sex': sex,
        'age': age,
        'cid': cid,
        's_next': s_next,
        't_next': t_next,
        'x_next': x_next,
        # 'ref_mu': ref_mu,
        # 'ref_var': ref_var,
        'pad_mask': pad_mask,
        'pids': list(pids)
    }

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

def load_and_split_data(sequences_path, source, num_patients=None, random_state=42, print_info=True, nstates=2):
    sequences_path = os.path.join(sequences_path, f'{source}_sequences_v2.pkl')
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)

    if num_patients is not None:
        sequences = sample_by_key(sequences, num_patients, key="cid", seed=random_state, replace=False)

    stratify_labels = get_stratify_labels(sequences, nstates=nstates)

    train_val_seq, test_seq = partial_stratified_split(
        sequences, stratify_labels, test_size=0.2, random_state=random_state
    )

    train_seq, val_seq = partial_stratified_split(
        train_val_seq, get_stratify_labels(train_val_seq, nstates=nstates), test_size=0.125, random_state=random_state
    )
    
    sequences_ids = set([seq['pid'] for seq in sequences])
    train_ids = set([seq['pid'] for seq in train_seq])
    val_ids = set([seq['pid'] for seq in val_seq])
    test_ids = set([seq['pid'] for seq in test_seq])
    
    if print_info:
        print(f"{source} Dataset Split Summary")
        if num_patients:
            print(f"{'Sampled Set':<18}: {len(sequences):>4} sequences, {len(sequences_ids):>4} patients")
        else:
            print(f"{'Total Set':<18}: {len(sequences):>4} sequences, {len(sequences_ids):>4} patients")
        print(f"{'Training Set':<18}: {len(train_seq):>4} sequences, {len(train_ids):>4} patients")
        print(f"{'Validation Set':<18}: {len(val_seq):>4} sequences, {len(val_ids):>4} patients")
        print(f"{'Test Set':<18}: {len(test_seq):>4} sequences, {len(test_ids):>4} patients")
        print('=' * 90)
    return train_seq, val_seq, test_seq


def create_dataloaders(train_seq, val_seq, test_seq, nstates, batch_size=16, random_state=42):
    """Create train/val/test dataloaders."""

    train_loader = DataLoader(
        TimeSeriesDataset(train_seq, nstates),
        batch_size=batch_size,
        sampler=create_weighted_sampler(train_seq),
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        TimeSeriesDataset(val_seq, nstates), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        TimeSeriesDataset(test_seq, nstates), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader