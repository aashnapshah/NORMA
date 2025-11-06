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
sys.path.append('../../SETPOINT/')
from process.config import REFERENCE_INTERVALS

# Test vocabulary
TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        x = torch.from_numpy(seq["x"]).float().unsqueeze(-1)
        t = torch.from_numpy(seq["t"]).float().unsqueeze(-1)
        s = torch.from_numpy(seq["s"]).long()
        
        x_h = x[:-1]   
        t_h = t[:-1]    
        s_h = s[:-1]    
        
        t_next = t[-1]  
        s_next = torch.tensor([s[-1]], dtype=torch.long)   
        x_next = x[-1]   
        
        seq['sex'] = 1 if seq['sex'] == 'F' or seq['sex'] == 1 else seq['sex']
        sex = torch.tensor([seq["sex"]], dtype=torch.long)
        cid = torch.tensor([seq["cid"]], dtype=torch.long)
        pids = seq["pid"] 
        
        return x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, pids

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

def get_stratify_labels(sequences):
    stratify_labels = []
    for i, seq in enumerate(sequences):
        cid = seq["cid"] 
        #s_next = seq["s"][-1]
        source = seq["source"]
        stratify_labels.append(f"{cid}_{source}")
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
    x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, pids = zip(*batch)

    lengths = [xh.shape[0] for xh in x_h]
    x_h = pad_sequence(x_h, batch_first=True)
    t_h = pad_sequence(t_h, batch_first=True)
    s_h = pad_sequence(s_h, batch_first=True)

    sex = torch.stack(sex)
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
        'cid': cid,
        's_next': s_next,
        't_next': t_next,
        'x_next': x_next,
        # 'ref_mu': ref_mu,
        # 'ref_var': ref_var,
        'pad_mask': pad_mask,
        'pids': list(pids)
    }

def load_and_split_data(sequences_path, source, num_patients=None, random_state=42, print_info=True):
    sequences_path = os.path.join(sequences_path, f'{source}_sequences.pkl')
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)
        
    if num_patients is not None:
        sequences = sample_by_key(sequences, num_patients, key="cid", seed=random_state, replace=False)
  
    stratify_labels = get_stratify_labels(sequences)    
    train_val_seq, test_seq = train_test_split(
        sequences, test_size=0.2, stratify=stratify_labels, random_state=random_state
    )

    stratify_labels = get_stratify_labels(train_val_seq)
    train_seq, val_seq = train_test_split(
        train_val_seq, test_size=0.125, stratify=stratify_labels, random_state=random_state
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


def create_dataloaders(train_seq, val_seq, test_seq, batch_size=16, random_state=42):
    """Create train/val/test dataloaders."""

    train_loader = DataLoader(
        TimeSeriesDataset(train_seq),
        batch_size=batch_size,
        sampler=create_weighted_sampler(train_seq),
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        TimeSeriesDataset(val_seq), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        TimeSeriesDataset(test_seq), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
