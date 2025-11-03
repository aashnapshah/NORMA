import numpy as np
import pandas as pd
import random
from collections import defaultdict
import sys
import warnings
from torch.utils.data import WeightedRandomSampler

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

sys.path.append('../../SETPOINT/')
from process.config import REFERENCE_INTERVALS

# Test vocabulary
TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}

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

def is_normal(row, code_col='test_name', pred_col='pred_mean', mu_col=None, var_col=None):
    if mu_col is None and var_col is None:
        """Check if measurement is within normal range."""
        sex = 'F' if row['sex'] == 1 else 'M'
        low, high, _ = REFERENCE_INTERVALS[row[code_col]][sex]
    else:
        low = row[mu_col] - 2 * np.sqrt(row[var_col])
        high = row[mu_col] + 2 * np.sqrt(row[var_col])
        row['range'] = f"{low} - {high}"
    return low < row[pred_col] < high

def get_ref_stats(test_name, sex):
    """Get reference mean and variance."""
    sex_key = 'F' if sex == 1 else 'M'
    low, high, _ = REFERENCE_INTERVALS[test_name][sex_key]
    return (low + high) / 2, ((high - low) / 4) ** 2

def get_stratify_labels(sequences):
    stratify_labels = []
    for i, seq in enumerate(sequences):
        cid = seq["cid"] 
        #s_next = seq["s"][-1]
        source = seq["source"]
        stratify_labels.append(f"{cid}_{source}")
    return stratify_labels

def load_data(filename, num_patients=None, min_points=3):
    """Load and filter data."""
    df = pd.read_csv(filename)
    
    counts = df.groupby(['subject_id', 'test_name']).size()
    valid = counts[counts >= min_points].reset_index()
    df = df.merge(valid, on=['subject_id', 'test_name']).drop(columns=[0])
    
    if num_patients:
        patients = df['subject_id'].unique()
        sample_size = min(num_patients, len(patients))
        sampled = np.random.choice(patients, sample_size, replace=False)
        df = df[df['subject_id'].isin(sampled)]
    
    return df

def create_sequences(df, min_context=2):
    """Create training sequences."""
    df['test_name'] = df['test_name'].fillna('NA')
    df['condition'] = df.apply(lambda row: is_normal(row, pred_col='numeric_value'), axis=1)
    sequences = []
    
    for (source, pid, test_name), group in df.groupby(['source', 'subject_id', 'test_name']):
        if len(group) < min_context + 1:
            continue
        source = group['source'].iloc[0]
        group = group.sort_values('time')
        times = pd.to_datetime(group['time'], errors='coerce')
        t = ((times - times.iloc[0]).dt.total_seconds() / 86400).astype(np.float32).values
        x = group['numeric_value'].astype(np.float32).values
        s = group['condition'].astype(np.int32).values
        
        # replace F with 1 and M with 0
        sex = 1 if group['sex'].iloc[0] == 'F' else 0
        
        cid = TEST_VOCAB[test_name]
        ref_mu, ref_var = get_ref_stats(test_name, sex)
        
        sequences.append({
            "source": source,
            "x": x,  # Historical measurements (all but last)
            "t": t,  # Historical timestamps
            "s": s,  # Historical condition states
            "sex": sex,        # Gender
            "cid": cid,        # Lab code ID
            "ref_mu": np.array([ref_mu], dtype=np.float32),
            "ref_var": np.array([ref_var], dtype=np.float32),
            "subject_id": pid,
        })
    
    return sequences

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