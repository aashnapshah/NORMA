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