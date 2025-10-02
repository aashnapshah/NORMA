import numpy as np
import pandas as pd
import sys
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

sys.path.append('../../SETPOINT/')
from process.config import REFERENCE_INTERVALS

# Test vocabulary
TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}

def is_normal(row):
    """Check if measurement is within normal range."""
    sex = 'F' if row['gender_concept_id'] == 1 else 'M'
    low, high, _ = REFERENCE_INTERVALS[row['test_name']][sex]
    return low < row['numeric_value'] < high

def get_ref_stats(test_name, sex):
    """Get reference mean and variance."""
    sex_key = 'F' if sex == 1 else 'M'
    low, high, _ = REFERENCE_INTERVALS[test_name][sex_key]
    return (low + high) / 2, ((high - low) / 4) ** 2

def get_stratify_labels(sequences):
    stratify_labels = []
    for seq in sequences:
        cid = seq["cid"] 
        s_next = seq["s"][-1]
        stratify_labels.append(f"{cid}_{s_next}")
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
    df['condition'] = df.apply(is_normal, axis=1)
    sequences = []
    
    for (pid, test_name), group in df.groupby(['subject_id', 'test_name']):
        if len(group) < min_context + 1:
            continue
            
        group = group.sort_values('time')
        times = pd.to_datetime(group['time'], errors='coerce')
        t = ((times - times.iloc[0]).dt.total_seconds() / 86400).astype(np.float32).values
        x = group['numeric_value'].astype(np.float32).values
        s = group['condition'].astype(np.int32).values
        
        sex = group['gender_concept_id'].iloc[0]
        cid = TEST_VOCAB[test_name]
        ref_mu, ref_var = get_ref_stats(test_name, sex)
        
        sequences.append({
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