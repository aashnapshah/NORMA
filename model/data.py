import numpy as np
import pandas as pd
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import *

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Get complete sequences
        x = torch.from_numpy(seq["x"]).float().unsqueeze(-1)
        t = torch.from_numpy(seq["t"]).float().unsqueeze(-1)
        s = torch.from_numpy(seq["s"]).long()
        
        # Split into historical and query parts
        x_h = x[:-1]  # All but last measurement
        t_h = t[:-1]  # All but last timestamp
        s_h = s[:-1]  # All but last condition state
        
        # Query/next step data
        t_next = t[-1]  # Last timestamp (keep as tensor)
        s_next = torch.tensor([s[-1]], dtype=torch.long)   # Last condition state
        x_next = x[-1]   # Last measurement value (for target)
        
        seq['sex'] = 1 if seq['sex'] == 'F' or seq['sex'] == 1 else 0
        sex = torch.tensor([seq["sex"]], dtype=torch.long)
        cid = torch.tensor([seq["cid"]], dtype=torch.long)
    
        # Reference statistics
        #ref_mu = torch.tensor(seq["ref_mu"][0], dtype=torch.float32)
        #ref_var = torch.tensor(seq["ref_var"][0], dtype=torch.float32)
        subject_id = seq["subject_id"]
        
        # can you check if there are nan values in the sequence
        if torch.isnan(x_h).any() or torch.isnan(s_h).any() or torch.isnan(t_h).any() or torch.isnan(x_next).any():
            print('nan values in the sequence')
            print(seq)
            raise ValueError('nan values in the sequence')
        
        return x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, subject_id

def collate_fn(batch):
    """Collate function for DataLoader."""
    x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, subject_id = zip(*batch)

    # Pad sequences and stack tensors
    x_h = pad_sequence(x_h, batch_first=True)
    t_h = pad_sequence(t_h, batch_first=True)
    s_h = pad_sequence(s_h, batch_first=True)

    sex = torch.stack(sex)
    cid = torch.stack(cid)
    s_next = torch.stack(s_next)
    t_next = torch.stack(t_next)
    x_next = torch.stack(x_next)

    # Create padding mask
    lengths = [seq.shape[0] for seq in x_h]
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
        'subject_ids': list(subject_id)
    }

def load_and_split_data(sequences_path, random_state=42):
    print("Loading sequences...")
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} sequences")
    print(sequences[0:2])
    #print if there are any nan values in the sequences
    
    print("Splitting data...")
    stratify_labels = get_stratify_labels(sequences)
    from collections import Counter
    label_counts = Counter(stratify_labels)
    print(f"Number of unique stratify labels: {len(label_counts)}")
    # print("Stratify label value counts (top 20):")
    # for label, count in sorted(label_counts.items(), key=lambda x: x[0]):  # Sort by label
    #     print(f"{label}: {count}")

    train_val_seq, test_seq = train_test_split(
        sequences, test_size=0.2, stratify=stratify_labels, random_state=random_state
    )

    stratify_labels = get_stratify_labels(train_val_seq)
    train_seq, val_seq = train_test_split(
        train_val_seq, test_size=0.125, stratify=stratify_labels, random_state=random_state
    )
    
    return train_seq, val_seq, test_seq


def create_dataloaders(train_seq, val_seq, test_seq, batch_size=16, random_state=42):
    """Create train/val/test dataloaders."""

    print(f"Split sizes - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
    print('Using weighted sampler for training')

    train_loader = DataLoader(
        TimeSeriesDataset(train_seq),
        batch_size=batch_size,
        sampler=create_weighted_sampler(train_seq),
        collate_fn=collate_fn,
        num_workers=4,
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sequences_path = '../data/processed/sequences.pkl'
    train_seq, val_seq, test_seq = load_and_split_data(sequences_path) #, num_patients=100, min_points=3)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_seq, val_seq, test_seq, batch_size=16, random_state=42
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        print("Shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {len(value)} items")
        print(f"Device: {batch['x_h'].device}")
        break

if __name__ == "__main__":
    main()
