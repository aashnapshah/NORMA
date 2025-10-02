import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
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
        
        # Demographics and lab info
        sex = torch.tensor([seq["sex"]], dtype=torch.long)
        cid = torch.tensor([seq["cid"]], dtype=torch.long)
    
        # Reference statistics
        ref_mu = torch.tensor(seq["ref_mu"][0], dtype=torch.float32)
        ref_var = torch.tensor(seq["ref_var"][0], dtype=torch.float32)
        subject_id = seq["subject_id"]
        
        return x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, ref_mu, ref_var, subject_id

def collate_fn(batch):
    """Collate function for DataLoader."""
    x_h, s_h, t_h, sex, cid, s_next, t_next, x_next, ref_mu, ref_var, subject_id = zip(*batch)

    # Pad sequences and stack tensors
    x_h = pad_sequence(x_h, batch_first=True)
    t_h = pad_sequence(t_h, batch_first=True)
    s_h = pad_sequence(s_h, batch_first=True)

    sex = torch.stack(sex)
    cid = torch.stack(cid)
    s_next = torch.stack(s_next)
    t_next = torch.stack(t_next)
    x_next = torch.stack(x_next)
    ref_mu = torch.stack(ref_mu)
    ref_var = torch.stack(ref_var)

    # Create padding mask
    lengths = [seq.shape[0] for seq in x_h]
    max_len = x_h.shape[1]
    pad_mask = torch.ones(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        pad_mask[i, :l] = False

    # Return as dictionary for cleaner unpacking
    return {
        'x_h': x_h,
        't_h': t_h, 
        's_h': s_h,
        'sex': sex,
        'cid': cid,
        's_next': s_next,
        't_next': t_next,
        'x_next': x_next,
        'ref_mu': ref_mu,
        'ref_var': ref_var,
        'pad_mask': pad_mask,
        'subject_ids': list(subject_id)
    }

def create_dataloaders(df, batch_size=16, random_state=42):
    """Create train/val/test dataloaders with stratified splitting."""
    sequences = create_sequences(df)
    
    stratify_labels = get_stratify_labels(sequences)
    train_val_seq, test_seq = train_test_split(
        sequences, test_size=0.2, stratify=stratify_labels, random_state=random_state
    )
    
    stratify_labels = get_stratify_labels(train_val_seq)
    train_seq, val_seq = train_test_split(
        train_val_seq, test_size=0.125, stratify=stratify_labels, random_state=random_state
    )
    
    print(f"Split sizes - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

    # Create DataLoaders
    train_loader = DataLoader(
        TimeSeriesDataset(train_seq),
        batch_size=batch_size,
        shuffle=True,
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

    df = load_data("../data/processed/lab_measurements.csv", min_points=3)
    train_loader, val_loader, test_loader = create_dataloaders(df, batch_size=16, random_state=42)

    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        print("Shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {len(value)} items")
        print(f"Device: {batch['x'].device}")
        break

if __name__ == "__main__":
    main()
