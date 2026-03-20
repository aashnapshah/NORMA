#!/usr/bin/env python
"""
Step 11: Age-dependent reference interval analysis.

Samples observations across age strata and computes mean PerRI and NORMA-RI
bounds by age for each analyte. Outputs CSV for plotting age-dependent trends.

Output: results/{dataset}/raw/age_ri.csv
  Columns: analyte, age, method, ri_low, ri_high, ri_mid, n

Usage:
    python 11_age_ri.py --dataset eicu
    python 11_age_ri.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS
from config import NORMA_RUN_ID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}
NORMA_LOW = f'norma_{NORMA_RUN_ID}_low'
NORMA_HIGH = f'norma_{NORMA_RUN_ID}_high'


def compute_age_ri(ref_df, n_sample=2000, seed=42):
    """Compute mean RI bounds by age for PerRI (GMM) and NORMA.

    Samples n_sample observations per analyte across age strata.
    Returns DataFrame with: analyte, age, method, ri_low, ri_high, ri_mid, n
    """
    ref_df['analyte'] = ref_df['analyte'].replace('', 'NA').fillna('NA')
    ref_df['age'] = pd.to_numeric(ref_df['age'], errors='coerce')
    ref_df = ref_df.dropna(subset=['age'])

    analytes = sorted(a for a in ref_df['analyte'].unique() if a not in EXCLUDE_ANALYTES)
    rng = np.random.RandomState(seed)

    all_rows = []
    for analyte in analytes:
        lab = ref_df[ref_df['analyte'] == analyte].copy()
        if len(lab) < 50:
            continue

        # Sample up to n_sample
        if len(lab) > n_sample:
            lab = lab.sample(n=n_sample, random_state=rng)

        # Bin ages into integer years
        lab['age_bin'] = lab['age'].astype(int)

        for age_bin, grp in lab.groupby('age_bin'):
            if len(grp) < 5:
                continue

            # PerRI (GMM)
            if 'gmm_low' in grp.columns and 'gmm_high' in grp.columns:
                gmm_low = grp['gmm_low'].mean()
                gmm_high = grp['gmm_high'].mean()
                all_rows.append({
                    'analyte': analyte, 'age': age_bin, 'method': 'PerRI',
                    'ri_low': gmm_low, 'ri_high': gmm_high,
                    'ri_mid': (gmm_low + gmm_high) / 2, 'n': len(grp),
                })

            # NORMA
            if NORMA_LOW in grp.columns and NORMA_HIGH in grp.columns:
                norma_low = grp[NORMA_LOW].dropna().mean()
                norma_high = grp[NORMA_HIGH].dropna().mean()
                if not (np.isnan(norma_low) or np.isnan(norma_high)):
                    all_rows.append({
                        'analyte': analyte, 'age': age_bin, 'method': 'NORMA',
                        'ri_low': norma_low, 'ri_high': norma_high,
                        'ri_mid': (norma_low + norma_high) / 2, 'n': len(grp),
                    })

            # PopRI (constant, for reference)
            from process.config import REFERENCE_INTERVALS
            if analyte in REFERENCE_INTERVALS:
                ri = REFERENCE_INTERVALS[analyte]
                # Use average of M/F
                m = ri.get('M', (None, None))
                f = ri.get('F', (None, None))
                pop_low = (m[0] + f[0]) / 2 if m[0] and f[0] else m[0] or f[0]
                pop_high = (m[1] + f[1]) / 2 if m[1] and f[1] else m[1] or f[1]
                if pop_low and pop_high:
                    all_rows.append({
                        'analyte': analyte, 'age': age_bin, 'method': 'PopRI',
                        'ri_low': pop_low, 'ri_high': pop_high,
                        'ri_mid': (pop_low + pop_high) / 2, 'n': len(grp),
                    })

        print(f"    {analyte}: {len(lab)} samples")

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--n_sample', type=int, default=2000)
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    # Load ref intervals
    val_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(val_dir, 'cache', ds.name, 'ref_intervals.csv')
    print(f"  Loading {ref_path}")
    ref_df = pd.read_csv(ref_path, keep_default_na=False, low_memory=False)
    # Convert numeric columns from strings
    for col in ['age', 'gmm_low', 'gmm_high', NORMA_LOW, NORMA_HIGH, 'baseline_mean', 'baseline_std']:
        if col in ref_df.columns:
            ref_df[col] = pd.to_numeric(ref_df[col], errors='coerce')

    print(f"  {len(ref_df)} patient-analyte pairs")

    age_df = compute_age_ri(ref_df, n_sample=args.n_sample)

    out_path = os.path.join(out_dir, 'age_ri.csv')
    age_df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(age_df)} rows to {out_path}")


if __name__ == '__main__':
    main()
