#!/usr/bin/env python
"""
Step 6: Compute mortality rate by analyte value quintile.

Groups index measurements into quintiles of raw analyte values and computes
mortality rate (with Wilson score 95% CI) within each quintile.

Output: results/{dataset}/raw/mortality_by_quintile.csv
  Columns: analyte, q_label, mortality_pct, ci_lo, ci_hi, n

Usage:
    python 06_mortality.py --dataset eicu
    python 06_mortality.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}


def compute_quintile_mortality(df, analyte_col, value_col, event_col):
    """Compute mortality rate by quintile per analyte.

    Returns DataFrame with: analyte, q_label, mortality_pct, ci_lo, ci_hi, n
    """
    df[analyte_col] = df[analyte_col].replace('', 'NA').fillna('NA')
    analytes = sorted(a for a in df[analyte_col].unique() if a not in EXCLUDE_ANALYTES)

    all_rows = []
    for analyte in analytes:
        lab_df = df[df[analyte_col] == analyte].dropna(subset=[value_col, event_col]).copy()
        if len(lab_df) < 20:
            continue

        lab_df['quintile'] = pd.qcut(lab_df[value_col], 5, labels=False, duplicates='drop')
        mort = lab_df.groupby('quintile').agg(
            mortality_rate=(event_col, 'mean'),
            n=(event_col, 'count'),
        ).reset_index()
        mort['q_label'] = mort['quintile'] + 1

        # Wilson score 95% CI
        z = 1.96
        p = mort['mortality_rate']
        n_q = mort['n']
        denom = 1 + z**2 / n_q
        center = (p + z**2 / (2 * n_q)) / denom
        halfwidth = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_q)) / n_q) / denom
        mort['ci_lo'] = (center - halfwidth) * 100
        mort['ci_hi'] = (center + halfwidth) * 100
        mort['mortality_pct'] = mort['mortality_rate'] * 100
        mort['analyte'] = analyte

        all_rows.append(mort[['analyte', 'q_label', 'mortality_pct', 'ci_lo', 'ci_hi', 'n']])

    if not all_rows:
        return pd.DataFrame(columns=['analyte', 'q_label', 'mortality_pct', 'ci_lo', 'ci_hi', 'n'])
    return pd.concat(all_rows, ignore_index=True)


def compute_deviation_mortality(df, event_col, n_bins=10):
    """Compute mortality by standardized deviation from personal baseline.

    z = |value - baseline_mean| / baseline_std for each index measurement.
    Groups into deciles of z and computes mortality rate.

    Returns DataFrame with: analyte, decile, z_median, mortality_pct, ci_lo, ci_hi, n
    """
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    analytes = sorted(a for a in df['analyte'].unique() if a not in EXCLUDE_ANALYTES)

    all_rows = []
    for analyte in analytes:
        lab_df = df[df['analyte'] == analyte].dropna(
            subset=['value', event_col, 'baseline_mean', 'baseline_std']
        ).copy()
        lab_df = lab_df[lab_df['baseline_std'] > 0]
        if len(lab_df) < 20:
            continue

        lab_df['z_score'] = (lab_df['value'] - lab_df['baseline_mean']).abs() / lab_df['baseline_std']

        lab_df['decile'] = pd.qcut(lab_df['z_score'], n_bins, labels=False, duplicates='drop')
        mort = lab_df.groupby('decile').agg(
            mortality_rate=(event_col, 'mean'),
            n=(event_col, 'count'),
            z_median=('z_score', 'median'),
        ).reset_index()
        mort['decile'] = mort['decile'] + 1

        # Wilson score 95% CI
        z = 1.96
        p = mort['mortality_rate']
        n_q = mort['n']
        denom = 1 + z**2 / n_q
        center = (p + z**2 / (2 * n_q)) / denom
        halfwidth = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_q)) / n_q) / denom
        mort['ci_lo'] = (center - halfwidth) * 100
        mort['ci_hi'] = (center + halfwidth) * 100
        mort['mortality_pct'] = mort['mortality_rate'] * 100
        mort['analyte'] = analyte

        all_rows.append(mort[['analyte', 'decile', 'z_median', 'mortality_pct', 'ci_lo', 'ci_hi', 'n']])

    if not all_rows:
        return pd.DataFrame(columns=['analyte', 'decile', 'z_median', 'mortality_pct', 'ci_lo', 'ci_hi', 'n'])
    return pd.concat(all_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)
    val_dir = os.path.dirname(os.path.abspath(__file__))

    # Load split data
    split_path = os.path.join(val_dir, 'data', 'split_df.pkl')
    print(f"  Loading {split_path}")
    split_df = pd.read_pickle(split_path)

    # Use index measurements only
    split_df = split_df[split_df['split'] == 'index'].copy()
    split_df['analyte'] = split_df['analyte'].replace('', 'NA').fillna('NA')

    # Get mortality outcome — only attach mortality, not all diseases
    mort_key = ds.mortality_outcome
    event_col = ds.outcomes[mort_key]['event_col']

    if event_col not in split_df.columns:
        print(f"  Attaching mortality...")
        from utils import get_mortality
        mort = get_mortality()
        split_df = split_df.merge(mort, on='patientunitstayid', how='left')

    print(f"  {len(split_df)} index measurements, event_col={event_col}")
    print(f"  Event rate: {split_df[event_col].mean():.3f}")

    # 1. Quintile mortality (raw values)
    print("\n  Computing quintile mortality...")
    mort_df = compute_quintile_mortality(split_df, 'analyte', 'value', event_col)
    out_path = os.path.join(out_dir, 'mortality_by_quintile.csv')
    mort_df.to_csv(out_path, index=False)
    print(f"  Saved {len(mort_df)} rows to {out_path}")

    # 2. Deviation mortality (standardized distance from baseline)
    # Need baseline_mean and baseline_std from ref_intervals
    ref_path = os.path.join(val_dir, 'cache', ds.name, 'ref_intervals.csv')
    print(f"\n  Loading {ref_path}")
    ref_df = pd.read_csv(ref_path, keep_default_na=False, low_memory=False)
    ref_df['analyte'] = ref_df['analyte'].replace('', 'NA').fillna('NA')

    # Merge baseline stats onto index measurements
    ref_cols = ['patient_id', 'analyte', 'baseline_mean', 'baseline_std']
    merged = split_df.merge(ref_df[ref_cols], on=['patient_id', 'analyte'], how='inner')
    print(f"  {len(merged)} measurements with baseline stats")

    print("\n  Computing deviation mortality...")
    dev_df = compute_deviation_mortality(merged, event_col)
    dev_path = os.path.join(out_dir, 'mortality_by_deviation.csv')
    dev_df.to_csv(dev_path, index=False)
    print(f"  Saved {len(dev_df)} rows to {dev_path}")


if __name__ == '__main__':
    main()
