#!/usr/bin/env python
"""
Step 5: Classify index measurements against reference intervals.

For each index measurement, classifies it as low (0), normal (1), or high (2)
under PopRI, PerRI (GMM), and NORMA reference intervals.

Reads:
  - split_df.pkl (from 02_split_data.py)
  - ref_intervals.csv (from 03_compute_refs.py)

Output: results/{dataset}/raw/classification.csv
  Each row = one index measurement with ref range bounds and classifications.

Usage:
    python 06_classify.py --dataset eicu
    python 06_classify.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import DATASETS
from config import NORMA_RUN_ID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

NORMA_RUN = f'norma_{NORMA_RUN_ID}'


def classify_value(value, low, high):
    """Classify a value against reference interval bounds."""
    try:
        value = float(value)
        low = float(low)
        high = float(high)
    except (ValueError, TypeError):
        return np.nan
    if np.isnan(value) or np.isnan(low) or np.isnan(high):
        return np.nan
    if value < low:
        return 0  # low
    elif value > high:
        return 2  # high
    return 1  # normal


def load_split(dataset_name):
    """Load split_df.pkl for a dataset."""
    val_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(val_dir, 'data', 'split_df.pkl')
    print(f"  Loading {path}")
    return pd.read_pickle(path)


def load_ref_intervals(dataset_name):
    """Load ref_intervals.csv for a dataset."""
    val_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(val_dir, 'cache', dataset_name, 'ref_intervals.csv')
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    return df


def build_classification(split_df, ref_df):
    """Classify index measurements against all available reference intervals.

    Returns DataFrame with classification columns (pop_class, gmm_class, norma_*_class).
    """
    # Fix sodium
    split_df['analyte'] = split_df['analyte'].replace('', 'NA').fillna('NA')

    # Get index measurements
    index_df = split_df[split_df['split'] == 'index'].copy()
    print(f"  {len(index_df)} index measurements")

    # Merge
    result = index_df.merge(
        ref_df, on=['patient_id', 'analyte'],
        suffixes=('', '_ref')
    )
    print(f"  {len(result)} matched to ref intervals")

    # Drop _ref suffix columns from merge
    drop_cols = [c for c in result.columns if c.endswith('_ref')]
    result = result.drop(columns=drop_cols, errors='ignore')

    # Classify: PopRI
    if 'pop_low' in result.columns and 'pop_high' in result.columns:
        result['PopRI_class'] = result.apply(
            lambda r: classify_value(r['value'], r['pop_low'], r['pop_high']), axis=1
        )
        valid = result['PopRI_class'].dropna()
        print(f"  PopRI: {(valid != 1).mean()*100:.1f}% abnormal ({len(valid)} measurements)")

    # Classify: PerRI (GMM)
    if 'gmm_low' in result.columns and 'gmm_high' in result.columns:
        result['PerRI_class'] = result.apply(
            lambda r: classify_value(r['value'], r['gmm_low'], r['gmm_high']), axis=1
        )
        valid = result['PerRI_class'].dropna()
        print(f"  PerRI: {(valid != 1).mean()*100:.1f}% abnormal ({len(valid)} measurements)")

    # Classify: NORMA (use NORMA_RUN_ID from config)
    norma_low = f'norma_{NORMA_RUN_ID}_low'
    norma_high = f'norma_{NORMA_RUN_ID}_high'
    if norma_low in result.columns and norma_high in result.columns:
        result['NORMA_class'] = result.apply(
            lambda r: classify_value(r['value'], r[norma_low], r[norma_high]), axis=1
        )
        valid = result['NORMA_class'].dropna()
        print(f"  NORMA: {(valid != 1).mean()*100:.1f}% abnormal ({len(valid)} measurements)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    split_df = load_split(ds.name)
    ref_df = load_ref_intervals(ds.name)

    result = build_classification(split_df, ref_df)

    # Save CSV
    csv_path = os.path.join(out_dir, 'classification.csv')
    result.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(result)} rows to {csv_path}")
    print(f"  {result['patient_id'].nunique()} patients, {result['analyte'].nunique()} analytes")


if __name__ == '__main__':
    main()
