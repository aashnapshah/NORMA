#!/usr/bin/env python
"""
Step 6: Compute abnormality prevalence and reclassification per analyte.

Reads classification.csv (from 05_classify.py) and computes:
  - Overall abnormality prevalence per method (PopRI, PerRI, NORMA)
  - Reclassification: among PopRI-normal measurements, % flagged by PerRI/NORMA

Outputs one CSV per dataset: prevalence.csv
  Columns: analyte, n, PopRI_pct, PerRI_pct, NORMA_pct,
           n_popri_normal, PerRI_reclass_pct, NORMA_reclass_pct

Usage:
    python 05_prevalence.py --dataset eicu
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

METHODS = ['PopRI', 'PerRI', 'NORMA']


def load_classification(dataset_name):
    """Load classification data."""
    path = os.path.join(ROOTDIR, 'results', dataset_name, 'raw', 'classification.csv')
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for col in df.columns:
        if col.endswith('_class'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def abnormal_pct(series):
    """Fraction classified as abnormal (not 1) in percent."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid != 1).mean() * 100


def compute_prevalence(df):
    """Compute combined prevalence + reclassification table."""
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')

    analytes = sorted(a for a in df['analyte'].unique() if a not in EXCLUDE_ANALYTES)

    rows = []
    for analyte in analytes:
        lab = df[df['analyte'] == analyte]
        n = len(lab)

        row = {
            'analyte': analyte,
            'n': n,
            'PopRI_pct': abnormal_pct(lab['PopRI_class']),
            'PerRI_pct': abnormal_pct(lab['PerRI_class']),
            'NORMA_pct': abnormal_pct(lab['NORMA_class']),
        }

        # Reclassification: among PopRI-normal
        pop_normal = lab[lab['PopRI_class'] == 1]
        row['n_popri_normal'] = len(pop_normal)
        row['PerRI_reclass_pct'] = abnormal_pct(pop_normal['PerRI_class'])
        row['NORMA_reclass_pct'] = abnormal_pct(pop_normal['NORMA_class'])

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    classified = load_classification(ds.name)
    print(f"  {len(classified)} classified measurements")

    df = compute_prevalence(classified)

    # Add overall summary row
    num_cols = ['n', 'PopRI_pct', 'PerRI_pct', 'NORMA_pct',
                'n_popri_normal', 'PerRI_reclass_pct', 'NORMA_reclass_pct']
    overall = {'analyte': 'Overall'}
    for col in num_cols:
        if col in df.columns:
            if col.startswith('n'):
                overall[col] = df[col].sum()
            else:
                overall[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    out_path = os.path.join(out_dir, 'prevalence.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")

    # Print summary
    o = df[df['analyte'] == 'Overall'].iloc[0]
    print(f"\n  Overall (mean across analytes):")
    print(f"    PopRI: {o['PopRI_pct']:.1f}%  PerRI: {o['PerRI_pct']:.1f}%  NORMA: {o['NORMA_pct']:.1f}%")
    print(f"    RR (PopRI-normal): PerRI {o['PerRI_reclass_pct']:.1f}%  NORMA {o['NORMA_reclass_pct']:.1f}%")


if __name__ == '__main__':
    main()
