#!/usr/bin/env python
"""
Step 10: Lead time analysis.

For each analyte, computes how much earlier NORMA flags an abnormality
compared to PopRI. Among measurements that NORMA flags as abnormal but
PopRI calls normal, the lead time is the time until PopRI also flags abnormal
(or until the end of follow-up).

Output: results/{dataset}/raw/lead_time.csv
  Columns: analyte, n_earlier, median_lead_hours, iqr_lead_hours_25, iqr_lead_hours_75,
           mean_lead_hours

Usage:
    python 10_lead_time.py --dataset eicu
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


def load_classification(dataset_name):
    """Load classification.csv with numeric class columns."""
    path = os.path.join(ROOTDIR, 'results', dataset_name, 'raw', 'classification.csv')
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for col in df.columns:
        if col.endswith('_class'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_lead_time(df, time_col):
    """Compute lead time: how much earlier NORMA flags vs PopRI.

    For each (patient, analyte), finds measurements where NORMA flags abnormal
    but PopRI says normal. Then finds the next PopRI-abnormal measurement
    for that patient-analyte pair. Lead time = time difference.

    Args:
        df: classification DataFrame with PopRI_class, NORMA_class, time_col
        time_col: column with measurement timestamps (e.g., labresultoffset in minutes)

    Returns:
        DataFrame with per-analyte lead time statistics.
    """
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    analytes = sorted(a for a in df['analyte'].unique() if a not in EXCLUDE_ANALYTES)

    rows = []
    for analyte in analytes:
        lab = df[df['analyte'] == analyte].sort_values(['patient_id', time_col])

        # Measurements where NORMA flags but PopRI doesn't
        norma_only = lab[(lab['NORMA_class'] != 1) & (lab['PopRI_class'] == 1)]
        if len(norma_only) == 0:
            rows.append({
                'analyte': analyte,
                'n_norma_only': 0,
                'n_with_later_pop_flag': 0,
                'median_lead_hours': np.nan,
                'iqr_lead_hours_25': np.nan,
                'iqr_lead_hours_75': np.nan,
                'mean_lead_hours': np.nan,
            })
            continue

        # For each NORMA-only flag, find the next PopRI-abnormal measurement
        pop_abnormal = lab[lab['PopRI_class'] != 1]
        lead_times = []

        for pid in norma_only['patient_id'].unique():
            pid_norma = norma_only[norma_only['patient_id'] == pid]
            pid_pop_abn = pop_abnormal[pop_abnormal['patient_id'] == pid]

            if len(pid_pop_abn) == 0:
                continue

            for _, norma_row in pid_norma.iterrows():
                norma_time = norma_row[time_col]
                # Find next PopRI-abnormal measurement after this one
                later_pop = pid_pop_abn[pid_pop_abn[time_col] > norma_time]
                if len(later_pop) > 0:
                    next_pop_time = later_pop[time_col].iloc[0]
                    lead = next_pop_time - norma_time  # in original units (minutes for eICU)
                    lead_times.append(lead)

        lead_times = np.array(lead_times)

        # Convert to hours (eICU timestamps are in minutes)
        lead_hours = lead_times / 60 if len(lead_times) > 0 else np.array([])

        rows.append({
            'analyte': analyte,
            'n_norma_only': len(norma_only),
            'n_with_later_pop_flag': len(lead_hours),
            'median_lead_hours': np.median(lead_hours) if len(lead_hours) > 0 else np.nan,
            'iqr_lead_hours_25': np.percentile(lead_hours, 25) if len(lead_hours) > 0 else np.nan,
            'iqr_lead_hours_75': np.percentile(lead_hours, 75) if len(lead_hours) > 0 else np.nan,
            'mean_lead_hours': np.mean(lead_hours) if len(lead_hours) > 0 else np.nan,
        })

        n_lead = len(lead_hours)
        med = np.median(lead_hours) if n_lead > 0 else 0
        print(f"    {analyte}: {len(norma_only)} NORMA-only flags, {n_lead} with later PopRI flag, "
              f"median lead = {med:.1f}h")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    df = load_classification(ds.name)

    # Detect time column
    time_col = 'timestamp' if 'timestamp' in df.columns else 'labresultoffset'
    print(f"  Using time column: {time_col}")
    print(f"  {len(df)} measurements, {df['patient_id'].nunique()} patients")

    lead_df = compute_lead_time(df, time_col)

    # Add overall summary row
    valid = lead_df.dropna(subset=['median_lead_hours'])
    if len(valid) > 0:
        overall_median = valid['median_lead_hours'].median()
        overall_q25 = valid['iqr_lead_hours_25'].median()
        overall_q75 = valid['iqr_lead_hours_75'].median()
        lead_df = pd.concat([lead_df, pd.DataFrame([{
            'analyte': 'Overall',
            'n_norma_only': lead_df['n_norma_only'].sum(),
            'n_with_later_pop_flag': lead_df['n_with_later_pop_flag'].sum(),
            'median_lead_hours': overall_median,
            'iqr_lead_hours_25': overall_q25,
            'iqr_lead_hours_75': overall_q75,
            'mean_lead_hours': valid['mean_lead_hours'].mean(),
        }])], ignore_index=True)

    out_path = os.path.join(out_dir, 'lead_time.csv')
    lead_df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(lead_df)} rows to {out_path}")

    # Print summary
    overall = lead_df[lead_df['analyte'] == 'Overall']
    if len(overall) > 0:
        o = overall.iloc[0]
        print(f"\n  Overall: median lead time = {o['median_lead_hours']:.1f}h "
              f"(IQR: {o['iqr_lead_hours_25']:.1f}–{o['iqr_lead_hours_75']:.1f}h)")


if __name__ == '__main__':
    main()
