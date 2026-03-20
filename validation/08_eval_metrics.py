#!/usr/bin/env python
"""
Step 8: Compute classification evaluation metrics per analyte and outcome.

For each (outcome, analyte, method), computes PPV, sensitivity, specificity, F1
using the classification CSV from step 06.

Two analyses:
  1. Overall: all classified measurements
  2. PopRI-restricted: only measurements PopRI called normal (isolates added value of PerRI/NORMA)

Output: results/{dataset}/raw/eval_metrics.csv
        results/{dataset}/raw/eval_metrics_popri_normal.csv

Usage:
    python 08_eval_metrics.py --dataset eicu
    python 08_eval_metrics.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS
from config import PRIMARY_OUTCOMES

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

METHODS = ['PopRI', 'PerRI', 'NORMA']


def compute_metrics(df, outcome_col, methods):
    """Compute PPV, sensitivity, specificity, F1 per (analyte, method).

    Pre-aggregates to patient level once per analyte, then slices by method.
    """
    analytes = sorted(a for a in df['analyte'].unique() if a not in EXCLUDE_ANALYTES)
    cls_cols = [f'{m}_class' for m in methods if f'{m}_class' in df.columns]
    rows = []

    for analyte in analytes:
        lab_df = df[df['analyte'] == analyte]
        if len(lab_df) == 0:
            continue

        # Pre-compute abnormal flags for all methods at once (vectorized)
        agg_dict = {outcome_col: 'first'}
        for cls_col in cls_cols:
            # For each method: patient is abnormal if any measurement != 1
            lab_df[f'_abn_{cls_col}'] = (lab_df[cls_col] != 1).astype(int)
            agg_dict[f'_abn_{cls_col}'] = 'max'

        patient_df = lab_df.groupby('patient_id').agg(agg_dict).dropna(subset=[outcome_col])
        if len(patient_df) < 5:
            continue
        patient_df[outcome_col] = patient_df[outcome_col].astype(int)

        for method in methods:
            cls_col = f'{method}_class'
            abn_col = f'_abn_{cls_col}'
            if abn_col not in patient_df.columns:
                continue

            abn = patient_df[abn_col].astype(int)
            evt = patient_df[outcome_col]

            tp = int(((abn == 1) & (evt == 1)).sum())
            fp = int(((abn == 1) & (evt == 0)).sum())
            tn = int(((abn == 0) & (evt == 0)).sum())
            fn = int(((abn == 0) & (evt == 1)).sum())

            n_flagged = tp + fp
            n_events = tp + fn
            n_no_events = tn + fp
            n_not_flagged = tn + fn

            n_total = tp + fp + tn + fn
            ppv = tp / n_flagged if n_flagged > 0 else np.nan
            npv = tn / n_not_flagged if n_not_flagged > 0 else np.nan
            sens = tp / n_events if n_events > 0 else np.nan
            spec = tn / n_no_events if n_no_events > 0 else np.nan
            acc = (tp + tn) / n_total if n_total > 0 else np.nan
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and ppv + sens > 0) else np.nan

            rows.append({
                'analyte': analyte,
                'method': method,
                'n': len(patient_df),
                'n_events': int(n_events),
                'n_flagged': int(n_flagged),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'ppv': ppv, 'npv': npv,
                'sensitivity': sens, 'specificity': spec,
                'accuracy': acc, 'f1': f1,
            })

    return rows


def attach_needed_outcomes(df, ds, needed):
    """Attach only the specific outcome columns needed, avoiding full attach_outcomes."""
    from utils import get_mortality, load_diagnosis_processed

    df = df.copy()

    # Mortality — just merge patient table
    if 'mortality' in needed and 'patientunitstayid' in df.columns:
        mort = get_mortality()
        cols_to_add = [c for c in mort.columns if c not in df.columns or c == 'patientunitstayid']
        df = df.merge(mort[cols_to_add], on='patientunitstayid', how='left')
        print(f"    Attached mortality")

    # Prolonged LOS
    if 'prolonged_los' in needed and 'unitdischargeoffset' in df.columns:
        df['prolonged_los'] = df['unitdischargeoffset'] > (7 * 24 * 60)
        print(f"    Attached prolonged_los")

    # Disease-based outcomes (AKI, sepsis, etc.) — only load needed diseases
    disease_outcomes = {o: col for o, col in needed.items()
                        if o not in ('mortality', 'prolonged_los', 'pop_abnormal')}
    if disease_outcomes and 'patientunitstayid' in df.columns:
        from config import DISEASES
        diagnosis = load_diagnosis_processed()
        for outcome_key, event_col in disease_outcomes.items():
            disease_name = event_col.replace('has_', '')
            if disease_name not in DISEASES:
                continue
            cfg = DISEASES[disease_name]
            dx_mask = diagnosis['diagnosisstring'].str.contains(cfg['dx_pattern'], case=False, na=False)
            icd_mask = diagnosis['icd9code'].str.contains(cfg['icd_pattern'], na=False)
            earliest = diagnosis.loc[dx_mask | icd_mask].groupby('patientunitstayid')['diagnosisoffset'].min()
            df[event_col] = df['patientunitstayid'].isin(earliest.index)
            df[f'{disease_name}_offset'] = df['patientunitstayid'].map(earliest)
            print(f"    Attached {outcome_key} ({len(earliest)} stays)")

    return df


def load_classification(dataset_name):
    """Load classification.csv with proper numeric types for class columns."""
    path = os.path.join(ROOTDIR, 'results', dataset_name, 'raw', 'classification.csv')
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for col in df.columns:
        if col.endswith('_class'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    df = load_classification(ds.name)

    # Fix sodium
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')

    # Attach only the outcome columns we need
    needed = {}
    for o in PRIMARY_OUTCOMES:
        if o not in ds.outcomes:
            continue
        event_col = ds.outcomes[o]['event_col']
        if event_col not in df.columns:
            needed[o] = event_col

    if needed:
        print(f"  Attaching outcomes: {list(needed.keys())}...")
        df = attach_needed_outcomes(df, ds, needed)
    else:
        print(f"  All outcome columns already present")

    methods = METHODS
    print(f"  Methods: {methods}")

    # Run for each outcome, collect all rows
    all_rows = []
    restr_rows = []

    for outcome_key in PRIMARY_OUTCOMES:
        if outcome_key not in ds.outcomes:
            continue
        event_col = ds.outcomes[outcome_key]['event_col']
        if event_col not in df.columns:
            print(f"  Skipping {outcome_key} ({event_col} not found)")
            continue

        event_rate = df.drop_duplicates('patient_id')[event_col].mean()
        print(f"\n  Outcome: {outcome_key} (event rate: {event_rate:.3f})")

        # Overall metrics — all patients
        rows = compute_metrics(df, event_col, methods)
        for row in rows:
            row['outcome'] = outcome_key
            row['subset'] = 'all'
        all_rows.extend(rows)

        # Normal-setpoint patients: GMM mean within PopRI
        if 'gmm_mean' in df.columns and 'pop_low' in df.columns:
            normal_setpoint = df[
                (df['gmm_mean'] >= df['pop_low']) &
                (df['gmm_mean'] <= df['pop_high'])
            ]
            n_patients = normal_setpoint['patient_id'].nunique()
            print(f"    Normal-setpoint patients: {n_patients}")

            rows = compute_metrics(normal_setpoint, event_col, methods)
            for row in rows:
                row['outcome'] = outcome_key
                row['subset'] = 'normal_setpoint'
            all_rows.extend(rows)

        # PopRI-normal measurements (all patients)
        restricted_methods = [m for m in methods if m != 'PopRI']
        if 'PopRI_class' in df.columns:
            pop_normal_meas = df[df['PopRI_class'] == 1]
            rows = compute_metrics(pop_normal_meas, event_col, restricted_methods)
            for row in rows:
                row['outcome'] = outcome_key
                row['subset'] = 'popri_normal'
            restr_rows.extend(rows)

        # PopRI-normal measurements among normal-setpoint patients
        if 'gmm_mean' in df.columns and 'PopRI_class' in df.columns:
            ns_pop_normal = normal_setpoint[normal_setpoint['PopRI_class'] == 1]
            print(f"    Normal-setpoint + PopRI-normal: {len(ns_pop_normal)} measurements")
            rows = compute_metrics(ns_pop_normal, event_col, restricted_methods)
            for row in rows:
                row['outcome'] = outcome_key
                row['subset'] = 'normal_setpoint_popri_normal'
            restr_rows.extend(rows)

    # Save combined
    if all_rows:
        all_df = pd.DataFrame(all_rows)
        out_path = os.path.join(out_dir, 'eval_metrics.csv')
        all_df.to_csv(out_path, index=False)
        print(f"\n  Saved {len(all_df)} rows to {out_path}")

    if restr_rows:
        restr_df = pd.DataFrame(restr_rows)
        out_path = os.path.join(out_dir, 'eval_metrics_restricted.csv')
        restr_df.to_csv(out_path, index=False)
        print(f"  Saved {len(restr_df)} rows to {out_path}")


if __name__ == '__main__':
    main()
