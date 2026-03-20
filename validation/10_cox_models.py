#!/usr/bin/env python
"""
Step 9: Cox proportional hazards models.

For each (outcome, analyte, method), fits a Cox PH model with:
  - age, sex (covariates)
  - abnormal flag (0/1) from each classification method

Reports hazard ratios, p-values, and concordance indices.

Output: results/{dataset}/raw/cox_results.csv

Usage:
    python 09_cox_models.py --dataset eicu
    python 09_cox_models.py --dataset chs
"""
import argparse
import os
import sys
import warnings

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

from datasets import DATASETS
from config import PRIMARY_OUTCOMES

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

METHODS = ['PopRI', 'PerRI', 'NORMA']

warnings.filterwarnings("ignore")


def load_classification(dataset_name):
    """Load classification.csv with numeric class columns."""
    path = os.path.join(ROOTDIR, 'results', dataset_name, 'raw', 'classification.csv')
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for col in df.columns:
        if col.endswith('_class'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    return df


def fit_cox(df, outcome_cfg, analyte, method, test_frac=0.4, seed=42):
    """Fit one Cox model for a single (analyte, method, outcome).

    Returns dict with HR, CI, p-value, concordance, or None.
    """
    cls_col = f'{method}_class'
    event_col = outcome_cfg['event_col']
    time_col = outcome_cfg.get('time_col')
    censor_col = outcome_cfg.get('censor_col')

    lab_df = df[df['analyte'] == analyte].copy()
    if len(lab_df) < 20:
        return None

    # Patient-level aggregation
    lab_df['_abnormal'] = (lab_df[cls_col] != 1).astype(int)

    agg_dict = {
        'age': ('age', 'first'),
        'sex': ('sex', 'first'),
        'abnormal': ('_abnormal', 'max'),
        'event': (event_col, 'first'),
    }

    agg_dict['event_time'] = (time_col, 'first')
    agg_dict['censor_time'] = (censor_col, 'first')

    patient_df = lab_df.groupby('patient_id').agg(**agg_dict).reset_index()
    patient_df = patient_df.dropna(subset=['event', 'age', 'sex'])
    patient_df['event'] = patient_df['event'].astype(int)
    patient_df['sex'] = patient_df['sex'].astype(int)

    # Duration: time to event (if event) or censor time (if no event)
    patient_df['duration'] = np.where(
        patient_df['event'] == 1,
        patient_df['event_time'],
        patient_df['censor_time']
    )
    patient_df['duration'] = pd.to_numeric(patient_df['duration'], errors='coerce')
    patient_df = patient_df.dropna(subset=['duration', 'age', 'sex', 'abnormal'])
    patient_df = patient_df[patient_df['duration'] > 0]

    if len(patient_df) < 10 or patient_df['event'].sum() < 3 or patient_df['abnormal'].nunique() < 2:
        return None

    # Stratified train/test split by event status and duration bin
    from sklearn.model_selection import train_test_split
    patient_df['_strat'] = patient_df['event'].astype(str)
    try:
        train_idx, test_idx = train_test_split(
            patient_df.index, test_size=test_frac, random_state=seed,
            stratify=patient_df['_strat']
        )
    except ValueError:
        # Fallback if stratification fails (too few samples in a stratum)
        train_idx, test_idx = train_test_split(
            patient_df.index, test_size=test_frac, random_state=seed
        )
    cols = ['duration', 'event', 'age', 'sex', 'abnormal']
    train_df = patient_df.loc[train_idx, cols]
    test_df = patient_df.loc[test_idx, cols]

    if len(train_df) < 10 or train_df['event'].sum() < 3:
        return None

    # Fit
    cph = CoxPHFitter()
    try:
        cph.fit(train_df, duration_col='duration', event_col='event')
    except Exception:
        return None

    hr = cph.hazard_ratios_['abnormal']
    ci = cph.confidence_intervals_.loc['abnormal']
    p = cph.summary.loc['abnormal', 'p']

    try:
        c_test = cph.score(test_df, scoring_method='concordance_index')
    except Exception:
        c_test = np.nan

    return {
        'analyte': analyte,
        'method': method,
        'HR': hr,
        'HR_lower': np.exp(ci.iloc[0]),
        'HR_upper': np.exp(ci.iloc[1]),
        'p_value': p,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_events_train': int(train_df['event'].sum()),
        'n_events_test': int(test_df['event'].sum()),
        'concordance_train': cph.concordance_index_,
        'concordance_test': c_test,
    }


def apply_fdr(results_df):
    """Apply BH FDR correction per outcome and globally."""
    if len(results_df) == 0:
        return results_df
    df = results_df.copy()
    df['p_fdr'] = np.nan
    for outcome in df['outcome'].unique():
        mask = df['outcome'] == outcome
        pvals = df.loc[mask, 'p_value'].values
        if len(pvals) > 0:
            _, corrected, _, _ = multipletests(pvals, method='fdr_bh')
            df.loc[mask, 'p_fdr'] = corrected
    _, df['p_fdr_global'], _, _ = multipletests(df['p_value'].values, method='fdr_bh')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, 'results', ds.name, 'raw')
    os.makedirs(out_dir, exist_ok=True)

    df = load_classification(ds.name)
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')

    # Attach only needed outcomes
    needed = {}
    for o in PRIMARY_OUTCOMES:
        if o in ds.outcomes:
            event_col = ds.outcomes[o]['event_col']
            if event_col not in df.columns:
                needed[o] = event_col

    if needed:
        print(f"  Attaching outcomes: {list(needed.keys())}...")
        from utils import get_mortality, load_diagnosis_processed
        df = df.copy()

        if 'mortality' in needed and 'patientunitstayid' in df.columns:
            mort = get_mortality()
            cols_to_add = [c for c in mort.columns if c not in df.columns or c == 'patientunitstayid']
            df = df.merge(mort[cols_to_add], on='patientunitstayid', how='left')
            print(f"    Attached mortality")

        if 'prolonged_los' in needed and 'unitdischargeoffset' in df.columns:
            df['prolonged_los'] = df['unitdischargeoffset'] > (7 * 24 * 60)
            print(f"    Attached prolonged_los")

        disease_needed = {o: col for o, col in needed.items()
                          if o not in ('mortality', 'prolonged_los')}
        if disease_needed and 'patientunitstayid' in df.columns:
            from config import DISEASES
            diagnosis = load_diagnosis_processed()
            for outcome_key, event_col in disease_needed.items():
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
    else:
        print(f"  All outcome columns already present")

    methods = METHODS
    analytes = sorted(a for a in df['analyte'].unique() if a not in EXCLUDE_ANALYTES)
    print(f"  {len(df)} measurements, {len(analytes)} analytes, methods: {methods}")

    all_rows = []
    for outcome_key in PRIMARY_OUTCOMES:
        if outcome_key not in ds.outcomes:
            continue
        outcome_cfg = ds.outcomes[outcome_key]
        event_col = outcome_cfg['event_col']
        if event_col not in df.columns:
            print(f"  Skipping {outcome_key} ({event_col} not found)")
            continue

        event_rate = df.drop_duplicates('patient_id')[event_col].mean()
        print(f"\n  {outcome_key} (event rate: {event_rate:.3f})")

        for analyte in analytes:
            for method in methods:
                result = fit_cox(df, outcome_cfg, analyte, method)
                if result is None:
                    continue
                result['outcome'] = outcome_key
                all_rows.append(result)

                sig = '*' if result['p_value'] < 0.05 else ' '
                print(f"    {analyte:>5s} | {result['method']:>6s} | "
                      f"HR={result['HR']:.2f} ({result['HR_lower']:.2f}-{result['HR_upper']:.2f}) "
                      f"p={result['p_value']:.3f}{sig} C={result['concordance_test']:.3f}")

    if all_rows:
        results = pd.DataFrame(all_rows)
        results = apply_fdr(results)
        out_path = os.path.join(out_dir, 'cox_results.csv')
        results.to_csv(out_path, index=False)
        print(f"\n  Saved {len(results)} results to {out_path}")

        sig = results[results['p_fdr'] < 0.05]
        print(f"  Significant: {len(sig)}/{len(results)} (FDR q<0.05)")


if __name__ == '__main__':
    main()
