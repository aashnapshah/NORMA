"""
Compute cohort summary statistics per dataset.

Produces two CSVs per dataset with identical schemas:
  - cohort_demographics.csv: total patients, age, sex distribution
  - cohort_summary.csv: per-analyte stats (N patients, mean, std, time span, tests/person)

Usage:
    python scripts/cohort_summary.py --dataset ehrshot
    python scripts/cohort_summary.py --dataset mimiciv
    python scripts/cohort_summary.py --dataset eicu
    python scripts/cohort_summary.py --dataset chs
    python scripts/cohort_summary.py --all
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# plots.py is in the same directory (scripts/)
from plots import EXCLUDE_ANALYTES, ANALYTE_NAMES, get_unit, get_pop_ri


def compute_summary(df):
    """Compute per-analyte summary from standardized DataFrame.

    Expected columns: patient_id, analyte, value, timestamp
    timestamp should be in days.
    """
    df = df[~df['analyte'].isin(EXCLUDE_ANALYTES)].copy()
    analytes = sorted(df['analyte'].unique())

    rows = []
    for analyte in analytes:
        sub = df[df['analyte'] == analyte]
        per_patient = sub.groupby('patient_id')
        counts = per_patient.size()
        spans = per_patient['timestamp'].agg(['min', 'max'])
        spans_days = spans['max'] - spans['min']

        rows.append({
            'analyte': analyte,
            'n_patients': sub['patient_id'].nunique(),
            'value_mean': sub['value'].mean(),
            'value_std': sub['value'].std(),
            'time_span_mean': spans_days.mean(),
            'time_span_std': spans_days.std(),
            'tests_per_person_mean': counts.mean(),
            'tests_per_person_std': counts.std(),
        })

    return pd.DataFrame(rows)


def compute_demographics(df):
    """Compute cohort demographics from standardized DataFrame.

    Expected columns: patient_id, sex (0=M, 1=F), age, timestamp (days)
    """
    pat = df.drop_duplicates('patient_id')
    n = pat['patient_id'].nunique()
    n_female = (pat['sex'] == 1).sum()
    n_male = (pat['sex'] == 0).sum()
    # Overall time span per patient (days)
    per_patient = df.groupby('patient_id')['timestamp'].agg(['min', 'max'])
    spans_days = per_patient['max'] - per_patient['min']
    return pd.DataFrame([{
        'n_patients': int(n),
        'n_sequences': df.groupby(['patient_id', 'analyte']).ngroups,
        'age_mean': pat['age'].mean(),
        'age_std': pat['age'].std(),
        'pct_female': n_female / (n_female + n_male) * 100 if (n_female + n_male) > 0 else None,
        'pct_male': n_male / (n_female + n_male) * 100 if (n_female + n_male) > 0 else None,
        'span_days_median': spans_days.median(),
        'span_days_q25': spans_days.quantile(0.25),
        'span_days_q75': spans_days.quantile(0.75),
    }])


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_sequences(source):
    """Load from combined_sequences_v2.pkl, filter by source."""
    path = os.path.join(ROOTDIR, 'data', 'processed', 'combined_sequences_v2.pkl')
    with open(path, 'rb') as f:
        seqs = pickle.load(f)
    seqs = [s for s in seqs if s['source'] == source]
    rows = []
    for s in seqs:
        pid = s['pid']
        analyte = s['test_name']
        sex = s['sex']
        age = s['age']
        for x_val, t_val in zip(s['x'], s['t']):
            rows.append({
                'patient_id': pid, 'analyte': analyte,
                'value': float(x_val), 'timestamp': float(t_val),
                'sex': sex, 'age': age,
            })
    return pd.DataFrame(rows)


def load_eicu():
    """Load eICU from split_df.pkl."""
    path = os.path.join(ROOTDIR, 'validation', 'data', 'split_df.pkl')
    df = pd.read_pickle(path)
    df = df.rename(columns={
        'uniquepid': 'patient_id', 'lab_code': 'analyte',
        'labresult': 'value', 'labresultoffset': 'timestamp',
        'gender': 'sex_str',
    })
    df['sex'] = (df['sex_str'] == 'Female').astype(int)
    df['timestamp'] = df['timestamp'] / (60 * 24)  # minutes → days
    return df[['patient_id', 'analyte', 'value', 'timestamp', 'sex', 'age']]


def chs_hardcoded():
    """CHS values from manuscript — placeholder until data access."""
    data = {
        'A1C':  (336494, 7.01, 1.35, None, None, 11.03, 5.55),
        'ALB':  (790490, 4.29, 0.89, None, None, 8.62, 3.58),
        'ALP':  (1099044, 75.09, 24.50, None, None, 9.44, 4.06),
        'ALT':  (1233902, 20.88, 12.71, None, None, 9.94, 4.35),
        'AST':  (1236781, 21.66, 8.87, None, None, 9.92, 4.33),
        'BUN':  (1251453, 15.22, 5.24, None, None, 9.97, 4.32),
        'CA':   (909036, 9.43, 0.44, None, None, 9.02, 3.82),
        'CL':   (31171, 102.70, 3.12, None, None, 6.08, 1.44),
        'CO2':  (299, 32.41, 13.79, None, None, 5.68, 1.05),
        'CRE':  (1310465, 0.83, 0.23, None, None, 10.09, 4.36),
        'GLU':  (1351168, 101.19, 28.64, None, None, 10.17, 4.47),
        'HCT':  (1407912, 40.59, 4.05, None, None, 9.71, 3.94),
        'HDL':  (1002204, 50.40, 13.29, None, None, 10.04, 4.40),
        'HGB':  (1408118, 13.44, 1.43, None, None, 9.71, 3.95),
        'K':    (1182544, 4.48, 0.41, None, None, 9.72, 4.15),
        'LDL':  (1187844, 111.04, 31.84, None, None, 10.45, 4.72),
        'MCH':  (1250759, 28.85, 2.11, None, None, 9.28, 3.70),
        'MCHC': (1407568, 33.10, 1.16, None, None, 9.71, 3.94),
        'MCV':  (1400648, 87.10, 5.57, None, None, 9.71, 3.94),
        'NA':   (1187537, 140.32, 2.62, None, None, 9.75, 4.18),
        'PLT':  (1407716, 251.05, 64.28, None, None, 9.71, 3.94),
        'RBC':  (1404794, 4.67, 0.48, None, None, 9.69, 3.93),
        'RDW':  (1403339, 13.62, 1.19, None, None, 9.66, 3.91),
        'TBIL': (815774, 0.62, 0.30, None, None, 8.57, 3.56),
        'TC':   (1260730, 188.92, 37.87, None, None, 10.56, 4.76),
        'TGL':  (1244518, 133.99, 65.58, None, None, 10.57, 4.77),
        'TP':   (791481, 7.30, 0.47, None, None, 8.61, 3.57),
        'WBC':  (1409027, 7.12, 2.16, None, None, 9.71, 3.94),
    }
    rows = []
    for analyte, (n, mean, std, ts_mean, ts_std, tpp_mean, tpp_std) in data.items():
        rows.append({
            'analyte': analyte, 'n_patients': n,
            'value_mean': mean, 'value_std': std,
            'time_span_mean': ts_mean, 'time_span_std': ts_std,
            'tests_per_person_mean': tpp_mean, 'tests_per_person_std': tpp_std,
        })

    summary = pd.DataFrame(rows)
    demographics = pd.DataFrame([{
        'n_patients': 903022,
        'n_sequences': 16301161,
        'age_mean': None, 'age_std': None,
        'pct_female': None, 'pct_male': None,
        'span_days_median': None, 'span_days_q25': None, 'span_days_q75': None,
    }])
    return summary, demographics


def run_dataset(dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if dataset == 'chs':
        summary, demographics = chs_hardcoded()
    else:
        if dataset in ('ehrshot', 'mimiciv'):
            source = 'ehrshot' if dataset == 'ehrshot' else 'mimiciv'
            print(f"  Loading {source} sequences...")
            df = load_sequences(source)
        elif dataset == 'eicu':
            print(f"  Loading eICU...")
            df = load_eicu()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients")
        summary = compute_summary(df)
        demographics = compute_demographics(df)

    summary_path = os.path.join(out_dir, 'cohort_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  {summary_path}")

    demo_path = os.path.join(out_dir, 'cohort_demographics.csv')
    demographics.to_csv(demo_path, index=False)
    print(f"  {demo_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ehrshot', 'mimiciv', 'eicu', 'chs'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        datasets = ['ehrshot', 'mimiciv', 'eicu', 'chs']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['ehrshot', 'mimiciv', 'eicu', 'chs']

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        out_dir = os.path.join(ROOTDIR, 'results', dataset, 'raw')
        run_dataset(dataset, out_dir)


if __name__ == '__main__':
    main()
