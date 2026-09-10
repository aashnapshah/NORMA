#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
from typing import Dict, Tuple, List, Optional
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

sys.path.append('../../NORMA/process/')
from config import REFERENCE_INTERVALS

MIMIC_LABEL_TO_CODE = {
    'Alanine Aminotransferase (ALT)': 'ALT',
    'Albumin': 'ALB',
    '<Albumin>': 'ALB',
    'Calcium, Total': 'CA',
    'Asparate Aminotransferase (AST)': 'AST',
    'Bilirubin, Direct': 'DBIL',
    'Bilirubin, Total': 'TBIL',
    'Total CO2': 'CO2',
    'Bicarbonate': 'CO2',
    'White Blood Cells': 'WBC',
    'WBC': 'WBC',
    'WBC Count': 'WBC',
    'Chloride': 'CL',
    'Cholesterol, HDL': 'HDL',
    'Cholesterol, Total': 'TC',
    'Creatinine': 'CRE',
    'Creatinine, Serum': 'CRE',
    'Glucose': 'GLU',
    'Potassium': 'K',
    'Protein, Total': 'TP',
    'Urea Nitrogen': 'BUN',
    'Hematocrit': 'HCT',
    '% Hemoglobin A1c': 'A1C',
    'Alkaline Phosphatase': 'ALP',
    'Hemoglobin': 'HGB',
    'Platelet Count': 'PLT',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'MCV': 'MCV',
    'RDW': 'RDW',
    'Red Blood Cells': 'RBC',
    'Gamma Glutamyltransferase': 'GGT',
    'PT': 'PT',
    'Cholesterol, LDL, Calculated': 'LDL',
    'Cholesterol, LDL, Measured': 'LDL',
    'C-Reactive Protein': 'CRP',
    'Lactate Dehydrogenase (LD)': 'LDH',
    'Triglycerides': 'TGL',
    'Triglycer': 'TGL',
    'MPV': 'MPV',
    'Mean Platelet Volume': 'MPV',
    'Bilirubin': 'TBIL',
    'Sodium': 'NA',
}

EHRSHOT_LABEL_TO_CODE = {
    'TG': 'TGL',
}

DATA_DIR_DEFAULT = '../../NORMA/data/processed/'
EHR_PATH_DEFAULT = os.path.join(DATA_DIR_DEFAULT, 'ehrshot/lab_measurements.csv')
MIMIC_DIR_DEFAULT = os.path.join(DATA_DIR_DEFAULT, 'mimiciv/')

def pretty_int(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'NA'
    try:
        return f"{int(val):,}"
    except Exception:
        return str(val)
    

def sex_to01(series: pd.Series) -> pd.Series:
    def map_one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            s = x.strip().upper()
            if s in {'M', 'MALE'}:
                return 0
            if s in {'F', 'FEMALE'}:
                return 1
            if s in {'0', '1'}:
                return int(s)
        if isinstance(x, (int, np.integer)):
            if x in (0, 1):
                return int(x)
            if x == 8507:
                return 0
            if x == 8532:
                return 1
        return np.nan
    return series.apply(map_one)

def load_ehrshot(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data['test_name'] = data['test_name'].fillna('NA')
    data['test_name'] = data['test_name'].apply(lambda x: EHRSHOT_LABEL_TO_CODE[x] if x in EHRSHOT_LABEL_TO_CODE else x)
    # convert code 'nan' to "NA"
    data['source'] = 'ehrshot'
    if 'gender_concept_id' in data.columns:
        data['sex'] = sex_to01(data['gender_concept_id'])
    elif 'sex' in data.columns:
        data['sex'] = sex_to01(data['sex'])
    elif 'gender' in data.columns:
        data['sex'] = sex_to01(data['gender'])
    else:
        data['sex'] = np.nan
    return data


def load_mimic(mimic_dir: str) -> pd.DataFrame:
    labs_path = os.path.join(mimic_dir, 'filtered_lab_data.csv')
    patients_path = os.path.join(mimic_dir, 'patients.csv')
    
    lab_data = pd.read_csv(labs_path, delimiter='\t')
    lab_data = lab_data.dropna(subset=['valuenum'])

    patient_data = pd.read_csv(patients_path)
    lab_data = lab_data.merge(patient_data[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left')
    
    lab_data = lab_data[lab_data['label'].isin(MIMIC_LABEL_TO_CODE)].copy()
    lab_data['test_name'] = lab_data['label'].map(MIMIC_LABEL_TO_CODE)

    lab_data = lab_data.rename(columns={'valuenum': 'numeric_value', 'valueuom': 'unit', 'charttime': 'time'})
    print(lab_data['time'].head())
    lab_data['source'] = 'mimiciv'
    lab_data['year_chart'] = pd.to_datetime(lab_data['time']).dt.year
    print(lab_data['time'].head())
    
    lab_data['age'] = lab_data['anchor_age'] + (lab_data['year_chart'] - lab_data['anchor_year'])
    lab_data['sex'] = sex_to01(lab_data['gender'])
    print('Age: ', lab_data['age'].describe())
    print('Anchor Age: ', lab_data['anchor_age'].mean(), lab_data['anchor_age'].std())
    print('Anchor Year: ', lab_data['anchor_year'].mean(), lab_data['anchor_year'].std())
    print('Year Chart: ', lab_data['year_chart'].mean(), lab_data['year_chart'].std())
    return lab_data[['source', 'subject_id', 'time', 'test_name', 'label', 'numeric_value', 'unit', 'sex', 'age']]

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    def process_each(group):
        q1 = group['numeric_value'].quantile(0.25)
        q3 = group['numeric_value'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        new_df = group[(group['numeric_value'] >= lower_bound) & (group['numeric_value'] <= upper_bound)]

        # To avoid errors when the filtered group is empty (all outliers):
        new_mean = new_df['numeric_value'].mean() if not new_df.empty else float('nan')
        new_std = new_df['numeric_value'].std() if not new_df.empty else float('nan')
        new_min = new_df['numeric_value'].min() if not new_df.empty else float('nan')
        new_max = new_df['numeric_value'].max() if not new_df.empty else float('nan')
        mean = group['numeric_value'].mean()
        std = group['numeric_value'].std()
        before_min = group['numeric_value'].min()
        before_max = group['numeric_value'].max()

        print(
            f"{group['test_name'].iloc[0]:<6} | "
            f"{group.shape[0]:>6} -> {new_df.shape[0]:<6} | "
            f"{mean:>6.2f} ({std:<6.2f}) -> {new_mean:>6.2f} ({new_std:<6.2f}) | "
            f"({before_min:>6.2f}, {before_max:<6.2f}) -> ({new_min:>6.2f}, {new_max:<6.2f})"
        )
        return new_df

    header = (
    f"{'Test Name':<6} | {'Rows':^13} | {'Mean (Std)':^22} | {'Range':^25}\n"
    + "=" * 75
    )
    print(header)
    df = df.copy()
    df = df.groupby(['source', 'test_name'], group_keys=False).apply(process_each)
    df = df.reset_index(drop=True)
    return df

def remove_duplicate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['source', 'subject_id', 'sex', 'test_name', 'time', 'age'])
    df = df.drop_duplicates(subset=['source', 'subject_id', 'sex', 'test_name', 'time', 'numeric_value'], keep='first')
    
    # if there are duplicates, with subject, test and time_delta, get the average of the numeric_value
    df = df.groupby(['source', 'subject_id', 'sex', 'test_name', 'time', 'age']).agg({'numeric_value': 'mean'}).reset_index()
    df = df.reset_index(drop=True)
    return df

# -----------------------
# Stats
# -----------------------
def per_source_test_stats(df: pd.DataFrame):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    stats = {}

    for src, group in df.groupby('source'):
        src_stats = {}

        lab_stats = group.groupby('test_name')['numeric_value'].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()

        n_seq = group.groupby('test_name')['subject_id'].nunique().reset_index().rename(columns={'subject_id': 'n_sequences'})
        lab_stats = lab_stats.merge(n_seq, on='test_name', how='left')
        src_stats['values'] = lab_stats

        subj_grp = group.groupby(['test_name', 'subject_id'])
        ct = subj_grp.size().reset_index(name='num')
        ct_stats = ct.groupby('test_name')['num'].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
        src_stats['counts'] = ct_stats

        first = subj_grp['time'].min()
        last = subj_grp['time'].max()
        span_days = (last - first).dt.days.reset_index(name='days')
        span_stats = span_days.groupby('test_name')['days'].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
        src_stats['spans'] = span_stats

        num_sequences = subj_grp.size().groupby('test_name').size().reset_index(name='num_sequences')
        src_stats['num_sequences'] = num_sequences

        stats[src] = src_stats

    return stats


def flat_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute concise per-source/per-test summary statistics (flat table), including number of sequences."""
    if df.empty:
        return pd.DataFrame(columns=[
            'source', 'test_name',
            'value_mean', 'value_std', 'value_min', 'value_max', 'value_median',
            'count_mean', 'count_std', 'count_min', 'count_max', 'count_median',
            'days_mean', 'days_std', 'days_min', 'days_max', 'days_median',
            'num_sequences'
        ])
    df = df.copy()
    df['numeric_value'] = pd.to_numeric(df['numeric_value'], errors='coerce')
    agg = per_source_test_stats(df)
    tables: List[pd.DataFrame] = []

    for src, seg in agg.items():
        vals = seg['values'].rename(columns={
            'mean': 'value_mean', 'std': 'value_std', 'min': 'value_min',
            'max': 'value_max', 'median': 'value_median'
        })
        cnt = seg['counts'].rename(columns={
            'mean': 'count_mean', 'std': 'count_std', 'min': 'count_min',
            'max': 'count_max', 'median': 'count_median'
        })
        days = seg['spans'].rename(columns={
            'mean': 'days_mean', 'std': 'days_std', 'min': 'days_min',
            'max': 'days_max', 'median': 'days_median'
        })
        num_sequences = seg['num_sequences']
        merged = (
            vals.merge(cnt, on='test_name', how='outer')
                .merge(days, on='test_name', how='outer')
                .merge(num_sequences, on='test_name', how='left')
        )
        merged.insert(0, 'source', src)
        tables.append(merged)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(['source', 'test_name']).reset_index(drop=True)
    return result


def stats(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for backward compatibility."""
    return flat_stats(df)


# -----------------------
# Preprocessing and sequences
# -----------------------
TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}

def filter_min_points(df: pd.DataFrame, min_points: int) -> pd.DataFrame:
    """Keep only (subject_id, test_name) pairs with at least min_points measurements per source."""
    frames = []
    for src in df['source'].unique():
        data = df[df['source'] == src].copy()
        counts = data.groupby(['subject_id', 'test_name']).size().reset_index(name='n')
        keep = set(zip(*counts[counts['n'] >= min_points][['subject_id', 'test_name']].values.T))
        data_multi = data[data[['subject_id', 'test_name']].apply(tuple, axis=1).isin(keep)]
        frames.append(data_multi)
    return pd.concat(frames, ignore_index=True)


def add_time_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    group_cols = ['source', 'subject_id', 'test_name']
    df.sort_values(group_cols + ['time'], inplace=True)
    df['time_delta'] = (df['time'] - df.groupby(group_cols)['time'].transform('first')).dt.total_seconds() / 86400
    df['time_delta'] = df['time_delta'].astype(np.float32)
    return df

def is_normal(values, test_name, sex):
    sex = 'M' if sex == 0 else 'F'
    low, high, _= REFERENCE_INTERVALS[test_name][sex]
    return (low <= values) & (values <= high)

def is_high_low(values, test_name, sex):
    sex = 'M' if sex == 0 else 'F'
    low, high, _= REFERENCE_INTERVALS[test_name][sex]
    is_low = values < low
    is_high = values > high
    # i want to return -1, 0, 1 for low, normal, high
    return np.where(is_low, -1, np.where(is_high, 1, 0))
    
def create_sequences(df, quiet=False):
    group_cols = ['source', 'subject_id', 'test_name']
    grouped = df.groupby(group_cols, sort=False)
    exclude = ['CRP', 'LDH', 'GGT', 'PT']
    
    sequences: List[dict] = []
    for i, (key, group) in enumerate(tqdm(grouped, desc="Creating sequences", disable=quiet)):
        t = group['time_delta'].to_numpy(dtype=np.float32)
        x = pd.to_numeric(group['numeric_value'], errors='coerce').to_numpy(dtype=np.float32)
        source, pid, test_name = key
        sex01 = group['sex'].iloc[0]
        age = group['age'].iloc[0]
        cid = TEST_VOCAB.get(test_name)
        if cid is None or pd.isna(sex01) or test_name in exclude:
            continue
        s = is_normal(x, test_name, sex01)
        s3 = is_high_low(x, test_name, sex01)
            
        sequences.append({
            "source": source,
            "pid": pid,
            "test_name": test_name,
            "cid": cid,
            "sex": sex01,
            "age": age,
            "x": x,
            "t": t,
            "s": s,
            "s3": s3,
        })
    return sequences


def process_lab_data(
    load_func,
    data_path: str,
    min_points: int,
    n_head: Optional[int],
    name: str,
    save_dir: Optional[str] = None,
):
    df = load_func(data_path)
    print(f"[{name}] Loaded {df.shape[0]:,} rows ({df.shape[1]} columns)")

    df['numeric_value'] = pd.to_numeric(df['numeric_value'], errors='coerce')
    df = df.dropna(subset=['numeric_value', 'subject_id', 'test_name', 'time', 'sex', 'age'])

    if n_head is not None and n_head >= 0:
        sample_pids = pd.unique(df['subject_id'])[:n_head]
        df = df[df['subject_id'].isin(sample_pids)]

    print(f"[{name}] Loaded {df.shape[0]:,} rows ({df.shape[1]} columns)")
    
    before = df.shape[0]
    df = df[df['numeric_value'] != 0]
    print(f"[{name}] Filtered zero values: {before - df.shape[0]:,} rows removed. Now {df.shape[0]:,} rows.")


    # Remove duplicate sequences
    before = df.shape[0]
    df = remove_duplicate_sequences(df)
    print(f"[{name}] Removed duplicate sequences: {before - df.shape[0]:,} rows removed. Now {df.shape[0]:,} rows.")

    before_df = df.copy()
    df = remove_outliers(df)
    print(f"[{name}] Removed outliers: {before_df.shape[0] - df.shape[0]:,} rows removed. Now {df.shape[0]:,} rows.")

    # Filter by min_points
    before = df.shape[0]
    df = filter_min_points(df, min_points=min_points)
    print(f"[{name}] Removed <{min_points} point sequences: {before - df.shape[0]:,} rows removed. Now {df.shape[0]:,} rows.")

    df = add_time_delta_columns(df)

    # Preview
    head_preview = df.head(3)
    if not head_preview.empty:
        print(f"[{name}] First few rows after preprocessing:")
        print(head_preview.to_string(index=False))

    # Sequences
    seqs = create_sequences(df)
    print(f"[{name}] Created {len(seqs)} sequences")

    # Save dataframe
    if save_dir is not None:
        df_save_path = os.path.join(save_dir, f'{name}_processed_df.csv')
        os.makedirs(os.path.dirname(df_save_path), exist_ok=True)
        df.to_csv(df_save_path, index=False)
        print(f"[{name}] Dataframe saved to {df_save_path}")

    # Save sequences
    if save_dir is not None:
        seq_save_path = os.path.join(save_dir, f'{name}_sequences.pkl')
        os.makedirs(os.path.dirname(seq_save_path), exist_ok=True)
        with open(seq_save_path, 'wb') as f:
            pickle.dump(seqs, f)
        print(f"[{name}] Sequences saved to {seq_save_path}")
        print(f"[{name}] Sequences length: {len(seqs)}")
        print(f"[{name}] Sequences: {seqs[0]}")
    return df, seqs


def print_summary(
    ehr_path: str,
    mimic_dir: str,
    merged_path: str,
    stats_path: str,
    merged: pd.DataFrame,
    stats_tbl: pd.DataFrame
) -> None:
    print('Inputs:')
    print(f'  EHRShot: {ehr_path}')
    print(f'  MIMIC-IV: {mimic_dir}\n')

    print('Merged dataset:')
    print(f'  Rows: {pretty_int(len(merged))}, Columns: {pretty_int(merged.shape[1])}')
    for src, grp in sorted(merged.groupby('source')):
        rows = pretty_int(len(grp))
        subjects = pretty_int(grp["subject_id"].nunique(dropna=True))
        tests = pretty_int(grp["test_name"].nunique(dropna=True))
        print(f'  {src}: rows={rows}, subjects={subjects}, tests={tests}')
    print('\nPer-test statistics:')
    print(f'  Rows: {pretty_int(len(stats_tbl))}\n')

    print('Outputs:')
    print(f'  Merged CSV: {merged_path}')
    print(f'  Stats CSV:  {stats_path}')


def main():
    parser = argparse.ArgumentParser(description='Merge lab datasets, compute stats, and build sequences.')
    parser.add_argument('--ehr-path', default=EHR_PATH_DEFAULT, help='Path to EHRShot CSV.')
    parser.add_argument('--mimic-dir', default=MIMIC_DIR_DEFAULT, help='Path to MIMIC-IV filtered lab data directory.')
    parser.add_argument('--save-dir', default=DATA_DIR_DEFAULT, help='Output directory for processed dataframes and sequences.')
    parser.add_argument('--min-points', type=int, default=2, help='Minimum points per (subject_id, test_name) to keep.')
    parser.add_argument('--head-n', type=int, default=None, help='Limit to first N patients per dataset (by subject_id order). Use -1 to disable.')
    args = parser.parse_args()

    # # Process EHRShot
    ehrshot_df, ehrshot_seq = process_lab_data(
        load_func=load_ehrshot,
        data_path=args.ehr_path,
        min_points=args.min_points,
        n_head=args.head_n,
        name="EHRSHOT",
        save_dir=args.save_dir,
    )
    
    # Process MIMIC-IV
    mimiciv_df, mimiciv_seq = process_lab_data(
        load_func=load_mimic,
        data_path=args.mimic_dir,
        min_points=args.min_points,
        n_head=args.head_n,
        name="MIMIC-IV",
        save_dir=args.save_dir,
    )

    # Merge and save merged dataframe
    merged_df = pd.concat([mimiciv_df, ehrshot_df], ignore_index=True)

    merged_path = os.path.join(args.save_dir, 'combined_lab_data_v2.csv')
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    merged_df.to_csv(merged_path, index=False)
    print(f"[MERGED] Dataframe saved to {merged_path}")

    # Save merged sequences
    all_sequences = mimiciv_seq + ehrshot_seq
    sequences_path = os.path.join(args.save_dir, 'combined_sequences_v2.pkl')
    if os.path.exists('sequences_path'):
        with open(sequences_path, "rb") as f:
            all_sequences = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(sequences_path), exist_ok=True)
        with open(sequences_path, "wb") as f:
            pickle.dump(all_sequences, f)
        print(f"[MERGED] Sequences saved to {sequences_path}")

    # Stats (compute if file not present)
    stats_path = os.path.join(args.save_dir, 'combined_lab_stats_by_test_v2.csv')
    if os.path.exists('stats_path'):
        stats_df = pd.read_csv(stats_path)
    else:
        stats_df = stats(merged_df)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)

    print_summary(
        ehr_path=args.ehr_path,
        mimic_dir=args.mimic_dir,
        merged_path=merged_path,
        stats_path=stats_path,
        merged=merged_df,
        stats_tbl=stats_df,
    )


if __name__ == '__main__':
    main()
