import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys

sys.path.append('../')
from process.config import LOINC_CODES, REVERSE_LOINC_CODES, DATAPATH
import logging
import os
from tqdm import tqdm

def log_status(df: pd.DataFrame, message: str) -> None:
    """Log the current status of the DataFrame with a custom message."""
    n_patients, n_measurements = total_ms_pts(df)
    logging.info(f"{message} | {n_patients} Patients | {n_measurements} Measurements")

def filter_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only relevant measurement rows and remove NaNs and non-positive values."""
    df = df[df['table'] == 'measurement']
    # add LOINC/ to the code in loinc_codes
    LOINC_CODES_LIST = [f"LOINC/{code}" for code in REVERSE_LOINC_CODES.keys()]
    df = df[df['code'].isin(LOINC_CODES_LIST)]
    log_status(df, "Removed non-CBC measurements")
    df = df.dropna(subset=['numeric_value'])
    log_status(df, "Removed NaN values")
    df = df[df['numeric_value'] > 0]
    log_status(df, "Removed values <= 0")
    # Fix WBC units
    wbc_mask = (df['code'].isin(LOINC_CODES['WBC'])) & (df['unit'] == "/uL")
    df.loc[wbc_mask, 'numeric_value'] /= 1000
    # Add test name column
    # need to add LOINC/ to the code in loinc_codes
    df['test_name'] = df['code'].str.replace('LOINC/', '').map(REVERSE_LOINC_CODES)
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove values outside 3 standard deviations for each test."""
    for test_name in LOINC_CODES.keys():
        test_mask = df['test_name'] == test_name
        test_vals = df.loc[test_mask, 'numeric_value']
        mean, std = test_vals.mean(), test_vals.std()
        lower, upper = mean - 3 * std, mean + 3 * std
        mask = ~((test_mask) & ~df['numeric_value'].between(lower, upper))
        df = df[mask]
    log_status(df, "Removed outliers (3 std outside mean)")
    return df

def filter_by_age(df: pd.DataFrame, input_path: str = "../data/table_csv/person.csv") -> pd.DataFrame:
    """Filter out measurements for patients not between 13 and 100 years old at time of test."""
    person_df = pd.read_csv(input_path)
    person_df = person_df.rename(columns={"person_id": "subject_id"})
    df = df.merge(person_df[['subject_id', 'birth_DATETIME']], on='subject_id', how='left')
    df['birth_DATETIME'] = pd.to_datetime(df['birth_DATETIME'])
    df['age'] = (df['time'] - df['birth_DATETIME']).dt.days / 365.25
    df = df[df['age'].between(13, 100)]
    log_status(df, "Enforced 13 < Age < 100 at test")
    return df

def get_sex(df: pd.DataFrame, input_path: str = "../data/table_csv/person.csv") -> pd.DataFrame:
    """Get sex for each patient."""
    person_df = pd.read_csv(input_path)
    person_df = person_df.rename(columns={"person_id": "subject_id"})
    df = df.merge(person_df[['subject_id', 'gender_concept_id']], on='subject_id', how='left')
    # Convert gender_concept_id to binary
    df['gender_concept_id'] = df['gender_concept_id'].map({8507: 0, 8532: 1})
    df['sex'] = df['gender_concept_id'].map(lambda x: 'M' if x == 0 else 'F')
    return df

def total_ms_pts(db):
    return (db['subject_id'].unique().shape[0], db[db['table'] == 'measurement'].shape[0])

def df_search(db, substring):
    mask = db.apply(lambda row: row.astype(str).str.contains(substring, case=False, na=False)).any(axis=1)
    return db[mask]

def filter_by_frequency(df: pd.DataFrame, day_gap: int = 30, min_labs: int = 5, cutoff_date: str = '2018-01-01') -> pd.DataFrame:
    """Keep only patients/tests with at least min_labs labs day_gap days apart before cutoff_date."""
    rows_to_keep = []
    grouped = df.groupby(['subject_id', 'test_name'])
    for (subject_id, test_name), group in grouped:
        pre_cutoff_group = group[group['time'] < pd.Timestamp(cutoff_date)]
        dates = np.sort(pd.to_datetime(pre_cutoff_group['time']))
        if len(dates) < min_labs:
            continue
        subsequence = [dates[0]]
        i = 1
        while i < len(dates):
            if (dates[i] - subsequence[-1]) / np.timedelta64(1, 'D') >= day_gap:
                subsequence.append(dates[i])
                if len(subsequence) == min_labs:
                    rows_to_keep.extend(group.index)
                    break
            i += 1
    df = df.loc[np.unique(rows_to_keep)]
    log_status(df, f"Ensured {min_labs}+ labs with {day_gap}+ day gaps (before {cutoff_date})")
    # Log number of setpoints by test
    for test_name in LOINC_CODES.keys():
        num_patients = df[df['test_name'] == test_name]['subject_id'].nunique()
        logging.info(f"Setpoints for {test_name}: {num_patients} Patients")
    return df


def filter_test_sequences(df, index_year=2018, day_gap=30, min_pre_tests=5, save_path=None):
    grouped = df.groupby(['subject_id', 'test_name'])
    rows_to_keep = []
    cutoff_date = pd.Timestamp(f'{index_year}-01-01')

    for (pt, test), group in tqdm(grouped, desc="Filtering sequences"):
        # Convert time column to datetime with error handling
        group['time'] = pd.to_datetime(group['time'], errors='coerce')
                
        pre_group = group[group['time'] < cutoff_date]
        post_group = group[group['time'] >= cutoff_date]

        if post_group.empty:
            continue

        pre_group_sorted = pre_group.sort_values('time')
        dates = pre_group_sorted['time'].values

        if len(dates) < min_pre_tests:
            continue

        subsequence_indices = [pre_group_sorted.index[0]]
        last_date = dates[0]

        for i in range(1, len(dates)):
            if (dates[i] - last_date) / np.timedelta64(1, 'D') >= day_gap:
                subsequence_indices.append(pre_group_sorted.index[i])
                last_date = dates[i]

        if len(subsequence_indices) >= min_pre_tests:
            rows_to_keep.extend(subsequence_indices)
            #rows_to_keep.extend(post_group.index)

    df_filtered = df.loc[np.unique(rows_to_keep)]

    if save_path:
        df_filtered.to_pickle(save_path)

    return df_filtered.sort_values(by=['subject_id', 'test_name', 'time'])


def filter_measurements_foy(df, time_col='time', min_gap=30,
                                                min_tests=3, index_year=2022):
    """
    Filters DataFrame for (subject_id, test_name) groups that have at least `min_tests`
    measurements before `index_year`, each at least `min_gap_days` apart.
    Retains those and all post-`index_year` measurements from qualifying groups.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    isolated_rows = []
    post2018_rows = []
    grouped = df.groupby(['subject_id', 'test_name'])

    for (pt, test), group in tqdm(grouped):
        pre2018_group = group[group['time'] < pd.Timestamp('2018-01-01')]
        post2018_group = group[group['time'] >= pd.Timestamp('2018-01-01')]

        if len(pre2018_group) < 5:
            continue

        # Sort by date
        sorted_group = pre2018_group.sort_values('time')
        sorted_dates = sorted_group['time'].values
        sorted_indices = sorted_group.index

        isolated_indices = []

        for i in range(len(sorted_dates)):
            left_ok = (i == 0) or (sorted_dates[i] - sorted_dates[i - 1] > np.timedelta64(30, 'D'))
            right_ok = (i == len(sorted_dates) - 1) or (sorted_dates[i + 1] - sorted_dates[i] > np.timedelta64(30, 'D'))

            if left_ok and right_ok:
                isolated_indices.append(sorted_indices[i])

        if len(isolated_indices) >= 5:
            isolated_rows.extend(isolated_indices)

    # Final filtered df: isolated pre-2018 + all post-2018 from qualifying groups
    df = df.loc[np.unique(isolated_rows + post2018_rows)]
    return df


def cutoff_measurements_df(df, percent=0.8, min_tests=5):
    
    df['time'] = pd.to_datetime(df['time'])
    grouped = df.groupby(['subject_id', 'test_name'])
    i = 0
    rows_to_keep = []
    for (pt, test), group in tqdm(grouped, desc="Filtering Sequences"):
        # Sort by time
        group_sorted = group.sort_values('time')
        if len(group_sorted) < min_tests:
            continue

        # Get time range
        start_time = group_sorted['time'].min()
        end_time = group_sorted['time'].max()
        total_time_range = end_time - start_time

        # Calculate cutoff time by converting timedelta to hours first
        time = total_time_range.total_seconds() 
        cutoff = float(time) * percent
        
        cutoff_time = start_time + pd.Timedelta(seconds=cutoff)
        measurements_in_range = group_sorted[group_sorted['time'] <= cutoff_time]
        measurements_post_range = group_sorted[group_sorted['time'] > cutoff_time]
        
        if len(measurements_in_range) >= min_tests and len(measurements_post_range) > 0:
            rows_to_keep.extend(measurements_in_range.index)
        
    df_filtered = df.loc[np.unique(rows_to_keep)]
    return df_filtered.sort_values(['subject_id', 'test_name', 'time'])

def filter_measurements_yash(df, index_year=2018, min_gap=30, min_tests=5):
    grouped = df.groupby(['subject_id', 'test_name'])   
    df['time'] = pd.to_datetime(df['time'])

    # i find the longest subsequence of tests before the index year
    # i check the subseq is at least 5 long and there is at least 1 more test post index
    rows_to_keep = []
    for (pt, test), group in tqdm(grouped):
            pre_group = group[group['time'] < pd.Timestamp(f'{index_year}-01-01')]
            #post_group = group[group['time'] >= pd.Timestamp(f'{index_year}-01-01')]

            #if (post_group.shape[0] <= 0):
            #    continue # skip if no test after index and required

            # Sort pre_group by time to get ordered dates with preserved index
            pre_group_sorted = pre_group.sort_values('time')
            dates = pre_group_sorted['time'].values

            if len(dates) < 5:
                continue  # skip if not enough test entries

            # Build the subsequence and track corresponding indices
            subsequence_indices = [pre_group_sorted.index[0]]
            last_date = dates[0]

            for i in range(1, len(dates)):
                if (dates[i] - last_date) / np.timedelta64(1, 'D') >= min_gap:
                    subsequence_indices.append(pre_group_sorted.index[i])
                    last_date = dates[i]

            # Keep if at least 5 tests are in subsequence
            if len(subsequence_indices) >= 5:
                rows_to_keep.extend(subsequence_indices)

    df = df.loc[np.unique(rows_to_keep)]
    return df
    
def filter_measurements_df(df, index_year=2018, min_gap=30, min_tests=5):
    df['time'] = pd.to_datetime(df['time'])
    # remove any rows after index year
    grouped = df.groupby(['subject_id', 'test_name'])

    # i find the longest subsequence of tests before the index year
    # i check the subseq is at least 5 long and there is at least 1 more test post index
    rows_to_keep = []
    for (pt, test), group in tqdm(grouped, desc="Filtering sequences"):
            pre_group = group[group['time'] < pd.Timestamp(f'{index_year}-01-01')]
            post_group = group[group['time'] >= pd.Timestamp(f'{index_year}-01-01')]

            #if not (post_group.shape[0] > 0):
            #        continue # skip if no test afetr index

            # Sort pre_group by time to get ordered dates with preserved index
            pre_group_sorted = pre_group.sort_values('time')
            dates = pre_group_sorted['time'].values

            if len(dates) < min_tests:
                continue  # skip if not enough test entries

            # Build the subsequence and track corresponding indices
            subsequence_indices = [pre_group_sorted.index[0]]
            last_date = dates[0]

            for i in range(1, len(dates)):
                if (dates[i] - last_date) / np.timedelta64(1, 'D') >= min_gap:
                    subsequence_indices.append(pre_group_sorted.index[i])
                    last_date = dates[i]

            # Keep if at least 5 tests are in subsequence
            if len(subsequence_indices) >= min_tests:
                rows_to_keep.extend(subsequence_indices)
                # Always include all post-2018 rows
                #rows_to_keep.extend(post_group.index)

    df_filtered = df.loc[np.unique(rows_to_keep)]
    return df_filtered

def preprocess_pipeline(
    input_path: str = DATAPATH,
    output_path: str = "../data/all_lab_measurements.pkl",
    person_csv_path: str = "../data/raw/tables_csv/person.csv",
    day_gap: int = 30,
    min_labs: int = 5,
    cutoff_date: str = '2018-01-01'
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the result."""
    df = pd.read_parquet(input_path)
    log_status(df, "Starting point")
    df = filter_measurements(df)
    df = remove_outliers(df)
    df = filter_by_age(df, person_csv_path)
    print(df.head())
    #df = filter_by_frequency(df, day_gap, min_labs, cutoff_date)
    df = get_sex(df, person_csv_path)
    pd.to_pickle(df, output_path)
    df.to_csv(output_path.replace(".pkl", ".csv"), index=False)
    print(f"Final preprocessed data saved to {output_path}")
    return df

if __name__ == '__main__':
    preprocess_pipeline()