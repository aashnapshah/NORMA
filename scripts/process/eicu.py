#!/usr/bin/env python
"""Process raw eICU CSVs: filter labs, remove outliers, merge demographics and label
the care setting of every measurement (`setting`, see process/covariates.py).

Outputs (data/eICU/): processed.parquet (raw eICU column names + `setting`),
diagnosis.parquet. The baseline/index split is scripts/02_index_labs.py.
"""

# Run from anywhere: the repo root (process.config, process.covariates) and
# scripts/ (process.covariates) and scripts/lib (datasets: paths and run
# constants) go on sys.path.
import os as _os, sys as _sys
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))   # norma/scripts
_ROOT = _os.path.dirname(_SCRIPTS_DIR)                                          # norma (repo root)
_VAL_DIR = _ROOT
for _p in (_os.path.join(_SCRIPTS_DIR, "lib"), _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import os
import pandas as pd

from datasets import EICU_DATA_DIR, data_dir
from process.covariates import eicu_setting, report_settings

# data/<cohort>/<stage>/: the raw-table pickles this stage caches
CACHE_DIR = os.path.join(data_dir("eicu"), "01_process")

# eICU lab names -> analyte codes
EICU_LAB_MAP = {
    # CBC
    'HGB': ['Hgb'], 'HCT': ['Hct'], 'RBC': ['RBC'],
    'PLT': ['platelets x 1000'], 'MCH': ['MCH'], 'MCHC': ['MCHC'],
    'MCV': ['MCV'], 'MPV': ['MPV'], 'RDW': ['RDW'], 'WBC': ['WBC x 1000'],
    # BMP
    'NA': ['sodium'], 'K': ['potassium'], 'CL': ['chloride'],
    'CO2': ['bicarbonate', 'Total CO2', 'HCO3'], 'BUN': ['BUN'],
    'CRE': ['creatinine'], 'GLU': ['glucose', 'bedside glucose'],
    'A1C': [], 'CA': ['calcium', 'ionized calcium'],
    # HFP
    'ALT': ['ALT (SGPT)'], 'GGT': [], 'AST': ['AST (SGOT)'],
    'LDH': ['LDH'], 'PT': ['PT'], 'ALP': ['alkaline phos.'],
    'TBIL': ['total bilirubin'], 'DBIL': ['direct bilirubin'],
    'ALB': ['albumin'], 'TP': ['total protein'], 'CRP': ['CRP', 'CRP-hs'],
    # Lipids
    'TC': ['total cholesterol'], 'HDL': ['HDL'], 'LDL': ['LDL'],
    'TGL': ['triglycerides'],
}

REVERSE_LAB_MAP = {}
for code, names in EICU_LAB_MAP.items():
    for name in names:
        REVERSE_LAB_MAP[name] = code
ALL_EICU_NAMES = set(REVERSE_LAB_MAP.keys())

def load_patient():
    """Load patient table with pickle caching."""
    cache_path = os.path.join(CACHE_DIR, "patient.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached patient data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading patient.csv (first time, will cache)...")
    patient = pd.read_csv(os.path.join(EICU_DATA_DIR, "patient.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    patient.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return patient


def load_labs(min_tests=3):
    """Filter to 34 target labs, remove IQR outliers, require min_tests per patient-lab."""
    cache_path = os.path.join(CACHE_DIR, "labs_filtered.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached lab data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading lab.csv (first time, will cache)...")
    lab = pd.read_csv(os.path.join(EICU_DATA_DIR, "lab.csv"))

    # Track attrition through processing steps
    processing_attrition = []
    n_raw = len(lab)
    n_raw_patients = lab["patientunitstayid"].nunique()
    processing_attrition.append({
        "Step": "Raw eICU lab table",
        "N Measurements": n_raw,
        "N Stays": n_raw_patients,
    })

    lab = lab[lab["labname"].isin(ALL_EICU_NAMES)].copy()
    processing_attrition.append({
        "Step": "Filter to 34 target lab codes",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    lab["lab_code"] = lab["labname"].map(REVERSE_LAB_MAP)
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab["days_from_admit"] = lab["labresultoffset"] / (60 * 24)
    lab = lab.sort_values(["patientunitstayid", "lab_code", "labresultoffset"])
    lab["days_since_last"] = lab.groupby(["patientunitstayid", "lab_code"])["days_from_admit"].diff()

    # Remove outliers (1.5*IQR per lab)
    outlier_log = []
    cleaned = []
    for lab_code, group in lab.groupby("lab_code"):
        vals = group["labresult"]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (vals >= lower) & (vals <= upper)
        n_removed = (~mask).sum()
        outlier_log.append({
            "lab_code": lab_code, "n_before": len(group),
            "n_removed": n_removed,
            "pct_removed": round(n_removed / len(group) * 100, 2) if len(group) > 0 else 0,
            "iqr_lower": lower, "iqr_upper": upper,
        })
        cleaned.append(group[mask])
    lab = pd.concat(cleaned, ignore_index=True)
    outlier_df = pd.DataFrame(outlier_log)
    print("Outlier removal per lab:")
    print(outlier_df.to_string(index=False))
    processing_attrition.append({
        "Step": "Remove IQR outliers",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    # Keep only patients with enough measurements
    counts = lab.groupby(["patientunitstayid", "lab_code"]).size().reset_index(name="n")
    valid = counts[counts["n"] >= min_tests][["patientunitstayid", "lab_code"]]
    before = len(lab)
    lab = lab.merge(valid, on=["patientunitstayid", "lab_code"], how="inner")
    print(f"\nMin tests filter: {before} -> {len(lab)} rows (min {min_tests} per patient-lab)")
    processing_attrition.append({
        "Step": f"Require ≥{min_tests} tests per stay-lab",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    os.makedirs(CACHE_DIR, exist_ok=True)
    lab.to_pickle(cache_path)
    outlier_df.to_pickle(os.path.join(CACHE_DIR, "outlier_log.pkl"))
    # Save processing attrition for downstream scripts
    pd.DataFrame(processing_attrition).to_csv(
        os.path.join(CACHE_DIR, "processing_attrition.csv"), index=False
    )
    print(f"Cached to {cache_path}")
    return lab


def load_diagnosis():
    """Load diagnosis table with pickle caching."""
    cache_path = os.path.join(CACHE_DIR, "diagnosis.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached diagnosis data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading diagnosis.csv (first time, will cache)...")
    diagnosis = pd.read_csv(os.path.join(EICU_DATA_DIR, "diagnosis.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    diagnosis.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return diagnosis


def merge_labs_patients(lab, patient):
    """Merge labs with patient demographics (age, sex, ethnicity, year)."""
    patient_cols = patient[["patientunitstayid", "uniquepid", "gender", "age",
                            "ethnicity", "hospitaldischargeyear"]].copy()
    patient_cols["age"] = pd.to_numeric(patient_cols["age"], errors="coerce")
    patient_cols.rename(columns={"hospitaldischargeyear": "year"}, inplace=True)
    return lab.merge(patient_cols, on="patientunitstayid", how="left")


def main():
    parser = argparse.ArgumentParser(description="Process raw eICU data")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    args = parser.parse_args()

    out_dir = data_dir("eicu")                 # = EICUDataset.data_dir
    processed_path = os.path.join(out_dir, "processed.parquet")
    diagnosis_path = os.path.join(out_dir, "diagnosis.parquet")

    if not args.force and os.path.exists(processed_path):
        print(f"Processed data already exists at {processed_path}")
        print("Use --force to reprocess from scratch.")
        return

    # Load raw data
    print("Loading raw eICU data...")
    patient = load_patient()
    print(f"  Patient table: {patient.shape}")

    lab = load_labs()
    print(f"  Filtered lab table: {lab.shape}")

    diagnosis = load_diagnosis()
    print(f"  Diagnosis table: {diagnosis.shape}")

    # Merge
    print("Merging labs with patient demographics...")
    merged = merge_labs_patients(lab, patient)
    print(f"  Merged: {merged.shape} ({merged['uniquepid'].nunique()} patients, "
          f"{merged['lab_code'].nunique()} lab codes)")

    # Care setting: labs at offset >= 0 are in the ICU, earlier ones come from
    # the unit's admit source (ED / inpatient).
    merged["setting"] = eicu_setting(merged, patient)
    report_settings(merged["setting"], "eICU")

    # Save
    merged.to_parquet(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    diagnosis.to_parquet(diagnosis_path, index=False)
    print(f"Saved diagnosis to {diagnosis_path}")


if __name__ == "__main__":
    main()
