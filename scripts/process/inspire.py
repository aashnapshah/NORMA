#!/usr/bin/env python
"""Process raw INSPIRE CSVs: filter labs, remove outliers, merge demographics and label
the care setting of every measurement (`setting`, see process/covariates.py).

Outputs (INSPIREDataset.data_dir = data/inspire/25-75/): processed.parquet (raw
INSPIRE column names + `setting`), diagnosis.parquet. The baseline/index split is
scripts/02_index_labs.py.
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
import numpy as np

from datasets import EXCLUDE_LABS, INSPIRE_DATA_DIR, data_dir
from process.covariates import inspire_setting, report_settings

# data/<cohort>/<stage>/: the raw-table pickles this stage caches
CACHE_DIR = os.path.join(data_dir("inspire"), "01_process")

# ============================================================
# Paths
# ============================================================

INSPIRE_RAW_DIR = INSPIRE_DATA_DIR

# ============================================================
# INSPIRE lab name → standard lab code mapping
# ============================================================

INSPIRE_LAB_MAP = {
    # CBC
    "hb": "HGB",
    "hct": "HCT",
    "platelet": "PLT",
    "wbc": "WBC",
    # BMP
    "sodium": "NA",
    "potassium": "K",
    "chloride": "CL",
    "hco3": "CO2",
    "bun": "BUN",
    "creatinine": "CRE",
    "glucose": "GLU",
    "hba1c": "A1C",
    "calcium": "CA",
    # HFP
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "total_bilirubin": "TBIL",
    "albumin": "ALB",
    "total_protein": "TP",
    "crp": "CRP",
}

ALL_INSPIRE_NAMES = set(INSPIRE_LAB_MAP.keys())


def load_operations():
    """Load operations table (patient demographics + outcomes) with pickle caching."""
    cache_path = os.path.join(CACHE_DIR, "inspire_operations.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached operations from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading operations.csv (first time, will cache)...")
    ops = pd.read_csv(os.path.join(INSPIRE_RAW_DIR, "operations.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    ops.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return ops


def load_labs(min_tests=3):
    """Filter to target labs, remove IQR outliers, require min_tests per patient-lab."""
    cache_path = os.path.join(CACHE_DIR, "inspire_labs_filtered.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached lab data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading labs.csv (first time, will cache)...")
    lab = pd.read_csv(os.path.join(INSPIRE_RAW_DIR, "labs.csv"))

    # Track attrition
    processing_attrition = []
    n_raw = len(lab)
    n_raw_patients = lab["subject_id"].nunique()
    processing_attrition.append({
        "Step": "Raw INSPIRE lab table",
        "N Measurements": n_raw,
        "N Patients": n_raw_patients,
    })

    # Filter to target labs
    lab = lab[lab["item_name"].isin(ALL_INSPIRE_NAMES)].copy()
    lab["lab_code"] = lab["item_name"].map(INSPIRE_LAB_MAP)
    processing_attrition.append({
        "Step": "Filter to target lab codes",
        "N Measurements": len(lab),
        "N Patients": lab["subject_id"].nunique(),
    })

    lab["value"] = pd.to_numeric(lab["value"], errors="coerce")
    lab = lab.dropna(subset=["value"])

    # Deduplicate: same patient, lab, time → keep first
    before_dedup = len(lab)
    lab = lab.drop_duplicates(subset=["subject_id", "lab_code", "chart_time"], keep="first")
    print(f"Deduplication: {before_dedup:,} -> {len(lab):,} rows ({before_dedup - len(lab):,} removed)")
    processing_attrition.append({
        "Step": "Deduplicate same patient-lab-time",
        "N Measurements": len(lab),
        "N Patients": lab["subject_id"].nunique(),
    })

    lab["days_from_admit"] = lab["chart_time"] / (60 * 24)
    lab = lab.sort_values(["subject_id", "lab_code", "chart_time"])
    lab["days_since_last"] = lab.groupby(["subject_id", "lab_code"])["days_from_admit"].diff()

    # Remove outliers (1.5*IQR per lab)
    outlier_log = []
    cleaned = []
    for lab_code, group in lab.groupby("lab_code"):
        vals = group["value"]
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
        "N Patients": lab["subject_id"].nunique(),
    })

    # Keep only patients with enough measurements
    counts = lab.groupby(["subject_id", "lab_code"]).size().reset_index(name="n")
    valid = counts[counts["n"] >= min_tests][["subject_id", "lab_code"]]
    before = len(lab)
    lab = lab.merge(valid, on=["subject_id", "lab_code"], how="inner")
    print(f"\nMin tests filter: {before} -> {len(lab)} rows (min {min_tests} per patient-lab)")
    processing_attrition.append({
        "Step": f"Require ≥{min_tests} tests per patient-lab",
        "N Measurements": len(lab),
        "N Patients": lab["subject_id"].nunique(),
    })

    os.makedirs(CACHE_DIR, exist_ok=True)
    lab.to_pickle(cache_path)
    outlier_df.to_pickle(os.path.join(CACHE_DIR, "inspire_outlier_log.pkl"))
    pd.DataFrame(processing_attrition).to_csv(
        os.path.join(CACHE_DIR, "inspire_processing_attrition.csv"), index=False
    )
    print(f"Cached to {cache_path}")
    return lab


def load_diagnosis():
    """Load diagnosis table with pickle caching."""
    cache_path = os.path.join(CACHE_DIR, "inspire_diagnosis.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached diagnosis from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading diagnosis.csv (first time, will cache)...")
    diagnosis = pd.read_csv(os.path.join(INSPIRE_RAW_DIR, "diagnosis.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    diagnosis.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return diagnosis


def merge_labs_operations(lab, ops):
    """Merge labs with patient demographics from operations table.

    A patient may have multiple operations. We take the first operation's
    demographics (age, sex, race) per subject_id.
    """
    # Deduplicate: one row per patient (first operation)
    ops_sorted = ops.sort_values(["subject_id", "opdate"])
    patient_demo = ops_sorted.drop_duplicates(subset=["subject_id"], keep="first")

    patient_cols = patient_demo[[
        "subject_id", "sex", "age", "race",
        "admission_time", "discharge_time",
        "icuin_time", "icuout_time",
        "inhosp_death_time", "allcause_death_time",
    ]].copy()

    patient_cols["age"] = pd.to_numeric(patient_cols["age"], errors="coerce")

    return lab.merge(patient_cols, on="subject_id", how="left")


def attach_setting(out_dir, force=False):
    """Add the `setting` column to an existing index_labs.parquet IN PLACE, without
    recutting the baseline/index split (rows, order and every other column stay
    untouched; a .bak-setting backup is written first).

    index_labs files built before 2026-08-28 lack the column, which the NORMA
    covariate arms (q_set, q_age_set, ...) need at inference (04_refs.py).
    Same labeller as the full run: lib/settings.inspire_setting over the ops
    table's admission / ICU windows.  (The dev cohorts cannot be patched post hoc:
    their index_labs keep only the re-anchored day clock, so the admission windows
    cannot be joined; rerun process/dev_cohort.py --source {mimiciv,ehrshot} + 02_index_labs.)
    """
    import shutil
    path = os.path.join(out_dir, "index_labs.parquet")
    df = pd.read_parquet(path)
    if "setting" in df.columns and not force:
        print(f"{path} already has a setting column; use --force to recompute")
        return
    print(f"{path}: {len(df):,} rows")
    df["setting"] = inspire_setting(df, load_operations())
    report_settings(df["setting"], "INSPIRE")
    bak = path + ".bak-setting"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup: {bak}")
    df.to_parquet(path, index=False)
    print(f"  wrote {path} (setting column added, all other columns unchanged)")


def main():
    parser = argparse.ArgumentParser(description="Process raw INSPIRE data")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--attach_setting", action="store_true",
                        help="only add the `setting` column to the existing index_labs.parquet "
                             "(no reprocessing, split untouched)")
    args = parser.parse_args()

    out_dir = data_dir("inspire")                          # = INSPIREDataset.data_dir
    os.makedirs(out_dir, exist_ok=True)
    processed_path = os.path.join(out_dir, "processed.parquet")
    diagnosis_path = os.path.join(out_dir, "diagnosis.parquet")

    if args.attach_setting:
        attach_setting(out_dir, force=args.force)
        return

    if not args.force and os.path.exists(processed_path):
        print(f"Processed data already exists at {processed_path}")
        print("Use --force to reprocess from scratch.")
        return

    # Load raw data
    print("Loading raw INSPIRE data...")
    ops = load_operations()
    print(f"  Operations table: {ops.shape}")

    lab = load_labs()
    print(f"  Filtered lab table: {lab.shape}")

    diagnosis = load_diagnosis()
    print(f"  Diagnosis table: {diagnosis.shape}")

    # Merge
    print("Merging labs with patient demographics...")
    merged = merge_labs_operations(lab, ops)
    print(f"  Merged: {merged.shape} ({merged['subject_id'].nunique()} patients, "
          f"{merged['lab_code'].nunique()} lab codes)")

    # Care setting from the admission / ICU windows of *all* of the patient's
    # operations (merge_labs_operations keeps only the first one).
    merged["setting"] = inspire_setting(merged.rename(columns={"subject_id": "patient_id"}), ops)
    report_settings(merged["setting"], "INSPIRE")

    # Save
    merged.to_parquet(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    diagnosis.to_parquet(diagnosis_path, index=False)
    print(f"Saved diagnosis to {diagnosis_path}")


if __name__ == "__main__":
    main()
