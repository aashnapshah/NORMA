#!/usr/bin/env python
"""
Step 1: Process raw eICU data.

Loads raw CSVs, filters labs, removes outliers, merges with patient demographics,
and saves processed data to data/.

Usage:
    python 01_process_eicu.py [--force]
"""
import argparse
import os

from config import DATA_DIR
from utils import load_patient, load_labs, load_diagnosis, merge_labs_patients


def main():
    parser = argparse.ArgumentParser(description="Process raw eICU data")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    args = parser.parse_args()

    processed_path = os.path.join(DATA_DIR, "eicu_processed.pkl")
    diagnosis_path = os.path.join(DATA_DIR, "eicu_diagnosis.pkl")

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

    # Save
    merged.to_pickle(processed_path)
    print(f"Saved processed data to {processed_path}")

    diagnosis.to_pickle(diagnosis_path)
    print(f"Saved diagnosis to {diagnosis_path}")


if __name__ == "__main__":
    main()
