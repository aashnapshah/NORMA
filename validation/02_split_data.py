#!/usr/bin/env python
"""
Step 2: Split each patient-lab group into baseline and index measurements,
filter to patients with sufficient baseline data, and compute per-analyte
summary statistics.

Works on both eICU and CHS via --dataset flag.

Usage:
    python 02_split_data.py --dataset eicu
    python 02_split_data.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from process.config import REFERENCE_INTERVALS

PANELS = {
    "CBC": ["HGB", "HCT", "RBC", "PLT", "MCH", "MCHC", "MCV", "MPV", "RDW", "WBC"],
    "BMP": ["NA", "K", "CL", "CO2", "BUN", "CRE", "GLU", "A1C", "CA"],
    "HFP": ["ALT", "GGT", "AST", "LDH", "PT", "ALP", "TBIL", "DBIL", "ALB", "TP", "CRP"],
    "Lipid": ["TC", "HDL", "LDL", "TGL"],
}
CODE_TO_PANEL = {code: panel for panel, codes in PANELS.items() for code in codes}


# ── Splitting ────────────────────────────────────────────────────────────────

def split_eicu(df, baseline_pct=0.75, min_baseline_count=5, min_baseline_days=0):
    """eICU split: first baseline_pct of measurements chronologically = baseline,
    remainder = index. Filter for min baseline count and span."""
    df = df.sort_values(["patient_id", "analyte", "timestamp"])

    splits = []
    for _, g in df.groupby(["patient_id", "analyte"]):
        n = len(g)
        n_bl = max(int(n * baseline_pct), 1)
        splits.extend(["baseline"] * n_bl + ["index"] * (n - n_bl))
    df["split"] = splits

    # Filter baseline
    bl = df[df["split"] == "baseline"]
    bl_agg = bl.groupby(["patient_id", "analyte"]).agg(
        n_meas=("value", "count"),
        t_min=("timestamp", "min"),
        t_max=("timestamp", "max"),
    )
    # timestamp is in minutes for eICU → convert to days
    bl_agg["span_days"] = (bl_agg["t_max"] - bl_agg["t_min"]) / (60 * 24)

    keep = bl_agg[
        (bl_agg["n_meas"] >= min_baseline_count) &
        (bl_agg["span_days"] >= min_baseline_days)
    ].index

    n_before = len(bl_agg)
    n_after = len(keep)
    print(f"  Baseline filter: {n_after}/{n_before} patient-lab pairs kept "
          f"(≥{min_baseline_count} meas, ≥{min_baseline_days}d span)")

    keep_set = set(keep)
    df = df[df.set_index(["patient_id", "analyte"]).index.map(lambda x: x in keep_set)].copy()
    return df


def split_chs(df, min_baseline_count=5, min_baseline_days=90):
    """CHS split: baseline = outpatient labs before Jan 1 2015 (≥5 measurements,
    ≥90 days apart). Index = first measurement between Jan 1 2015 and Jan 1 2016.

    NOT YET IMPLEMENTED — placeholder for CHS server.
    """
    raise NotImplementedError(
        "CHS splitting must be run on the CHS server. "
        "Import the resulting split_df.pkl into validation/data/."
    )


# ── Loading ──────────────────────────────────────────────────────────────────

def load_split_df(dataset_name):
    """Load split_df.pkl, checking new location first, then legacy."""
    val_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(val_dir, "cache", dataset_name, "split_df.pkl"),
        os.path.join(val_dir, "data", "split_df.pkl"),  # legacy
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  Loading {p}")
            return pd.read_pickle(p)
    raise FileNotFoundError(f"split_df.pkl not found for {dataset_name}")


# ── Stats ────────────────────────────────────────────────────────────────────

def compute_analyte_stats(df, which_split="baseline"):
    """Per-analyte summary stats. Works on standardized columns."""
    sub = df[df["split"] == which_split].copy()
    rows = []
    for analyte in sorted(sub["analyte"].unique()):
        lab_df = sub[sub["analyte"] == analyte]
        vals = lab_df["value"].dropna()
        per_patient = lab_df.groupby("patient_id")
        counts = per_patient.size()
        spans = per_patient["timestamp"].agg(["min", "max"])
        # Assume timestamp already in consistent units per dataset
        span_range = spans["max"] - spans["min"]

        unit, ref_str = "", ""
        if analyte in REFERENCE_INTERVALS:
            ri = REFERENCE_INTERVALS[analyte]
            unit = ri["M"][2]
            m_lo, m_hi = ri["M"][0], ri["M"][1]
            f_lo, f_hi = ri["F"][0], ri["F"][1]
            ref_str = f"{m_lo}–{m_hi}" if (m_lo, m_hi) == (f_lo, f_hi) else f"F:{f_lo}–{f_hi} M:{m_lo}–{m_hi}"

        rows.append({
            "Panel": CODE_TO_PANEL.get(analyte, ""),
            "Analyte": analyte, "Unit": unit, "Ref Interval": ref_str,
            "N Patients": lab_df["patient_id"].nunique(),
            "Value Mean": vals.mean(), "Value Std": vals.std(),
            "Meas/Patient Mean": counts.mean(), "Meas/Patient Std": counts.std(),
            "Span Mean": span_range.mean(), "Span Std": span_range.std(),
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--baseline_pct", type=float, default=0.75)
    parser.add_argument("--min_baseline_count", type=int, default=5)
    parser.add_argument("--min_baseline_days", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(ROOTDIR, "results", ds.name, "raw")
    os.makedirs(out_dir, exist_ok=True)

    # Load
    print(f"Loading {ds.name} data...")
    df = ds.load_processed()
    print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients")

    # Split
    print("Splitting into baseline/index...")
    if ds.name == "eicu":
        split_df = split_eicu(df, args.baseline_pct, args.min_baseline_count, args.min_baseline_days)
    elif ds.name == "chs":
        split_df = split_chs(df, args.min_baseline_count, args.min_baseline_days)
    else:
        raise ValueError(f"No split logic for {ds.name}")

    # Save split
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", ds.name)
    os.makedirs(cache_dir, exist_ok=True)
    split_path = os.path.join(cache_dir, "split_df.pkl")
    split_df.to_pickle(split_path)
    print(f"  Saved split data to {split_path} ({len(split_df):,} rows)")

    # Stats
    print("Computing baseline stats...")
    stats = compute_analyte_stats(split_df, "baseline")
    stats_path = os.path.join(out_dir, "baseline_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"  Saved to {stats_path}")

    total = split_df["patient_id"].nunique()
    n_bl = (split_df["split"] == "baseline").sum()
    n_ix = (split_df["split"] == "index").sum()
    print(f"\n  {total:,} patients, {n_bl:,} baseline / {n_ix:,} index measurements")


if __name__ == "__main__":
    main()
