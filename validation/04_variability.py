#!/usr/bin/env python
"""
Step 8: Compute intra- and inter-individual variability per analyte.

Uses baseline mean and std from ref_intervals.csv:
  - Intra-individual CV: baseline_std / baseline_mean per patient, then summarize
  - Inter-individual CV: std(baseline_means) / mean(baseline_means) across patients
  - Index of Individuality: CV_intra / CV_inter

Usage:
    python 08_variability.py --dataset eicu
    python 08_variability.py --dataset chs
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import DATASETS

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

PANELS = {
    "CBC": ["HGB", "HCT", "RBC", "PLT", "MCH", "MCHC", "MCV", "MPV", "RDW", "WBC"],
    "BMP": ["NA", "K", "CL", "CO2", "BUN", "CRE", "GLU", "A1C", "CA"],
    "HFP": ["ALT", "AST", "ALP", "TBIL", "DBIL", "ALB", "TP"],
    "Lipid": ["TC", "HDL", "LDL", "TGL"],
}
CODE_TO_PANEL = {code: panel for panel, codes in PANELS.items() for code in codes}


def load_ref_intervals(dataset_name):
    val_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(val_dir, "cache", dataset_name, "ref_intervals.csv")
    print(f"  Loading {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    return df


def bootstrap_variability(means, stds, n_bootstrap=1000, seed=42):
    """Bootstrap CV_intra, CV_inter, and II with 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(means)
    cv_intra_vals = stds / means

    boot_intra = np.empty(n_bootstrap)
    boot_inter = np.empty(n_bootstrap)
    boot_ii = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_means = means[idx]
        b_stds = stds[idx]
        b_cv_intra = (b_stds / b_means).mean()
        b_cv_inter = b_means.std() / b_means.mean()
        boot_intra[b] = b_cv_intra
        boot_inter[b] = b_cv_inter
        boot_ii[b] = b_cv_intra / b_cv_inter if b_cv_inter > 0 else np.nan

    def ci(arr):
        return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))

    return {
        'cv_intra': cv_intra_vals.mean(),
        'cv_intra_ci_lower': ci(boot_intra)[0],
        'cv_intra_ci_upper': ci(boot_intra)[1],
        'cv_inter': means.std() / means.mean(),
        'cv_inter_ci_lower': ci(boot_inter)[0],
        'cv_inter_ci_upper': ci(boot_inter)[1],
        'ii': cv_intra_vals.mean() / (means.std() / means.mean()),
        'ii_ci_lower': ci(boot_ii)[0],
        'ii_ci_upper': ci(boot_ii)[1],
    }


def compute_variability(ref_df, n_bootstrap=1000):
    rows = []
    for analyte in sorted(c for c in ref_df['analyte'].unique() if c not in EXCLUDE_ANALYTES):
        lab_df = ref_df[ref_df['analyte'] == analyte].copy()
        lab_df = lab_df[(lab_df["baseline_mean"] > 0) & (lab_df["baseline_std"] >= 0)]
        if len(lab_df) < 5:
            continue

        means = lab_df["baseline_mean"].values
        stds = lab_df["baseline_std"].values
        stats = bootstrap_variability(means, stds, n_bootstrap)

        rows.append({
            "Panel": CODE_TO_PANEL.get(analyte, ""),
            "analyte": analyte,
            "n_patients": len(lab_df),
            "cv_intra": stats['cv_intra'],
            "cv_intra_ci_lower": stats['cv_intra_ci_lower'],
            "cv_intra_ci_upper": stats['cv_intra_ci_upper'],
            "cv_inter": stats['cv_inter'],
            "cv_inter_ci_lower": stats['cv_inter_ci_lower'],
            "cv_inter_ci_upper": stats['cv_inter_ci_upper'],
            "individuality_index": stats['ii'],
            "ii_ci_lower": stats['ii_ci_lower'],
            "ii_ci_upper": stats['ii_ci_upper'],
        })

    result = pd.DataFrame(rows)
    panel_order = {"CBC": 0, "BMP": 1, "HFP": 2, "Lipid": 3, "": 4}
    result["_sort"] = result["Panel"].map(panel_order).fillna(4)
    return result.sort_values(["_sort", "analyte"]).drop(columns="_sort").reset_index(drop=True)


def load_split_pairs(dataset_name):
    """Load patient-analyte pairs from split_df.pkl."""
    val_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(val_dir, "data", "split_df.pkl")
    df = pd.read_pickle(path)
    # Fix sodium
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    pairs = df[['patient_id', 'analyte']].drop_duplicates()
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap replicates")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    out_dir = os.path.join(ROOTDIR, "results", ds.name, "raw")
    os.makedirs(out_dir, exist_ok=True)

    ref_df = load_ref_intervals(ds.name)
    print(f"  {len(ref_df)} patient-analyte pairs in ref_intervals")

    # Filter to only patient-analyte pairs present in split_df
    split_pairs = load_split_pairs(ds.name)
    ref_df = ref_df.merge(split_pairs, on=['patient_id', 'analyte'], how='inner')
    print(f"  {len(ref_df)} pairs after filtering to split")
    print(f"  Bootstrap replicates: {args.bootstrap}")

    var_df = compute_variability(ref_df, n_bootstrap=args.bootstrap)

    out_path = os.path.join(out_dir, "variability.csv")
    var_df.to_csv(out_path, index=False)
    print(f"  Saved {len(var_df)} analytes to {out_path}")


if __name__ == "__main__":
    main()
