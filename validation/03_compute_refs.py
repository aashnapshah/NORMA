#!/usr/bin/env python
"""
Step 3: Compute reference intervals from baseline data.

For each (patient, analyte), computes:
  - Population reference range (PopRI)
  - GMM setpoint reference range (PerRI)
  - NORMA personalized reference range (for each run_id)

Supports checkpointing for resumable computation.

Usage:
    python 03_compute_refs.py --dataset eicu --run_ids 334f7e21 167f05e8
    python 03_compute_refs.py --dataset eicu  # pop + GMM only
    python 03_compute_refs.py --dataset chs --run_ids 334f7e21
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import DATASETS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import population_reference_range, gmm_setpoint, load_norma_model, norma_reference_range


VAL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_split_df(dataset_name):
    candidates = [
        os.path.join(VAL_DIR, "cache", dataset_name, "split_df.pkl"),
        os.path.join(VAL_DIR, "data", "split_df.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  Loading {p}")
            return pd.read_pickle(p)
    raise FileNotFoundError(f"split_df.pkl not found for {dataset_name}")


def compute_reference_intervals(split_df, run_ids=None, gmm_n_std=2, device="cpu",
                                checkpoint_path=None, save_every=500):
    if run_ids is None:
        run_ids = []
    baseline = split_df[split_df["split"] == "baseline"].copy()

    # Detect column names (support both old and new)
    pid_col = "patient_id" if "patient_id" in baseline.columns else "uniquepid"
    analyte_col = "analyte" if "analyte" in baseline.columns else "lab_code"
    value_col = "value" if "value" in baseline.columns else "labresult"
    time_col = "timestamp" if "timestamp" in baseline.columns else "labresultoffset"
    sex_col = "sex" if "sex" in baseline.columns else "gender"

    agg = baseline.groupby([pid_col, analyte_col]).agg(
        sex=(sex_col, "first"),
        age=("age", "first"),
        n_baseline=(value_col, "count"),
        baseline_mean=(value_col, "mean"),
        baseline_std=(value_col, "std"),
        t_min=(time_col, "min"),
        t_max=(time_col, "max"),
    ).reset_index()

    # Convert time span to days
    ds_time = split_df.attrs.get("time_unit", "minutes")
    if ds_time == "minutes":
        agg["baseline_span_days"] = (agg["t_max"] - agg["t_min"]) / (60 * 24)
    else:
        agg["baseline_span_days"] = agg["t_max"] - agg["t_min"]

    print(f"Computing ref intervals for {len(agg)} patient-analyte pairs "
          f"({agg[pid_col].nunique()} patients, {agg[analyte_col].nunique()} analytes)")

    # Check for existing checkpoint
    done_keys = set()
    partial_rows = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path, keep_default_na=False, na_values=[''])
        partial_rows = existing.to_dict("records")
        done_keys = set(zip(existing["patient_id"], existing["analyte"]))
        print(f"Resuming: {len(done_keys)} done, {len(agg) - len(done_keys)} remaining")

    val_groups = baseline.groupby([pid_col, analyte_col])[value_col].apply(
        lambda s: s.dropna().values
    )
    grp_groups = baseline.groupby([pid_col, analyte_col])

    # Load NORMA models
    models = {}
    for run_id in run_ids:
        model, hparams = load_norma_model(run_id, device=device)
        models[run_id] = (model, hparams)
        print(f"  Loaded {run_id}: {getattr(hparams, 'model', '?')}, {getattr(hparams, 'loss', '?')}")

    # Resolve sex for population_reference_range
    def get_sex_str(sex_val):
        if isinstance(sex_val, str):
            return sex_val[0].upper()  # "Male"→"M", "Female"→"F"
        return "F" if sex_val == 1 else "M"

    new_rows = []
    for _, row in tqdm(agg.iterrows(), total=len(agg), desc="Computing ref intervals"):
        pid = row[pid_col]
        analyte = row[analyte_col]
        if (pid, analyte) in done_keys:
            continue

        vals = val_groups.get((pid, analyte), np.array([]))
        if len(vals) < 2:
            continue

        sex_str = get_sex_str(row["sex"])
        pop_low, pop_high = population_reference_range(analyte, sex_str)

        sp_mean, sp_std = gmm_setpoint(pd.Series(vals))
        gmm_low = sp_mean - gmm_n_std * sp_std
        gmm_high = sp_mean + gmm_n_std * sp_std

        rec = {
            "patient_id": pid, "analyte": analyte,
            "sex": row["sex"], "age": row["age"],
            "n_baseline": row["n_baseline"],
            "baseline_span_days": row["baseline_span_days"],
            "baseline_mean": row["baseline_mean"],
            "baseline_std": row["baseline_std"],
            "pop_low": pop_low, "pop_high": pop_high,
            "gmm_mean": sp_mean, "gmm_std": sp_std,
            "gmm_low": gmm_low, "gmm_high": gmm_high,
        }

        if models:
            grp = grp_groups.get_group((pid, analyte))
            for run_id, (model, hparams) in models.items():
                low, high = norma_reference_range(
                    model, hparams,
                    grp[value_col], grp[time_col],
                    analyte, sex_str, row["age"], device=device,
                )
                rec[f"norma_{run_id}_low"] = low
                rec[f"norma_{run_id}_high"] = high

        new_rows.append(rec)

        if checkpoint_path and len(new_rows) % save_every == 0:
            pd.DataFrame(partial_rows + new_rows).to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint: {len(partial_rows) + len(new_rows)}/{len(agg)} done")

    all_rows = partial_rows + new_rows
    ref_df = pd.DataFrame(all_rows)
    if checkpoint_path:
        ref_df.to_csv(checkpoint_path, index=False)
    print(f"Computed ref intervals for {len(ref_df)} patient-analyte pairs")
    return ref_df


def augment_norma(ref_df, split_df, run_ids, device="cpu",
                  checkpoint_path=None, save_every=500):
    baseline = split_df[split_df["split"] == "baseline"].copy()

    pid_col = "patient_id" if "patient_id" in baseline.columns else "uniquepid"
    analyte_col = "analyte" if "analyte" in baseline.columns else "lab_code"
    value_col = "value" if "value" in baseline.columns else "labresult"
    time_col = "timestamp" if "timestamp" in baseline.columns else "labresultoffset"

    missing_ids = [rid for rid in run_ids if f"norma_{rid}_low" not in ref_df.columns]
    if not missing_ids:
        print("All NORMA run_ids already present.")
        return ref_df

    print(f"Will compute NORMA for: {missing_ids}")
    grp_groups = baseline.groupby([pid_col, analyte_col])

    models = {}
    for run_id in missing_ids:
        model, hparams = load_norma_model(run_id, device=device)
        models[run_id] = (model, hparams)
        print(f"  Loaded {run_id}: {getattr(hparams, 'model', '?')}, {getattr(hparams, 'loss', '?')}")

    for run_id in missing_ids:
        ref_df[f"norma_{run_id}_low"] = np.nan
        ref_df[f"norma_{run_id}_high"] = np.nan

    def get_sex_str(sex_val):
        if isinstance(sex_val, str):
            return sex_val[0].upper()
        return "F" if sex_val == 1 else "M"

    n_computed = 0
    for idx, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Augmenting NORMA"):
        pid = row["patient_id"]
        analyte = row["analyte"]
        try:
            grp = grp_groups.get_group((pid, analyte))
        except KeyError:
            continue

        if len(grp[value_col].dropna()) < 2:
            continue

        sex_str = get_sex_str(row.get("sex", row.get("gender", "M")))
        for run_id, (model, hparams) in models.items():
            low, high = norma_reference_range(
                model, hparams,
                grp[value_col], grp[time_col],
                analyte, sex_str, row["age"], device=device,
            )
            ref_df.at[idx, f"norma_{run_id}_low"] = low
            ref_df.at[idx, f"norma_{run_id}_high"] = high

        n_computed += 1
        if checkpoint_path and n_computed % save_every == 0:
            ref_df.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint: {n_computed}/{len(ref_df)} done")

    if checkpoint_path:
        ref_df.to_csv(checkpoint_path, index=False)

    for run_id in missing_ids:
        n_filled = ref_df[f"norma_{run_id}_low"].notna().sum()
        print(f"  {run_id}: computed {n_filled}/{len(ref_df)} ranges")

    return ref_df


def check_coverage(split_df, ref_df):
    """Check which patient-analyte pairs in split_df are missing from ref_df.
    Returns missing pairs DataFrame and prints per-analyte summary."""
    pid_col = "patient_id" if "patient_id" in split_df.columns else "uniquepid"
    analyte_col = "analyte" if "analyte" in split_df.columns else "lab_code"

    # Fix sodium in split_df
    split_df[analyte_col] = split_df[analyte_col].replace('', 'NA').fillna('NA')

    # All unique pairs in split
    split_pairs = split_df[[pid_col, analyte_col]].drop_duplicates()
    split_pairs = split_pairs.rename(columns={pid_col: 'patient_id', analyte_col: 'analyte'})

    # Fix sodium in ref_df
    ref_df['analyte'] = ref_df['analyte'].replace('', 'NA').fillna('NA')

    ref_pairs = ref_df[['patient_id', 'analyte']].drop_duplicates()

    merged = split_pairs.merge(ref_pairs, on=['patient_id', 'analyte'],
                               how='left', indicator=True)
    missing = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    print(f"\n  Coverage check:")
    print(f"    Split pairs: {len(split_pairs):,}")
    print(f"    Ref pairs:   {len(ref_pairs):,}")
    print(f"    Missing:     {len(missing):,}")

    if len(missing) > 0:
        print(f"\n  Missing per analyte:")
        for analyte in sorted(missing['analyte'].unique()):
            n_miss = (missing['analyte'] == analyte).sum()
            n_total = (split_pairs['analyte'] == analyte).sum()
            print(f"    {analyte}: {n_miss:,} / {n_total:,} missing")

    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--run_ids", type=str, nargs="*", default=["334f7e21", "167f05e8"])
    parser.add_argument("--gmm_n_std", type=float, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--check_only", action="store_true",
                        help="Only check coverage, don't compute")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    cache_dir = os.path.join(VAL_DIR, "cache", ds.name)
    os.makedirs(cache_dir, exist_ok=True)

    split_df = load_split_df(ds.name)

    checkpoint_path = os.path.join(cache_dir, "ref_df_checkpoint.csv")
    output_path = os.path.join(cache_dir, "ref_intervals.csv")

    existing_path = output_path if os.path.exists(output_path) else (
        checkpoint_path if os.path.exists(checkpoint_path) else None
    )

    if existing_path and args.run_ids:
        print(f"Found existing ref intervals at {existing_path}")
        ref_df = pd.read_csv(existing_path, keep_default_na=False, low_memory=False)
        # Fix sodium
        ref_df['analyte'] = ref_df['analyte'].replace('', 'NA').fillna('NA')
        print(f"  {len(ref_df)} patient-analyte pairs")

        # Check coverage against split
        missing = check_coverage(split_df, ref_df)

        if args.check_only:
            return

        # Compute refs for missing pairs that have baseline data
        if len(missing) > 0:
            pid_col = "patient_id" if "patient_id" in split_df.columns else "uniquepid"
            analyte_col = "analyte" if "analyte" in split_df.columns else "lab_code"
            split_df[analyte_col] = split_df[analyte_col].replace('', 'NA').fillna('NA')

            baseline = split_df[split_df["split"] == "baseline"]
            # Filter to only missing pairs that have baseline data
            baseline_pairs = baseline[[pid_col, analyte_col]].drop_duplicates()
            baseline_pairs = baseline_pairs.rename(columns={pid_col: 'patient_id', analyte_col: 'analyte'})
            to_compute = missing.merge(baseline_pairs, on=['patient_id', 'analyte'], how='inner')
            print(f"\n  {len(to_compute):,} missing pairs have baseline data — computing...")

            if len(to_compute) > 0:
                # Filter split_df to just these pairs for computation
                to_compute_set = set(zip(to_compute['patient_id'], to_compute['analyte']))
                mask = split_df.apply(lambda r: (r[pid_col], r[analyte_col]) in to_compute_set, axis=1)
                subset_df = split_df[mask].copy()

                new_refs = compute_reference_intervals(
                    subset_df, run_ids=args.run_ids, gmm_n_std=args.gmm_n_std,
                    device=args.device, save_every=args.save_every,
                )
                ref_df = pd.concat([ref_df, new_refs], ignore_index=True)

        missing = [rid for rid in args.run_ids if f"norma_{rid}_low" not in ref_df.columns]
        if missing:
            ref_df = augment_norma(
                ref_df, split_df, args.run_ids, device=args.device,
                checkpoint_path=checkpoint_path, save_every=args.save_every,
            )
        else:
            print("All requested NORMA run_ids already present.")
    else:
        ref_df = compute_reference_intervals(
            split_df, run_ids=args.run_ids, gmm_n_std=args.gmm_n_std,
            device=args.device, checkpoint_path=checkpoint_path,
            save_every=args.save_every,
        )

    ref_df.to_csv(output_path, index=False)
    print(f"Saved {len(ref_df)} ref intervals to {output_path}")


if __name__ == "__main__":
    main()
