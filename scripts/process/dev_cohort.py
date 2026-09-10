"""Build a NORMA development cohort (EHRSHOT or MIMIC-IV) for the validation pipeline:
data/<source>/processed.parquet with the columns every other cohort has
(patient_id, analyte, timestamp [days since the patient's first lab], value, sex, age,
setting), restricted to NORMA's held-out TEST sequences.

    python process/dev_cohort.py --source ehrshot [--force]
    python process/dev_cohort.py --source mimiciv [--force]
    python process/dev_cohort.py --source ehrshot --filter_existing [--dry_run]

Source: data/processed/<SRC>_processed_df.csv (process/process_data.py output):
    subject_id, sex, test_name, time, age, numeric_value
Test membership: (pid, code) pairs with split == 'test' in model/logs/<run>/predictions_combined.csv
    (patient ids are unique per source; model/predictions/pid_source.csv maps pid -> source).

--filter_existing applies the same test-sequence filter, in place, to artifacts
already computed from an older all-patient cohort (originals kept as
<file>.bak-allpatients): data/<source>/{processed,index_labs}.parquet and every
parquet/csv under data/<source>/{02_index_labs,04_refs,05_forecasting}/.  Every
method's output is keyed per (patient_id, analyte) and a test sequence's prediction
never depended on that sequence being in the cohort, so nothing has to be re-fit.
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
import glob
import os
import shutil

import numpy as np
import pandas as pd

from datasets import NORMA_RUN_ID
from process.covariates import ehrshot_setting, mimic_setting, report_settings

REPO = _ROOT
PROCESSED_DIR = os.path.join(os.path.dirname(REPO), "data", "processed")
SOURCE_FILES = {"ehrshot": "EHRSHOT_processed_df.csv", "mimiciv": "MIMIC-IV_processed_df.csv"}
SETTING_FN = {"ehrshot": ehrshot_setting, "mimiciv": mimic_setting}
PID_SOURCE = os.path.join(REPO, "model", "predictions", "pid_source.csv")


def test_sequences(source, run_id):
    """{(pid, analyte)} of NORMA's held-out TEST sequences from this source.

    NORMA's split is per patient-analyte sequence, so a patient with a test
    sodium series may have a train creatinine series. Selecting *patients* with
    any test sequence (the previous behaviour) pulled in those train/val
    sequences — on MIMIC-IV 633k of the 1.42M dev sequences were NORMA train/val
    and a further 523k were never in NORMA at all — so NORMA (and Cohen, which is
    fit on NORMA's EHRSHOT train split) were scored on data they had trained on.
    Aashna, 2026-08-28: "use the measurements that were used to develop NORMA".
    """
    pred = os.path.join(REPO, "model", "logs", run_id, "predictions_combined.csv")
    p = pd.read_csv(pred, usecols=["pid", "code", "split"], keep_default_na=False, na_values=[""])
    p = p[p["split"].astype(str).str.lower() == "test"].drop_duplicates(["pid", "code"])
    src = pd.read_csv(PID_SOURCE).set_index("pid")["source"]
    p = p[p["pid"].map(src) == source]
    code = p["code"].astype(str).replace("", "NA")      # sodium is literally "NA"
    return set(zip(p["pid"].astype(int), code))


FILTER_DIRS = ["data/{ds}", "data/{ds}/04_refs", "data/{ds}/05_forecasting", "data/{ds}/02_index_labs"]


def _filter(df, keep):
    a = df["analyte"].astype(str).replace("", "NA")
    key = pd.MultiIndex.from_arrays([df["patient_id"].astype(int), a])
    return df[key.isin(keep)]


def filter_existing(source, keep, dry_run=False):
    """Restrict already-computed artifacts of `source` to the test sequences `keep`, in place."""
    for d in FILTER_DIRS:
        d = os.path.join(_VAL_DIR, d.format(ds=source))
        for path in sorted(glob.glob(os.path.join(d, "*.parquet")) + glob.glob(os.path.join(d, "*.csv"))):
            is_pq = path.endswith(".parquet")
            df = pd.read_parquet(path) if is_pq else pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
            if not {"patient_id", "analyte"} <= set(df.columns):
                print(f"  skip {os.path.relpath(path, _VAL_DIR)} (no patient_id/analyte)"); continue
            sub = _filter(df, keep)
            print(f"  {os.path.relpath(path, _VAL_DIR)}: {len(df):,} -> {len(sub):,} rows "
                  f"({df['patient_id'].nunique():,} -> {sub['patient_id'].nunique():,} patients)")
            if dry_run:
                continue
            bak = path + ".bak-allpatients"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            if is_pq:
                sub.to_parquet(path, index=False)
            else:
                sub.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_FILES))
    ap.add_argument("--run_id", default=NORMA_RUN_ID, help="run whose test split defines the cohort")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--filter_existing", action="store_true",
                    help="restrict existing artifacts to the test sequences instead of building the cohort")
    ap.add_argument("--dry_run", action="store_true", help="with --filter_existing: report row counts only")
    args = ap.parse_args()

    if args.filter_existing:
        keep = test_sequences(args.source, args.run_id)
        print(f"{args.source}: {len(keep):,} NORMA test sequences, {len({k[0] for k in keep}):,} patients")
        return filter_existing(args.source, keep, dry_run=args.dry_run)

    out_dir = os.path.join(_VAL_DIR, "data", args.source)
    out_path = os.path.join(out_dir, "processed.parquet")
    if os.path.exists(out_path) and not args.force:
        print(f"  {out_path} exists; use --force to rebuild")
        return
    os.makedirs(out_dir, exist_ok=True)

    keep = test_sequences(args.source, args.run_id)
    print(f"  {args.source}: {len(keep):,} NORMA test sequences "
          f"({len({k[0] for k in keep}):,} patients)")

    src_path = os.path.join(PROCESSED_DIR, SOURCE_FILES[args.source])
    usecols = ["subject_id", "sex", "test_name", "time", "age", "numeric_value"]
    parts = []
    for chunk in pd.read_csv(src_path, usecols=usecols, chunksize=2_000_000,
                             keep_default_na=False, na_values=[""]):
        key = pd.MultiIndex.from_arrays([chunk["subject_id"].astype(int),
                                         chunk["test_name"].astype(str).replace("", "NA")])
        chunk = chunk[key.isin(keep)]
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"  {len(df):,} measurements read for those patients")

    df = df.rename(columns={"subject_id": "patient_id", "test_name": "analyte", "numeric_value": "value"})
    df["analyte"] = df["analyte"].astype(str).replace("", "NA")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    t = pd.to_datetime(df["time"], errors="coerce")
    df = df[t.notna()].copy(); t = t[t.notna()]
    t0 = t.groupby(df["patient_id"]).transform("min")
    df["timestamp"] = (t - t0).dt.total_seconds() / 86400.0        # days since the patient's first lab
    # Care setting from the source admission/visit tables (keyed on subject_id, time, test_name)
    df["setting"] = SETTING_FN[args.source](pd.DataFrame({
        "subject_id": df["patient_id"].values, "time": t.values, "test_name": df["analyte"].values}))
    report_settings(df["setting"], args.source)
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce").fillna(0).astype(int)   # 0 = M, 1 = F
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = (df[["patient_id", "analyte", "timestamp", "value", "sex", "age", "setting"]]
          .sort_values(["patient_id", "analyte", "timestamp"], kind="stable")
          .drop_duplicates(["patient_id", "analyte", "timestamp", "value"])
          .reset_index(drop=True))
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}: {len(df):,} rows, {df['patient_id'].nunique():,} patients, "
          f"{df['analyte'].nunique()} analytes")


if __name__ == "__main__":
    main()
