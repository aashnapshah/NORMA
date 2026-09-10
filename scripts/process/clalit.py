#!/usr/bin/env python
"""Process raw Clalit parquet chunks into processed.parquet, diagnosis, and ref_intervals
(one directory per chunk).

processed.parquet keeps the server's baseline-window flag `meets_2015_gap90_tests5`
(scripts/02_index_labs.py --dataset chs turns it into `split`) and carries
`setting` (outpatient / inpatient from the per-lab `inpatient` flag; process/covariates.py).

--legacy_tables DIR  runs here, not in Clalit: converts the April-2026 CHS tables
(data/clalit/legacy_2026-04/README.md) into the current per-stage result files under
results/raw/chs/ so the pooled figures show CHS until the pipeline is rerun there.
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
import re

import sys
import pandas as pd
import numpy as np


from datasets import result_path, results_dir  # noqa: E402
from process.covariates import chs_setting, report_settings  # noqa: E402

# ============================================================
# Paths — {i} is replaced with chunk index at runtime
# ============================================================
VAL_DIR = _VAL_DIR
SANDBOX = os.path.join(VAL_DIR, "data", "clalit", "sandbox")

# --- Real data paths ---
DATA_DIR = r"\\10.100.117.220\Projects$\R01-MainResearch\R01-Aashna\data\processed"
INTERVAL_DIR = r"\\10.100.117.220\Projects$\R01-MainResearch\R01-Aashna\results\intervals\death\pre"
# The same window the split uses: 02_index_labs assigns baseline/index from
# meets_2015_gap90_tests5, so the server intervals must be fitted on that
# window too or the reference intervals and the history disagree.
BASELINE_FLAG = "meets_2015_gap90_tests5"
FILTER = BASELINE_FLAG
NORMA_DIR = r"\\10.100.117.220\Projects$\R01-MainResearch\R01-Aashna\NORMA_v2\model\logs\q_age_set\edits"

REAL_PATHS = {
    "labs":     os.path.join(DATA_DIR, "chunk_{i}", "labs_{i}_flag.parquet"),
    "outcomes": os.path.join(DATA_DIR, "chunk_{i}", "outcomes.parquet"),
    "bayes":    os.path.join(INTERVAL_DIR, "bayes", FILTER, "bayes_{i}.parquet"),
    "setpoint": os.path.join(INTERVAL_DIR, "setpoint", FILTER, "setpoint_{i}.parquet"),
    "norma":    os.path.join(NORMA_DIR, "norma_{i}.parquet"),
}

# --- Sandbox paths (for testing with fake data) ---
SANDBOX_PATHS = {
    "labs":     os.path.join(SANDBOX, "chunk_{i}", "labs_0_flagged.parquet"),
    "outcomes": os.path.join(SANDBOX, "chunk_{i}", "outcomes.parquet"),
    "bayes":    os.path.join(SANDBOX, "chunk_{i}", "bayes_0.parquet"),
    "setpoint": os.path.join(SANDBOX, "chunk_{i}", "setpoint_0.parquet"),
    "norma":    os.path.join(SANDBOX, "chunk_{i}", "norma_0.parquet"),
}



def _dist_row(vals):
    """Summary stats (n, mean, median, p95, max) for before/after comparison."""
    if len(vals) == 0:
        return (0, 0, 0, 0, 0)
    return (len(vals), vals.mean(), vals.median(), vals.quantile(0.95), vals.max())


def fix_values(df):
    """Apply unit corrections (RBC, A1C) and outlier filters (WBC, ALB)."""
    fixes = []  # collect (analyte, before, after) tuples

    # RBC: reported in 10³/µL, convert to 10⁶/µL (M/µL)
    # Skip if already converted (median < 10 means already in M/µL)
    rbc_mask = df["analyte"] == "RBC"
    if rbc_mask.any():
        before = _dist_row(df.loc[rbc_mask, "value"])
        if df.loc[rbc_mask, "value"].median() > 10:
            df.loc[rbc_mask, "value"] = df.loc[rbc_mask, "value"] / 1000
        fixes.append(("RBC", before, _dist_row(df.loc[rbc_mask, "value"])))

    # A1C: standardize to NGSP (%). Some chunks report in IFCC (mmol/mol).
    # Detect IFCC by p95 > 20. Convert: NGSP = IFCC / 10.929 + 2.15
    a1c_mask = df["analyte"] == "A1C"
    if a1c_mask.any():
        before = _dist_row(df.loc[a1c_mask, "value"])
        if df.loc[a1c_mask, "value"].quantile(0.95) > 20:
            df.loc[a1c_mask, "value"] = df.loc[a1c_mask, "value"] / 10.929 + 2.15
        df = df[~((df["analyte"] == "A1C") & (df["value"] > 20))]
        fixes.append(("A1C", before, _dist_row(df.loc[df["analyte"] == "A1C", "value"])))

    # WBC: drop values > 500 K/µL
    wbc_mask = df["analyte"] == "WBC"
    if wbc_mask.any():
        before = _dist_row(df.loc[wbc_mask, "value"])
        df = df[~((df["analyte"] == "WBC") & (df["value"] > 500))]
        fixes.append(("WBC", before, _dist_row(df.loc[df["analyte"] == "WBC", "value"])))

    # ALB: drop values > 10 g/dL
    alb_mask = df["analyte"] == "ALB"
    if alb_mask.any():
        before = _dist_row(df.loc[alb_mask, "value"])
        df = df[~((df["analyte"] == "ALB") & (df["value"] > 10))]
        fixes.append(("ALB", before, _dist_row(df.loc[df["analyte"] == "ALB", "value"])))

    # Print summary table per analyte
    for name, (bn, bm, bmed, bp, bx), (an, am, amed, ap, ax) in fixes:
        print(f"  {name}:")
        print(f"    {'':>8} {'n':>10} {'mean':>10} {'median':>10} {'p95':>10} {'max':>12}")
        print(f"    {'before':>8} {bn:>10,} {bm:>10.2f} {bmed:>10.2f} {bp:>10.2f} {bx:>12.2f}")
        print(f"    {'after':>8} {an:>10,} {am:>10.2f} {amed:>10.2f} {ap:>10.2f} {ax:>12.2f}")

    return df


def process_labs(labs):
    """Standardize columns, apply value fixes and label the care setting.
    The baseline/index cut is 02_index_labs (flag column kept for it)."""
    df = labs.rename(columns={
        "code": "analyte",
        "time_stamp": "timestamp",
        "numeric_value": "value",
        "age_at_test": "age",
    })

    # Keep standard columns + extras needed downstream (the baseline-window flag
    # is what 02_index_labs uses to assign `split`)
    keep = [
        "patient_id", "analyte", "value", "timestamp", "gender", "age",
        BASELINE_FLAG,
        "dob", "death_date", "membership_end", "inpatient", "alive_2015",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Standardize analyte codes
    df["analyte"] = df["analyte"].replace({"TG": "TGL"})

    df = fix_values(df).reset_index(drop=True)

    # Care setting (CHS has no ED/ICU signal: inpatient flag -> inpatient, else outpatient)
    if "inpatient" in df.columns:
        df["setting"] = chs_setting(df)
        report_settings(df["setting"], "CHS")

    return df


def process_outcomes(outcomes, labs):
    """Merge death_date, membership_end, and dob from labs into outcomes."""
    patient_info = (
        labs.groupby("patient_id")[["death_date", "membership_end", "dob"]]
        .first()
        .reset_index()
    )
    diagnosis = outcomes.merge(patient_info, on="patient_id", how="left")
    return diagnosis


def _build_pop_table():
    """Build population RI lookup table from REFERENCE_INTERVALS config."""
    from process.config import REFERENCE_INTERVALS
    rows = []
    for analyte, sexes in REFERENCE_INTERVALS.items():
        for sex, (lo, hi, _unit) in sexes.items():
            rows.append({"analyte": analyte, "sex": sex, "ri_low": lo, "ri_high": hi})
    return pd.DataFrame(rows)


def build_ref_intervals_long(bayes, setpoint, norma, run_id):
    """Build long-format ref_intervals (base/pop/per/norma) via vectorized merges."""
    print("  Building ref_intervals...")

    # Shared patient-analyte info from bayes
    df = bayes.rename(columns={
        "code": "analyte",
        "num_measurements": "n_bl",
        "time_period_days": "t_span",
    })
    df["analyte"] = df["analyte"].replace({"TG": "TGL"})
    shared_cols = ["patient_id", "analyte", "sex", "age", "n_bl", "t_span"]
    shared = df[shared_cols]

    # Base method
    print("    base...")
    base_df = shared.copy()
    base_df["method"] = "base"
    base_df["ri_mean"] = df["obs_mean"].values
    base_df["ri_std"] = np.sqrt(df["obs_var"].values)
    base_df["ri_low"] = np.nan
    base_df["ri_high"] = np.nan

    # Pop method (merge lookup table on analyte + sex)
    print("    pop...")
    pop_table = _build_pop_table()
    pop_df = shared.merge(pop_table, on=["analyte", "sex"], how="left")
    pop_df["method"] = "pop"
    pop_df["ri_mean"] = np.nan
    pop_df["ri_std"] = np.nan

    parts = [base_df, pop_df]

    # Per method (merge setpoint onto shared)
    if setpoint is not None:
        print("    per...")
        sp = setpoint.rename(columns={"code": "analyte"})
        sp["analyte"] = sp["analyte"].replace({"TG": "TGL"})
        per_df = shared.merge(
            sp[["patient_id", "analyte", "setpoint_mean", "setpoint_std"]],
            on=["patient_id", "analyte"], how="inner",
        )
        per_df["method"] = "per"
        per_df["ri_mean"] = per_df["setpoint_mean"]
        per_df["ri_std"] = per_df["setpoint_std"]
        per_df["ri_low"] = per_df["setpoint_mean"] - 2 * per_df["setpoint_std"]
        per_df["ri_high"] = per_df["setpoint_mean"] + 2 * per_df["setpoint_std"]
        per_df = per_df.drop(columns=["setpoint_mean", "setpoint_std"])
        parts.append(per_df)

    # Norma method (merge norma onto shared)
    # Norma cols: pid, cid, code, x_next, t_next, s_next, state, mu, std
    if norma is not None and run_id:
        print(f"    norma_{run_id}...")
        norma = norma[norma["state"] == True].copy()
        n = norma.rename(columns={"pid": "patient_id", "code": "analyte"})
        n["analyte"] = n["analyte"].replace({"TG": "TGL"})
        norma_df = shared.merge(
            n[["patient_id", "analyte", "mu", "std"]],
            on=["patient_id", "analyte"], how="inner",
        )
        norma_df["method"] = f"norma_{run_id}"
        norma_df["ri_mean"] = np.nan
        norma_df["ri_std"] = np.nan
        norma_df["ri_low"] = norma_df["mu"] - 2 * norma_df["std"]
        norma_df["ri_high"] = norma_df["mu"] + 2 * norma_df["std"]
        norma_df = norma_df.drop(columns=["mu", "std"])
        parts.append(norma_df)

    ref_df = pd.concat(parts, ignore_index=True)

    # RBC: bayes/setpoint source data is in 10³/µL; convert base and per
    # to 10⁶/µL (M/µL) to match pop RI and index_labs values.
    rbc_fix = (ref_df["analyte"] == "RBC") & ref_df["method"].isin(["base", "per"])
    for col in ["ri_mean", "ri_std", "ri_low", "ri_high"]:
        ref_df.loc[rbc_fix, col] = ref_df.loc[rbc_fix, col] / 1000

    return ref_df


def resolve_path(template, chunk_idx):
    """Replace {i} placeholder in path template with chunk index."""
    return template.format(i=chunk_idx)


def get_paths(sandbox=True):
    return SANDBOX_PATHS if sandbox else REAL_PATHS


def process_chunk(chunk_idx, out_dir, paths, run_id, force=False,
                   labs_only=False, analytes=None, refs=True):
    """Process one chunk. If analytes is set, patch only those into the existing processed.parquet."""
    os.makedirs(out_dir, exist_ok=True)
    processed_path = os.path.join(out_dir, "processed.parquet")
    diagnosis_path = os.path.join(out_dir, "diagnosis.parquet")
    ref_path = os.path.join(out_dir, "ref_intervals.parquet")

    if not force and os.path.exists(processed_path):
        print(f"  Already processed: {processed_path} (use --force to redo)")
        return

    # Resolve source paths (each can point to a different location)
    labs_path = resolve_path(paths["labs"], chunk_idx)

    labs = pd.read_parquet(labs_path)
    print(f"  Labs: {len(labs):,} rows, {labs['patient_id'].nunique():,} patients")

    # Filter raw labs to requested analytes before processing
    if analytes is not None:
        col = "code" if "code" in labs.columns else "analyte"
        labs = labs[labs[col].isin(analytes)]
        print(f"  Filtered to {', '.join(analytes)}: {len(labs):,} rows")

    # Process labs → processed.parquet
    processed = process_labs(labs)

    if analytes is not None:
        # Patch mode: replace only the specified analytes in the existing processed.parquet
        new_rows = processed[processed["analyte"].isin(analytes)]
        if os.path.exists(processed_path):
            existing = pd.read_parquet(processed_path)
            existing = existing[~existing["analyte"].isin(analytes)]
            processed = pd.concat([existing, new_rows], ignore_index=True)
            print(f"  Patched {len(analytes)} analyte(s): {', '.join(analytes)}")
        else:
            processed = new_rows

    n_flag = int(processed[BASELINE_FLAG].fillna(False).astype(bool).sum())
    print(f"  Processed: {len(processed):,} rows ({n_flag:,} in the baseline window), "
          f"{processed['patient_id'].nunique():,} patients")
    processed.to_parquet(processed_path, index=False)
    print(f"  Saved {processed_path}  (run 02_index_labs.py --dataset chs for index_labs)")

    if labs_only:
        return

    outcomes_path = resolve_path(paths["outcomes"], chunk_idx)
    outcomes = pd.read_parquet(outcomes_path)
    print(f"  Outcomes: {len(outcomes):,} patients")

    diagnosis = process_outcomes(outcomes, labs)
    diagnosis.to_parquet(diagnosis_path, index=False)
    print(f"  Saved {diagnosis_path}")

    if not refs:
        print("  --no_refs: ref_intervals left to 04_refs/{norma,baselines}.py")
        return

    # Build long-format ref_intervals
    print("  Loading ref interval sources...")
    bayes_path = resolve_path(paths["bayes"], chunk_idx)
    setpoint_path = resolve_path(paths["setpoint"], chunk_idx)
    norma_path = resolve_path(paths["norma"], chunk_idx)

    bayes = pd.read_parquet(bayes_path) if os.path.exists(bayes_path) else None
    if bayes is not None:
        print(f"    bayes: {len(bayes):,} rows")
    setpoint = pd.read_parquet(setpoint_path) if os.path.exists(setpoint_path) else None
    if setpoint is not None:
        print(f"    setpoint: {len(setpoint):,} rows")
    norma = pd.read_parquet(norma_path) if os.path.exists(norma_path) else None
    if norma is not None:
        print(f"    norma: {len(norma):,} rows")

    if bayes is not None:
        ref_df = build_ref_intervals_long(bayes, setpoint, norma, run_id)
        n_pairs = ref_df[["patient_id", "analyte"]].drop_duplicates().shape[0]
        methods = ref_df["method"].unique().tolist()
        print(f"  Ref intervals: {len(ref_df):,} rows, {n_pairs} patient-analyte pairs, methods={methods}")
        ref_df.to_parquet(ref_path, index=False)
        print(f"  Saved {ref_path}")
    else:
        print("  No bayes_0.parquet found, skipping ref_intervals")


# ============================================================
# Legacy import: the April-2026 CHS tables -> current result files
# ============================================================
LEGACY_METHODS = ["PopRI", "PerRI", "NORMA"]
LEGACY_OUTCOMES = ["mortality", "ckd", "t2d", "anemia_unspecified"]


def _read_table(path):
    """keep_default_na=False: the sodium analyte is spelled NA."""
    return pd.read_csv(path, keep_default_na=False)


def _num(text):
    """'1,234' -> 1234.0; '---' -> nan."""
    text = str(text).strip().replace(",", "")
    return float(text) if text not in ("", "---", "nan") else np.nan


def _value_ci(text):
    """'8.6 [8.5, 8.6]' -> (8.6, 8.5, 8.6); anything else -> nans."""
    m = re.match(r"\s*([-\d.]+)\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", str(text))
    return tuple(float(g) for g in m.groups()) if m else (np.nan, np.nan, np.nan)


def _mean_sd(text):
    """'7.1 ± 1.3' -> (7.1, 1.3)."""
    parts = str(text).split("±")
    return (float(parts[0]), float(parts[1])) if len(parts) == 2 else (np.nan, np.nan)


def _count_events(text):
    """'9517 (3118)' -> (9517, 3118)."""
    m = re.match(r"\s*([\d,]+)\s*\(\s*([\d,]+)\s*\)", str(text))
    return (_num(m.group(1)), _num(m.group(2))) if m else (np.nan, np.nan)


def _legacy_out(stage, name):
    """results/raw/chs/<nn>_<name> -- prefixed like a real run of `stage`, even
    though this importer is not itself a numbered stage."""
    out_dir = results_dir("chs", stage, "raw")
    os.makedirs(out_dir, exist_ok=True)
    return result_path(out_dir, name, stage)


def legacy_cohort(src):
    """cohort.csv (all cohorts, formatted) -> analyte, n_patients, value_mean, value_std."""
    t = _read_table(os.path.join(src, "cohort.csv"))
    rows = []
    for _, r in t.iterrows():
        n = _num(r["CHS N"])
        if not np.isfinite(n):
            continue
        mean, sd = _mean_sd(r["CHS Mean"])
        rows.append({"analyte": r["Analyte"], "n_patients": int(n), "value_mean": mean, "value_std": sd})
    return pd.DataFrame(rows)


def legacy_variability(src, cohort):
    """variability.csv (formatted 'value [lo, hi]') -> the 08_variability column set."""
    from figlib import ANALYTE_PANELS
    t = _read_table(os.path.join(src, "variability.csv"))
    panel_of = {a: panel for panel, members in ANALYTE_PANELS.items() for a in members}
    n_of = cohort.set_index("analyte")["n_patients"]
    rows = []
    for _, r in t.iterrows():
        intra = _value_ci(r["CHS CV_intra"])
        if not np.isfinite(intra[0]):
            continue
        inter = _value_ci(r["CHS CV_inter"])
        ii = _value_ci(r["CHS II"])
        a = r["Analyte"]
        rows.append({
            "Panel": panel_of.get(a, "Other"), "analyte": a, "n_patients": n_of.get(a, np.nan),
            "cv_intra": intra[0], "cv_intra_ci_lower": intra[1], "cv_intra_ci_upper": intra[2],
            "cv_inter": inter[0], "cv_inter_ci_lower": inter[1], "cv_inter_ci_upper": inter[2],
            "individuality_index": ii[0], "ii_ci_lower": ii[1], "ii_ci_upper": ii[2],
        })
    return pd.DataFrame(rows)


def legacy_prevalence(src):
    """chs_prevalence.csv -> per-analyte rates plus the Overall row the classify step writes
    (sums for counts, mean and quartiles across analytes for rates)."""
    t = _read_table(os.path.join(src, "chs_prevalence.csv"))
    df = pd.DataFrame({
        "analyte": t["Analyte"],
        "n": t["N"].map(_num),
        "PopRI_pct": t["PopRI (%)"].map(_num),
        "PerRI_pct": t["PerRI (%)"].map(_num),
        "PerRI_reclass_pct": t["PerRI RR (%)"].map(_num),
        "NORMA_pct": t["NORMA (%)"].map(_num),
        "NORMA_reclass_pct": t["NORMA RR (%)"].map(_num),
    }).dropna(subset=["PopRI_pct"])
    overall = {"analyte": "Overall", "n": df["n"].sum()}
    for col in [c for c in df.columns if c.endswith("_pct")]:
        overall[col] = df[col].mean()
        overall[f"{col}_median"] = df[col].median()
        overall[f"{col}_q25"] = df[col].quantile(0.25)
        overall[f"{col}_q75"] = df[col].quantile(0.75)
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


def legacy_eval(src):
    """chs_eval_<outcome>.csv (Per_RI / NORMA precision, sensitivity, specificity, accuracy
    within Pop_RI-normal tests) -> the long eval format."""
    rows = []
    for outcome in LEGACY_OUTCOMES:
        path = os.path.join(src, f"chs_eval_{outcome}.csv")
        if not os.path.exists(path):
            continue
        for _, r in _read_table(path).iterrows():
            for m in ("PerRI", "NORMA"):
                ppv = _num(r[f"{m} Precision"])
                sens = _num(r[f"{m} Sensitivity"])
                f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens > 0 else np.nan
                rows.append({
                    "outcome": outcome, "analyte": r["Analyte"], "method": m,
                    "n": _num(r["N"]), "n_events": _num(r["Events"]),
                    "ppv": ppv, "sensitivity": sens,
                    "specificity": _num(r[f"{m} Specificity"]),
                    "accuracy": _num(r[f"{m} Accuracy"]),
                    "f1": f1,
                    "per_100_flagged_with_event": 100 * ppv,
                    "number_needed_to_flag": 1 / ppv if ppv > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def legacy_cox(src):
    """chs_cox_<outcome>.csv (HR [95% CI] per method) -> the long cox format for the primary
    definition (first index measurement, binary flag).  p-values are reconstructed from
    the CI on the log scale; BH-FDR within outcome and globally, as 13_cox.py does."""
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    rows = []
    for outcome in LEGACY_OUTCOMES:
        path = os.path.join(src, f"chs_cox_{outcome}.csv")
        if not os.path.exists(path):
            continue
        for _, r in _read_table(path).iterrows():
            n_train, ev_train = _count_events(r["N (train)"])
            n_test, ev_test = _count_events(r["N (test)"])
            for m in LEGACY_METHODS:
                hr, lo, hi = _value_ci(r[f"{m} HR"])
                if not np.isfinite(hr) or lo <= 0 or hi <= lo:
                    continue
                se = (np.log(hi) - np.log(lo)) / (2 * 1.96)
                z = np.log(hr) / se
                rows.append({
                    "analyte": r["Analyte"], "method": m, "outcome": outcome,
                    "exposure": "first", "encoding": "binary", "level": "abnormal",
                    "n": n_train + n_test, "n_events": ev_train + ev_test,
                    "HR": hr, "HR_lower": lo, "HR_upper": hi,
                    "p_value": 2 * norm.sf(abs(z)),
                })
    df = pd.DataFrame(rows)
    df["p_fdr"] = np.nan
    for _, idx in df.groupby("outcome").groups.items():
        df.loc[idx, "p_fdr"] = multipletests(df.loc[idx, "p_value"].to_numpy(), method="fdr_bh")[1]
    df["p_fdr_global"] = multipletests(df["p_value"].to_numpy(), method="fdr_bh")[1]
    return df


def import_legacy_tables(src):
    """The April-2026 CHS tables -> the current per-stage result files under
    results/raw/chs/ (see data/clalit/legacy_2026-04/README.md for what is covered)."""
    import pandas as _pd
    cohort = legacy_cohort(src)
    ev = legacy_eval(src)
    # The legacy tables do not distinguish the eval subsets, so the same rows stand
    # in for both -- one file now, with `subset` as the column (see eval.csv).
    ev_both = _pd.concat([ev.assign(subset=s) for s in ("pop_normal", "per_normal_pop_normal")],
                         ignore_index=True)
    outputs = [
        ("03_cohort_summary", "cohort.csv", cohort.assign(split="all")),
        ("08_variability", "variability.csv", legacy_variability(src, cohort)),
        ("07_classify", "prevalence.csv", legacy_prevalence(src).assign(subset="all")),
        ("12_eval", "eval.csv", ev_both),
        ("13_cox", "cox.csv", legacy_cox(src).assign(subset="all")),
    ]
    for stage, name, df in outputs:
        path = _legacy_out(stage, name)
        df.to_csv(path, index=False)
        print(f"  {len(df):4d} rows -> {os.path.relpath(path, _VAL_DIR)}")


def main():
    parser = argparse.ArgumentParser(description="Process raw Clalit parquet data")
    parser.add_argument("--n_chunks", type=int, default=3,
                        help="Number of chunks to process (0..n_chunks-1)")
    parser.add_argument("--out_root", default=None,
                        help="Path to output chunk directories")
    parser.add_argument("--run_id", default="q_age_set",
                        help="NORMA run ID to filter from norma parquet")
    parser.add_argument("--sandbox", action="store_true",
                        help="Use sandbox fake data (default if no --out_root)")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--labs_only", action="store_true",
                        help="Only reprocess processed.parquet (skip diagnosis, ref_intervals)")
    parser.add_argument("--no_refs", action="store_true",
                        help="Skip the server Bayes/setpoint/NORMA-edits ref_intervals: "
                             "04_refs/{norma,baselines}.py own every method now")
    parser.add_argument("--analytes", type=str, default=None,
                        help="Comma-separated analytes to patch in processed.parquet (implies --labs_only --force)")
    parser.add_argument("--legacy_tables", default=None, metavar="DIR",
                        help="convert the April-2026 CHS tables in DIR into results/raw/chs/ and stop")
    args = parser.parse_args()

    if args.legacy_tables:
        import_legacy_tables(args.legacy_tables)
        return

    # --analytes implies labs_only and force
    if args.analytes is not None:
        args.labs_only = True
        args.force = True
        args.analytes = [a.strip() for a in args.analytes.split(",")]

    val_dir = _VAL_DIR

    # Default to real data; use --sandbox to opt into sandbox
    use_sandbox = args.sandbox

    if args.out_root:
        out_root = args.out_root
    elif use_sandbox:
        out_root = os.path.join(val_dir, "data", "clalit", "sandbox")
    else:
        out_root = os.path.join(val_dir, "data", "clalit")

    print(f"Processing {args.n_chunks} chunk(s)")
    print(f"  Output: {out_root}")

    paths = get_paths(sandbox=use_sandbox)
    print(f"  Source: {'sandbox' if use_sandbox else 'real'}")
    for i in range(args.n_chunks):
        out_dir = os.path.join(out_root, f"chunk_{i}")
        print(f"\nProcessing chunk_{i}...")
        process_chunk(chunk_idx=i, out_dir=out_dir, paths=paths,
                      run_id=args.run_id, force=args.force,
                      labs_only=args.labs_only, analytes=args.analytes,
                      refs=not args.no_refs)


if __name__ == "__main__":
    main()
