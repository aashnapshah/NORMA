#!/usr/bin/env python
"""How well does each model predict a patient's first index measurement?

Targets are built by lib/metrics.py (baseline history -> first index
measurement), the same rows the norma step of 04_refs.py scored, so three
families of predictors are compared on identical targets:

  NORMA                   04_refs (norma)       oracle (query = realized state, leaks
                                                the future state), normal (query fixed
                                                to "normal"), marginal / marginal_freq
                                                (mixture over states)
  history-only baselines  computed here         Last, Mean, ARIMA(1,1,1)
                                                [+ state-informed variants, --with_state]
  reference-interval      04_refs (baselines)   each method's centre as a point forecast:
  centres                                       PopRI midpoint, PerRI mean,
                                                Gaussian_{mle,trunc,eb}, Cohen_{m2,m3,m4}
                                                (NORMA's centre is NORMA_normal)

Every method is scored per analyte on the rows common to the methods that cover
the analyte, on all targets and on targets whose realized state is normal.

Outputs:
  results/raw/<cohort>/05_forecast_baselines.parquet
                                             Last / Mean / ARIMA per target (reused
                                             unless --force)
  results/raw/<cohort>/05_forecast.csv          n, MAE, MAPE, RMSE, R2, bias per
                                             (analyte, method, target_state)
                                            analyte, plus analyte="pooled" (all
                                            rows at once) and analyte="median"
                                            (median over analytes) rows

--norma_versions compares the NORMA model versions (covariate-ablation arms) with
EACH OTHER on the development test split -- no baselines, no centres -- from
model/logs/<version>/predictions_combined.csv, on the rows every version
predicted, split by development source -> results/raw/dev/05_norma_versions.csv
(fig_norma_versions).  Metrics are the ones above, so the numbers are comparable
with the cohort figures.

--no_norma scores the baselines and the interval centres alone, on targets built
here from index_labs instead of taken from the norma step's predictions -- for a
cohort the model was never run on.

Usage:
    python 05_forecasting.py --dataset eicu --workers 16
    python 05_forecasting.py --dataset mimiciv --max_patients 40000
    python 05_forecasting.py --norma_versions [--versions q_age_set 334f7e21 ...]

Figures and tables
------------------
Composites with one row per cohort (EHRSHOT, MIMIC-IV, eICU, INSPIRE, CHS):

  summary[_normal]     averaged over analytes — the six forecasters in FC_SHOW (same set
                       on both target sets) as colours; columns = MAE / MAPE / R² (weighted
                       mean ± SD across analytes); rows = cohorts, labelled vertically.
                       Open marker = given the state of the next value.
                       No in-figure legend: the manuscript legend explains the labels.
  by_analyte_<metric>[_normal]  methods × analytes heatmap, one block per cohort.
                       MAE is coloured relative to the best method for that analyte
                       (raw units differ across analytes); grey text = fewer than 50 targets.

Every cohort reads `<cohort>/05_forecast.csv` from `05_forecasting.py` (NORMA from
04_refs.py (norma step), history baselines computed there, interval centres from
04_refs.py (baselines step); the development cohorts run the same three steps).

The NORMA model versions (covariate-ablation arms) are compared only with each other,
on the development test split, from `results/raw/dev/05_norma_versions.csv` (05_forecasting.py --norma_versions):

  summary_norma        each arm's paired change from plain NORMA in MAE / MAPE / R²,
                       with a 95% bootstrap interval over analytes; rows = EHRSHOT /
                       MIMIC-IV, each on its own x scale.
  by_analyte_<metric>_norma  versions × analytes, one block per source,
                       absolute values in the cells.

The external cohorts are absent from these two: on them the arms are compared through
the pipeline (04_refs.py --runs <arms>, then the *_norma figures of each stage).
"""
import bootstrap

import argparse
import importlib.util
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from constants import MEDIAN_ROW, POOLED_ROW, PSEUDO_ANALYTES
import datasets
from datasets import (already_done, EXCLUDE_LABS, MODEL_LOG_DIR, NORMA_RUN_ID, dev_results_dir, add_dataset_args,
                      get_dataset, result_path)
from metrics import TARGET_KEYS, build_pairs, pairs_frame, subsample_patients, describe

MODEL_DIR = bootstrap.MODEL_DIR
BASELINES_FILE = "forecast_baselines.parquet"

STATIC_METHODS = {
    "PopRI": None,  # midpoint of pop_ri_low / pop_ri_high
    "PerRI": "per_ri_mean",
    "Gaussian_mle": "gaussian_mle_ri_mean",
    "Gaussian_trunc": "gaussian_trunc_ri_mean",
    "Gaussian_eb": "gaussian_eb_ri_mean",
    "Cohen_m2": "cohen_m2_ri_mean",
    "Cohen_m3": "cohen_m3_ri_mean",
    "Cohen_m4": "cohen_m4_ri_mean",
}
NORMA_VARIANTS = {"NORMA_normal": "normal", "NORMA_marginal": "marginal",
                  "NORMA_marginal_freq": "marginal_freq", "NORMA_oracle": "oracle"}
BASELINE_METHODS = {"Last": "last", "Mean": "mean", "ARIMA": "arima"}
PRINT_ORDER = ["PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb",
               "Cohen_m2", "Cohen_m3", "Cohen_m4", "Last", "Mean", "ARIMA",
               "NORMA_normal", "NORMA_marginal", "NORMA_marginal_freq", "NORMA_oracle"]

# --norma_versions: display order; the keys must match NORMA_VERSIONS in figures.py
VERSIONS = [NORMA_RUN_ID] + [r for r in ["334f7e21", "q_age", "q_set", "q_co", "q_age_set", "q_age_co",
                                         "q_set_co", "q_age_set_co", "q_co_q"] if r != NORMA_RUN_ID]
PID_SOURCE = os.path.join(MODEL_DIR, "predictions", "pid_source.csv")
NV_KEYS = ["pid", "code", "t_next"]


def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fix_analyte(df):
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df[~df["analyte"].isin(set(EXCLUDE_LABS))]


# ----------------------------------------------------------------- NORMA (04_refs, norma step)

def load_norma(ds, run_id, path=None):
    cols = TARGET_KEYS + ["x_next", "s_next", "n_hist", "horizon_days"]
    cols += [f"{run_id}_{v}" for v in NORMA_VARIANTS.values()]
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"No NORMA predictions at {path}.")
        print(f"  Loading {path}")
        df = pd.read_parquet(path, columns=cols)
    else:
        df = ds.load_norma_predictions(columns=cols)
        if df is None or not len(df):
            raise SystemExit(f"No NORMA predictions at {ds.norma_predictions_path()}; "
                             f"run 04_refs.py --dataset {ds.name} --only norma first.")
    return _fix_analyte(df).rename(columns={f"{run_id}_{v}": f"pred_{m}" for m, v in NORMA_VARIANTS.items()})


def build_targets(ds, args):
    """The forecasting targets straight from index_labs — the same
    (patient, analyte, target) records the norma step scores.  Used by --no_norma,
    where there are no NORMA predictions to take the target rows from, and by
    load_baselines when its cache has to be rebuilt."""
    df = ds.load_index_labs()
    time_unit = df.attrs.get("time_unit", getattr(ds, "time_unit", "days"))
    df = subsample_patients(df, args.max_patients)
    return build_pairs(df, time_unit, target=args.target, max_hist=args.max_hist,
                       exclude=getattr(ds, "exclude_labs", ()))


# ----------------------------------------------------------------- history-only baselines

_FB = None


def _baseline_one(args):
    global _FB
    if _FB is None:
        _FB = _import("_forecast_baselines", os.path.join(MODEL_DIR, "baselines", "forecast.py"))
    x_h, s_h, s_next, with_state, skip_arima = args
    return _FB.forecast_pair(x_h, s_h, s_next, with_state=with_state, skip_arima=skip_arima)


def run_baselines(recs, workers=4, with_state=False, skip_arima=False):
    t0 = time.time()
    jobs = ((r["x_h"], r["s_h"], r["s_next"], with_state, skip_arima) for r in recs)
    with Pool(workers) as pool:
        rows = list(pool.imap(_baseline_one, jobs, chunksize=500))
    print(f"  baselines: {len(rows):,} targets in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def load_baselines(ds, targets, args, results_dir, recs=None):
    """Last / Mean / ARIMA per target, computed once per cohort and reused.

    `recs` is the already-built target records (--no_norma builds them to get the
    target rows in the first place), so index_labs is not read and paired twice."""
    path = result_path(results_dir, BASELINES_FILE)
    if os.path.exists(path) and not args.force:
        print(f"  Loading {path}")
        base = pd.read_parquet(path)
        missing = targets.merge(base[TARGET_KEYS], on=TARGET_KEYS, how="left", indicator=True)
        n_missing = int((missing["_merge"] == "left_only").sum())
        if n_missing == 0:
            return base
        print(f"    {n_missing:,} of {len(targets):,} targets not in the cache — recomputing")

    if recs is None:
        recs = build_targets(ds, args)
        # only the targets NORMA scored (e.g. the --max_patients subset)
        wanted = set(map(tuple, targets[TARGET_KEYS].to_numpy()))
        recs = [r for r in recs if (r["patient_id"], r["analyte"], r["target_idx"]) in wanted]
        describe(recs, args.target)

    base = pairs_frame(recs)[TARGET_KEYS]
    out = run_baselines(recs, workers=args.workers, with_state=args.with_state, skip_arima=args.skip_arima)
    for c in out.columns:
        base[c] = out[c].values
    if "arima_fallback" in base.columns:
        print(f"  ARIMA guard: {base['arima_fallback'].mean():.2%} of targets fell back to last value")
    os.makedirs(results_dir, exist_ok=True)
    base.to_parquet(path, index=False)
    print(f"  wrote {path}")
    return base


# ----------------------------------------------------------------- reference-interval centres

def load_static_centres(ds):
    """One predicted centre per (patient_id, analyte) per reference-interval method,
    pivoted from ref_intervals.parquet (method -> <method>_ri_{mean,low,high})."""
    ref = _fix_analyte(ds.load_ref_intervals())
    ref = ref[ref["method"] != "base"]
    wide = ref.pivot_table(index=["patient_id", "analyte"], columns="method",
                           values=["ri_mean", "ri_low", "ri_high"], aggfunc="first")
    wide.columns = [f"{m}_{v}" for v, m in wide.columns]      # -> pop_ri_low, cohen_m4_ri_mean, ...
    df = wide.reset_index()

    out = df[["patient_id", "analyte"]].copy()
    out["pred_PopRI"] = (pd.to_numeric(df["pop_ri_low"], errors="coerce")
                         + pd.to_numeric(df["pop_ri_high"], errors="coerce")) / 2.0
    absent = []
    for method, col in STATIC_METHODS.items():
        if col is None:
            continue
        if col not in df.columns:
            absent.append(method)
            continue
        centre = pd.to_numeric(df[col], errors="coerce")
        # A null `ri_mean` with bounds present is not a missing prediction: the
        # Gaussian baselines use it to MARK a Pop_RI fallback (too few Pop_RI-normal
        # history points, or no EB prior) and still emit the population interval
        # as the bounds — the centre is then the interval midpoint.  Where the
        # bounds are missing too (Cohen skips analytes with too little healthy
        # training data) there is genuinely no prediction and the centre stays NaN.
        prefix = col[:-len("_mean")]
        low_col, high_col = f"{prefix}_low", f"{prefix}_high"
        if low_col in df.columns and high_col in df.columns:
            midpoint = (pd.to_numeric(df[low_col], errors="coerce")
                        + pd.to_numeric(df[high_col], errors="coerce")) / 2.0
            centre = centre.fillna(midpoint)
        out[f"pred_{method}"] = centre
    if absent:
        print(f"    not in this cohort's ref_intervals: {absent}")
    return out


# ----------------------------------------------------------------- scoring

def _metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 2:   # R² needs two points; small n is reported via the `n` column, not hidden
        return dict(n=len(y), mae=np.nan, mape=np.nan, rmse=np.nan, r2=np.nan, bias=np.nan)
    err = yhat - y
    denom = np.abs(y)
    mape = float(np.mean(np.abs(err[denom > 0]) / denom[denom > 0]) * 100) if (denom > 0).any() else np.nan
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(
        n=int(len(y)),
        mae=float(np.mean(np.abs(err))),
        mape=mape,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        bias=float(np.mean(err)),
    )


def score_frame(df, dataset, min_n):
    """Per-analyte metrics for every `pred_<method>` column of df, on rows common to
    the methods that cover the analyte; plus a `pooled` row on rows common to all."""
    methods = [c[5:] for c in df.columns if c.startswith("pred_")]
    pred_cols = [f"pred_{m}" for m in methods]

    # Common rows, per analyte: every method that covers the analyte must have a
    # prediction, so methods are compared on identical targets.  A method with
    # NO prediction for an analyte (Cohen skips analytes with too little healthy
    # training data, e.g. LDL / MPV / TC on eICU) is left out for that analyte only.
    rows = []
    for analyte, grp in df.groupby("analyte"):
        covered = [m for m in methods if grp[f"pred_{m}"].notna().any()]
        absent = [m for m in methods if m not in covered]
        common = grp.dropna(subset=[f"pred_{m}" for m in covered])
        if absent:
            print(f"    {analyte}: no prediction from {absent} — scored on the other "
                  f"{len(covered)} methods ({len(common):,} common targets)")
        for state_label, subset in [("normal", common[common["s_next"] == 1]),
                                    ("all_states", common)]:
            if len(subset) < min_n:
                continue
            for m in covered:
                r = _metrics(subset["x_next"], subset[f"pred_{m}"])
                r.update(analyte=analyte, method=m, target_state=state_label, dataset=dataset)
                rows.append(r)
    # pooled across analytes: rows every method predicts
    common_all = df.dropna(subset=pred_cols)
    print(f"  Pooled rows (all {len(methods)} methods non-null): {len(common_all):,} "
          f"({len(common_all) / max(len(df), 1) * 100:.1f}%), "
          f"{common_all['analyte'].nunique()} analytes")
    for state_label, subset in [("normal", common_all[common_all["s_next"] == 1]),
                                ("all_states", common_all)]:
        for m in methods:
            r = _metrics(subset["x_next"], subset[f"pred_{m}"])
            r.update(analyte=POOLED_ROW, method=m, target_state=state_label, dataset=dataset)
            rows.append(r)
    return rows


def write_outputs(rows, results_dir):
    detail = pd.DataFrame(rows)
    order = ["dataset", "target_state", "analyte", "method", "n", "mae", "mape", "rmse", "r2", "bias"]
    detail = detail[order]
    per_analyte = detail[detail["analyte"] != POOLED_ROW]
    g = per_analyte.groupby(["dataset", "target_state", "method"])
    summary = g[["mae", "mape", "rmse", "r2"]].median().reset_index()
    summary["n_analytes"] = g["analyte"].nunique().to_numpy()
    summary["n_targets"] = g["n"].sum().to_numpy()
    summary["analyte"] = MEDIAN_ROW
    # one file: per analyte, plus the pooled and median pseudo-analyte rows
    out_a = out_s = result_path(results_dir, "forecast.csv")
    pd.concat([detail, summary], ignore_index=True).to_csv(out_a, index=False)
    for state_label in ["normal", "all_states"]:
        s = summary[summary["target_state"] == state_label].set_index("method")
        print(f"\n  ── targets with realized state = {state_label} (median over analytes) ──")
        for m in PRINT_ORDER:
            if m in s.index:
                print(f"      {m:<20s} MAE={s.loc[m, 'mae']:8.3f}  "
                      f"MAPE={s.loc[m, 'mape']:6.1f}%  R2={s.loc[m, 'r2']:6.3f}")
    print(f"\nWrote {out_a} ({len(detail)} rows)")
    print(f"Wrote {out_s} ({len(summary)} rows)")


def run_cohort(args):
    ds = get_dataset(args)
    run_id = args.run_id or (ds.run_ids[0] if getattr(ds, "run_ids", None) else NORMA_RUN_ID)
    results_dir = ds.setup_output()
    if already_done(args, results_dir, "forecast.csv", label="forecast scores"):
        return
    if args.out_root:
        results_dir = os.path.join(args.out_root, "results", "raw", ds.output_sub())
        os.makedirs(results_dir, exist_ok=True)

    recs = None
    if ds.no_norma:
        # No NORMA predictions to take the target rows from, so the targets are built
        # here; the baselines and the interval centres are scored on exactly those.
        print("  Targets from index_labs (--no_norma: NORMA is not scored)")
        recs = build_targets(ds, args)
        describe(recs, args.target)
        targets = _fix_analyte(pairs_frame(recs))
        norma = targets[TARGET_KEYS + ["x_next", "s_next", "n_hist", "horizon_days"]]
    else:
        print("  NORMA forecasts (04_refs, norma step)")
        norma = load_norma(ds, run_id, path=args.norma_predictions)
    print(f"    {len(norma):,} targets, {norma['patient_id'].nunique():,} patients")

    print("  History-only baselines")
    base = load_baselines(ds, norma[TARGET_KEYS], args, results_dir, recs=recs)
    base = base.rename(columns={v: f"pred_{m}" for m, v in BASELINE_METHODS.items()})
    keep = TARGET_KEYS + [c for c in base.columns if c.startswith("pred_")]
    df = norma.merge(base[keep], on=TARGET_KEYS, how="inner")

    print("  Reference-interval centres (04_refs, baselines step)")
    static = load_static_centres(ds)
    print(f"    {len(static):,} (patient, analyte) pairs")
    df = df.merge(static, on=["patient_id", "analyte"], how="inner")
    print(f"  Merged: {len(df):,} targets with "
          + ("baseline and interval-centre predictions" if ds.no_norma
             else "NORMA, baseline and interval-centre predictions"))

    rows = score_frame(df, ds.name, args.min_n)
    write_outputs(rows, results_dir)


# ----------------------------------------------------------------- --norma_versions (dev test split)

def _load_version(run_id, log_dir):
    """Test-split rows of one version: pid, code, t_next, x_next, q50."""
    path = os.path.join(log_dir, run_id, "predictions_combined.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=NV_KEYS + ["x_next", "q50", "split"],
                     keep_default_na=False, na_values=[""])
    df = df[df.split == "test"].drop(columns="split")
    df["code"] = df["code"].replace("", "NA").fillna("NA")   # sodium reads as NaN
    return df[~df.code.isin(set(EXCLUDE_LABS))]


def run_norma_versions(args):
    """Every version scored on the rows ALL of them predicted (same split, so the
    test rows are the same sequences), so a version cannot look better by having
    been evaluated on an easier subset."""
    preds = {}
    for v in args.versions:
        d = _load_version(v, args.log_dir)
        if d is None:
            print(f"  skip {v}: no predictions_combined.csv")
            continue
        preds[v] = d
        print(f"  {v}: {len(d):,} test rows, {d.code.nunique()} analytes")
    if not preds:
        raise SystemExit("No version has predictions_combined.csv — nothing to compare.")

    common = None
    for d in preds.values():
        idx = pd.MultiIndex.from_frame(d[NV_KEYS])
        common = idx if common is None else common.intersection(idx)
    print(f"  common test rows across {len(preds)} versions: {len(common):,}")
    if len(common) == 0:
        # Writing here would replace a good file with an empty one, which is what
        # happened when the patient-split arms (p_base, p_co, p_causal, p_full,
        # trained with --split_by patient) were passed alongside the
        # sequence-split covariate arms: their test rows are different sequences,
        # so the intersection is empty by construction.
        raise SystemExit(
            "No test row is shared by all versions, so nothing can be compared "
            "pairwise. Versions trained with different --split_by values do not "
            "share a test split; compare them separately.")

    src = pd.read_csv(PID_SOURCE).set_index("pid")["source"]
    rows = []
    for v, d in preds.items():
        d = d[pd.MultiIndex.from_frame(d[NV_KEYS]).isin(common)].copy()
        d["source"] = d.pid.map(src)
        for s in DEV_COHORTS + ["all"]:
            sub = d if s == "all" else d[d.source == s]
            for code, g in sub.groupby("code"):
                r = _metrics(g.x_next, g.q50)
                rows.append({"version": v, "source": s, "analyte": code,
                             "n": r["n"], "mae": r["mae"], "mape": r["mape"], "r2": r["r2"]})

    out = result_path(dev_results_dir("05_forecasting"), "norma_versions.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")
    for s in DEV_COHORTS + ["all"]:
        sub = df[df.source == s]
        if not len(sub):
            continue
        # skip analytes _metrics could not score (n < 2, e.g. EHRSHOT has one
        # TGL target): np.average propagates a single NaN to the whole mean
        def _wmean(g):
            ok = np.isfinite(g.mae) & np.isfinite(g.n) & (g.n > 0)
            return np.average(g.mae[ok], weights=g.n[ok]) if ok.any() else np.nan
        w = sub.groupby("version")[["mae", "n"]].apply(_wmean).sort_values()
        print(f"  {s:8s} weighted mean MAE: " + "  ".join(f"{k} {v:.3f}" for k, v in w.items()))


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_dataset_args(p, required=False)
    p.add_argument("--run_id", default=None, help="NORMA run id in norma_predictions (default: dataset.run_ids[0])")
    p.add_argument("--target", default="first", choices=["first", "all"])
    p.add_argument("--max_hist", type=int, default=128)
    p.add_argument("--max_patients", type=int, default=None,
                   help="same patient subsample as 04_refs.py --max_patients (seed 42)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--with_state", action="store_true", help="also fit state-informed baselines")
    p.add_argument("--skip_arima", action="store_true")
    p.add_argument("--norma_predictions", default=None,
                   help="override path of the norma step's predictions file")
    p.add_argument("--out_root", default=None,
                   help="write results/raw/<cohort>/ under this folder instead (smoke tests)")
    p.add_argument("--min_n", type=int, default=2,
                   help="Minimum targets per analyte (default 2: score everything, `n` is in the output)")
    g = p.add_argument_group("norma_versions")
    g.add_argument("--norma_versions", action="store_true",
                   help="compare the NORMA model versions with each other on the dev test split "
                        "(no dataset) -> results/norma_versions.csv")
    g.add_argument("--versions", nargs="+", default=VERSIONS)
    g.add_argument("--log_dir", default=MODEL_LOG_DIR)
    args = p.parse_args()

    if args.norma_versions:
        run_norma_versions(args)
        return
    if args.dataset is None:
        p.error("--dataset is required (or pass --norma_versions)")
    run_cohort(args)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

import warnings

from figlib import *  # noqa: F401,F403
from datasets import NORMA_RUN_ID, dev_results_dir   # figlib's explicit re-export list does not carry them


# ─────────────────────────────────────────────────────────────────────────────
# Method registry: key -> (label, family, colour, uses realized future state)
# Colour = family hue, variant = lightness; identity is always carried by the
# axis label as well, never by colour alone.
# ─────────────────────────────────────────────────────────────────────────────
# (key, heatmap column header, short label for the summary's row groups)
FC_FAMILIES = [("norma", "NORMA", "NORMA"), ("history", "History-only baselines", "Baselines"),
               ("ri", "Reference-interval centres", "RI centres")]
# Which methods this analysis can draw, in display order. Labels and colours come
# from lib/models.py — this file used to carry its own copies, which is how
# Cohen ended up terracotta here and brown in 06_calibration/07_classify.
# `_group` is the bracket a method sits under in the composites, and is not the
# same thing as the registry's family (which groups Gaussian/Cohen variants).
_FC_GROUP = {"NORMA_oracle": "norma", "NORMA_marginal": "norma",
             "Last": "history", "Mean": "history", "ARIMA": "history"}
FC_ORDER = ["NORMA_oracle", "NORMA_marginal", "Last", "Mean", "ARIMA",
            "PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb", "Cohen_m4"]
FC_METHODS = {k: (models.label(k), _FC_GROUP.get(k, "ri"), models.color(k),
                  k in models.USES_REALIZED_STATE) for k in FC_ORDER}
# The interval centres are used here as point forecasts, so they get a forecast
# reading of the same method: the midpoint of Pop_RI, the mean of Per_RI.
FC_METHODS["PopRI"] = ("Pop$_{RI}$ Midpoint",) + FC_METHODS["PopRI"][1:]
FC_METHODS["PerRI"] = ("Per$_{RI}$ Mean",) + FC_METHODS["PerRI"][1:]

# What the figures show (Aashna, 2026-08-27: "going forward I only care about Cohen (m4),
# ARIMA, Gaussian, last value, NORMA leak-free and realized state").  Short labels on
# the axes; what they mean goes in the figure legend of the manuscript, not the PDF.
FC_SHOW = ["NORMA_oracle", "NORMA_marginal", "Last", "ARIMA", "Gaussian_mle", "Cohen_m4"]
# The same six on both target sets (Aashna, 2026-08-28: "the same baselines should be in
# summary_normal and summary").  Cohen and the Gaussian fit predict a *normal* value
# (Gaussian = mean of the Pop_RI-normal history), so on all next values they are
# expected to trail the history-only baselines.  Gaussian_eb / _trunc stay in the
# registry for the tables.
FC_SHOW_BY_TARGET = {"all_states": FC_SHOW, "normal": FC_SHOW}
FC_SHORT = models.labels(FC_ORDER, short=True)
FC_METRICS = [("mae", "MAE"), ("mape", "MAPE (%)"), ("r2", r"$R^2$")]
FC_TARGETS = [("all_states", "All next values"), ("normal", "Next normal values")]
_R2_FLOOR = -1.0   # R² of constant-centre methods can reach -30; clip and annotate

# dev-split file model names -> registry keys
_DEV_RENAME = {"NORMA-Quantile": "NORMA_oracle", "NORMA-Quantile (marginal)": "NORMA_marginal",
               "Last": "Last", "Mean": "Mean", "ARIMA": "ARIMA"}


def _fc_label(m, target_state="all_states"):
    # On next-normal targets the realized-state query IS the "normal" query.
    if target_state == "normal" and m == "NORMA_oracle":
        return "NORMA-N"
    return FC_SHORT.get(m, FC_METHODS[m][0])


def _fc_color(m):
    return FC_METHODS[m][2]


# ─────────────────────────────────────────────────────────────────────────────
# Loading: long frame with columns analyte, method, target_state, mape, r2
# ─────────────────────────────────────────────────────────────────────────────
def load_forecast_ext(ds):
    d = load_result(ds, "forecast.csv", normalize=False)
    if d is None:
        return None
    d = to_numeric(d)
    d["analyte"] = d["analyte"].replace("", "NA").fillna("NA")   # sodium reads as NaN
    d = d[(~d.analyte.isin(PSEUDO_ANALYTES)) & (~d.analyte.isin(EXCLUDE_ANALYTES))
          & d.method.isin(FC_ORDER)]
    return d if len(d) else None


def load_forecast_dev(cohort):
    d = load_prediction("forecasting_by_analyte_state_variants.csv")
    if d is None:
        return None
    d = d.copy()
    d["analyte"] = d["analyte"].replace("", "NA").fillna("NA")
    d = d[(d.split == cohort) & d.model.isin(_DEV_RENAME) & (~d.analyte.isin(EXCLUDE_ANALYTES))]
    if len(d) == 0:
        return None
    d = to_numeric(d)
    d["method"] = d.model.map(_DEV_RENAME)
    wide = d.pivot_table(index=["analyte", "method"], columns="metric", values="point_estimate").reset_index()
    wide = wide.rename(columns={"MAPE": "mape", "R2": "r2", "MAE": "mae"})
    wide["target_state"] = "all_states"
    return wide


def _methods_present(d, target_state="all_states"):
    have = set(d.method)
    return [m for m in FC_SHOW_BY_TARGET.get(target_state, FC_SHOW) if m in have]



def load_forecast(cohort):
    d = load_forecast_ext(cohort)
    if d is None and cohort in DEV_COHORTS:   # 05_forecasting.py not run for this dev cohort yet
        d = load_forecast_dev(cohort)
    return d


def _cohort_title(cohort):
    return DATASET_DISPLAY.get(cohort, cohort)


# ─────────────────────────────────────────────────────────────────────────────
# summary: one axis per metric, cohorts down the y axis, models as colours
# ─────────────────────────────────────────────────────────────────────────────
def _weighted_stats(d, metric, methods, key="method"):
    """Mean across analytes weighted by the number of targets, ± weighted SD."""
    out = {}
    for m in methods:
        sub = d[d[key] == m].dropna(subset=[metric])
        if len(sub) == 0:
            continue
        w = sub["n"].to_numpy(float) if "n" in sub.columns else np.ones(len(sub))
        v = sub[metric].to_numpy(float)
        mean = np.average(v, weights=w); sd = np.sqrt(np.average((v - mean) ** 2, weights=w))
        lo, hi = mean - sd, mean + sd
        if metric == "r2":   # constant-centre methods reach R² of -30: floor at -1
            lo, mean, hi = (max(x, _R2_FLOOR) for x in (lo, mean, hi))
        else:                # error metrics cannot be negative: SD bar clipped at 0
            lo = max(lo, 0)
        out[m] = (mean, lo, hi)
    return out


def _summary_grid(target_state, cohorts):
    frames = {c: load_forecast(c) for c in cohorts}
    frames = {c: (d[d.target_state == target_state] if d is not None else None) for c, d in frames.items()}
    frames = {c: (d if d is not None and len(d) else None) for c, d in frames.items()}
    if not any(d is not None for d in frames.values()):
        return None
    shown = [m for m in FC_SHOW_BY_TARGET[target_state]
             if any(d is not None and (d.method == m).any() for d in frames.values())]
    rows = []
    for c in cohorts:
        d = frames[c]
        if d is None:
            rows.append((c, None)); continue
        rows.append((c, {m: {metric: _weighted_stats(d, metric, [m]).get(m, (np.nan, np.nan, np.nan))
                             for metric, _ in FC_METRICS} for m in shown}))
    metrics = [(k, lab, (_R2_FLOOR - 0.05, 1.0) if k == "r2" else (0, None)) for k, lab in FC_METRICS]
    fig = dot_blocks(rows, metrics, shown, {m: _fc_color(m) for m in shown},
                     {m: _fc_label(m, target_state) for m in shown},
                     open_marker=[m for m in shown if FC_METHODS[m][3]],
                     label_rotation=270, row_labels=True)
    for ax in fig.axes:
        if ax.get_xlabel() == FC_METRICS[-1][1]:   # R² axis: zero line and floor tick
            ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["≤−1", "−0.5", "0", "0.5", "1"])
    return fig


def fig_summary():
    figs = {}
    f = _summary_grid("all_states", COHORT_ORDER)
    if f is not None:
        figs[""] = f
    f = _summary_grid("normal", COHORT_ORDER)
    if f is not None:
        figs["normal"] = f
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# by_analyte_<metric>[_normal]: methods × analytes heatmap, one block per cohort
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_cell(metric):
    def f(v):
        if metric == "r2":
            return "<−1" if v < -1 else f"{v + 0.0:.2f}".replace("-0.00", "0.00").replace("-", "−")
        if metric == "mape":
            return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        return f"{v:.2g}" if v < 10 else f"{v:.0f}"     # MAE in raw units
    return f


def _by_analyte_grid(metric, cbar_label, target_state="all_states", cohorts=COHORT_ORDER):
    frames = {c: load_forecast(c) for c in cohorts}
    frames = {c: (d[d.target_state == target_state] if d is not None else None) for c, d in frames.items()}
    frames = {c: (d if d is not None and len(d) else None) for c, d in frames.items()}
    if not any(d is not None for d in frames.values()):
        return None
    analytes = analyte_panel_order(set().union(*[set(d.analyte) for d in frames.values() if d is not None]))
    all_mape = np.concatenate([d["mape"].to_numpy(float) for d in frames.values() if d is not None])
    if metric == "r2":
        cmap, vmin, vmax, extend, neg = TEAL_CMAP, 0, 1, "neither", GREY_CELL
    elif metric == "mape":
        cmap, vmin, vmax, extend, neg = CORAL_CMAP, 0, float(np.nanpercentile(all_mape, 90)), "max", None
    else:   # MAE coloured relative to the best method for that analyte
        cmap, vmin, vmax, extend, neg = CORAL_CMAP, 1, 3, "max", None
    blocks = []
    for c in cohorts:
        d = frames[c]
        if d is None:
            blocks.append((c, None)); continue
        methods = _methods_present(d, target_state)
        piv = lambda col: d.pivot_table(index="method", columns="analyte", values=col).reindex(index=methods, columns=analytes).to_numpy(float)
        mat = piv(metric)
        if metric == "r2":
            shown = np.clip(mat, 0, 1)
        elif metric == "mae":
            import warnings
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (analyte absent in this cohort)
                shown = mat / np.nanmin(mat, axis=0, keepdims=True)
        else:
            shown = mat
        blocks.append((c, dict(mat=mat, shown=shown, nmat=piv("n") if "n" in d.columns else None,
                               rows=[_fc_label(m, target_state) for m in methods],
                               row_groups=[FC_METHODS[m][1] for m in methods])))
    extra = [Patch(facecolor=GREY_CELL, label=r"$R^2 < 0$")] if metric == "r2" else []
    return heatmap_blocks(blocks, analytes, cmap, vmin, vmax, _fmt_cell(metric), cbar_label, extend=extend,
                          neg_color=neg, col_groups=ANALYTE_PANELS, legend_extra=extra)


_BA_SPECS = [("mae", "MAE relative to the best method for that analyte"), ("mape", "MAPE (%)"), ("r2", r"$R^2$")]


def fig_by_analyte():
    figs = {}
    for metric, cbl in _BA_SPECS:
        f = _by_analyte_grid(metric, f"{cbl}, all next values")
        if f is not None:
            figs[metric] = f
        # next normal values: the task the reference-interval centres (Cohen) are built for
        f = _by_analyte_grid(metric, f"{cbl}, next normal values", "normal")
        if f is not None:
            figs[f"{metric}_normal"] = f
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# summary_norma / by_analyte_<metric>_norma: the NORMA model versions against each
# other on the development test split, no baselines and no interval centres.
# The versions differ only in which per-measurement covariates enter the
# encoder, so they are scored on the rows all of them predicted
# (05_forecasting.py --norma_versions).
# ─────────────────────────────────────────────────────────────────────────────
# norma_versions.csv is keyed by bare training run id; the registry keys the same
# arms as the rest of the pipeline sees them (NORMA / NORMA_<arm>), so the labels
# and colours here are the ones 06_calibration and 07_classify use for the arms.
# Every covariate arm, in the additive-ladder order of run_names.RUN_ORDER, not
# just the three that datasets.NORMA_ABLATION_RUN_IDS carries through the rest of
# the pipeline. This figure is the one place the arms are compared with each
# other, so showing the whole ladder is the point of it; _nv_shown() still drops
# any arm that norma_versions.csv does not contain.
from run_names import RUN_ORDER  # noqa: E402
NV_RUNS = [NORMA_RUN_ID] + [r for r in RUN_ORDER if r != NORMA_RUN_ID]
NV_KEY = {NORMA_RUN_ID: "NORMA", **{r: f"NORMA_{r}" for r in NV_RUNS if r != NORMA_RUN_ID}}
# Short labels: this figure is about the forecast, so the baseline arm reads
# "NORMA" rather than the interval's NORMA_RI.
NORMA_VERSIONS = {r: (models.label(NV_KEY[r], short=True), models.color(NV_KEY[r]))
                  for r in NV_RUNS}


def load_norma_versions():
    path = find_in(dev_results_dir("05_forecasting"), "norma_versions.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    d["analyte"] = d["analyte"].replace("", "NA").fillna("NA")
    d = to_numeric(d[~d.analyte.isin(EXCLUDE_ANALYTES) & d.version.isin(list(NORMA_VERSIONS))].copy())
    return d if len(d) else None


def _nv_shown(d):
    return [v for v in NORMA_VERSIONS if (d.version == v).any()]


def _nv_paired(sub, metric, arms, n_boot=2000, seed=0):
    """Paired change from the baseline arm, over the analytes both versions scored.

    Every version is scored on identical target rows (05_forecasting.py --norma_versions inner-joins
    them), so the comparison is paired per analyte and the change is what carries
    signal -- the absolute values differ by ~0.3% within a cohort. Error bar = 95%
    bootstrap interval over analytes (2,000 resamples, n-weighted mean), i.e. how
    much the direction of the change depends on which analytes were included.
    Relative (%) for the error metrics, whose units differ across analytes;
    absolute for R2, which is already unitless.
    """
    base = sub[sub.version == NORMA_RUN_ID].set_index("analyte")
    rng = np.random.default_rng(seed)
    out = {}
    for v in arms:
        cur = sub[sub.version == v].set_index("analyte")
        common = base.index.intersection(cur.index)
        if not len(common):
            continue
        b = base.loc[common, metric].to_numpy(float)
        c = cur.loc[common, metric].to_numpy(float)
        w = base.loc[common, "n"].to_numpy(float)
        ok = np.isfinite(b) & np.isfinite(c) & np.isfinite(w) & (w > 0)
        if metric == "r2":
            delta = c - b
        else:
            ok &= b > 0
            with np.errstate(all="ignore"):
                delta = (c - b) / b * 100.0
        delta, w = delta[ok], w[ok]
        if len(delta) < 2:
            continue
        mean = np.average(delta, weights=w)
        idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
        boot = (delta[idx] * w[idx]).sum(1) / w[idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[v] = (mean, lo, hi)
    return out


# The change is what the figure is about, so the axes say so; the absolute values
# are in the by_analyte heatmaps and in the tables.
NV_METRICS = [("mae", "Change in MAE (%)"), ("mape", "Change in MAPE (%)"),
              ("r2", r"Change in $R^2$")]


def fig_summary_norma():
    """Each arm's paired change from plain NORMA, one row per development source.

    Left of the dashed line is better for MAE and MAPE, right of it for R2. The
    baseline arm is the dashed line itself rather than a row of zeros. Each source
    gets its own x axis (share_x=False): EHRSHOT and MIMIC-IV differ by ~20% on MAE
    and the arms by ~0.3%, so one shared scale collapsed every arm onto one point.
    """
    d = load_norma_versions()
    if d is None or NORMA_RUN_ID not in set(d.version):
        return None
    arms = [v for v in _nv_shown(d) if v != NORMA_RUN_ID]
    if not arms:
        return None
    rows = []
    for s in DEV_COHORTS:
        sub = d[d.source == s]
        if not len(sub):
            rows.append((s, None)); continue
        stats = {metric: _nv_paired(sub, metric, arms) for metric, _ in NV_METRICS}
        rows.append((s, {v: {metric: stats[metric].get(v, (np.nan,) * 3)
                             for metric, _ in NV_METRICS}
                         for v in arms if any(v in stats[m] for m, _ in NV_METRICS)}))
    metrics = [(k, lab, None) for k, lab in NV_METRICS]
    return {None: dot_blocks(rows, metrics, arms, {v: NORMA_VERSIONS[v][1] for v in arms},
                             {v: NORMA_VERSIONS[v][0] for v in arms},
                             label_rotation=270, row_labels=True, share_x=False,
                             zero_line=True)}


# The arms differ by a fraction of a percent, so an analyte needs a lot of targets
# before its cells mean anything: EHRSHOT has 74 MPV targets and 1 TGL target, and MPV
# is where the arms look most different (a 29% spread) purely because of that. Cells
# below this many targets are greyed as unreliable rather than dropped.
NV_SMALL_N = 500


def _nv_fmt(metric):
    """Three significant digits: the versions differ in the third one, so
    _fmt_cell's two would print the same number for every version of ALP."""
    def f(v):
        if metric == "r2":
            return "<−1" if v < -1 else f"{v:.3f}".replace("-", "−")
        return f"{v:.3g}"
    return f


def fig_by_analyte_norma():
    """Versions × analytes, one block per source: where the covariates actually move MAE."""
    d = load_norma_versions()
    if d is None:
        return None
    shown = _nv_shown(d)
    analytes = analyte_panel_order(set(d.analyte))
    figs = {}
    for metric, cbar in [("mae", "MAE relative to the best NORMA version for that analyte"),
                         ("mape", "MAPE (%)"), ("r2", r"$R^2$")]:
        if metric == "r2":
            cmap, vmin, vmax, extend, neg = TEAL_CMAP, 0, 1, "neither", GREY_CELL
        elif metric == "mape":
            cmap, vmin, vmax, extend, neg = CORAL_CMAP, 0, float(np.nanpercentile(d.mape, 90)), "max", None
        else:   # the versions sit within a few percent of each other: tight scale
            cmap, vmin, vmax, extend, neg = CORAL_CMAP, 1, 1.10, "max", None
        blocks = []
        for s in DEV_COHORTS:
            sub = d[d.source == s]
            if not len(sub):
                blocks.append((s, None)); continue
            piv = lambda col: sub.pivot_table(index="version", columns="analyte", values=col
                                              ).reindex(index=shown, columns=analytes).to_numpy(float)
            mat = piv(metric)
            if metric == "r2":
                sh = np.clip(mat, 0, 1)
            elif metric == "mae":
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # analyte absent for every version
                    sh = mat / np.nanmin(mat, axis=0, keepdims=True)
            else:
                sh = mat
            blocks.append((s, dict(mat=mat, shown=sh, nmat=piv("n"),
                                   rows=[NORMA_VERSIONS[v][0] for v in shown])))
        extra = [Patch(facecolor=GREY_CELL, label=r"$R^2 < 0$")] if metric == "r2" else []
        figs[metric] = heatmap_blocks(blocks, analytes, cmap, vmin, vmax, _nv_fmt(metric), cbar,
                                      extend=extend, neg_color=neg, col_groups=ANALYTE_PANELS,
                                      legend_extra=extra, small_n=NV_SMALL_N)
    return figs


FIGURES = [
    FigSpec("05_forecasting", "summary",    fig_summary,    False, (), None),   # summary, summary_normal
    FigSpec("05_forecasting", "by_analyte", fig_by_analyte, False, (), None),   # by_analyte_{mae,mape,r2}[_normal]
    FigSpec("05_forecasting", "summary_norma", fig_summary_norma, False, (), None),
    FigSpec("05_forecasting", "by_analyte_norma", fig_by_analyte_norma, False, (), None),
]


# ══════════════════════════════════════════════════════════════════════════
# Tables — 05_forecasting: table_* definitions and registry slice.
# ══════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403

# save_table()'s first argument is the folder the table is written into,
# so it must match this directory name. Keeping the literal here (rather
# than only in the TableSpec) is what drifted during the restructure.  # noqa: F401,F403


def table_prediction_performance():
    df = load_prediction("forecasting_overall.csv")
    if df is None:
        return []
    df = to_numeric(df); models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    rows = []
    for metric in METRIC_ORDER:
        for split in DEV_SPLITS:
            row = {"Metric": metric, "Split": SPLIT_LABELS[split]}
            for model in models:
                match = df[(df["model"] == model) & (df["split"] == split) & (df["metric"] == metric)]
                row[model_label(model)] = _fmt_val(match.iloc[0]["mean"], match.iloc[0]["ci_lower"], match.iloc[0]["ci_upper"], metric) if len(match) == 1 else "---"
            rows.append(row)
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{ll" + "r" * len(models) + "}", r"\toprule",
             "Metric & & " + " & ".join(model_label(m) for m in models) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        vals = [row[model_label(m)] for m in models]
        best = _bold_best(vals, lower_is_better=LOWER_BETTER[row["Metric"]])
        cells = [r"\textbf{" + v + "}" if b is not None else v for v, b in zip(vals, best)]
        lines.append((r"\multirow{3}{*}{" + row["Metric"] + "}" if i % 3 == 0 else "") + " & " + row["Split"] + " & " + " & ".join(cells) + r" \\")
        if i % 3 == 2 and i < len(rows) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("05_forecasting", "prediction_performance", lines, pd.DataFrame(rows))
    return ["prediction_performance"]

def table_analyte_performance():
    df = load_prediction("forecasting_by_analyte.csv")
    if df is None:
        return []
    df = to_numeric(df[(df["split"] == "test") & (df["metric"].isin(METRIC_ORDER)) & (~df["analyte"].isin(EXCLUDE_ANALYTES))].dropna(subset=["analyte"]).copy())
    models = [m for m in [NORMA_MODEL, "ARIMA"] if m in df["model"].unique()]
    if not models:
        return []
    analytes = sorted(df["analyte"].unique()); rows = []
    for a in analytes:
        row = {"Analyte": a}
        for model in models:
            for metric in METRIC_ORDER:
                match = df[(df["model"] == model) & (df["analyte"] == a) & (df["metric"] == metric)]
                v = match.iloc[0]["point_estimate"] if len(match) == 1 else np.nan
                row[f"{model_label(model)}_{metric}"] = ("---" if pd.isna(v) else (f"{v:.2f}" if metric == "R2" else f"{v:.1f}"))
        rows.append(row)
    n_m = len(METRIC_ORDER)
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l" + "r" * (len(models) * n_m) + "}", r"\toprule"]
    h1, cmid = " ", []
    for i, model in enumerate(models):
        h1 += r" & \multicolumn{" + str(n_m) + r"}{c}{" + model_label(model) + "}"; cmid.append(f"\\cmidrule(lr){{{2 + i * n_m}-{1 + (i + 1) * n_m}}}")
    lines += [h1 + r" \\", " ".join(cmid), "Analyte" + "".join(f" & {m}" for _ in models for m in METRIC_ORDER) + r" \\", r"\midrule"]
    for row in rows:
        cells = [row["Analyte"]]
        best = {metric: _bold_best([row[f"{model_label(m)}_{metric}"] for m in models], LOWER_BETTER[metric]) for metric in METRIC_ORDER}
        for mi, model in enumerate(models):
            for metric in METRIC_ORDER:
                v = row[f"{model_label(model)}_{metric}"]
                cells.append(r"\textbf{" + v + "}" if best[metric][mi] is not None else v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("05_forecasting", "analyte_performance", lines, pd.DataFrame(rows))
    return ["analyte_performance"]

TABLES = [
    TableSpec("05_forecasting",   "prediction_performance",  table_prediction_performance,  False, (), None),
    TableSpec("05_forecasting",   "analyte_performance",     table_analyte_performance,     False, (), None),
]


if __name__ == "__main__":
    main()
