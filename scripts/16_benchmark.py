#!/usr/bin/env python
"""Head-to-head benchmark of every reference-interval method on the classification
table.  The native flag rates differ enormously (Per_RI flags ~66% of eICU
measurements, Cohen_m3 ~8% of the Pop_RI-normal ones), so native PPV / sensitivity
are not comparable: a method looks "precise" simply by flagging less.  Every step
therefore works on the continuous deviation z = |value - centre| / halfwidth
(z = 1 at the method's own boundary; 07_classify stores `<method>_z`), summarised
per patient as the max over that analyte's index measurements, on the patients
ALL methods can score.

  comparison       threshold-free AUROC of z per analyte x method x outcome, the
                   native operating point (flag rate, PPV, lift) and PPV / lift at
                   matched alert budgets (top 5 / 10 / 20 % of patients)
                   -> 16_method_comparison.csv (per analyte + analyte="median")
  operating_point  every method re-thresholded to the SAME operating point (R1-6,
                   R3-M2/M3): anchors native | rate:<r> (5/10/20/30 %) |
                   rate_of:<M> (M's native rate; M = PopRI, NORMA, PerRI) |
                   sensitivity:<s> (0.5, 0.8) | specificity:<s> (0.9, 0.95), each
                   with flag rate, sensitivity, specificity, PPV, NPV, lift
                   -> 16_matched_operating_point.csv (per analyte + analyte="median")
  burden           does a method degrade on patients with an abnormal history?
                   Pairs stratified by the fraction of their BASELINE values outside
                   Pop_RI (0 % / 1-25 / 26-50 / >50); flag rate and interval width
                   per stratum and method (the centre is 05_forecasting's concern)
                   -> 16_abnormal_burden.csv (per analyte + analyte="median" rows)
  significance     where does NORMA beat each comparator (Referee 2.2)?  Paired
                   DeLong test of the two AUCs on the same patients, per analyte x
                   outcome, BH-FDR within (outcome, comparator) plus a global
                   correction; on the Pop_RI-normal subset, the tests a population
                   interval calls normal and a personalised one can reclassify
                   -> 16_significance.csv

Usage:
    python 16_benchmark.py --dataset eicu
    python 16_benchmark.py --dataset eicu --only operating_point --subsets pop_normal

Figures and tables
------------------
Only what this folder's own scripts produce lives here. The per-method PPV /
balanced-accuracy / sensitivity / specificity panels are 12_eval/methods_<ds>,
the HR-fraction / concordance panels are 13_cox/methods_<ds>, and the
reclassification rate is 07_classify/reclassification.

  circos_matched_<ds>      the 12_eval circos redrawn with NORMA re-thresholded to Per_RI's flag
                           rate (16_benchmark.py, operating_point step): read next to circos_<ds>, it
                           shows how much of the native difference is operating point rather than
                           estimation (R1-6 / R3-M2 / R3-M3)
The pooled-bar matched_operating_point figures were dropped 2026-09-04 (pooled medians could
not be read against the per-lab circos); a per-analyte dumbbell version was tried the same
day and dropped as hard to read; method_comparison[_all,_norma] went the same day because
12_eval/methods_all already shows the AUROC and the 10 % flag (lift there = relative risk).
The comparison step's CSVs are still written for the tables.
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd

from constants import MEDIAN_ROW
from datasets import already_done, EXCLUDE_LABS, add_dataset_args, get_dataset, result_path, save_csv
from metrics import delong_test, population_reference_range
from metrics import (auroc, bh_fdr, method_prefix, operating_point, patient_level, ppv_at_budget,
                        threshold_for_rate, threshold_for_sensitivity, threshold_for_specificity)

# Reuse is keyed on these: a step whose files are all present is skipped
# unless --force (datasets.already_done).
STEP_OUTPUTS = {
    "comparison": ["method_comparison.csv"],
    "operating_point": ["matched_operating_point.csv"],
    "burden": ["abnormal_burden.csv"],
    "significance": ["significance.csv"],
}
STEPS = ("comparison", "operating_point", "burden", "significance")
BUDGETS = [0.05, 0.10, 0.20]                   # comparison
RATES = [0.05, 0.10, 0.20, 0.30]               # operating_point
SENSITIVITIES = [0.5, 0.8]
SPECIFICITIES = [0.9, 0.95]
ANCHOR_METHODS = ["PopRI", "NORMA", "PerRI"]   # "NORMA" matches NORMA_<run_id>
OP_METRICS = ["flag_rate", "sensitivity", "specificity", "ppv", "npv", "lift"]
BURDEN_BINS = [-0.001, 0.0, 0.25, 0.5, 1.0]    # burden
BURDEN_LABELS = ["0% abnormal", "1-25%", "26-50%", ">50%"]


def _fix_analyte(df):
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df[~df["analyte"].isin(set(EXCLUDE_LABS))]


def subset_of(classified, methods, name):
    """(frame, methods) for a subset; Pop_RI is left out of pop_normal, where its flag
    is constant by construction.  (None, None) for an unknown or impossible subset."""
    if name == "all":
        return classified, methods
    if name == "pop_normal":
        if "PopRI_class" not in classified.columns:
            print("  No PopRI_class column, skipping pop_normal subset")
            return None, None
        return classified[classified["PopRI_class"] == 1], [m for m in methods if m != "PopRI"]
    print(f"  Unknown subset {name}, skipping")
    return None, None


def common_patients(grp, methods, event_col):
    """{method: (patient-level frame on the patients ALL methods score, n scored by this
    method alone)}.  Without the restriction the AUROCs tabulated side by side come
    from different patient sets; the differences are not random (Cohen has no
    interval where the healthy history is too thin)."""
    per_method = {}
    for method in methods:
        patients = patient_level(grp, method, event_col)
        if patients.empty:
            continue
        scored = patients[patients["z"].notna()]
        if len(scored) >= 20:
            per_method[method] = scored
    if not per_method:
        return {}
    shared = sorted(set.intersection(*(set(p.index) for p in per_method.values())))
    if len(shared) < 20:
        return {}
    return {m: (p.loc[shared], len(p)) for m, p in per_method.items()}


def save_with_median_rows(detail, results_dir, stem, keys, cols):
    """One file per stem: the per-analyte rows, plus the median across analytes as
    analyte="median" rows.  Two files for one table meant every reader had to know
    which of the pair to open."""
    summary = detail.groupby(keys, dropna=False)[cols].median().reset_index()
    summary["n_analytes"] = detail.groupby(keys, dropna=False)["analyte"].nunique().to_numpy()
    summary["analyte"] = MEDIAN_ROW
    path = os.path.join(results_dir, f"{stem}.csv")
    save_csv(pd.concat([detail, summary], ignore_index=True), path)
    print(f"\nWrote {path} ({len(detail)} per-analyte + {len(summary)} median rows)")


def per_subset_and_outcome(ds, classified, methods, outcomes, subsets, compute, report):
    """Run compute(subset, frame, methods, outcome, event_col) -> rows for every subset
    and outcome; report(rows frame, outcome, methods) prints a summary of each."""
    rows = []
    for subset in subsets:
        frame, subset_methods = subset_of(classified, methods, subset)
        if frame is None:
            continue
        print(f"\n  -- subset={subset} ({len(frame):,} measurements) --")
        for outcome in outcomes:
            cfg = ds.outcomes.get(outcome)
            if cfg is None or cfg["event_col"] not in frame.columns:
                print(f"    {outcome}: no event column, skipping")
                continue
            got = compute(subset, frame, subset_methods, outcome, cfg["event_col"])
            rows += got
            if got:
                report(pd.DataFrame(got), outcome, subset_methods)
    return rows


# =============================================================================
# comparison
# =============================================================================

def native_operating_point(patients):
    """Flag rate, PPV and lift of the method's own flag rule, or NaNs."""
    native = patients.dropna(subset=["abn"])
    if len(native) < 20 or native["abn"].sum() == 0:
        return {"native_flag_rate": np.nan, "native_ppv": np.nan, "native_lift": np.nan}
    flagged = native["abn"] == 1
    ppv = round(float(native.loc[flagged, "event"].mean()), 4)
    base = float(native["event"].mean())
    return {"native_flag_rate": round(float(flagged.mean()), 4), "native_ppv": ppv,
            "native_lift": round(ppv / base, 3) if base > 0 else np.nan}


def comparison_rows(subset, classified, methods, outcome, event_col):
    rows = []
    for analyte, grp in classified.groupby("analyte"):
        if len(grp) < 20:
            continue
        for method, (patients, n_scored) in common_patients(grp, methods, event_col).items():
            # the base rate comes from the same (finite-z) patients the AUROC and PPV use
            base_rate = float(patients["event"].mean())
            row = {
                "analyte": analyte, "method": method, "outcome": outcome, "subset": subset,
                "n_patients": len(patients), "n_events": int(patients["event"].sum()),
                "n_scored_by_this_method": n_scored, "event_rate": round(base_rate, 4),
                "auroc": auroc(patients["z"], patients["event"]),
            }
            row.update(native_operating_point(patients))
            for budget in BUDGETS:
                ppv, realised = ppv_at_budget(patients["z"], patients["event"], budget)
                tag = f"{int(budget * 100):02d}"
                has_ppv = np.isfinite(ppv)
                row[f"ppv_at_{tag}"] = round(ppv, 4) if has_ppv else np.nan
                row[f"lift_at_{tag}"] = round(ppv / base_rate, 3) if has_ppv and base_rate > 0 else np.nan
                row[f"realised_rate_at_{tag}"] = round(realised, 4) if np.isfinite(realised) else np.nan
            rows.append(row)
    return rows


def report_comparison(rows, outcome, methods):
    medians = rows.groupby("method")[["auroc", "native_flag_rate", "lift_at_10"]].median()
    print(f"    {outcome}: {rows['analyte'].nunique()} analytes")
    for m in methods:
        if m in medians.index:
            print(f"        {m:<16s} AUROC={medians.loc[m, 'auroc']:.3f}  "
                  f"native_flag_rate={medians.loc[m, 'native_flag_rate']:.3f}  "
                  f"lift@10%={medians.loc[m, 'lift_at_10']:.2f}")


def run_comparison(ds, classified, methods, outcomes, args, results_dir):
    rows = per_subset_and_outcome(ds, classified, methods, outcomes, args.subsets,
                                  comparison_rows, report_comparison)
    if not rows:
        print("No rows computed.")
        return
    detail = pd.DataFrame(rows)
    metric_cols = [c for c in detail.columns if c not in ("analyte", "method", "outcome", "subset")]
    save_with_median_rows(detail, results_dir, "method_comparison", ["subset", "outcome", "method"],
                            metric_cols)


# =============================================================================
# operating_point
# =============================================================================

def anchor_method(methods, name):
    hits = [m for m in methods if m == name or m.startswith(name + "_")]
    return hits[0] if hits else None


def anchors_for(patients, anchor_rates):
    """(kind, target, operating point) for every anchor of one method."""
    z = patients["z"].to_numpy()
    event = patients["event"].to_numpy()
    anchors = []
    if patients["abn"].notna().any():          # native: the method's own class
        anchors.append(("native", np.nan, operating_point(patients["abn"].to_numpy(), event, 1.0)))
    for rate in RATES:
        anchors.append(("rate", rate, operating_point(z, event, threshold_for_rate(z, rate))))
    for name, rate in anchor_rates.items():
        anchors.append((f"rate_of:{name}", rate, operating_point(z, event, threshold_for_rate(z, rate))))
    for s in SENSITIVITIES:
        anchors.append(("sensitivity", s, operating_point(z, event, threshold_for_sensitivity(z, event, s))))
    for s in SPECIFICITIES:
        anchors.append(("specificity", s, operating_point(z, event, threshold_for_specificity(z, event, s))))
    return anchors


def operating_point_rows(subset, classified, methods, outcome, event_col):
    rows = []
    for analyte, grp in classified.groupby("analyte"):
        if len(grp) < 20:
            continue
        frames = {m: patients for m, (patients, _) in common_patients(grp, methods, event_col).items()}
        if not frames or not any(f["event"].sum() > 0 for f in frames.values()):
            continue
        anchor_rates = {}                       # native rates of the anchor methods
        for name in ANCHOR_METHODS:
            m = anchor_method(list(frames), name)
            if m is not None and frames[m]["abn"].notna().any():
                anchor_rates[name] = float((frames[m]["abn"] == 1).mean())
        for method, patients in frames.items():
            base = {
                "analyte": analyte, "method": method, "outcome": outcome, "subset": subset,
                "n_patients": len(patients), "n_events": int(patients["event"].sum()),
                "event_rate": float(patients["event"].mean()),
                "auroc": auroc(patients["z"], patients["event"]),
            }
            for kind, target, op in anchors_for(patients, anchor_rates):
                rows.append({**base, "anchor": kind, "target": target,
                             **{k: op[k] for k in OP_METRICS}, "n_flagged": op["n_flagged"]})
    return rows


def report_operating_point(rows, outcome, methods):
    print(f"    {outcome}: {rows['analyte'].nunique()} analytes; median PPV at 10% flagged / "
          f"at PopRI's rate / native (native rate):")
    for m in methods:
        d = rows[rows["method"] == m]
        if d.empty:
            continue
        at_10 = d[(d["anchor"] == "rate") & (d["target"] == 0.10)]["ppv"].median()
        at_pop = d[d["anchor"] == "rate_of:PopRI"]["ppv"].median()
        native = d[d["anchor"] == "native"]
        print(f"        {m:<16s} {at_10:.3f} / {at_pop:.3f} / "
              f"{native['ppv'].median():.3f} ({native['flag_rate'].median():.3f})")


def run_operating_point(ds, classified, methods, outcomes, args, results_dir):
    rows = per_subset_and_outcome(ds, classified, methods, outcomes, args.subsets,
                                  operating_point_rows, report_operating_point)
    if not rows:
        print("No rows computed.")
        return
    save_with_median_rows(pd.DataFrame(rows), results_dir, "matched_operating_point",
                            ["subset", "outcome", "method", "anchor", "target"], ["auroc"] + OP_METRICS)


# =============================================================================
# burden
# =============================================================================

def baseline_burden(ds):
    """Per (patient, analyte): fraction of the BASELINE values outside Pop_RI."""
    baseline = ds.load_index_labs().query("split == 'baseline'").copy()
    baseline = _fix_analyte(baseline)
    baseline["value"] = pd.to_numeric(baseline["value"], errors="coerce")
    baseline = baseline.dropna(subset=["value"])
    pairs = baseline[["analyte", "sex"]].drop_duplicates().itertuples(index=False)
    ranges = {(a, sex): population_reference_range(a, sex) for a, sex in pairs}
    keys = list(zip(baseline["analyte"], baseline["sex"]))
    low = np.array([ranges[k][0] for k in keys], dtype=float)
    high = np.array([ranges[k][1] for k in keys], dtype=float)
    # an analyte without a published interval gives (None, None); comparing against NaN
    # is False both ways, which would score every such value as normal
    value = baseline["value"].to_numpy()
    known = np.isfinite(low) & np.isfinite(high)
    baseline["_abnormal"] = np.where(known, ((value < low) | (value > high)).astype(float), np.nan)
    baseline = baseline.dropna(subset=["_abnormal"])
    burden = (baseline.groupby(["patient_id", "analyte"])
              .agg(abn_frac=("_abnormal", "mean"), n_baseline=("_abnormal", "size")).reset_index())
    burden["stratum"] = pd.cut(burden["abn_frac"], bins=BURDEN_BINS, labels=BURDEN_LABELS)
    print("  baseline abnormality burden per patient-analyte pair:")
    print(burden["stratum"].value_counts().reindex(BURDEN_LABELS).to_string())
    return burden


def flag_and_width(df, prefix):
    low = pd.to_numeric(df[f"{prefix}_low"], errors="coerce")
    high = pd.to_numeric(df[f"{prefix}_high"], errors="coerce")
    flag = ((df["value"] < low) | (df["value"] > high)).astype(float)
    flag[low.isna() | high.isna()] = np.nan
    return flag, high - low


def run_burden(ds, classified, methods, args, results_dir):
    burden = baseline_burden(ds)[["patient_id", "analyte", "abn_frac", "stratum"]]
    df = classified.merge(burden, on=["patient_id", "analyte"], how="inner")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    print(f"  {len(df):,} index measurements matched to a baseline burden")
    methods = [m for m in methods if f"{method_prefix(m)}_low" in df.columns]

    rows, analyte_rows = [], []
    for stratum in BURDEN_LABELS:
        sub = df[df["stratum"] == stratum]
        if len(sub) < args.min_n:
            continue
        for method in methods:
            prefix = method_prefix(method)
            flag, width = flag_and_width(sub, prefix)
            rows.append({"stratum": stratum, "method": method, "dataset": args.dataset,
                         "n": int(flag.notna().sum()), "flag_rate": float(flag.mean(skipna=True)),
                         "median_width": float(width.median(skipna=True)),
                         "mean_abn_frac": float(sub["abn_frac"].mean())})
            for analyte, grp in sub.groupby("analyte"):
                flag, width = flag_and_width(grp, prefix)
                if flag.notna().sum() < 20:
                    continue
                analyte_rows.append({"dataset": args.dataset, "stratum": stratum, "method": method,
                                     "analyte": analyte, "n": int(flag.notna().sum()),
                                     "flag_rate": float(flag.mean(skipna=True)),
                                     "median_width": float(width.median(skipna=True))})
    if not rows:
        raise SystemExit(f"No stratum reached --min_n={args.min_n}. Nothing written; check that "
                         f"the classification table and index_labs share a patient_id format.")
    out = pd.DataFrame(rows)
    out["analyte"] = MEDIAN_ROW        # `out` is already the across-analyte row
    save_csv(pd.concat([pd.DataFrame(analyte_rows), out], ignore_index=True),
             os.path.join(results_dir, "abnormal_burden.csv"))

    for metric, label in [("flag_rate", "fraction of index tests flagged abnormal"),
                          ("median_width", "median interval width")]:
        print(f"\n  -- {label}, by baseline abnormality burden --")
        table = out.pivot_table(index="method", columns="stratum", values=metric)
        table = table.reindex([m for m in methods if m in table.index])
        table = table[[c for c in BURDEN_LABELS if c in table.columns]]
        print(table.round(3).to_string())
        if metric == "median_width" and len(table.columns) >= 2:
            first, last = table.columns[0], table.columns[-1]
            print(f"\n    width adaptation ({last} / {first}):")
            print((table[last] / table[first]).round(2).to_string())
    print(f"\nWrote {results_dir}/abnormal_burden.csv")
    print(f"Wrote {results_dir}/abnormal_burden.csv")


# =============================================================================
# significance
# =============================================================================

def paired_delong(grp, reference, comparator, event_col, min_patients, min_events):
    """DeLong test of reference vs comparator on the patients both score, or None."""
    ref = patient_level(grp, reference, event_col)
    comp = patient_level(grp, comparator, event_col)
    if ref.empty or comp.empty:
        return None
    joined = ref[["z", "event"]].join(comp[["z"]], how="inner", rsuffix="_comp").dropna()
    if len(joined) < min_patients:
        return None
    n_events = int(joined["event"].sum())
    # both classes need enough members: DeLong's variance is undefined with a single
    # member in either class, which used to surface as p = 0
    if min(n_events, len(joined) - n_events) < min_events:
        return None
    r = delong_test(joined["z"].to_numpy(), joined["z_comp"].to_numpy(), joined["event"].to_numpy())
    return {"n_patients": len(joined), "n_events": n_events,
            "auroc_reference": r["auc_a"], "auroc_comparator": r["auc_b"], "delta_auroc": r["delta"],
            "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "z": r["z"], "p_value": r["p"]}


def run_significance(ds, classified, methods, outcomes, args, results_dir):
    # NORMA is the reference the manuscript tests against; on a cohort run with
    # --no_norma the personalised baseline Per_RI takes its place, so the step still
    # produces the pairwise DeLong table (the `reference` column names which it was).
    reference = (args.reference
                 or next((m for m in methods if m.startswith("NORMA")), None)
                 or next((m for m in ("PerRI", "PopRI") if m in methods), None))
    if reference is None:
        raise SystemExit("No NORMA or Per_RI method found; pass --reference explicitly.")
    comparators = [m for m in methods if m != reference]
    frame, _ = subset_of(classified, methods, "pop_normal")
    if frame is None:
        return
    print(f"\n  -- pop_normal ({len(frame):,} measurements), reference = {reference} --")
    print(f"  comparators: {comparators}")
    rows = []
    for outcome in outcomes:
        cfg = ds.outcomes.get(outcome)
        if cfg is None or cfg["event_col"] not in frame.columns:
            continue
        tested = set()
        for analyte, grp in frame.groupby("analyte"):
            if len(grp) < args.min_patients:
                continue
            for comparator in comparators:
                result = paired_delong(grp, reference, comparator, cfg["event_col"],
                                       args.min_patients, args.min_events)
                if result is None:
                    continue
                tested.add(analyte)
                rows.append({"dataset": args.dataset, "subset": "pop_normal", "outcome": outcome,
                             "analyte": analyte, "reference": reference, "comparator": comparator,
                             **result})
        print(f"    {outcome}: {len(tested)} analytes x {len(comparators)} comparators tested")

    if not rows:
        print("No comparisons met the minimum patient/event thresholds.")
        return
    df = pd.DataFrame(rows)
    df["p_fdr"] = np.nan                       # BH within each (outcome, comparator) family
    for _, idx in df.groupby(["outcome", "comparator"]).groups.items():
        df.loc[idx, "p_fdr"] = bh_fdr(df.loc[idx, "p_value"].to_numpy())
    df["p_fdr_global"] = bh_fdr(df["p_value"].to_numpy())
    df["favours"] = np.where(df["delta_auroc"] > 0, "reference", "comparator")
    df["significant"] = df["p_fdr"] < 0.05
    path = result_path(results_dir, "significance.csv")
    df.round(6).to_csv(path, index=False)

    print(f"\n  -- {reference} vs each comparator (analytes with FDR q < 0.05) --")
    for comparator in comparators:
        d = df[df["comparator"] == comparator]
        if d.empty:
            continue
        wins = int(((d["delta_auroc"] > 0) & d["significant"]).sum())
        losses = int(((d["delta_auroc"] < 0) & d["significant"]).sum())
        print(f"    {comparator:<16s} {wins:3d} wins / {losses:3d} losses / "
              f"{len(d) - wins - losses:3d} ns   median dAUROC {d['delta_auroc'].median():+.4f}")
    print(f"\nWrote {path} ({len(df)} comparisons)")


# =============================================================================
# main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--subsets", nargs="+", default=["all", "pop_normal"],
                   help="measurement subsets for comparison / operating_point")
    p.add_argument("--outcomes", nargs="+", default=None, help="outcome keys (default: the primary outcomes)")
    g = p.add_argument_group("burden")
    g.add_argument("--min_n", type=int, default=100, help="minimum index measurements per stratum")
    g = p.add_argument_group("significance")
    g.add_argument("--reference", default=None, help="method to test (default: the dataset's NORMA run)")
    g.add_argument("--min_patients", type=int, default=50, help="minimum patients with a paired score")
    g.add_argument("--min_events", type=int, default=10)
    args = p.parse_args()

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    # Reuse is the default: drop any step whose output is already written.
    # This runs BEFORE the classification is loaded, so a fully-cached run
    # costs nothing rather than paying the read and then skipping.
    todo = [s for s in args.only
            if not already_done(args, results_dir, *STEP_OUTPUTS[s], label=s)]
    if not todo:
        return
    classified = _fix_analyte(ds.load_classification())
    outcomes = args.outcomes or ds.primary_outcomes
    missing = [o for o in outcomes
               if o in ds.outcomes and ds.outcomes[o]["event_col"] not in classified.columns]
    if missing:
        print(f"  Attaching outcomes: {missing}")
        classified = ds.attach_outcomes(classified)
    # only methods whose interval columns exist in this cohort (some NORMA arms are
    # scored on eICU before the external cohorts catch up)
    methods = [m for m in ds.methods
               if f"{m}_class" in classified.columns or f"{method_prefix(m)}_low" in classified.columns]
    print(f"  {len(classified):,} measurements, {classified['patient_id'].nunique():,} patients, "
          f"{classified['analyte'].nunique()} analytes\n  Methods: {methods}")

    if "comparison" in todo:
        print("=== comparison ===")
        run_comparison(ds, classified, methods, outcomes, args, results_dir)
    if "operating_point" in todo:
        print("=== operating_point ===")
        run_operating_point(ds, classified, methods, outcomes, args, results_dir)
    if "burden" in todo:
        print("=== burden ===")
        run_burden(ds, classified, methods, args, results_dir)
    if "significance" in todo:
        print("=== significance ===")
        run_significance(ds, classified, methods, outcomes, args, results_dir)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403


# ── matched operating points, per analyte, in the circos style of 12_eval ───
_CIRCOS_METRICS = ["ppv", "sensitivity", "specificity"]   # AUROC is threshold-free, so it cannot change


def _matched_frame(ds):
    """Per_RI at its own threshold next to NORMA re-thresholded to Per_RI's flag rate,
    within Pop_RI-normal tests, in the column layout the circos helper reads."""
    d = load_result(ds, "matched_operating_point.csv")
    if d is None:
        return None
    d = d[d["analyte"].astype(str) != MEDIAN_ROW]        # per-analyte rows only
    d["method"] = d["method"].map(_bm_method)
    to_numeric(d)
    d = d[(d.subset == "pop_normal") & (~d.analyte.isin(EXCLUDE_ANALYTES))]
    perri = d[(d.anchor == "native") & (d.method == "PerRI")]
    norma = d[(d.anchor == "rate_of:PerRI") & (d.method == "NORMA")]
    if perri.empty or norma.empty:
        return None
    drawn = pd.concat([perri, norma])
    out = drawn.rename(columns={"n_patients": "n"})
    return out[["outcome", "analyte", "method", "n", "n_events", "n_flagged", *_CIRCOS_METRICS]]


def fig_circos_matched(ds):
    """The 12_eval circos (NORMA minus Per_RI per analyte, rows = metrics, columns = outcomes)
    with NORMA scored at Per_RI's flag rate instead of its own threshold: what is left of
    the circos difference once both methods flag the same fraction of tests (R1-6)."""
    df = _matched_frame(ds)
    if df is None:
        return {}
    outcomes = [o for o in OUTCOMES.get(ds, []) if o in set(df.outcome)]
    nr, nc = len(_CIRCOS_METRICS), len(outcomes)
    fig, axes = plt.subplots(nr, nc, figsize=(1.8 * nc + 0.5, 1.9 * nr + 0.3),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for i, metric in enumerate(_CIRCOS_METRICS):
        for j, outcome in enumerate(outcomes):
            ax = axes[i][j]
            if not _draw_circos(ax, df[df.outcome == outcome], metric, EVAL_METRIC_COLORS[metric]):
                ax.set_axis_off()
                continue
            if i == 0:
                ax.set_title(OUTCOME_DISPLAY.get(outcome, outcome), fontsize=FONT_TITLE, pad=8)
            if j == 0:
                ax.text(-0.25, 0.5, r"$\Delta$ " + EVAL_METRIC_LABELS[metric], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=FONT_AXIS, color=DARK)
    fig.suptitle(f"{DATASET_DISPLAY.get(ds, ds)}: NORMA$_{{RI}}$ at Per$_{{RI}}$'s flag rate $-$ Per$_{{RI}}$, "
                 f"within Pop$_{{RI}}$-normal tests", x=0.02, ha="left", fontsize=FONT_TITLE)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    return {None: fig}


FIGURES = [
    FigSpec("16_benchmark", "circos_matched", fig_circos_matched, True,
            ("matched_operating_point.csv",), _one("circos_matched")),
]


if __name__ == "__main__":
    main()
