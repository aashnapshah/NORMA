#!/usr/bin/env python
"""Outcome discrimination of every reference-interval method, from the classes and
deviation scores 07_classify stored.  Unit of analysis: one patient x analyte with
the patient's worst measurement (any abnormal flag; max z for the continuous form).

  metrics    PPV, NPV, sensitivity, specificity, F1 of each method's OWN flag per
             analyte x method x outcome, on all measurements, on the Pop_RI-normal
             ones (where personalised intervals can reclassify) and on the
             per_normal subsets -> 12_eval.csv, one `subset` column (per-chunk counts cached on CHS)

  auroc      threshold-free discrimination (R1-6, R3-M2/M3): the native flag rates
             differ by more than 20x, so native sensitivity / specificity mostly
             measure stringency.  Sweeping the threshold on z = |value - centre| /
             halfwidth walks the one-parameter family of intervals the coverage
             level indexes, and AUC is invariant to any monotone relabelling of it.
             Per (subset, outcome, analyte, method): AUC with DeLong 95% CI, a paired
             DeLong test against NORMA, average precision with its prevalence
             baseline, and the case-mix adjusted reading (age + sex logistic model
             on a 40% stratified holdout, with and without the score: auc_base /
             auc_adj / auc_gain).  `Overall` = the worst deviation over all analytes
             -> 12_auroc.csv

  deviation  patient-level deviation score, no model fitted (Cohen Fig. 5e design):
             the patient's WORST deviation across all analytes and draws (pooled,
             analyte = 'all') and per analyte; `any` = max |z|, `toward` = max of
             zs * pop_side floored at 0 (only deviations pointing at the nearer
             Pop_RI bound count).  --landmark_h L > 0 uses only draws within L hours
             of the patient's first scored draw and excludes patients whose event
             or follow-up ends at or before it; L = 0 scores the whole stay.
             Per outcome, subset, score and landmark: AUROC (DeLong CI), AUPRC /
             prevalence (bootstrap CI), and at MATCHED flag rates (top 5/10/20 %
             of patients, at each method's native flag, at Pop_RI's native rate):
             PPV, NPV, lift, RR flagged vs unflagged, sensitivity, specificity
             -> 12_deviation_score.csv

Usage:
    python 12_eval.py --dataset eicu
    python 12_eval.py --dataset eicu --only auroc deviation --landmark_h 0 24

Figures and tables
------------------
Within Pop_RI-normal tests of patients with a stable baseline:
  circos_<ds>              radial bars of delta(NORMA - comparator) per analyte at each method's
                           own threshold; rows = metrics, columns = outcomes; three rings per
                           panel on one scale: inner vs Per_RI, middle vs Cohen (model 4), outer vs
                           Empirical Bayes.  16_benchmark/circos_matched_<ds> is the single-ring
                           Per_RI comparison with NORMA at Per_RI's flag rate
  methods_all              every RI method on every cohort (rows) at a MATCHED flag rate: the
                           top 10 % of patients by each method's deviation score in the first
                           24 h are flagged, outcomes strictly after 24 h.  Columns: AUROC of
                           the score (threshold-free), then relative risk, precision,
                           sensitivity and specificity of that flag.  Written by
                           12_eval.py (deviation step)
  methods_all_norma        the NORMA covariate arms instead of the RI methods
  roc                      ROC space, rows = cohorts, columns = outcomes: each method's median
                           operating point over analytes (IQR whiskers), filled on all tests
                           and hollow within Pop_RI-normal tests, joined by a line (the subset
                           a referee read as random -- it sits on the diagonal by construction)
Reduced 2026-09-04 from 23 families (roc, methods, auroc, by_outcome and 14 deviation
variants, then the per-analyte relative-risk figure): the native-threshold bar figures
confounded stringency with accuracy, the deviation variants (every draw / native / Pop_RI's
rate, PPV / NPV per analyte) repeated one design at other anchors, and the per-analyte
relative risks were too noisy to read.
"""
import bootstrap  # noqa: F401

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from constants import ANCHOR, MIN_EVENTS, MIN_PATIENTS
from datasets import already_done, cached_chunk_frames, read_chunk_classification, EXCLUDE_LABS, NORMA_RUN_ID, add_dataset_args, get_dataset, save_csv
from metrics import delong_auc_cov, delong_test, hours_from_admit, to_hours, pop_side, signed_z

# Reuse is keyed on these: a step whose files are all present is skipped
# unless --force (datasets.already_done).
STEP_OUTPUTS = {
    "metrics": ["eval.csv"],
    "auroc": ["auroc.csv"],
    "deviation": ["deviation_score.csv"],
}
STEPS = ("metrics", "auroc", "deviation")
OVERALL = "Overall"     # auroc: per-patient worst deviation across all analytes
Z_CLIP = 10.0           # a logistic fit needs finite inputs
SEED = 42
RATES = (0.05, 0.10, 0.20)


def _fix_analyte(df):
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df[~df["analyte"].isin(set(EXCLUDE_LABS))]


def _outcomes(ds, df):
    return [o for o in ds.primary_outcomes
            if o in ds.outcomes and ds.outcomes[o]["event_col"] in df.columns]


def _wilson(x, n, z=1.96):
    """Wilson 95% interval for a proportion x/n."""
    if n <= 0:
        return np.nan, np.nan
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def _ci_pair(key, x, n):
    lo, hi = _wilson(x, n)
    return {f"{key}_lo": lo, f"{key}_hi": hi}


# =============================================================================
# metrics
# =============================================================================

def _patient_flags(grp, cls_cols, outcome_col):
    """One row per patient: outcome, and per method whether ANY measurement was abnormal
    (NaN when the method classified none of the patient's measurements)."""
    agg = {outcome_col: "first"}
    for cls_col in cls_cols:
        abnormal = pd.Series(np.nan, index=grp.index)
        valid = grp[cls_col].notna()
        abnormal[valid] = (grp.loc[valid, cls_col] != 1).astype(int)
        grp[f"_abn_{cls_col}"] = abnormal
        agg[f"_abn_{cls_col}"] = "max"
    patients = grp.groupby("patient_id").agg(agg).dropna(subset=[outcome_col])
    patients[outcome_col] = patients[outcome_col].astype(int)
    return patients


def confusion_counts(df, outcome_col, methods, subsets):
    """{(analyte, method, subset): {tp, fp, tn, fn}} of each method's own flag."""
    df = _fix_analyte(df.copy())
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.dropna(subset=[outcome_col])
    for col in ["per_ri_mean", "pop_ri_low", "pop_ri_high"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cls_cols = [f"{m}_class" for m in methods if f"{m}_class" in df.columns]
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    for subset, select, subset_methods in subsets:
        sub = select(df) if select else df
        for analyte, grp in sub.groupby("analyte"):
            patients = _patient_flags(grp, cls_cols, outcome_col)
            if len(patients) < 2:
                continue
            for method in subset_methods:
                abn_col = f"_abn_{method}_class"
                if abn_col not in patients.columns:
                    continue
                flagged = patients[abn_col].dropna().astype(int)
                if len(flagged) < 2:
                    continue
                event = patients.loc[flagged.index, outcome_col]
                c = counts[(analyte, method, subset)]
                c["tp"] += int(((flagged == 1) & (event == 1)).sum())
                c["fp"] += int(((flagged == 1) & (event == 0)).sum())
                c["tn"] += int(((flagged == 0) & (event == 0)).sum())
                c["fn"] += int(((flagged == 0) & (event == 1)).sum())
    return counts


def metrics_row(outcome, analyte, method, subset, c):
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n_flagged = tp + fp
    n_events = tp + fn
    n_no_events = tn + fp
    n_not_flagged = tn + fn
    n_total = tp + fp + tn + fn
    if n_total < 5:
        return None
    ppv = tp / n_flagged if n_flagged > 0 else np.nan
    npv = tn / n_not_flagged if n_not_flagged > 0 else np.nan
    sens = tp / n_events if n_events > 0 else np.nan
    spec = tn / n_no_events if n_no_events > 0 else np.nan
    balanced = (sens + spec) / 2 if (n_events > 0 and n_no_events > 0) else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and ppv + sens > 0) else np.nan
    return {
        "outcome": outcome, "analyte": analyte, "method": method, "subset": subset,
        "n": n_total, "n_events": n_events, "n_flagged": n_flagged,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "ppv": ppv, "npv": npv, "sensitivity": sens, "specificity": spec,
        "accuracy": balanced, "f1": f1,
        "per_100_flagged_with_event": round(ppv * 100, 1) if not np.isnan(ppv) else np.nan,
        "number_needed_to_flag": round(1 / ppv, 1) if ppv and ppv > 0 else np.nan,
    }


def subset_configs(methods, df):
    """(name, selector, methods) per subset; Pop_RI is left out where its flag is constant."""
    personalised = [m for m in methods if m != "PopRI"]
    has_per = "per_ri_mean" in df.columns and "pop_ri_low" in df.columns

    def per_normal(d):
        return d[(d["per_ri_mean"] >= d["pop_ri_low"]) & (d["per_ri_mean"] <= d["pop_ri_high"])]

    def pop_normal(d):
        return d[d["PopRI_class"] == 1]

    configs = [("all", None, methods)]
    if has_per:
        configs.append(("per_normal", per_normal, methods))
    configs.append(("pop_normal", pop_normal, personalised))
    if has_per:
        configs.append(("per_normal_pop_normal", lambda d: pop_normal(per_normal(d)), personalised))
    return configs


def chs_outcomes(chunk_dir, ds):
    """The chunk's diagnosis table as outcome flags (the CHS classification lacks them)."""
    path = os.path.join(chunk_dir, "diagnosis.parquet")
    if not os.path.exists(path):
        path = os.path.join(chunk_dir, "diagnosis.pkl")
    if not os.path.exists(path):
        return None
    diag = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)
    diag = diag.drop_duplicates(subset=["patient_id"])
    if "death_date" in diag.columns:
        diag["death_date"] = pd.to_datetime(diag["death_date"])
        diag["membership_end"] = pd.to_datetime(diag["membership_end"])
        death_days = (diag["death_date"] - pd.Timestamp("2015-01-01")).dt.days
        diag["died_5yr"] = diag["death_date"].notna() & (death_days <= 1825)
        diag["died_10yr"] = diag["death_date"].notna() & (death_days <= 3650)
    for key, cfg in ds.outcomes.items():
        if key != "mortality" and key in diag.columns:
            diag[key] = pd.to_datetime(diag[key])
            diag[cfg["event_col"]] = diag[key].notna()
    return diag


def counts_frame(df, ds, methods):
    """confusion_counts for every outcome of one frame, as a long frame."""
    subsets = subset_configs(methods, df)
    rows = []
    for outcome in _outcomes(ds, df):
        counts = confusion_counts(df, ds.outcomes[outcome]["event_col"], methods, subsets)
        for (analyte, method, subset), c in counts.items():
            rows.append({"outcome": outcome, "analyte": analyte, "method": method, "subset": subset, **c})
    return pd.DataFrame(rows)


def run_metrics(ds, cls, args, results_dir):
    methods = ds.methods
    keys = ["outcome", "analyte", "method", "subset"]

    if args.dataset == "chs":
        def compute(chunk_dir):
            chunk = read_chunk_classification(chunk_dir)
            if chunk is None:
                return None
            if ds._analytes is not None:
                chunk = chunk[chunk["analyte"].replace("", "NA").fillna("NA").isin(ds._analytes)]
            diag = chs_outcomes(chunk_dir, ds)
            if diag is not None:
                chunk = chunk.merge(diag, on="patient_id", how="left")
            return counts_frame(chunk, ds, methods)

        frames = list(cached_chunk_frames(ds, "eval_counts.parquet", compute, force=args.force))
        if not frames:
            print("  No results to save.")
            return
        totals = pd.concat(frames).groupby(keys)[["tp", "fp", "tn", "fn"]].sum()
    else:
        for outcome in _outcomes(ds, cls):
            event_rate = cls.drop_duplicates("patient_id")[ds.outcomes[outcome]["event_col"]].mean()
            print(f"  Outcome: {outcome} (event rate: {event_rate:.3f})")
        totals = counts_frame(cls, ds, methods).set_index(keys)

    rows = [metrics_row(*key, c) for key, c in totals.to_dict("index").items()]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("  No results to save.")
        return
    decimals = {"ppv": 3, "npv": 3, "sensitivity": 3, "specificity": 3, "accuracy": 3, "f1": 3,
                "per_100_flagged_with_event": 1, "number_needed_to_flag": 1}
    eval_df = pd.DataFrame(rows).round(decimals)
    # one file, `subset` kept as the column that distinguishes the restrictions
    out_path = os.path.join(results_dir, "eval.csv")
    save_csv(eval_df, out_path, analytes=ds._analytes, keys=("subset",))
    for subset, sub_df in eval_df.groupby("subset"):
        print(f"  Saved {len(sub_df)} rows (subset={subset}) to {out_path}")

    summary = (eval_df[eval_df["subset"] == "all"].groupby(["outcome", "method"])
               .agg(mean_per_100=("per_100_flagged_with_event", "mean"),
                    mean_nnf=("number_needed_to_flag", "mean")).round(1))
    print("\n  Per 100 patients flagged (mean across analytes):")
    for (outcome, method), row in summary.iterrows():
        print(f"    {outcome} / {method}: {row['mean_per_100']} had event (NNF: {row['mean_nnf']})")


# =============================================================================
# auroc
# =============================================================================

def _rank_auc(y, s):
    aucs, _ = delong_auc_cov(np.asarray(s, float)[None, :], np.asarray(y, float))
    return float(aucs[0])


def adjusted_auc(score, y, covariates):
    """Holdout AUC of age + sex, and of age + sex + the score (same 40% stratified holdout
    as the Cox models).  Returns (auc_base, auc_adj) or (nan, nan)."""
    if covariates is None:
        return np.nan, np.nan
    ok = np.isfinite(score) & np.isfinite(covariates).all(axis=1) & np.isfinite(y)
    if ok.sum() < MIN_PATIENTS or int((y[ok] == 1).sum()) < MIN_EVENTS:
        return np.nan, np.nan
    z = np.clip(score[ok], None, Z_CLIP)
    X, yy = covariates[ok], y[ok]
    try:
        train, test = train_test_split(np.arange(len(yy)), test_size=0.4, random_state=SEED, stratify=yy)
    except ValueError:
        return np.nan, np.nan
    if yy[test].sum() < MIN_EVENTS or yy[test].sum() == len(test):
        return np.nan, np.nan
    aucs = []
    for features in (X, np.column_stack([X, z])):
        try:
            model = LogisticRegression(max_iter=1000).fit(features[train], yy[train])
            aucs.append(_rank_auc(yy[test], model.decision_function(features[test])))
        except Exception:
            aucs.append(np.nan)
    return aucs[0], aucs[1]


def patient_scores(df, z_cols, event_col, by):
    """Worst deviation per method and the outcome, one row per `by` group."""
    agg = {c: "max" for c in z_cols}
    agg[event_col] = "first"
    for c in ("age", "sex"):
        if c in df.columns:
            agg[c] = "first"
    return df.groupby(by, observed=True).agg(agg).dropna(subset=[event_col])


def auc_rows(pat, z_cols, methods, event_col, ref_col, meta):
    """AUC + DeLong CI per method on one (analyte, outcome, subset) cell."""
    y = pat[event_col].to_numpy(float)
    covariates = None
    if {"age", "sex"} <= set(pat.columns):
        covariates = np.column_stack([pd.to_numeric(pat[c], errors="coerce").to_numpy(float)
                                      for c in ("age", "sex")])
    n_events = int(np.nansum(y == 1))
    if len(pat) < MIN_PATIENTS or n_events < MIN_EVENTS or n_events == len(pat):
        return []
    reference = pat[ref_col].to_numpy(float) if ref_col is not None else None

    rows = []
    for method, col in zip(methods, z_cols):
        s = pat[col].to_numpy(float)
        ok = np.isfinite(s)
        if ok.sum() < MIN_PATIENTS or int(np.nansum(y[ok] == 1)) < MIN_EVENTS:
            continue
        auc, cov = delong_auc_cov(s[ok][None, :], y[ok])
        se = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
        prevalence = float((y[ok] == 1).mean())
        try:
            ap = float(average_precision_score(y[ok], s[ok]))
        except Exception:
            ap = np.nan
        base, adj = adjusted_auc(s, y, covariates)
        row = dict(meta, method=method, n=int(ok.sum()), n_events=int((y[ok] == 1).sum()),
                   auc=float(auc[0]), auc_lo=float(auc[0] - 1.96 * se), auc_hi=float(auc[0] + 1.96 * se),
                   auc_base=base, auc_adj=adj, auc_gain=adj - base,
                   prevalence=prevalence, ap=ap, ap_lift=ap / prevalence if prevalence > 0 else np.nan)
        if reference is not None and col != ref_col:
            paired = np.isfinite(s) & np.isfinite(reference)
            if paired.sum() >= MIN_PATIENTS:
                test = delong_test(s[paired], reference[paired], y[paired])
                row["delta_vs_norma"] = test["delta"]
                row["p_vs_norma"] = test["p"]
        rows.append(row)
    return rows


def run_auroc(ds, cls, results_dir):
    df = cls.copy()
    methods = [m for m in ds.methods if f"{m}_z" in df.columns]
    if not methods:
        print("  No _z columns: re-run 07_classify.py.")
        return
    z_cols = [f"{m}_z" for m in methods]
    for c in z_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ref_col = f"NORMA_{NORMA_RUN_ID}_z"
    if ref_col not in z_cols:
        ref_col = None
    print(f"  Methods: {methods}")
    subsets = [("all", None)]
    if "PopRI_class" in df.columns:
        subsets.append(("pop_normal", lambda d: d[d["PopRI_class"] == 1]))

    rows = []
    for outcome in _outcomes(ds, df):
        event_col = ds.outcomes[outcome]["event_col"]
        df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
        for subset, select in subsets:
            sub = select(df) if select else df
            if not len(sub):
                continue
            per_analyte = patient_scores(sub, z_cols, event_col, ["analyte", "patient_id"])
            for analyte, grp in per_analyte.groupby(level="analyte", observed=True):
                meta = dict(outcome=outcome, subset=subset, analyte=analyte)
                rows += auc_rows(grp, z_cols, methods, event_col, ref_col, meta)
            # one score per patient: the worst deviation over all their analytes
            pooled = patient_scores(per_analyte.reset_index(), z_cols, event_col, "patient_id")
            meta = dict(outcome=outcome, subset=subset, analyte=OVERALL)
            rows += auc_rows(pooled, z_cols, methods, event_col, ref_col, meta)
            print(f"  {outcome:16s} {subset:11s} {len(per_analyte):>8,} patient-analyte scores")

    if not rows:
        print("  No results to save.")
        return
    decimals = {"auc": 4, "auc_lo": 4, "auc_hi": 4, "auc_base": 4, "auc_adj": 4, "auc_gain": 4,
                "prevalence": 4, "ap": 4, "ap_lift": 3, "delta_vs_norma": 4, "p_vs_norma": 6}
    out = pd.DataFrame(rows).round(decimals)
    cols = ["subset", "outcome", "analyte", "method", "n", "n_events", "auc", "auc_lo", "auc_hi",
            "auc_base", "auc_adj", "auc_gain", "prevalence", "ap", "ap_lift"]
    cols += [c for c in ("delta_vs_norma", "p_vs_norma") if c in out.columns]
    path = os.path.join(results_dir, "auroc.csv")
    save_csv(out[cols].sort_values(cols[:4]), path)
    print(f"  Wrote {len(out):,} rows -> {path}")


# =============================================================================
# deviation
# =============================================================================

def _rr_ci(a, k, c, u, z=1.96):
    """Risk ratio (a/k) / (c/u) with a log-scale 95% interval; NaN if a cell is empty."""
    if min(a, c) <= 0 or k <= 0 or u <= 0:
        return np.nan, np.nan, np.nan
    rr = (a / k) / (c / u)
    se = np.sqrt(1 / a - 1 / k + 1 / c - 1 / u)
    return float(rr), float(rr * np.exp(-z * se)), float(rr * np.exp(z * se))


def _auprc_boot(s, y, n_boot=100, seed=0):
    """Average precision with a bootstrap 95% interval: one sort, then each resample
    is a multinomial reweighting of the sorted labels."""
    order = np.argsort(-s, kind="stable")
    yy = y[order].astype(float)
    n = len(yy)

    def average_precision(w):
        tp = np.cumsum(w * yy)
        flagged = np.cumsum(w)
        positives = tp[-1]
        if positives <= 0:
            return np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(flagged > 0, tp / flagged, 0.0)
        return float(np.sum(precision * (w * yy) / positives))

    point = average_precision(np.ones(n))
    if n_boot <= 0:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [average_precision(rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)) for _ in range(n_boot)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    if not len(boots):
        return point, np.nan, np.nan
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def matched_rate_metrics(s, y, tag, rate):
    """PPV / NPV / lift / RR / sensitivity / specificity when the top `rate` of patients
    by score are flagged, with 95% intervals."""
    order = np.argsort(-s, kind="stable")
    n = len(y)
    total = y.sum()
    k = max(int(round(rate * n)), 1)
    u = n - k
    a = float(y[order[:k]].sum())            # events among the flagged
    c = float(total - a)                     # events among the unflagged
    ppv = a / k
    npv = (u - c) / u if u else np.nan
    base = float(y.mean())
    ppv_lo, ppv_hi = _wilson(a, k)
    rr, rr_lo, rr_hi = _rr_ci(a, k, c, u)
    row = {
        f"flag_rate_{tag}": rate,
        f"ppv_at_{tag}": ppv, f"ppv_at_{tag}_lo": ppv_lo, f"ppv_at_{tag}_hi": ppv_hi,
        f"npv_at_{tag}": npv,
        f"lift_at_{tag}": ppv / base if base else np.nan,
        f"lift_at_{tag}_lo": ppv_lo / base if base else np.nan,
        f"lift_at_{tag}_hi": ppv_hi / base if base else np.nan,
        f"rr_at_{tag}": rr, f"rr_at_{tag}_lo": rr_lo, f"rr_at_{tag}_hi": rr_hi,
        f"sens_at_{tag}": a / total if total else np.nan,
        f"spec_at_{tag}": (u - c) / (n - total) if n > total else np.nan,
    }
    row.update(_ci_pair(f"npv_at_{tag}", u - c, u))
    row.update(_ci_pair(f"sens_at_{tag}", a, total))
    row.update(_ci_pair(f"spec_at_{tag}", u - c, n - total))
    return row


def score_metrics(s, y, with_boot, pop_rate):
    """AUROC, AUPRC and the matched-rate metrics for one (score, label) pair; `pop_rate`
    adds the flag rate matched to Pop_RI's own abnormality rate."""
    auc, cov = delong_auc_cov(s[None, :], y)
    se = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    base = float(y.mean())
    ap, ap_lo, ap_hi = _auprc_boot(s, y, n_boot=100 if with_boot else 0)
    row = {
        "n": int(len(y)), "n_events": int(y.sum()), "prevalence": base,
        "auc": float(auc[0]), "auc_lo": float(auc[0] - 1.96 * se), "auc_hi": float(auc[0] + 1.96 * se),
        "auprc": ap, "auprc_lo": ap_lo, "auprc_hi": ap_hi,
        "auprc_norm": ap / base, "auprc_norm_lo": ap_lo / base, "auprc_norm_hi": ap_hi / base,
    }
    for rate in RATES:
        row.update(matched_rate_metrics(s, y, f"{int(rate * 100):02d}", rate))
    if np.isfinite(pop_rate):
        row.update(matched_rate_metrics(s, y, "pop", pop_rate))
    return row


def native_metrics(flag, y, tag="native"):
    """The same metrics for a binary patient flag (the method's own interval)."""
    flagged = np.isfinite(flag) & (flag > 0)
    n = len(y)
    k = int(flagged.sum())
    u = n - k
    total = float(y.sum())
    a = float(y[flagged].sum())
    c = total - a
    rr, rr_lo, rr_hi = _rr_ci(a, k, c, u)
    row = {
        f"flag_rate_{tag}": k / n if n else np.nan,
        f"ppv_at_{tag}": a / k if k else np.nan,
        f"npv_at_{tag}": (u - c) / u if u else np.nan,
        f"sens_at_{tag}": a / total if total else np.nan,
        f"spec_at_{tag}": (u - c) / (n - total) if n > total else np.nan,
        f"lift_at_{tag}": (a / k) / (total / n) if k and total else np.nan,
        f"rr_at_{tag}": rr, f"rr_at_{tag}_lo": rr_lo, f"rr_at_{tag}_hi": rr_hi,
    }
    row.update(_ci_pair(f"ppv_at_{tag}", a, k))
    row.update(_ci_pair(f"npv_at_{tag}", u - c, u))
    row.update(_ci_pair(f"sens_at_{tag}", a, total))
    row.update(_ci_pair(f"spec_at_{tag}", u - c, n - total))
    return row


def patient_outcomes(ds, cls, outcomes, unit):
    """Per outcome: one row per patient with event, time of event and end of follow-up
    (hours from admission)."""
    first = cls.groupby("patient_id", observed=True)
    tables = {}
    for o in outcomes:
        cfg = ds.outcomes[o]
        event = pd.to_numeric(first[cfg["event_col"]].first(), errors="coerce")
        t_event = pd.Series(np.nan, index=event.index)
        t_censor = pd.Series(np.nan, index=event.index)
        if cfg.get("time_col") in cls.columns:
            t_event = to_hours(first[cfg["time_col"]].first(), unit)
        if cfg.get("censor_col") in cls.columns:
            t_censor = to_hours(first[cfg["censor_col"]].first(), unit)
        tables[o] = pd.DataFrame({"event": event, "t_event": t_event, "t_censor": t_censor})
    return tables


def at_risk_at_landmark(patients, t0, landmark_h):
    """Drop patients whose event already happened, or whose follow-up ended, at or before
    their landmark (first scored draw + landmark_h)."""
    if landmark_h <= 0:
        return patients
    end = (t0 + landmark_h).reindex(patients.index)
    prevalent = (patients["event"] == 1) & (patients["t_event"] <= end)
    still_followed = (patients["t_censor"] > end) | patients["t_censor"].isna()
    return patients[~prevalent & still_followed & end.notna()]


def deviation_rows(method, scores, cls, groups, outcomes, patients_by_outcome, t_draw, is_normal,
                   landmarks, pop_rate):
    """Every (landmark, analyte group, outcome, subset, score) cell for one method."""
    pid = cls["patient_id"].to_numpy()
    rows = []
    for landmark_h in landmarks:
        for group, in_group in groups:
            pooled = group == "all"
            # the window starts at the patient's first scored draw of this group, as
            # the incidence step's index draw does (classified draws begin after the history)
            t0 = pd.Series(t_draw[in_group]).groupby(pid[in_group]).min()
            in_window = np.ones(len(cls), bool)
            if landmark_h > 0:
                in_window = t_draw <= (t0 + landmark_h).reindex(pid).to_numpy(float)
            selected = in_window & in_group
            worst = {
                "all": scores[selected].groupby("patient_id", observed=True).max(),
                "pop_normal": scores[selected & is_normal].groupby("patient_id", observed=True).max(),
            }
            for outcome in outcomes:
                patients = at_risk_at_landmark(patients_by_outcome[outcome], t0, landmark_h)
                for subset in ("all", "pop_normal"):
                    pat = worst[subset].reindex(patients.index).dropna(how="all")
                    y = patients.loc[pat.index, "event"].to_numpy(float)
                    for score in (("any", "toward") if pooled else ("toward",)):
                        s = pat[score].to_numpy(float)
                        ok = np.isfinite(s) & np.isfinite(y)
                        if ok.sum() < MIN_PATIENTS or len(np.unique(y[ok])) < 2:
                            continue
                        key = (landmark_h, group, outcome, score)
                        row = {"analyte": group, "landmark_h": landmark_h, "subset": subset,
                               "outcome": outcome, "method": method, "score": score}
                        row.update(score_metrics(s[ok], y[ok], pooled, pop_rate.get(key, np.nan)))
                        row.update(native_metrics(pat[f"nat_{score}"].to_numpy(float)[ok], y[ok]))
                        if method == "PopRI" and subset == "all":
                            pop_rate[key] = row["flag_rate_native"]
                        rows.append(row)
    return rows


def run_deviation(ds, cls, args, results_dir):
    methods = [m for m in ds.methods if f"{m}_z" in cls.columns]
    methods = sorted(methods, key=lambda m: m != "PopRI")     # Pop_RI first: its rate anchors 'pop'
    outcomes = _outcomes(ds, cls)
    unit = getattr(ds, "outcome_time_unit", None) or getattr(ds, "time_unit", None) or "minutes"
    if "t_hours" in cls.columns:
        t_draw = pd.to_numeric(cls["t_hours"], errors="coerce").to_numpy(float)
    else:
        t_draw = hours_from_admit(cls, getattr(ds, "time_unit", None)).to_numpy(float)
    side = pop_side(cls).to_numpy()
    is_normal = (cls["PopRI_class"] == 1).to_numpy()
    patients_by_outcome = patient_outcomes(ds, cls, outcomes, unit)
    analytes = sorted(cls["analyte"].unique())
    groups = [("all", np.ones(len(cls), bool))]
    groups += [(a, (cls["analyte"] == a).to_numpy()) for a in analytes]

    rows = []
    pop_rate = {}      # (landmark, group, outcome, score) -> Pop_RI's native flag rate on all draws
    for method in methods:
        zs = signed_z(cls, method).to_numpy(float)
        cl = np.full(len(cls), np.nan)
        if f"{method}_class" in cls.columns:
            cl = pd.to_numeric(cls[f"{method}_class"], errors="coerce").to_numpy(float)
        flagged = np.isfinite(cl) & (cl != 1)                # the method's own flag (1 = normal)
        scores = pd.DataFrame({
            "patient_id": cls["patient_id"].to_numpy(),
            "any": np.abs(zs),
            "toward": np.clip(zs * side, 0, None),
            "nat_any": flagged.astype(float),
            "nat_toward": (flagged & (zs * side > 0)).astype(float),
        })
        rows += deviation_rows(method, scores, cls, groups, outcomes, patients_by_outcome, t_draw,
                               is_normal, args.landmark_h, pop_rate)
        print(f"  {method}: done")

    out = pd.DataFrame(rows)
    if out.empty:
        print("  No results to save.")     # nothing cleared MIN_PATIENTS
        return
    save_csv(out, os.path.join(results_dir, "deviation_score.csv"))
    show = out[(out.score == "toward") & (out.analyte == "all") & (out.subset == "pop_normal")]
    for landmark_h in args.landmark_h:
        print(f"\n-- landmark {landmark_h:g} h, Pop_RI-normal draws, RR at a 10% flag rate --")
        table = show[show.landmark_h == landmark_h].pivot_table(index="method", columns="outcome",
                                                                values="rr_at_10")
        print(table.round(2).to_string())


# =============================================================================
# main
# =============================================================================

def load_classified(ds):
    cls = _fix_analyte(ds.load_classification())
    missing = [o for o in ds.primary_outcomes
               if o in ds.outcomes and ds.outcomes[o]["event_col"] not in cls.columns]
    if missing:
        print(f"  Attaching outcomes: {missing}")
        cls = ds.attach_outcomes(cls)
    return cls


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--landmark_h", type=float, nargs="*", default=[0.0, 24.0],
                   help="deviation: assessment windows in hours from the first scored draw; 0 = whole stay")
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
    print(f"  Methods: {ds.methods}")
    cls = None
    if args.dataset != "chs" or set(args.only) - {"metrics"}:
        cls = load_classified(ds)
    if "metrics" in todo:
        print("=== metrics ===")
        run_metrics(ds, cls, args, results_dir)
    if "auroc" in todo:
        print("=== auroc ===")
        run_auroc(ds, cls, results_dir)
    if "deviation" in todo:
        print("=== deviation ===")
        run_deviation(ds, cls, args, results_dir)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403

COHORTS = VAL_COHORTS         # eicu, inspire, chs: one row each in the pooled figure


# Named apart from the compute half's _outcomes(ds, df): both halves live in this
# one module, so a shared name silently rebinds the one defined first.
def _outcome_order(df):
    return [o for o in OUTCOME_DISPLAY if o in set(df["outcome"])] + \
           sorted(set(df["outcome"]) - set(OUTCOME_DISPLAY))


def _prevalence(sub):
    return float(np.nanmedian(sub["prevalence"])) if "prevalence" in sub and len(sub) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# circos: NORMA - Per_RI per analyte at native thresholds
# ─────────────────────────────────────────────────────────────────────────────
_COMPARATORS = ["PerRI", "Cohen_m4", "Gaussian_eb"]   # rings, inner to outer


def fig_circos(ds):
    """Grid of circos panels: rows = metrics, columns = outcomes.  Each panel has one
    ring per comparator (inner Per_RI, then Cohen, then Empirical Bayes): NORMA minus
    that method per analyte, each method at its own threshold, all rings on one scale."""
    df = _load_eval_restricted(ds)
    if df is None or len(df) == 0:
        return {}
    comparators = [c for c in _COMPARATORS if c in set(df["method"])]
    if not comparators:
        return {}
    drawn = df[df["method"].isin(["NORMA", *comparators])]
    export_cols = ["outcome", "analyte", "method", "n", "n_events", "tp", "fp", "tn", "fn",
                   "ppv", "sensitivity", "specificity", "auroc", "auroc_lo", "auroc_hi"]
    outcomes = _outcome_order(df)
    nr, nc = len(EVAL_METRICS), len(outcomes)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc + 0.5, 2.4 * nr + 0.5),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for i, metric in enumerate(EVAL_METRICS):
        for j, outcome in enumerate(outcomes):
            ax = axes[i][j]
            sub = df[df["outcome"] == outcome]
            if not _draw_circos_rings(ax, sub, metric, EVAL_METRIC_COLORS[metric], comparators):
                ax.set_axis_off()
                continue
            if i == 0:
                ax.set_title(OUTCOME_DISPLAY.get(outcome, outcome), fontsize=FONT_TITLE, pad=8)
            if j == 0:
                ax.text(-0.2, 0.5, r"$\Delta$ " + EVAL_METRIC_LABELS[metric], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=FONT_AXIS, color=DARK)
    fig.text(0.02, 0.995, f"{DATASET_DISPLAY.get(ds, ds)}: NORMA$_{{RI}}$ $-$ comparator within "
             f"Pop$_{{RI}}$-normal tests", ha="left", va="top", fontsize=FONT_TITLE, color=DARK)
    handles = [Line2D([], [], color=_BM_COLORS.get(c, DARK), lw=1.2,
                      label=f"{'Inner' if k == 0 else 'Outer' if k == len(comparators) - 1 else 'Middle'} ring: "
                            f"vs {RI_LABELS[c]}") for k, c in enumerate(comparators)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=len(handles),
               frameon=False, fontsize=FONT_LEGEND, handlelength=1.4, columnspacing=1.3)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    return {None: fig}


# ─────────────────────────────────────────────────────────────────────────────
# patient-level deviation score at a matched flag rate (12_eval.py, deviation step)
# ─────────────────────────────────────────────────────────────────────────────
def _deviation_frame(ds, score="toward", subset="pop_normal", analyte=None, landmark=24):
    """Rows of deviation_score.csv for one score / subset / landmark. analyte None
    = the pooled ('all') rows, '*' = every per-analyte row. Falls back to the
    whole-stay rows when the landmark was not computed."""
    d = load_result(ds, "deviation_score.csv")
    if d is None or len(d) == 0:
        return None
    d["method"] = d["method"].map(_bm_method)
    to_numeric(d, skip=("analyte", "subset", "outcome", "method", "score"))
    d = d[(d["score"].astype(str) == score) & (d["subset"].astype(str) == subset)]
    if "landmark_h" in d.columns:
        at_landmark = d[d["landmark_h"] == landmark]
        d = at_landmark if len(at_landmark) else d[d["landmark_h"] == 0]
    if "analyte" not in d.columns:
        return d if analyte in (None, "all") and len(d) else None
    if analyte == "*":
        d = d[d["analyte"].astype(str) != "all"]
    else:
        d = d[d["analyte"].astype(str) == ("all" if analyte is None else analyte)]
        export_cols = ["analyte", "landmark_h", "subset", "outcome", "method", "score", "n", "n_events",
                       "prevalence", "auc", "auc_lo", "auc_hi"]
        export_cols += [c for c in d.columns if c.startswith(("rr_at_", "sens_at_", "spec_at_"))
                        and (c.endswith(f"_{ANCHOR}") or c.endswith((f"_{ANCHOR}_lo", f"_{ANCHOR}_hi")))]
    return d if len(d) else None


def _grouped_bars(ax, d, outcomes, methods, value, lo=None, hi=None, ref=None, ref_fn=None):
    """Outcome groups on x, one bar per method, CI whiskers. `ref` = one horizontal
    reference line; `ref_fn(sub)` = a per-outcome reference (e.g. prevalence)
    drawn as a short dashed line across that group."""
    x = np.arange(len(outcomes))
    w = 0.8 / max(len(methods), 1)
    for j, m in enumerate(methods):
        sub = d[d.method == m].set_index("outcome").reindex(outcomes)
        xs = x + (j - (len(methods) - 1) / 2) * w
        vals = sub[value].to_numpy(float)
        ax.bar(xs, vals, w * 0.92, color=_BM_COLORS.get(m, "#999"), alpha=0.85,
               edgecolor="white", linewidth=0.3, label=RI_LABELS.get(m, m))
        if lo and hi and lo in sub and hi in sub:
            err = [np.clip(vals - sub[lo].to_numpy(float), 0, None), np.clip(sub[hi].to_numpy(float) - vals, 0, None)]
            ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=DARK, elinewidth=0.6, capsize=1.5, alpha=0.7)
    if ref is not None:
        ax.axhline(ref, color=DARK, lw=0.7, alpha=0.5, zorder=0)
    if ref_fn is not None:
        for i, o in enumerate(outcomes):
            r = ref_fn(d[d.outcome == o])
            if np.isfinite(r):
                ax.hlines(r, i - 0.42, i + 0.42, color=DARK, lw=0.8, ls=(0, (2, 1.5)), alpha=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_SHORT.get(o, o) for o in outcomes], fontsize=FONT_TICK, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    hide_spines(ax)


def _cohort_label(ax, ds):
    """Cohort name on the right of the row's last axis."""
    ax.text(1.04, 0.5, DATASET_DISPLAY.get(ds, ds), transform=ax.transAxes,
            rotation=270, ha="left", va="center", fontsize=FONT_AXIS, color=DARK)


# (column, label, reference line, y limits).  Precision and specificity are not panels:
# at a fixed 10 % flag rate specificity is ~0.90 for every method by construction, and
# precision is base rate x relative risk, so it only re-scales the relative-risk panel
# by an outcome prevalence that ranges from 0.02 to 0.75.
_MATCHED_PANELS = [
    ("auc", "AUROC", 0.5, (0.45, None)),
    (f"rr_at_{ANCHOR}", "Relative risk", 1.0, (0, None)),
    (f"sens_at_{ANCHOR}", "Sensitivity", None, (0, None)),
]


def fig_eval_methods_all():
    """Every RI method on every cohort at a matched flag rate: one row per cohort,
    x = outcome, one bar per method with its 95% interval.  AUROC is threshold-free;
    relative risk and sensitivity score the flag on the top 10 % of patients by each
    method's deviation score in the first 24 h."""
    data = {ds: _deviation_frame(ds) for ds in COHORTS}
    present = set()
    for d in data.values():
        if d is not None:
            present |= set(d.method)
    methods = [m for m in bm_methods() if m in present]
    if len(methods) < 2:
        return {}
    n_rows, n_cols = len(COHORTS), len(_MATCHED_PANELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 2.1 * n_rows + 0.5), squeeze=False)
    for ri, ds in enumerate(COHORTS):
        d = data[ds]
        if d is None:
            for ax in axes[ri]:
                pending_axis(ax, ds)
            _cohort_label(axes[ri, -1], ds)
            continue
        outcomes = [o for o in OUTCOMES.get(ds, []) if o in set(d.outcome)]
        for ax, (col, label, ref, ylim) in zip(axes[ri], _MATCHED_PANELS):
            _grouped_bars(ax, d, outcomes, methods, col, f"{col}_lo", f"{col}_hi", ref=ref)
            ax.set_ylim(*ylim)
            ax.set_ylabel(label, fontsize=FONT_AXIS)
        _cohort_label(axes[ri, -1], ds)
    handles = [Patch(facecolor=_BM_COLORS.get(m, "#999"), alpha=0.85, label=RI_LABELS.get(m, m)) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 8), frameon=False,
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, handletextpad=0.4,
               columnspacing=1.2)
    fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0, 0.985, 1 - 0.3 / fig.get_figheight()))
    return {None: fig}


# ─────────────────────────────────────────────────────────────────────────────
# roc: each method's median operating point on all tests and within Pop_RI-normal tests
# ─────────────────────────────────────────────────────────────────────────────
_ROC_SUBSETS = ("all", "pop_normal")   # subsets of eval.csv the ROC panel contrasts


def _roc_frame(ds, subset):
    ev = load_eval(ds, subset)
    if ev is None or len(ev) == 0:
        return None
    ev["method"] = ev["method"].map(_bm_method)
    to_numeric(ev)
    ev = ev[(~ev.analyte.isin(EXCLUDE_ANALYTES)) & (ev.n >= 100)].copy()
    ev["fpr"] = 1 - ev["specificity"]
    if len(ev):
        # the medians the figure draws, one pseudo-analyte row per method x outcome
        g = ev.groupby(["outcome", "method"])
        summary = g[["sensitivity", "specificity"]].median()
        for col in ("sensitivity", "specificity"):
            summary[f"{col}_q25"] = g[col].quantile(0.25)
            summary[f"{col}_q75"] = g[col].quantile(0.75)
        summary["n"] = g["n"].sum()
    return ev if len(ev) else None


def _roc_summary(sub):
    """Per method: median and quartiles over analytes of (FPR, TPR); a transcribed summary
    (one 'median' row per method with sensitivity_q25 / specificity_q75 ...) is read back."""
    if "sensitivity_q25" in sub.columns:
        s = sub.set_index("method")
        med = s[["fpr", "sensitivity"]]
        q1 = pd.DataFrame({"fpr": 1 - s["specificity_q75"], "sensitivity": s["sensitivity_q25"]})
        q3 = pd.DataFrame({"fpr": 1 - s["specificity_q25"], "sensitivity": s["sensitivity_q75"]})
        return med, q1, q3
    g = sub.groupby("method")[["fpr", "sensitivity"]]
    return g.median(), g.quantile(0.25), g.quantile(0.75)


def _roc_point(ax, x, y, xq, yq, m, filled):
    c = _BM_COLORS.get(m, "#999")
    marker = FAMILY_MARKERS[m.split("_")[0]]
    ax.plot([xq[0], xq[1]], [y, y], color=c, lw=0.7, alpha=0.6, zorder=2)
    ax.plot([x, x], [yq[0], yq[1]], color=c, lw=0.7, alpha=0.6, zorder=2)
    ax.scatter(x, y, s=26, marker=marker, facecolor=c if filled else "white", edgecolor=c,
               linewidth=0.9, zorder=4)


def _roc_panel(ax, frames, methods):
    """frames: {'all': rows, 'pop_normal': rows} for one outcome.  Filled = all tests, hollow
    = within Pop_RI-normal tests, one line per method joining the two; whiskers = IQR over
    analytes.  Pop_RI is left out of the subset (it cannot flag there)."""
    stats = {k: _roc_summary(d) for k, d in frames.items() if d is not None and len(d)}
    for m in methods:
        pts = {}
        for k, (med, q1, q3) in stats.items():
            if m in med.index and not (k == "pop_normal" and m == "PopRI"):
                pts[k] = (med.loc[m, "fpr"], med.loc[m, "sensitivity"],
                          (q1.loc[m, "fpr"], q3.loc[m, "fpr"]), (q1.loc[m, "sensitivity"], q3.loc[m, "sensitivity"]))
        if "all" in pts and "pop_normal" in pts:
            ax.plot([pts["all"][0], pts["pop_normal"][0]], [pts["all"][1], pts["pop_normal"][1]],
                    color=_BM_COLORS.get(m, "#999"), lw=0.6, alpha=0.5, zorder=1)
        for k, (x, y, xq, yq) in pts.items():
            _roc_point(ax, x, y, xq, yq, m, filled=(k == "all"))
    ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=0.6, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=FONT_TICK)
    hide_spines(ax)


def fig_roc():
    """ROC space, rows = cohorts, columns = outcomes: each method's median operating point
    over analytes at its own threshold, filled on all tests and hollow within Pop_RI-normal
    tests, joined by a line.  Every method sits well above the diagonal on all tests; the
    subset drags all of them onto it, because it removes the very values the intervals
    are built to flag -- that subset is what a referee read as 'random'."""
    data = {ds: {k: _roc_frame(ds, k) for k in _ROC_SUBSETS} for ds in COHORTS}
    present = set()
    for frames in data.values():
        for d in frames.values():
            if d is not None:
                present |= set(d.method)
    methods = [m for m in bm_methods() if m in present]
    if not methods:
        return {}
    n_cols = max([d.outcome.nunique() for frames in data.values() for d in frames.values() if d is not None] + [1])
    fig, axes = plt.subplots(len(COHORTS), n_cols, figsize=(1.75 * n_cols + 0.6, 1.85 * len(COHORTS) + 0.7),
                             squeeze=False)
    for ri, ds in enumerate(COHORTS):
        frames = data[ds]
        if all(d is None for d in frames.values()):
            pending_axis(axes[ri, 0], ds)
            for ax in axes[ri, 1:]:
                ax.set_axis_off()
            _cohort_label(axes[ri, -1], ds)
            continue
        seen = set().union(*[set(d.outcome) for d in frames.values() if d is not None])
        outcomes = [o for o in OUTCOMES.get(ds, []) if o in seen]
        for ci, ax in enumerate(axes[ri]):
            if ci >= len(outcomes):
                ax.set_axis_off()
                continue
            per_outcome = {k: (d[d.outcome == outcomes[ci]] if d is not None else None) for k, d in frames.items()}
            _roc_panel(ax, per_outcome, methods)
            ax.set_title(OUTCOME_SHORT.get(outcomes[ci], outcomes[ci]), fontsize=FONT_TITLE, loc="left")
            ax.set_xlabel("False positive rate", fontsize=FONT_AXIS)
            if ci == 0:
                ax.set_ylabel("True positive rate", fontsize=FONT_AXIS)
            else:
                ax.tick_params(labelleft=False)
        _cohort_label(axes[ri, -1], ds)
    handles = [Line2D([], [], ls="", marker=FAMILY_MARKERS[m.split("_")[0]], color=_BM_COLORS.get(m, "#999"),
                      markersize=4, label=RI_LABELS.get(m, m)) for m in methods]
    handles += [Line2D([], [], ls="", marker="o", color=DARK, markersize=4, label="All tests"),
                Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=DARK, markersize=4,
                       label="Within Pop$_{RI}$-normal tests")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 9),
               frameon=False, fontsize=FONT_LEGEND, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.5, h_pad=1.4, rect=(0, 0, 0.985, 1 - 0.3 / fig.get_figheight()))
    return {None: fig}

FIGURES = [
    FigSpec("12_eval", "circos", fig_circos, True, (EVAL_CSV,), _one("circos")),
    FigSpec("12_eval", "methods_all", fig_eval_methods_all, False, (), None),
    FigSpec("12_eval", "methods_all_norma", ablation_variant(fig_eval_methods_all), False, (), None),
    FigSpec("12_eval", "roc", fig_roc, False, (), None),
]


# ══════════════════════════════════════════════════════════════════════════
# Tables — 12_eval: table_* definitions and registry slice.
# ══════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403

# save_table()'s first argument is the folder the table is written into,
# so it must match this directory name. Keeping the literal here (rather
# than only in the TableSpec) is what drifted during the restructure.  # noqa: F401,F403


def table_eval():
    """Mean PerRI vs NORMA metrics per dataset x outcome (Pop_RI-normal, Per_RI-normal subset)."""
    rows = []
    for ds in DATASETS:
        df = _load_eval_restricted(ds)
        if df is None or len(df) == 0:
            for outcome in OUTCOMES.get(ds, []):
                row = {"Dataset": DATASET_DISPLAY[ds], "Outcome": OUTCOME_DISPLAY.get(outcome, outcome)}
                for metric in EVAL_METRICS:
                    ml = EVAL_METRIC_LABELS[metric]
                    row[f"PerRI {ml}"] = row[f"NORMA {ml}"] = row[f"D {ml}"] = "---"
                rows.append(row)
            continue
        df = to_numeric(df)
        for outcome in sorted(df["outcome"].unique()):
            sub = df[df["outcome"] == outcome]
            row = {"Dataset": DATASET_DISPLAY[ds], "Outcome": OUTCOME_DISPLAY.get(outcome, outcome)}
            for metric in EVAL_METRICS:
                ml = EVAL_METRIC_LABELS[metric]
                per = sub[sub["method"] == "PerRI"][metric].dropna(); nor = sub[sub["method"] == "NORMA"][metric].dropna()
                row[f"PerRI {ml}"] = f"{per.mean():.2f}" if len(per) else "---"
                row[f"NORMA {ml}"] = f"{nor.mean():.2f}" if len(nor) else "---"
                row[f"D {ml}"] = f"{nor.mean() - per.mean():+.2f}" if len(per) and len(nor) else "---"
            rows.append(row)
    if not rows:
        return []
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{ll" + "rrr" * len(EVAL_METRICS) + "}", r"\toprule"]
    h1, cmid = " & ", []
    for i, metric in enumerate(EVAL_METRICS):
        h1 += r" & \multicolumn{3}{c}{" + EVAL_METRIC_LABELS[metric] + "}"; cmid.append(f"\\cmidrule(lr){{{3 + i * 3}-{5 + i * 3}}}")
    per_norma = f" & {METHOD_DISPLAY['PerRI']} & {METHOD_DISPLAY['NORMA']} & " + r"$\Delta$"
    lines += [h1 + r" \\", " ".join(cmid), r"Dataset & Outcome" + per_norma * len(EVAL_METRICS) + r" \\", r"\midrule"]
    prev_ds = None
    for row in rows:
        cells = [row["Dataset"] if row["Dataset"] != prev_ds else "", row["Outcome"]]; prev_ds = row["Dataset"]
        for metric in EVAL_METRICS:
            ml = EVAL_METRIC_LABELS[metric]; cells += [row[f"PerRI {ml}"], row[f"NORMA {ml}"], row[f"D {ml}"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("12_eval", "eval", lines, pd.DataFrame(rows))
    return ["eval"]

def table_eval_detail(ds):
    df = _load_eval_restricted(ds)
    if df is None or len(df) == 0:
        return []
    df = to_numeric(df); methods = ["PerRI", "NORMA"]; written = []
    for outcome in sorted(df["outcome"].unique()):
        sub = df[df["outcome"] == outcome]; rows = []
        for a in all_analytes():
            any_match = sub[sub["analyte"] == a]
            row = {"Analyte": a, "N": "---", "Events": "---"}
            if len(any_match):
                for key, col in (("N", "n"), ("Events", "n_events")):
                    value = any_match.iloc[0][col]
                    if pd.notna(value):
                        row[key] = f"{int(value):,}"
            for method in methods:
                match = sub[(sub["analyte"] == a) & (sub["method"] == method)]
                for k in EVAL_METRICS:
                    row[f"{method} {EVAL_METRIC_LABELS[k]}"] = (f"{match.iloc[0][k]:.2f}" if len(match) == 1 and pd.notna(match.iloc[0][k]) else "---")
            rows.append(row)
        lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lrr" + "rrrr" * len(methods) + "}", r"\toprule"]
        h1, cmid = r" & & ", []
        for i, method in enumerate(methods):
            h1 += r" & \multicolumn{4}{c}{" + _ri(method) + "}"; cmid.append(f"\\cmidrule(lr){{{4 + i * 4}-{7 + i * 4}}}")
        lines += [h1 + r" \\", " ".join(cmid), "Analyte & N & Events" + r" & Precision & Sensitivity & Specificity & Accuracy" * len(methods) + r" \\", r"\midrule"]
        for row in rows:
            lines.append(" & ".join([row['Analyte'], row['N'], row['Events']] + [row[f"{m} {EVAL_METRIC_LABELS[k]}"] for m in methods for k in EVAL_METRICS]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        name = f"eval_detail_{ds}_{outcome.lower().replace(' ', '_')}"
        save_table("12_eval", name, lines, pd.DataFrame(rows)); written.append(name)
    return written

def table_eval_interpretable(ds):
    """Per 100 flagged -> how many had the event, one table per dataset (columns = outcome x method)."""
    df = _load_eval_restricted(ds)
    if df is None or len(df) == 0:
        return []
    df = to_numeric(df); methods = ["PerRI", "NORMA"]; outcomes = sorted(df["outcome"].unique()); rows = []
    for a in all_analytes():
        row = {"Analyte": a}
        for outcome in outcomes:
            sub = df[df["outcome"] == outcome]
            for method in methods:
                match = sub[(sub["analyte"] == a) & (sub["method"] == method)]
                v = match.iloc[0]["per_100_flagged_with_event"] if len(match) == 1 else np.nan
                row[f"{outcome}_{method}"] = f"{v:.0f}" if pd.notna(v) else "---"
        rows.append(row)
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l" + "rr" * len(outcomes) + "}", r"\toprule"]
    h1, cmid = "", []
    for i, outcome in enumerate(outcomes):
        h1 += r" & \multicolumn{2}{c}{" + OUTCOME_DISPLAY.get(outcome, outcome) + "}"; cmid.append(f"\\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}")
    lines += [h1 + r" \\", " ".join(cmid), "Analyte" + "".join(" & " + _ri(m) for _ in outcomes for m in methods) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join([row['Analyte']] + [row[f"{o}_{m}"] for o in outcomes for m in methods]) + r" \\")
    lines.append(r"\midrule")
    overall = [r"\textbf{Overall}"]
    for outcome in outcomes:
        sub = df[df["outcome"] == outcome]
        for method in methods:
            v = sub[sub["method"] == method]["per_100_flagged_with_event"].dropna()
            overall.append(f"\\textbf{{{v.mean():.0f}}}" if len(v) else r"\textbf{---}")
    lines += [" & ".join(overall) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    name = f"eval_interpretable_{ds}"
    save_table("12_eval", name, lines, pd.DataFrame(rows))
    return [name]

TABLES = [
    TableSpec("12_eval",          "eval",                    table_eval,                    False, (), None),
    TableSpec("12_eval",          "eval_detail",             table_eval_detail,             True,  (EVAL_CSV,), lambda ds: [f"eval_detail_{ds}_{o}" for o in OUTCOMES.get(ds, [])]),
    TableSpec("12_eval",          "eval_interpretable",      table_eval_interpretable,      True,  (EVAL_CSV,), lambda ds: [f"eval_interpretable_{ds}"]),
]


if __name__ == "__main__":
    main()
