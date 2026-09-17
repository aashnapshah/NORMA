#!/usr/bin/env python
"""Patient-level multi-analyte survival models: one row per patient, features =
each analyte's abnormal flag under one RI method (first measurement within
--window_hours of the patient's first draw; unmeasured = not flagged), plus age
and sex.  Two designs answer "does a personalised interval improve patient-level
risk stratification?":

Usage:
    python 14_patient_level.py --dataset eicu --panels all --no-gbm --no-bootstrap --common-analytes
    python 14_patient_level.py --dataset eicu --only swap --common-analytes
"""
import bootstrap  # noqa: F401

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from constants import ALL_SPLIT
import datasets
from datasets import already_done, EXCLUDE_LABS, PANELS, add_dataset_args, get_dataset, result_path, save_csv


def import_sksurv():
    """Bind the survival-forest names the refit and swap steps use."""
    global GradientBoostingSurvivalAnalysis, concordance_index_censored
    global concordance_index_ipcw, cumulative_dynamic_auc
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                cumulative_dynamic_auc)
from metrics import hours_from_admit

warnings.filterwarnings("ignore")

# Reuse is keyed on these: a step whose files are all present is skipped unless --force
# (datasets.already_done).
STEP_OUTPUTS = {
    "refit": ["nri.csv"],
    "swap": ["concordance.csv"],
}
STEPS = ("refit", "swap")
REF_METHOD = "PopRI"


# shared: patient-wide frames, splits, survival helpers

def common_analytes(classified, analytes, methods):
    """Analytes for which EVERY method produced intervals.  Patient-level models pool
    analytes into one feature vector, so a method missing an analyte would be
    silently credited with "normal" for it."""
    covered = {}
    for m in methods:
        col = f"{m}_class"
        covered[m] = set()
        if col in classified.columns:
            covered[m] = set(classified.loc[classified[col].notna(), "analyte"].unique())
    common = set(analytes)
    for m in methods:
        common &= covered[m]
    dropped = sorted(set(analytes) - common)
    if dropped:
        print(f"  --common-analytes: dropping {len(dropped)} analyte(s) not covered by all methods: "
              f"{dropped}")
        for m in methods:
            missing = sorted(set(analytes) - covered[m])
            if missing:
                print(f"      {m}: missing {missing}")
    return sorted(common)


def patient_wide(classified, method, outcome_cfg, analytes, window_hours=48, time_unit=None):
    """One row per patient: abnormal_<a> (1 / 0 / NaN if not measured), measured_<a>,
    n_analytes_measured, age, sex, event, duration.  Uses the FIRST measurement per
    analyte within `window_hours` of the patient's first measurement (hours since
    admission via lib/exposure.hours_from_admit, so unit-independent)."""
    cls_col = f"{method}_class"
    if "t_hours" in classified.columns:
        t_hours = pd.to_numeric(classified["t_hours"], errors="coerce")
    else:
        t_hours = hours_from_admit(classified, time_unit)
    windowed = classified.assign(_t_hours=t_hours.values)
    windowed = windowed[windowed["_t_hours"].notna()]
    first_draw = windowed.groupby("patient_id")["_t_hours"].transform("min")
    windowed = windowed[windowed["_t_hours"] - first_draw <= window_hours]

    parts = []
    for analyte in analytes:
        lab = windowed[windowed["analyte"] == analyte].sort_values("timestamp").copy()
        if len(lab) == 0:
            continue
        valid = lab[cls_col].notna()
        lab["_abn"] = np.nan
        lab.loc[valid, "_abn"] = (lab.loc[valid, cls_col] != 1).astype(int)
        first = lab.groupby("patient_id").first()
        parts.append(pd.DataFrame({f"abnormal_{analyte}": first['_abn'],
                                   f"measured_{analyte}": 1}, index=first.index))
    if not parts:
        return pd.DataFrame()
    wide = parts[0].join(parts[1:], how="outer") if len(parts) > 1 else parts[0]
    measured_cols = [c for c in wide.columns if c.startswith("measured_")]
    wide[measured_cols] = wide[measured_cols].fillna(0)
    wide["n_analytes_measured"] = wide[measured_cols].sum(axis=1)   # abnormal_* stay NaN when unmeasured

    demographics = (classified.sort_values("timestamp").groupby("patient_id")
                    .agg(age=("age", "first"), sex=("sex", "first"),
                         event=(outcome_cfg["event_col"], "first"),
                         event_time=(outcome_cfg.get("time_col"), "first"),
                         censor_time=(outcome_cfg.get("censor_col"), "first")))
    patients = wide.join(demographics, how="inner").reset_index()
    for col in ("event", "sex", "age", "event_time", "censor_time"):
        patients[col] = pd.to_numeric(patients[col], errors="coerce")
    patients["duration"] = np.where(patients["event"] == 1, patients["event_time"], patients["censor_time"])
    patients = patients.dropna(subset=["duration", "event", "age", "sex"])
    patients = patients[patients["duration"] > 0]
    return patients.drop(columns=["event_time", "censor_time"])


def stratified_split(df, test_frac=0.4, seed=42):
    try:
        train, test = train_test_split(df.index, test_size=test_frac, random_state=seed, stratify=df["event"])
    except ValueError:
        train, test = train_test_split(df.index, test_size=test_frac, random_state=seed)
    return df.loc[train], df.loc[test]


def survival_array(event, duration):
    return np.array(list(zip(event.astype(bool), duration.astype(float))),
                    dtype=[("event", bool), ("duration", float)])


def valid_windows(y_train, windows):
    """Windows inside the range cumulative_dynamic_auc accepts."""
    event_times = y_train["duration"][y_train["event"]]
    if not len(windows) or not len(event_times):
        return []
    return [t for t in windows if event_times.min() < t < y_train["duration"].max() * 0.999]


def ipcw_concordance(y_train, y_test, risk, t):
    try:
        return concordance_index_ipcw(y_train, y_test, risk, tau=t)[0]
    except Exception:
        return np.nan


def bootstrap_concordance(test_df, risk, n_boot=200, seed=42):
    """Bootstrap 95% CI for the test-set concordance."""
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.choice(len(test_df), size=len(test_df), replace=True)
        boot = test_df.iloc[idx]
        y = survival_array(boot["event"], boot["duration"])
        try:
            values.append(concordance_index_censored(y["event"], y["duration"], risk[idx])[0])
        except Exception:
            continue
    if len(values) < 50:
        return np.nan, np.nan
    return round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)


def window_metrics(y_train, y_test, risk, eval_windows):
    """{t: {auc, c_ipcw}} for every eval window (NaN AUC outside the valid range)."""
    out = {}
    windows = valid_windows(y_train, eval_windows)
    if windows:
        try:
            aucs, _ = cumulative_dynamic_auc(y_train, y_test, risk, windows)
            for t, auc in zip(windows, aucs):
                out[t] = {"auc": round(float(auc), 4), "c_ipcw": ipcw_concordance(y_train, y_test, risk, t)}
        except Exception:
            pass
    for t in eval_windows:
        if t not in out:
            out[t] = {"auc": np.nan, "c_ipcw": ipcw_concordance(y_train, y_test, risk, t)}
    return out


def nri_from_moves(df, up, down):
    """Net reclassification from boolean up / down moves against the reference."""
    n_events = int(df["event"].sum())
    n_nonevents = len(df) - n_events
    if len(df) < 10 or n_events < 3 or n_nonevents < 3:
        return None
    is_event = df["event"] == 1
    events_up, events_down = int((up & is_event).sum()), int((down & is_event).sum())
    nonevents_up, nonevents_down = int((up & ~is_event).sum()), int((down & ~is_event).sum())
    nri_events = (events_up - events_down) / n_events
    nri_nonevents = (nonevents_down - nonevents_up) / n_nonevents
    return {
        "n": len(df), "n_events": n_events, "n_nonevents": n_nonevents,
        "n_reclassified": int(up.sum() + down.sum()),
        "events_up": events_up, "events_down": events_down,
        "nonevents_up": nonevents_up, "nonevents_down": nonevents_down,
        "NRI": round(nri_events + nri_nonevents, 4),
        "NRI_events": round(nri_events, 4), "NRI_nonevents": round(nri_nonevents, 4),
    }


def bootstrap_nri(df, moves, n_boot=1000, seed=42):
    """SE and p-value of the NRI from patient resamples; moves(boot) -> (up, down)."""
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_boot):
        boot = df.iloc[rng.choice(len(df), size=len(df), replace=True)]
        n_events = boot["event"].sum()
        n_nonevents = len(boot) - n_events
        if n_events < 1 or n_nonevents < 1:
            continue
        up, down = moves(boot)
        is_event = boot["event"] == 1
        nri_events = ((up & is_event).sum() - (down & is_event).sum()) / n_events
        nri_nonevents = ((down & ~is_event).sum() - (up & ~is_event).sum()) / n_nonevents
        values.append(nri_events + nri_nonevents)
    if len(values) < 100:
        return np.nan, np.nan
    values = np.array(values)
    se = values.std()
    p = 2 * (1 - stats.norm.cdf(abs(values.mean() / se))) if se > 0 else 1.0
    return round(se, 4), round(p, 4)


def load_classified(ds, args):
    classified = ds.load_classification()
    classified["age"] = pd.to_numeric(classified["age"], errors="coerce")
    classified["sex"] = pd.to_numeric(classified["sex"], errors="coerce")
    classified["analyte"] = classified["analyte"].replace("", "NA").fillna("NA")
    missing = [o for o in ds.primary_outcomes
               if o in ds.outcomes and ds.outcomes[o]["event_col"] not in classified.columns]
    if missing:
        print(f"  Attaching outcomes: {missing}")
        classified = ds.attach_outcomes(classified)
    analytes = sorted(a for a in classified["analyte"].unique() if a not in EXCLUDE_LABS)
    if args.common_analytes:
        analytes = common_analytes(classified, analytes, ds.methods)
    print(f"  {len(classified)} measurements, {classified['patient_id'].nunique()} patients, "
          f"{len(analytes)} analytes, methods: {ds.methods}")
    return classified, analytes


# refit

def fit_gbm(train_df, test_df, features, eval_windows):
    """GradientBoostingSurvivalAnalysis -> (metrics, importances, window_metrics) or Nones."""
    y_train = survival_array(train_df["event"], train_df["duration"])
    y_test = survival_array(test_df["event"], test_df["duration"])
    model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=3, learning_rate=0.1,
                                             subsample=0.8, min_samples_leaf=10, random_state=42)
    try:
        model.fit(train_df[features].values, y_train)
    except Exception as e:
        print(f"        [GBM fit failed]: {e}")
        return None, None, None
    risk_train = model.predict(train_df[features].values)
    risk_test = model.predict(test_df[features].values)
    c_train = concordance_index_censored(y_train["event"], y_train["duration"], risk_train)[0]
    c_test = concordance_index_censored(y_test["event"], y_test["duration"], risk_test)[0]
    metrics = {
        "n_train": len(train_df), "n_test": len(test_df),
        "n_events_train": int(train_df["event"].sum()), "n_events_test": int(test_df["event"].sum()),
        "concordance_train": round(c_train, 4), "concordance_test": round(c_test, 4),
    }
    importances = {f: float(v) for f, v in zip(features, model.feature_importances_)}
    return metrics, importances, window_metrics(y_train, y_test, risk_test, eval_windows)


def fit_penalized_cox(train_df, test_df, features, eval_windows, penalizer=0.1):
    """L2-penalised CoxPH -> (metrics, coefs, window_metrics, predictions) or Nones."""
    kept = [c for c in features if train_df[c].std() > 0]     # zero variance -> NaN gradient
    dropped = sorted(set(features) - set(kept))
    if dropped:
        shown = f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}"
        print(f"        [Cox] dropped {len(dropped)} zero-var cols: {shown}")
    features = kept
    cox_cols = ["duration", "event"] + features
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    try:
        cph.fit(train_df[cox_cols], duration_col="duration", event_col="event")
        c_train = cph.concordance_index_
        c_test = cph.score(test_df[cox_cols], scoring_method="concordance_index")
    except Exception as e:
        print(f"        [Cox fit failed]: {e}")
        return None, None, None, None

    y_train = survival_array(train_df["event"], train_df["duration"])
    y_test = survival_array(test_df["event"], test_df["duration"])
    risk_train = cph.predict_log_partial_hazard(train_df[features]).values
    risk_test = cph.predict_log_partial_hazard(test_df[features]).values
    c_lower, c_upper = bootstrap_concordance(test_df, risk_test)
    metrics = {
        "n_train": len(train_df), "n_test": len(test_df),
        "n_events_train": int(train_df["event"].sum()), "n_events_test": int(test_df["event"].sum()),
        "concordance_train": round(c_train, 4), "concordance_test": round(c_test, 4),
        "concordance_lower": c_lower, "concordance_upper": c_upper,
    }
    coefs = {}
    for feature in features:
        if feature not in cph.summary.index:
            continue
        ci = cph.confidence_intervals_.loc[feature]
        coefs[feature] = {
            "HR": round(float(cph.hazard_ratios_[feature]), 4),
            "HR_lower": round(float(np.exp(ci.iloc[0])), 4),
            "HR_upper": round(float(np.exp(ci.iloc[1])), 4),
            "p_value": float(cph.summary.loc[feature, "p"]),
        }
    predictions = []
    for split, d, risk in (("train", train_df, risk_train), ("test", test_df, risk_test)):
        p = d[["patient_id", "duration", "event"]].copy()
        p["risk_score"] = risk
        p["split"] = split
        predictions.append(p)
    return metrics, coefs, window_metrics(y_train, y_test, risk_test, eval_windows), pd.concat(predictions)


def metrics_rows(metrics, windows, model_type, method, outcome, subset, panel):
    """One row for the full-horizon model + one row per time window."""
    base = {"model_type": model_type, "method": method, "outcome": outcome,
            "subset": subset, "panel": panel, **metrics}
    rows = [{**base, "time_window": "all", "auc": np.nan, "c_ipcw": np.nan}]
    for t, w in windows.items():
        rows.append({**base, "time_window": t, **w})
    return rows


def importance_rows(values, model_type, method, outcome, subset, panel, is_cox):
    rows = []
    for feature, value in values.items():
        if feature.startswith("abnormal_"):
            analyte, kind = feature[len("abnormal_"):], "abnormal"
        elif feature.startswith("measured_"):
            analyte, kind = feature[len("measured_"):], "measured"
        else:
            analyte, kind = feature, "covariate"
        row = {"feature": feature, "analyte": analyte, "feat_type": kind, "model_type": model_type,
               "method": method, "outcome": outcome, "subset": subset, "panel": panel}
        if is_cox:
            row.update(importance=value["HR"], **value)
        else:
            row.update(importance=round(value, 6), HR=np.nan, HR_lower=np.nan, HR_upper=np.nan,
                       p_value=np.nan)
        rows.append(row)
    return rows


def fdr_by_outcome(importance):
    """BH FDR on the penalised-Cox p-values, per outcome."""
    if importance.empty or "p_value" not in importance.columns:
        return importance
    importance = importance.copy()
    importance["p_fdr"] = np.nan
    is_cox = importance["model_type"] == "penalized_cox"
    for outcome in importance.loc[is_cox, "outcome"].unique():
        mask = is_cox & (importance["outcome"] == outcome) & importance["p_value"].notna()
        if mask.sum():
            p_values = importance.loc[mask, "p_value"].values
            importance.loc[mask, "p_fdr"] = multipletests(p_values, method="fdr_bh")[1]
    return importance


def risk_flag_nri(ref_flags, new_flags, events, bootstrap=True):
    """Category NRI between the reference and new patient-level risk categories
    (1 = high risk, 0 = low risk, from the two Cox models' risk scores)."""
    df = pd.DataFrame({"ref": ref_flags, "new": new_flags, "event": events}).dropna()

    def moves(d):
        return (d['ref'] == 0) & (d['new'] == 1), (d['ref'] == 1) & (d['new'] == 0)

    result = nri_from_moves(df, *moves(df))
    if result is None:
        return None
    result["NRI_se"], result["NRI_p"] = bootstrap_nri(df, moves) if bootstrap else (np.nan, np.nan)
    result['ref_high_risk_rate'] = round(df['ref'].mean(), 4)
    result["new_high_risk_rate"] = round(df["new"].mean(), 4)
    return result


def refit_nri(predictions, new_methods, panel="all", bootstrap=True):
    """NRI from the refit models' test-set risk scores: high risk = above the reference
    model's median; the same cutoff is applied to each new model."""
    preds = predictions[(predictions["panel"] == panel) & (predictions["split"] == "test")]
    if len(preds) == 0:
        print(f"  NRI: no predictions found for panel={panel}")
        return []
    rows = []
    for (outcome, subset), grp in preds.groupby(["outcome", "subset"]):
        ref = grp[grp["method"] == REF_METHOD].set_index("patient_id")
        if len(ref) == 0:
            continue
        cutoff = np.median(ref["risk_score"])
        ref_flags = (ref["risk_score"] >= cutoff).astype(int)
        for method in new_methods:
            new = grp[grp["method"] == method].set_index("patient_id")   # exact match, not a prefix
            if len(new) == 0:
                continue
            new_flags = (new["risk_score"] >= cutoff).astype(int)
            common = ref_flags.index.intersection(new_flags.index)
            if len(common) < 10:
                continue
            result = risk_flag_nri(ref_flags.loc[common], new_flags.loc[common], ref.loc[common, "event"],
                                   bootstrap=bootstrap)
            if result is None:
                continue
            result.update(outcome=outcome, ref_method=REF_METHOD, new_method=method, subset=subset)
            rows.append(result)
            print(f"      → {method:<12s} NRI={result['NRI']:+.3f}  "
                  f"(+:{result['NRI_events']:+.3f}  -:{result['NRI_nonevents']:+.3f})")
    return rows


def panel_map(analytes, requested):
    """{panel name: analytes present}; 'all' = every analyte."""
    available = {"all": analytes}
    for name, codes in PANELS.items():
        present = [a for a in codes if a in analytes]
        if len(present) >= 2:
            available[name] = present
    if requested is None:
        return available
    unknown = [k for k in requested if k not in available]
    if unknown:
        print(f"  Warning: unknown panels {unknown}, available: {list(available)}")
    return {k: available[k] for k in requested if k in available}


def merge_into_existing(df, path, keys):
    """Keep rows of the existing file whose keys this run did not produce (e.g. the
    model type that was skipped), then overwrite.  datasets.save_csv does the
    upsert; this only reports it."""
    save_csv(df, path, keys=keys)
    print(f"  Saved {len(df)} rows -> {path}")


class RefitResults:
    def __init__(self):
        self.metrics, self.importance, self.predictions, self.nri = [], [], [], []


def fit_panel(results, args, eval_windows, train_df, test_df, features, method, outcome, subset, panel):
    """Fit the requested models on one (method, outcome, subset, panel) and record them."""
    if not args.no_cox:
        metrics, coefs, windows, preds = fit_penalized_cox(train_df, test_df, features, eval_windows,
                                                           args.penalizer)
        if metrics is not None:
            if not subset.endswith("_eval_pop_normal"):
                by_window = "  ".join(f"{t}d={w['c_ipcw']:.3f}" for t, w in sorted(windows.items())
                                      if not np.isnan(w["c_ipcw"]))
                print(f"      {method:<12s} C={metrics['concordance_test']:.3f} "
                      f"({metrics['concordance_lower']}-{metrics['concordance_upper']})  {by_window}")
            results.metrics += metrics_rows(metrics, windows, "penalized_cox", method, outcome, subset, panel)
            results.importance += importance_rows(coefs, "penalized_cox", method, outcome, subset, panel,
                                                  is_cox=True)
            preds = preds.assign(method=method, outcome=outcome, subset=subset, panel=panel)
            results.predictions.append(preds)
    if not args.no_gbm:
        metrics, importances, windows = fit_gbm(train_df, test_df, features, eval_windows)
        if metrics is not None:
            by_window = "  ".join(f"{t}d={w['c_ipcw']:.3f}" for t, w in sorted(windows.items())
                                  if not np.isnan(w["c_ipcw"]))
            print(f"      {method:<12s} GBM C={metrics['concordance_test']:.3f}  {by_window}")
            results.metrics += metrics_rows(metrics, windows, "gbm", method, outcome, subset, panel)
            results.importance += importance_rows(importances, "gbm", method, outcome, subset, panel,
                                                  is_cox=False)


def refit_models(ds, classified, analytes, args, results_dir):
    panels = panel_map(analytes, args.panels)
    eval_windows = getattr(ds, "eval_windows", [])
    if eval_windows:
        print(f"  Eval windows: {eval_windows} (days)")
    print(f"  Panels: {list(panels)}")
    new_methods = [m for m in ds.methods if m != REF_METHOD]
    pop_normal_pids = None
    if "PopRI_class" in classified.columns:
        pop_normal_pids = set(classified.loc[classified["PopRI_class"] == 1, "patient_id"])
    results = RefitResults()
    subset = "all"

    for panel, panel_analytes in panels.items():
        print(f"\n  ── panel={panel} ({len(panel_analytes)} analytes) ──")
        n_required = max(2, int(len(panel_analytes) * args.min_coverage))
        for outcome in ds.primary_outcomes:
            cfg = ds.outcomes.get(outcome)
            if cfg is None or cfg["event_col"] not in classified.columns:
                print(f"    Skipping {outcome} (column not found)")
                continue
            event_rate = classified.drop_duplicates("patient_id")[cfg["event_col"]].mean()
            print(f"\n    {outcome} (event rate: {event_rate:.3f})")
            for method in ds.methods:
                patients = patient_wide(classified, method, cfg, analytes, time_unit=ds.time_unit)
                if patients.empty:
                    continue
                # this fraction of the panel must be measured (no imputation for the rest)
                patients = patients[patients["n_analytes_measured"] >= n_required]
                if len(patients) < 20 or patients["event"].sum() < 5:
                    continue
                train_df, test_df = stratified_split(patients)
                if len(train_df) < 10 or train_df["event"].sum() < 3 or len(test_df) < 5:
                    continue
                abn_cols = sorted(f"abnormal_{a}" for a in panel_analytes
                                  if f"abnormal_{a}" in patients.columns)
                if len(abn_cols) < 2:
                    continue
                # unmeasured analytes (allowed by --min-coverage < 1) enter as "not flagged", as
                # in the swap design; lifelines rejects NaN features
                train_df = train_df.copy()
                test_df = test_df.copy()
                train_df[abn_cols] = train_df[abn_cols].fillna(0)
                test_df[abn_cols] = test_df[abn_cols].fillna(0)
                features = abn_cols + ["age", "sex"]
                fit_panel(results, args, eval_windows, train_df, test_df, features, method, outcome,
                          subset, panel)
                if pop_normal_pids is not None:
                    test_pop = test_df[test_df["patient_id"].isin(pop_normal_pids)]
                    if len(test_pop) >= 5 and test_pop["event"].sum() >= 1:
                        fit_panel(results, args, eval_windows, train_df, test_pop, features, method, outcome,
                                  f"{subset}_eval_pop_normal", panel)
            # NRI once every method has a model for this outcome
            if results.predictions:
                preds = pd.concat(results.predictions, ignore_index=True)
                here = preds[(preds["outcome"] == outcome) & (preds["subset"] == subset)
                             & (preds["panel"] == panel)]
                if len(here):
                    results.nri += refit_nri(here, new_methods, panel=panel, bootstrap=not args.no_bootstrap)

    if not results.metrics:
        print("\n  No results produced — check data coverage.")
        return results.nri
    keys = ["model_type", "method", "outcome", "subset", "panel", "time_window"]
    metrics = pd.DataFrame(results.metrics)
    importance = fdr_by_outcome(pd.DataFrame(results.importance))
    merge_into_existing(metrics, os.path.join(results_dir, "concordance.csv"), keys)
    merge_into_existing(importance, os.path.join(results_dir, "multi_analyte_importance.csv"), keys + ["feature"])
    if results.predictions:
        preds = pd.concat(results.predictions, ignore_index=True)
        path = result_path(results_dir, "multi_analyte_predictions.csv")
        preds.to_csv(path, index=False)
        print(f"  Saved {len(preds)} predictions -> {path}")
    return results.nri


def run_refit(ds, classified, analytes, args, results_dir):
    # The models are refit only when there are no saved predictions to score, or --force asks for
    # it: retraining every panel to write back predictions that are already on disk is the
    # expensive half of...
    path = datasets.stage_path(results_dir, "multi_analyte_predictions.csv", "14_patient_level")
    if not args.force and os.path.exists(path):
        preds = pd.read_csv(path)
        print(f"\n  NRI: loaded {len(preds)} saved predictions from {path} "
              f"(use --force to refit)")
        new_methods = [m for m in ds.methods if m != REF_METHOD]
        nri = refit_nri(preds, new_methods, panel="all", bootstrap=not args.no_bootstrap)
    else:
        nri = refit_models(ds, classified, analytes, args, results_dir)
    if not nri:
        print("\n  NRI: no valid comparisons found")
        return
    path = os.path.join(results_dir, "nri.csv")
    save_csv(pd.DataFrame(nri).assign(design="refit"), path, keys=("design",))
    print(f"\n  Saved {len(nri)} NRI rows -> {path}")


# swap

def count_nri(ref_counts, new_counts, events, bootstrap=True):
    """Continuous NRI on abnormal-analyte counts: up = the new method flags MORE
    analytes than Pop_RI (good for events), down = fewer (good for non-events)."""
    df = pd.DataFrame({"ref": ref_counts, "new": new_counts, "event": events}).dropna()

    def moves(d):
        return d['new'] > d['ref'], d['new'] < d['ref']

    result = nri_from_moves(df, *moves(df))
    if result is None:
        return None
    result["NRI_se"], result["NRI_p"] = bootstrap_nri(df, moves) if bootstrap else (np.nan, np.nan)
    result['ref_mean_abnormal'] = round(df['ref'].mean(), 2)
    result["new_mean_abnormal"] = round(df["new"].mean(), 2)
    return result


def swap_frames(ds, classified, analytes, cfg, n_required):
    """{method: patient-wide frame} of the methods with enough covered patients."""
    frames = {}
    for method in ds.methods:
        patients = patient_wide(classified, method, cfg, analytes, time_unit=ds.time_unit)
        if patients.empty:
            continue
        patients = patients[patients["n_analytes_measured"] >= n_required]
        if len(patients) >= 20:
            frames[method] = patients
    return frames


def score_swapped(cph, features, abn_cols, frames, test_df, y_train, y_test, eval_windows, outcome):
    """C-index (and windowed AUC) of every method's flags through the Pop_RI-trained model."""
    rows = {}
    test_pids = test_df["patient_id"].values
    for method, patients in frames.items():
        swapped = patients.set_index("patient_id").reindex(test_pids)
        swapped[abn_cols] = swapped[abn_cols].fillna(0)
        risk = cph.predict_log_partial_hazard(swapped[features]).values
        y = survival_array(swapped["event"], swapped["duration"])
        c_index = concordance_index_censored(y["event"], y["duration"], risk)[0]
        c_lower, c_upper = bootstrap_concordance(test_df, risk)
        by_window = []
        rows[method] = []
        windows = valid_windows(y_train, eval_windows)
        if windows:
            try:
                aucs, _ = cumulative_dynamic_auc(y_train, y_test, risk, windows)
                for t, auc in zip(windows, aucs):
                    by_window.append(f"{t}d={auc:.3f}")
                    rows[method].append({"method": method, "outcome": outcome, "time_window": t,
                                         "auc": round(float(auc), 4), "concordance_test": np.nan})
            except Exception:
                pass
        print(f"    {method:<16s} C={c_index:.3f} ({c_lower}-{c_upper})  {'  '.join(by_window)}")
        rows[method].append({
            "method": method, "outcome": outcome, "time_window": "all",
            "concordance_test": round(c_index, 4), "concordance_lower": c_lower, "concordance_upper": c_upper,
            "n_test": len(test_df), "n_events_test": int(test_df["event"].sum()), "auc": np.nan,
        })
    return [r for method_rows in rows.values() for r in method_rows]


def run_swap(ds, classified, analytes, args, results_dir):
    new_methods = [m for m in ds.methods if m != REF_METHOD]
    n_required = max(1, int(len(analytes) * args.min_coverage))
    print(f"  Require >= {n_required}/{len(analytes)} analytes measured ({args.min_coverage:.0%} coverage)")
    eval_windows = getattr(ds, "eval_windows", [])
    metric_rows, nri_rows = [], []

    for outcome in ds.primary_outcomes:
        cfg = ds.outcomes.get(outcome)
        if cfg is None or cfg["event_col"] not in classified.columns:
            continue
        print(f"\n  ── {outcome} ──")
        frames = swap_frames(ds, classified, analytes, cfg, n_required)
        if REF_METHOD not in frames:
            print(f"    No data for {REF_METHOD}, skipping")
            continue
        common_pids = set(frames[REF_METHOD]["patient_id"])
        for method in new_methods:
            if method in frames:
                common_pids &= set(frames[method]["patient_id"])

        ref = frames[REF_METHOD]
        ref = ref[ref["patient_id"].isin(common_pids)].copy()
        abn_cols = sorted(c for c in ref.columns if c.startswith("abnormal_"))
        ref[abn_cols] = ref[abn_cols].fillna(0)
        print(f"    {len(common_pids)} common patients, {len(abn_cols)} analytes, "
              f"{int(ref['event'].sum())} events")
        if len(ref) < 20 or ref["event"].sum() < 5:
            continue

        train_df, test_df = stratified_split(ref)
        features = [c for c in abn_cols + ["age", "sex"] if train_df[c].std() > 0]
        cph = CoxPHFitter(penalizer=args.penalizer, l1_ratio=0.0)
        try:
            cph.fit(train_df[["duration", "event"] + features], duration_col="duration", event_col="event")
        except Exception as e:
            print(f"    Cox fit failed: {e}")
            continue
        y_train = survival_array(train_df["event"], train_df["duration"])
        y_test = survival_array(test_df["event"], test_df["duration"])
        metric_rows += score_swapped(cph, features, abn_cols, frames, test_df, y_train, y_test,
                                     eval_windows, outcome)

        # NRI on the number of flagged analytes, Pop_RI vs each method, on the test patients
        ref_test = test_df.set_index("patient_id")
        ref_counts = ref_test[abn_cols].sum(axis=1)
        for method in new_methods:
            if method not in frames:
                continue
            patients = frames[method].set_index("patient_id")
            patients = patients[patients.index.isin(ref_test.index)]
            new_cols = [c for c in patients.columns if c.startswith("abnormal_")]
            new_counts = patients[new_cols].fillna(0).sum(axis=1)
            common = ref_counts.index.intersection(new_counts.index)
            if len(common) < 10:
                continue
            result = count_nri(ref_counts.loc[common], new_counts.loc[common], ref_test.loc[common, "event"],
                               bootstrap=not args.no_bootstrap)
            if result is None:
                continue
            result.update(outcome=outcome, ref_method=REF_METHOD, new_method=method)
            nri_rows.append(result)
            print(f"      → {method:<16s} NRI={result['NRI']:+.3f}  "
                  f"(+:{result['NRI_events']:+.3f}  -:{result['NRI_nonevents']:+.3f})  "
                  f"avg abnormal: {result['ref_mean_abnormal']:.1f}→{result['new_mean_abnormal']:.1f}")

    if metric_rows:
        # the same statistic as the panel models, so the same file: model_type says which
        swap = pd.DataFrame(metric_rows).assign(model_type="swap")
        for dim in ("panel", "subset"):
            if dim not in swap.columns:
                swap[dim] = ALL_SPLIT
        merge_into_existing(swap, os.path.join(results_dir, "concordance.csv"),
                            ["model_type", "method", "outcome", "subset", "panel", "time_window"])
    if nri_rows:
        path = os.path.join(results_dir, "nri.csv")
        swap = pd.DataFrame(nri_rows).assign(design="swap")
        if "subset" not in swap.columns:      # the swap design scores every stay
            swap["subset"] = ALL_SPLIT
        save_csv(swap, path, keys=("design",))
        print(f"  Saved {len(nri_rows)} NRI rows -> {path}")
    else:
        print("\n  No NRI results produced")


# main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--common-analytes", action="store_true",
                   help="restrict to analytes covered by every method, so the comparison is like-for-like")
    p.add_argument("--min-coverage", type=float, default=0.75,
                   help="fraction of the panel's analytes a patient must have measured within the window")
    p.add_argument("--penalizer", type=float, default=0.1, help="L2 penalty strength for CoxPH")
    p.add_argument("--no-bootstrap", action="store_true", help="skip the NRI bootstrap (faster, no SE / p-value)")
    g = p.add_argument_group("refit")
    g.add_argument("--panels", nargs="+", default=None,
                   help="panels to run (default: all CBC BMP HFP Lipid); 'all' = full-analyte only")
    g.add_argument("--no-gbm", action="store_true", help="skip the GradientBoosting model")
    g.add_argument("--no-cox", action="store_true", help="skip the penalised Cox model")
    args = p.parse_args()
    import_sksurv()          # only the analysis needs the survival forest

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    # Reuse is the default: drop any step whose output is already written.
    todo = [s for s in args.only
            if not already_done(args, results_dir, *STEP_OUTPUTS[s], label=s)]
    if not todo:
        return
    classified, analytes = load_classified(ds, args)
    if "refit" in todo:
        print("=== refit ===")
        run_refit(ds, classified, analytes, args, results_dir)
    if "swap" in todo:
        print("=== swap ===")
        run_swap(ds, classified, analytes, args, results_dir)


# Figures and tables

from figlib import *  # noqa: F401,F403

_RI_MARKER = FAMILY_MARKERS   # lib/models.py, via figlib
COHORTS = VAL_COHORTS         # eicu, inspire, chs: one row each

REFIT_TITLE = "Cox refit per method"
SWAP_TITLE = "Pop$_{RI}$-trained Cox, flags swapped"
POP_NORMAL_TITLE = "Cox refit, scored inside Pop$_{RI}$"


def _outcome_label(o, wrap=False):
    label = OUTCOME_SHORT.get(o, OUTCOME_DISPLAY.get(o, o))
    if wrap:   # two-word outcomes overlap on a narrow bar axis
        label = label.replace(" ", "\n")
    return label


def _cohort_label(ax, ds):
    """Cohort name on the right of the row's last axis (Aashna 2026-08-31)."""
    ax.text(1.04, 0.5, DATASET_DISPLAY.get(ds, ds), transform=ax.transAxes,
            rotation=270, ha="left", va="center", fontsize=FONT_AXIS, color=DARK)


def _top_legend(fig, handles, H):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / H),
               ncol=len(handles), frameon=False, fontsize=FONT_LEGEND,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)


# loading
def _refit_cindex(ds, subset):
    """Penalised Cox on the full panel, all follow-up. subset 'all' = scored on
    every test patient; 'all_eval_pop_normal' = scored only on patients whose
    index values are inside Pop_RI (the reclassification question)."""
    df = load_result(ds, "concordance.csv")
    if df is None or len(df) == 0:
        return None
    df["method"] = df["method"].map(_bm_method)
    to_numeric(df)
    keep = (
        (df.model_type == "penalized_cox")
        & (df.panel == "all")
        & (df.time_window.astype(str) == "all")
        & (df.subset == subset)
    )
    sub = df[keep]
    return sub if len(sub) else None


def _swap_cindex(ds):
    df = load_result(ds, "concordance.csv")
    if df is not None and "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == "swap"]
    if df is None or len(df) == 0:
        return None
    df["method"] = df["method"].map(_bm_method)
    to_numeric(df)
    sub = df[df.time_window.astype(str) == "all"]
    return sub if len(sub) else None


def _nri_raw(ds, design):
    """nri.csv rows for one design, unprocessed."""
    df = load_result(ds, "nri.csv", normalize=False)
    if df is None or "design" not in df.columns:
        return df
    df = df[df["design"].astype(str) == design]
    return df if len(df) else None


def _nri(ds, design):
    """One design of nri.csv: "refit" (Cox refit per method) or "swap" (the
    Pop_RI-trained model scored with each method's flags)."""
    df = load_result(ds, "nri.csv", normalize=False)
    if df is not None and "design" in df.columns:
        df = df[df["design"].astype(str) == design]
    if df is None or len(df) == 0:
        return None
    df["new_method"] = df["new_method"].map(_bm_method)
    to_numeric(df, skip=("outcome", "subset", "new_method", "ref_method"))
    if "subset" in df.columns:
        df = df[df.subset == "all"]
    return df if len(df) else None


def _methods_present(frames, col):
    have = set()
    for d in frames:
        if d is not None:
            have |= set(d[col])
    return have


def _outcomes_present(ds, frames):
    have = set()
    for d in frames:
        if d is not None:
            have |= set(d.outcome)
    return [o for o in OUTCOMES.get(ds, []) if o in have]


# cindex: rows = cohorts, columns = refit | swap | refit scored inside Pop_RI
def _cindex_bars(ax, sub, outcomes, methods):
    x = np.arange(len(outcomes))
    n_m = len(methods)
    bw = 0.8 / n_m
    for mi, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for o in outcomes:
            r = sub[(sub.outcome == o) & (sub.method == m)]
            if len(r) == 0:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
                continue
            r = r.iloc[-1]
            c = float(r.concordance_test)
            cl = r.get("concordance_lower", np.nan)
            cu = r.get("concordance_upper", np.nan)
            vals.append(c)
            lo.append(c - cl if pd.notna(cl) else 0)
            hi.append(cu - c if pd.notna(cu) else 0)
        ax.bar(x + (mi - (n_m - 1) / 2) * bw, vals, bw, color=_BM_COLORS[m], alpha=0.9, yerr=[lo, hi],
               error_kw=dict(ecolor=DARK, elinewidth=0.5, capsize=1.5, capthick=0.5))
    ax.axhline(0.5, color=DARK, lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_outcome_label(o, wrap=True) for o in outcomes], fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    hide_spines(ax)


def fig_cindex():
    columns = [
        (REFIT_TITLE, lambda ds: _refit_cindex(ds, "all")),
        (SWAP_TITLE, _swap_cindex),
        (POP_NORMAL_TITLE, lambda ds: _refit_cindex(ds, "all_eval_pop_normal")),
    ]
    rows = {ds: [load(ds) for _, load in columns] for ds in COHORTS}
    if all(d is None for frames in rows.values() for d in frames):
        return {}
    have = _methods_present([d for frames in rows.values() for d in frames], "method")
    methods = [m for m in bm_methods() if m in have]

    W, row_h, H = 7.2, 1.7, 0.35 + 1.7 * len(COHORTS)
    fig, axes = plt.subplots(len(COHORTS), len(columns), figsize=(W, H), squeeze=False)
    for ri, ds in enumerate(COHORTS):
        frames = rows[ds]
        outcomes = _outcomes_present(ds, frames)
        for ci, (title, _) in enumerate(columns):
            ax = axes[ri, ci]
            sub = frames[ci]
            if sub is None:
                pending_axis(ax, ds)
            else:
                _cindex_bars(ax, sub, outcomes, methods)
            if ri == 0:
                ax.set_title(title, fontsize=FONT_TITLE, loc="left")
        # one y scale per cohort row, floored just under its lowest C-index
        drawn = [axes[ri, ci] for ci, d in enumerate(frames) if d is not None]
        if drawn:
            lo = min(float(np.nanmin(d.concordance_test)) for d in frames if d is not None)
            for ax in drawn:
                ax.set_ylim(max(0.4, lo - 0.05), None)
            drawn[0].set_ylabel("C-index", fontsize=FONT_AXIS)
            for ax in drawn[1:]:
                ax.sharey(drawn[0])
                ax.tick_params(axis="y", labelleft=False)
        _cohort_label(axes[ri, -1], ds)

    handles = [Patch(facecolor=_BM_COLORS[m], label=RI_LABELS[m]) for m in methods]
    _top_legend(fig, handles, H)
    fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0, 0.98, 1 - 0.28 / H))
    return {None: fig}


# nri: rows = cohorts, columns = refit | swap
def _nri_dots(ax, sub, outcomes, methods):
    n_m = len(methods)
    step = 0.7 / max(n_m - 1, 1)
    for k, m in enumerate(methods):
        off = (k - (n_m - 1) / 2) * step
        for yi, o in enumerate(outcomes):
            r = sub[(sub.outcome == o) & (sub.new_method == m)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            se = r.get("NRI_se", np.nan)
            err = 1.96 * float(se) if pd.notna(se) else 0
            ax.errorbar(float(r.NRI), yi + off, xerr=err, fmt=_RI_MARKER[m.split("_")[0]],
                        color=_BM_COLORS[m], markersize=4, elinewidth=0.7, capsize=1.5,
                        capthick=0.5, markeredgecolor="white", markeredgewidth=0.3)
    for yi in range(1, len(outcomes)):
        ax.axhline(yi - 0.5, color="#EEEEEE", lw=0.4, zorder=0)
    ax.axvline(0, color=DARK, lw=0.5, ls="--")
    ax.set_yticks(np.arange(len(outcomes)))
    ax.set_yticklabels([_outcome_label(o) for o in outcomes], fontsize=FONT_TICK)
    ax.set_ylim(-0.6, len(outcomes) - 0.4)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    hide_spines(ax)


def fig_nri():
    columns = [
        (REFIT_TITLE, lambda ds: _nri(ds, "refit")),
        (SWAP_TITLE, lambda ds: _nri(ds, "swap")),
    ]
    rows = {ds: [load(ds) for _, load in columns] for ds in COHORTS}
    if all(d is None for frames in rows.values() for d in frames):
        return {}
    have = _methods_present([d for frames in rows.values() for d in frames], "new_method")
    methods = [m for m in bm_methods() if m != "PopRI" and m in have]
    outcomes = {ds: _outcomes_present(ds, rows[ds]) for ds in COHORTS}
    last_drawn = [max(ri for ri, ds in enumerate(COHORTS) if rows[ds][ci] is not None)
                  for ci in range(len(columns))]

    # a pending cohort gets the height of a typical outcome list
    n_rows = [len(outcomes[ds]) or 4 for ds in COHORTS]
    heights = [0.22 * n + 0.45 for n in n_rows]
    W, H = 7.2, 0.45 + sum(heights)
    fig, axes = plt.subplots(len(COHORTS), len(columns), figsize=(W, H), squeeze=False,
                             gridspec_kw=dict(height_ratios=heights))
    for ri, ds in enumerate(COHORTS):
        for ci, (title, _) in enumerate(columns):
            ax = axes[ri, ci]
            sub = rows[ds][ci]
            if sub is None:
                pending_axis(ax, ds)
            else:
                _nri_dots(ax, sub, outcomes[ds], methods)
                if ci > 0:
                    ax.tick_params(axis="y", labelleft=False)
                if ri == last_drawn[ci]:
                    ax.set_xlabel("NRI vs Pop$_{RI}$", fontsize=FONT_AXIS)
            if ri == 0:
                ax.set_title(title, fontsize=FONT_TITLE, loc="left")
        _cohort_label(axes[ri, -1], ds)

    handles = [Line2D([], [], marker=_RI_MARKER[m.split("_")[0]], ls="", color=_BM_COLORS[m],
                      ms=4, label=RI_LABELS[m]) for m in methods]
    _top_legend(fig, handles, H)
    fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0, 0.98, 1 - 0.28 / H))
    return {None: fig}


FIGURES = [
    FigSpec("14_patient_level", "cindex", fig_cindex, False, (), None),
    FigSpec("14_patient_level", "cindex_norma", ablation_variant(fig_cindex), False, (), None),
    FigSpec("14_patient_level", "nri", fig_nri, False, (), None),
    FigSpec("14_patient_level", "nri_norma", ablation_variant(fig_nri), False, (), None),
]


# Tables — 14_patient_level: table_* definitions and registry slice.

from figlib import *  # noqa: F401,F403
from figlib import RI_LABELS, _bm_method

# save_table()'s first argument is the folder the table is written into, so it must match this
# directory name.

def table_nri():
    """Both designs of nri.csv: Cox refit per method ("refit", Per_RI-normal
    subset) and the Pop_RI-trained model scored with swapped flags ("swap")."""
    rows = []
    for ds in DATASETS:
        parts = []
        for design in ("refit", "swap"):
            df = _nri_raw(ds, design)
            if df is None or len(df) == 0:
                continue
            df = df.copy(); df["subset"] = df["subset"] if "subset" in df.columns else "all"
            df = df[df["subset"].isin(["per_normal", "all"])]
            df["design"] = design; parts.append(df)
        if not parts:
            rows.append({"Dataset": DATASET_DISPLAY[ds], "Outcome": "---", "Subset": "---", "Method": "---", "N": "---",
                         "Events": "---", "NRI": "---", "NRI (events)": "---", "NRI (non-events)": "---", "SE": "---", "p-value": "---"})
            continue
        df = pd.concat(parts, ignore_index=True)
        df = to_numeric(df, skip=("outcome", "subset", "new_method", "ref_method", "method", "design"))
        df["new_method"] = df["new_method"].map(_bm_method)
        df = df[df["new_method"].isin(RI_LABELS) & ~df["new_method"].isin(["Cohen_m2", "Cohen_m3"])]
        for _, r in df.iterrows():
            rows.append({"Dataset": DATASET_DISPLAY[ds], "Outcome": OUTCOME_DISPLAY.get(r["outcome"], r["outcome"]),
                         "Subset": r["design"], "Method": RI_LABELS[r["new_method"]],
                         "N": f'{int(r["n"]):,}', "Events": f'{int(r["n_events"]):,}', "NRI": f'{r["NRI"]:.4f}',
                         "NRI (events)": f'{r["NRI_events"]:.4f}', "NRI (non-events)": f'{r["NRI_nonevents"]:.4f}',
                         "SE": f'{r["NRI_se"]:.4f}', "p-value": fmt_pval(r["NRI_p"])})
    csv_df = pd.DataFrame(rows)
    tx = lambda s: tex_escape(s).replace(">", r"$>$").replace("<", r"$<$")
    header = [r"Dataset & Outcome & Design & Method & $N$ & NRI & NRI (events) & NRI (non-events) & $p$ \\"]
    body = [f'{tx(r["Dataset"])} & {tx(r["Outcome"])} & {tx(r["Subset"])} & {r["Method"]} & {r["N"]} & {r["NRI"]} & '
            f'{r["NRI (events)"]} & {r["NRI (non-events)"]} & {r["p-value"]} \\\\' for r in rows]
    save_table("14_patient_level", "nri", _table("ll l l r r r r r", header, body), csv_df)
    return ["nri"]

TABLES = [
    TableSpec("14_patient_level", "nri",                     table_nri,                     False, (), None),
]


if __name__ == "__main__":
    main()
