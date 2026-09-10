#!/usr/bin/env python
"""Does a personalised interval see a Pop_RI-abnormal value coming?  Two designs,
both disease-agnostic (the endpoint is the lab value itself leaving Pop_RI), both
DIRECTIONAL (a deviation toward the population centre is never a positive;
lib/ri_metrics.deviates_toward_bound) and both with Cohen's testing-bias control
(a pair enters only if it is retested inside the horizon / window).  Clinical
endpoints live in 17_outcomes.

  future_abnormal  Cohen et al. 2021 Fig. 5a/b on every RI method: among index
                   values inside Pop_RI, relative risk (PPV / prevalence) of a
                   Pop_RI-abnormal value within --horizon, cutoff per 10-year age
                   band x sex at matched --sensitivity (their 0.2).  Pop_RI defines
                   the endpoint, so it is a reference line, not a competitor.
                   Also the centre bias of every method at the Pop_RI-normal index.
                   -> 11_future_abnormal.csv (per analyte, analyte="median", age bands),
                      11_centre_bias.csv (age bands live in 11_future_abnormal.csv)

  lead_time        how many hours (and tests) BEFORE a value leaves Pop_RI does
                   each method already flag it, at equal alert burden?  Cohort =
                   pairs whose first index value is inside Pop_RI, fixed --window;
                   every follow-up measurement scored.  Anchors:
                     native      each method's own rule, z > 1 (NORMA's flags contain
                                 Pop_RI's here, so it is earlier by construction)
                     soc_rate    every method thresholded to the standard of care's
                                 MEASUREMENT-LEVEL flag rate -- equal burden
                     early_sens  every method at the same EARLY sensitivity
                                 (--early-sens): the false alarms this costs
                   plus a threshold sweep (target_rate rows of 11_lead_time.csv, an early-detection
                   ROC).  Among pairs reaching the endpoint: lead in hours AND in
                   tests, stratified by tests/day tertile, fraction flagged >= h
                   hours before; flags among non-events; alerts per 100 pair-days.
                   The standard of care fires exactly AT the endpoint (lead 0).
                   -> 11_lead_time.csv (age_band / target_rate = "all" for the pooled rows)

Usage:
    python 11_lead_time.py --dataset eicu
    python 11_lead_time.py --dataset eicu --only lead_time --window 72

Figures and tables
------------------
Everything here is disease-agnostic: the endpoint is the lab value itself
leaving Pop_RI.  Clinical-endpoint figures live in 17_outcomes.
  future_abnormal        RR of a future Pop_RI-abnormal value per method, one panel
                         per cohort (Cohen Fig. 5b pooled)
  future_abnormal_age_<ds>  the same RR against age, one panel per analyte
  lead_time              fraction of eventually-abnormal pairs each method had already
                         flagged h hours before the crossing, one panel per cohort
  lead_time_analyte      median lead (hours) of an early flag per analyte, one panel per
                         cohort, Per_RI / Cohen / NORMA only
  *_norma                the NORMA covariate arms instead of the RI methods
Pop_RI defines the endpoint, so it is never a series.  Trimmed 2026-09-04: the
flagged-early panel, the age variants of lead time, the early-detection ROC and
the centre-bias diagnostic were removed.
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd

from constants import ALL_SPLIT, MEDIAN_ROW
from datasets import already_done, add_dataset_args, get_dataset, save_csv, EXCLUDE_LABS
from metrics import cut_at_sensitivity, jitter, matched_sensitivity_flags, strata_of, threshold_at_rate, deviates_toward_bound, method_prefix

# Reuse is keyed on these: a step whose files are all present is skipped
# unless --force (datasets.already_done).
STEP_OUTPUTS = {
    "future_abnormal": ["future_abnormal.csv"],
    "lead_time": ["lead_time.csv"],
}
STEPS = ("future_abnormal", "lead_time")
TIME_COLUMNS = ("t_hours", "timestamp", "days_from_admit")
LEADS_H = (0, 6, 12, 24, 48, 72)
SWEEP_RATES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)


def time_in_hours(df):
    """Index time in hours, whichever column this cohort carries."""
    for col in TIME_COLUMNS:
        if col in df.columns and df[col].notna().any():
            t = pd.to_numeric(df[col], errors="coerce")
            return t if col == "t_hours" else t * 24.0
    raise KeyError(f"no usable time column; looked for {TIME_COLUMNS}")


def load_classified(ds):
    cls = ds.load_classification()
    cls["analyte"] = cls["analyte"].replace("", "NA").fillna("NA")
    cls = cls[~cls["analyte"].isin(EXCLUDE_LABS)]
    methods = [m for m in ds.methods if f"{m}_z" in cls.columns]
    return cls, methods


# =============================================================================
# future_abnormal
# =============================================================================


def future_pairs(cls, horizon_h):
    """One row per (patient, analyte): the first Pop_RI-normal measurement, labelled
    future_abnormal = any later Pop_RI-abnormal measurement within `horizon_h`; kept
    only if the pair has a follow-up measurement inside the horizon."""
    keys = ["patient_id", "analyte"]
    cls = cls.copy()
    cls["_t"] = time_in_hours(cls)
    cls = cls.dropna(subset=["_t", "PopRI_class"]).sort_values(keys + ["_t"])
    index = (cls[cls["PopRI_class"] == 1].groupby(keys, observed=True).head(1)
             .rename(columns={"_t": "t_index"}))
    if not len(index):
        return index
    later = cls[keys + ["_t", "PopRI_class"]].merge(index[keys + ["t_index"]], on=keys)
    later = later[(later["_t"] > later["t_index"]) & (later["_t"] <= later["t_index"] + horizon_h)]
    n_followup = later.groupby(keys, observed=True).size().rename("n_followup")
    n_abnormal = later[later["PopRI_class"] != 1].groupby(keys, observed=True).size().rename("n_abn")
    pairs = index.merge(n_followup, on=keys, how="inner")   # testing-bias control
    pairs = pairs.merge(n_abnormal, on=keys, how="left")
    pairs["future_abnormal"] = (pairs["n_abn"].fillna(0) > 0).astype(int)
    return pairs


def confusion(flag, y):
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    prevalence = (tp + fn) / n if n else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    return {
        "n": n, "n_events": tp + fn, "prevalence": prevalence,
        "flag_rate": (tp + fp) / n if n else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": ppv,
        "rr": ppv / prevalence if (prevalence and np.isfinite(ppv)) else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def relative_risk_rows(pairs, methods, sensitivity):
    """Cohen Fig. 5b: threshold per age band x sex at `sensitivity`, then RR (PPV /
    prevalence) pooled per analyte and per stratum.  Returns (pooled, by_age)."""
    pooled, by_age = [], []
    for method in methods:
        for analyte, g in pairs.groupby("analyte", observed=True):
            z_raw = pd.to_numeric(g[f"{method}_z"], errors="coerce")
            g = g[np.isfinite(z_raw)]
            if len(g) < 100 or g["future_abnormal"].nunique() < 2:
                continue
            z0 = jitter(z_raw[np.isfinite(z_raw)].to_numpy(float))
            y = g["future_abnormal"].to_numpy(int)
            strata = strata_of(g)
            # 'toward_bound' is the result (Cohen: abnormal = high OR low per test); 'any'
            # keeps the non-directional rule for transparency.  A method with a biased
            # centre gains under 'any' from values far below its centre that later rise,
            # and loses under 'toward_bound' because those look like reversion.
            for direction in ("toward_bound", "any"):
                z = z0
                if direction == "toward_bound" and method != "PopRI":
                    z = np.where(deviates_toward_bound(g, method), z0, -np.inf)
                flag, thresholded = matched_sensitivity_flags(z, y, strata, sensitivity)
                for (band, sex), idx in strata.items():
                    if (band, sex) not in thresholded:
                        continue
                    idx = np.asarray(idx)
                    c = confusion(flag[idx], y[idx])
                    if c["n_events"] >= 5:
                        by_age.append({"analyte": analyte, "method": method, "direction": direction,
                                       "age_band": band, "sex": sex,
                                       "target_sensitivity": sensitivity, **c})
                pooled.append({"analyte": analyte, "method": method, "direction": direction,
                               "target_sensitivity": sensitivity, "n_strata": len(strata),
                               **confusion(flag, y)})
    return pooled, by_age


def centre_bias_rows(pairs, methods):
    """(method centre - value) / Pop_RI half-width at the Pop_RI-normal index."""
    x = pd.to_numeric(pairs["value"], errors="coerce")
    pop_low = pd.to_numeric(pairs["pop_ri_low"], errors="coerce")
    pop_high = pd.to_numeric(pairs["pop_ri_high"], errors="coerce")
    half_width = (pop_high - pop_low) / 2
    rows = []
    for method in methods:
        prefix = method_prefix(method)
        if f"{prefix}_low" not in pairs.columns:
            continue
        low = pd.to_numeric(pairs[f"{prefix}_low"], errors="coerce")
        high = pd.to_numeric(pairs[f"{prefix}_high"], errors="coerce")
        bias = ((low + high) / 2 - x) / half_width
        for analyte, b in bias.groupby(pairs["analyte"]):
            b = b.dropna()
            if len(b) < 100:
                continue
            rows.append({"analyte": analyte, "method": method, "n": len(b),
                         "bias_median": float(b.median()),
                         "bias_iqr25": float(b.quantile(.25)), "bias_iqr75": float(b.quantile(.75)),
                         "abs_bias_median": float(b.abs().median())})
    return rows


def run_future_abnormal(ds, cls, methods, args, results_dir):
    pairs = future_pairs(cls, args.horizon)
    if not len(pairs):
        print("  no (patient, analyte) pairs with a Pop_RI-normal index and follow-up")
        return
    print(f"  {len(pairs):,} pairs | future-abnormal rate {pairs['future_abnormal'].mean():.3f} | "
          f"horizon {args.horizon:g} h")
    bias = centre_bias_rows(pairs, methods)
    if bias:
        save_csv(pd.DataFrame(bias), os.path.join(results_dir, "centre_bias.csv"), analytes=ds._analytes)

    pooled, by_age = relative_risk_rows(pairs, methods, args.sensitivity)
    per_analyte = pd.DataFrame(pooled)
    if not len(per_analyte):
        print("  no analyte had enough data")
        return
    # The across-analyte rows use the SAME metric column names as the per-analyte
    # rows (rr, not median_rr) -- `analyte` says which kind of row it is, so one
    # metric column serves both.
    summary = per_analyte.groupby(["method", "direction"]).agg(
        n_analytes=("analyte", "size"), n=("n", "sum"), n_events=("n_events", "sum"),
        rr=("rr", "median"), ppv=("ppv", "median"),
        sensitivity=("sensitivity", "median"), specificity=("specificity", "median"),
        flag_rate=("flag_rate", "median"),
    ).reset_index()
    summary["horizon_hours"] = args.horizon
    summary["target_sensitivity"] = args.sensitivity
    summary["analyte"] = MEDIAN_ROW
    summary = summary.sort_values(["direction", "rr"], ascending=[True, False])
    parts = [per_analyte, summary]
    if by_age:
        parts.append(pd.DataFrame(by_age))       # age_band / sex rows of the same table
    out = pd.concat(parts, ignore_index=True)
    for dim in ("age_band", "sex"):              # "all" = not broken out along it
        if dim in out.columns:
            out[dim] = out[dim].fillna(ALL_SPLIT).replace("", ALL_SPLIT)
        else:
            out[dim] = ALL_SPLIT
    save_csv(out, os.path.join(results_dir, "future_abnormal.csv"), analytes=ds._analytes)
    print("\n" + summary.to_string(index=False))


# =============================================================================
# lead_time
# =============================================================================

def lead_frames(cls, analyte, window_h):
    """(measurements, patients) for one analyte: every measurement after each pair's
    Pop_RI-normal index, inside the window and up to the endpoint (the first
    Pop_RI-abnormal value; censoring = the last measurement in the window)."""
    cls = cls[cls["analyte"] == analyte].copy()
    if not len(cls):
        return None, None
    cls["t"] = pd.to_numeric(cls["t_hours"], errors="coerce")
    cls = cls.dropna(subset=["t", "PopRI_class"]).sort_values(["patient_id", "t"])
    index = cls[cls["PopRI_class"] == 1].groupby("patient_id", observed=True).head(1)
    age = np.nan
    if "age" in index.columns:
        age = pd.to_numeric(index["age"], errors="coerce").to_numpy()
    patients = pd.DataFrame({"patient_id": index["patient_id"].to_numpy(),
                             "t0": index["t"].to_numpy(), "age": age})

    later = cls[["patient_id", "t", "PopRI_class"]].merge(patients, on="patient_id")
    later = later[(later.t > later.t0) & (later.t <= later.t0 + window_h)]
    if not len(later):
        return None, None
    first_abnormal = later[later.PopRI_class != 1].groupby("patient_id")["t"].min().rename("t_event")
    last_seen = later.groupby("patient_id")["t"].max().rename("t_censor")
    patients = patients.merge(last_seen, on="patient_id", how="inner")   # retested in the window
    patients = patients.merge(first_abnormal, on="patient_id", how="left")
    patients["event"] = patients.t_event.notna().astype(int)

    end = np.where(patients.event == 1, patients.t_event, patients.t_censor)
    patients["t_end"] = np.minimum(end, patients.t0 + window_h)
    in_window = (patients.event == 1) & (patients.t_event <= patients.t0 + window_h)
    patients["event_in_window"] = in_window.astype(int)
    patients = patients[patients.t_end > patients.t0]

    measurements = cls.merge(patients[["patient_id", "t0", "t_end"]], on="patient_id")
    measurements = measurements[(measurements.t > measurements.t0) & (measurements.t <= measurements.t_end)]
    return measurements, patients


def first_flag(measurements, flag):
    """Time of each pair's first flagged measurement (patients never flagged are absent)."""
    flagged = measurements[flag]
    return flagged.groupby("patient_id", observed=True)["t"].min().rename("t_flag")


def early_stats(measurements, patients, flag):
    """(early-flag rate among eventually-abnormal pairs, flag rate among pairs that never
    reach the endpoint, measurement-level flag rate) for one rule."""
    p = patients.merge(first_flag(measurements, flag), on="patient_id", how="left")
    events = p[p.event_in_window == 1]
    non_events = p[p.event_in_window == 0]
    early = float(((events.t_event - events.t_flag) > 0).mean()) if len(events) else np.nan
    false_alarm = float(non_events.t_flag.notna().mean()) if len(non_events) else np.nan
    return early, false_alarm, float(flag.mean())


def rate_for_early_sensitivity(z, measurements, patients, target, lo=0.005, hi=0.8, iters=14):
    """Measurement-level flag rate at which a method's early-flag rate among
    eventually-abnormal pairs equals `target` (bisection; monotone in the rate)."""
    def early_at(rate):
        threshold = threshold_at_rate(z, rate)
        if not np.isfinite(threshold):
            return np.nan
        return early_stats(measurements, patients, np.isfinite(z) & (z >= threshold))[0]

    if not (early_at(hi) >= target >= early_at(lo)):
        return np.nan
    for _ in range(iters):
        mid = (lo + hi) / 2
        e = early_at(mid)
        if not np.isfinite(e):
            return np.nan
        lo, hi = (lo, mid) if e >= target else (mid, hi)
    return (lo + hi) / 2


def landmark_counts(lm):
    tp = int((lm.flag_m1 & (lm.event_later == 1)).sum())
    fp = int((lm.flag_m1 & (lm.event_later == 0)).sum())
    return {"lm_n": len(lm), "lm_events": int(lm.event_later.sum()), "lm_tp": tp, "lm_fp": fp}


def lead_rows(measurements, patients, flag, method, anchor, analyte, outcome):
    """Detection curve + lead distribution for one flag rule -> (row, age_rows), or None."""
    flagged = measurements.assign(_flag=flag)
    n_tests = flagged.groupby("patient_id", observed=True).size().rename("n_tests")
    p = (patients.merge(first_flag(measurements, flag), on="patient_id", how="left")
         .merge(n_tests, on="patient_id", how="left"))
    p["n_tests"] = p["n_tests"].fillna(0)
    p["days"] = (p.t_end - p.t0) / 24.0
    p["tests_per_day"] = p.n_tests / p.days.clip(lower=1 / 24)
    events = p[p.event_in_window == 1].copy()
    non_events = p[p.event_in_window == 0]
    if len(events) < 20:
        return None

    events["lead_h"] = events.t_event - events.t_flag                    # NaN if never flagged
    events["lead_h"] = events["lead_h"].where(events["lead_h"] >= 0)     # flag at or before the endpoint
    # lead in TESTS: draws strictly between the first flag and the event
    draws = flagged[["patient_id", "t"]].merge(events[["patient_id", "t_flag", "t_event"]], on="patient_id")
    between = draws[(draws.t > draws.t_flag) & (draws.t < draws.t_event)].groupby("patient_id").size()
    flagged_before = events.lead_h.notna()
    events["lead_tests"] = events["patient_id"].map(between).fillna(0).where(flagged_before)
    early = events[events.lead_h > 0]

    row = {
        "analyte": analyte, "outcome": outcome, "method": method, "anchor": anchor,
        "n_patients": len(p), "n_events": len(events), "n_nonevents": len(non_events),
        "meas_flag_rate": float(flag.mean()),
        "alerts_per_100_patient_days": 100 * float(flag.sum()) / max(float(p.days.sum()), 1e-9),
        "nonevents_flagged_frac": float(non_events.t_flag.notna().mean()) if len(non_events) else np.nan,
        "events_flagged_frac": float(events.t_flag.notna().mean()),
        "events_flagged_before_frac": float((events.lead_h > 0).mean()),
        "median_lead_h": float(events.lead_h.median()),
        "iqr25_lead_h": float(events.lead_h.quantile(.25)),
        "iqr75_lead_h": float(events.lead_h.quantile(.75)),
        # conditional on firing strictly BEFORE the endpoint: the lead an early flag buys
        "median_lead_h_early": float(early.lead_h.median()) if len(early) else np.nan,
        "iqr25_lead_h_early": float(early.lead_h.quantile(.25)) if len(early) else np.nan,
        "iqr75_lead_h_early": float(early.lead_h.quantile(.75)) if len(early) else np.nan,
        "median_lead_tests_early": float(early.lead_tests.median()) if len(early) else np.nan,
        "median_lead_tests": float(events.lead_tests.median()),
        "median_tests_per_day": float(p.tests_per_day.median()),
    }
    for h in LEADS_H:
        row[f"detected_{h}h_before"] = float((events.lead_h >= h).mean())
    # testing-intensity tertiles on the whole cohort, so strata are comparable across methods
    try:
        tertile = pd.qcut(p.tests_per_day, 3, labels=["low", "mid", "high"])
        for label in ("low", "mid", "high"):
            e = events[tertile.reindex(events.index) == label]
            enough = len(e) >= 10
            row[f"median_lead_h_{label}_intensity"] = float(e.lead_h.median()) if enough else np.nan
            row[f"detected_24h_{label}_intensity"] = float((e.lead_h >= 24).mean()) if enough else np.nan
    except ValueError:
        pass

    # landmark view: every pair judged on its FIRST follow-up draw, outcome = the value
    # leaves Pop_RI LATER in the window; pairs that cross on that very draw are excluded
    # (no lead is possible).  Counting "any flag before the crossing" instead gives
    # never-abnormal pairs the whole window and pushes every RR below 1.
    first_draw = (flagged.sort_values("t").groupby("patient_id", observed=True).head(1)
                  [["patient_id", "t", "_flag"]].rename(columns={"t": "t_m1", "_flag": "flag_m1"}))
    lm = p.merge(first_draw, on="patient_id", how="inner")
    lm = lm[~((lm.event_in_window == 1) & (lm.t_event <= lm.t_m1))]
    lm["event_later"] = (lm.event_in_window == 1).astype(int)
    counts = landmark_counts(lm)
    row.update(counts)
    if (counts["lm_tp"] + counts["lm_fp"]) and counts["lm_events"] and counts["lm_n"]:
        ppv = counts["lm_tp"] / (counts["lm_tp"] + counts["lm_fp"])
        row["rr_landmark"] = ppv / (counts["lm_events"] / counts["lm_n"])
    else:
        row["rr_landmark"] = np.nan

    age_rows = []
    if "age" in p.columns and p["age"].notna().any():
        band = np.floor(p["age"] / 10.0) * 10.0
        lm_band = np.floor(lm["age"] / 10.0) * 10.0
        for b, idx in p.groupby(band).groups.items():
            e = events[events.index.isin(idx)]
            ne = non_events[non_events.index.isin(idx)]
            if len(e) < 10:
                continue
            e_early = e[e.lead_h > 0]
            age_rows.append({
                "analyte": analyte, "outcome": outcome, "method": method, "anchor": anchor,
                "age_band": float(b), "n_events": len(e), "n_early": len(e_early),
                "n_24h": int((e.lead_h >= 24).sum()),
                "median_lead_h_early": float(e_early.lead_h.median()) if len(e_early) else np.nan,
                "n_nonevents": len(ne), "n_nonevents_flagged": int(ne.t_flag.notna().sum()),
                **landmark_counts(lm[lm_band == b]),
            })
    return row, age_rows


class LeadCollector:
    """Accumulates lead_rows results, tagging each with its anchor(s) and direction."""

    def __init__(self):
        self.rows, self.age_rows, self.sweep_rows = [], [], []

    def add(self, result, anchors=("soc_rate",), direction="toward_bound"):
        if result is None:
            return
        row, age_rows = result
        for anchor in anchors:
            self.rows.append({**row, "anchor": anchor, "direction": direction})
            self.age_rows.extend({**r, "anchor": anchor, "direction": direction} for r in age_rows)


def score_method(collector, method, measurements, patients, soc_rate, analyte, label, early_sens):
    """Every anchor for one method on one analyte."""
    z = pd.to_numeric(measurements[f"{method}_z"], errors="coerce").to_numpy(float)
    z_any = jitter(z)                                             # ties would inflate the matched rate
    # directional: only an exit toward the nearer Pop_RI bound can count as a flag;
    # exits toward the population centre are sent below every threshold
    z_toward = np.where(deviates_toward_bound(measurements, method), z_any, -np.inf)

    def flags(z, threshold):
        return np.isfinite(z) & (z >= threshold)

    def score(flag, anchor):
        return lead_rows(measurements, patients, flag, method, anchor, analyte, label)

    # non-directional rule, kept as direction='any' for the comparison figures
    t_any = threshold_at_rate(z_any, soc_rate)
    if np.isfinite(t_any):
        collector.add(score(flags(z_any, t_any), "soc_rate"), ("soc_rate",), "any")
    collector.add(score(np.isfinite(z_any) & (z_any > 1.0), "native"), ("native",), "any")

    t_soc = threshold_at_rate(z_toward, soc_rate)
    if np.isfinite(t_soc):
        collector.add(score(flags(z_toward, t_soc), "soc_rate"))
    # early-detection ROC: sweep the measurement-level flag rate
    for rate in SWEEP_RATES:
        t = threshold_at_rate(z_toward, rate)
        if not np.isfinite(t):
            continue
        early, false_alarm, flag_rate = early_stats(measurements, patients, flags(z_toward, t))
        collector.sweep_rows.append({
            "analyte": analyte, "outcome": label, "method": method, "target_rate": rate,
            "meas_flag_rate": flag_rate, "events_flagged_before_frac": early,
            "nonevents_flagged_frac": false_alarm,
            "n_events": int(patients.event_in_window.sum()),
            "n_nonevents": int((patients.event_in_window == 0).sum()),
        })
    # the same EARLY sensitivity for every method: what does it cost in false alarms?
    rate = rate_for_early_sensitivity(z_toward, measurements, patients, early_sens)
    if np.isfinite(rate):
        collector.add(score(flags(z_toward, threshold_at_rate(z_toward, rate)), "early_sens"), ("early_sens",))
    # strictly outside the interval: a value ON the bound is normal (z == 1 exactly)
    collector.add(score(np.isfinite(z_toward) & (z_toward > 1.0), "native"), ("native",))


def run_lead_time(ds, cls, methods, args, results_dir):
    analytes = list(ds._analytes) if ds._analytes else sorted(cls["analyte"].dropna().unique())
    label = "pop_abnormal"
    collector = LeadCollector()
    for analyte in analytes:
        measurements, patients = lead_frames(cls, analyte, args.window)
        if measurements is None or len(measurements) < 500 or patients.event_in_window.sum() < 20:
            continue
        soc_flag = (measurements["PopRI_class"] != 1).to_numpy()
        soc_rate = float(soc_flag.mean())
        print(f"  {analyte}: {len(patients):,} patients, {int(patients.event_in_window.sum()):,} events "
              f"in window, {len(measurements):,} follow-up measurements, "
              f"Pop_RI flags {100 * soc_rate:.1f}% of them")
        soc = lead_rows(measurements, patients, soc_flag, "StandardOfCare", "soc_rate", analyte, label)
        collector.add(soc, ("soc_rate", "native"))
        for method in methods:
            score_method(collector, method, measurements, patients, soc_rate, analyte, label, args.early_sens)

    if not collector.rows:
        print("  nothing with enough data")
        return
    out = pd.DataFrame(collector.rows)
    out["window_hours"] = args.window
    # One table: the pooled rows, the age-band rows and the threshold sweep are the
    # same analysis cut three ways, so the cut is a column ("all" = not cut).
    parts = [out]
    if collector.age_rows:
        parts.append(pd.DataFrame(collector.age_rows))
    if collector.sweep_rows:
        parts.append(pd.DataFrame(collector.sweep_rows))
    out = pd.concat(parts, ignore_index=True)
    for dim in ("age_band", "target_rate"):
        if dim in out.columns:
            out[dim] = out[dim].fillna(ALL_SPLIT).replace("", ALL_SPLIT)
        else:
            out[dim] = ALL_SPLIT
    save_csv(out, os.path.join(results_dir, "lead_time.csv"), analytes=ds._analytes)

    cols = ["events_flagged_before_frac", "detected_24h_before", "median_lead_h_early",
            "median_lead_tests_early", "nonevents_flagged_frac", "alerts_per_100_patient_days"]
    titles = {
        "native": "each method's own rule (z > 1)",
        "soc_rate": "every method at the standard of care's alert rate",
        "early_sens": f"every method at the same EARLY sensitivity ({args.early_sens:.0%} of "
                      "eventually-abnormal pairs flagged before the crossing)",
    }
    for anchor, title in titles.items():
        sel = out[(out.anchor == anchor) & (out.direction == "toward_bound")]
        summary = sel.groupby("method")[cols].median()
        summary = summary.sort_values("events_flagged_before_frac", ascending=False)
        print(f"\n  {title} -- median over analytes:")
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))


# =============================================================================
# main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    g = p.add_argument_group("future_abnormal")
    g.add_argument("--horizon", type=float, default=168.0,
                   help="hours after the index in which a Pop_RI-abnormal value counts as the "
                        "endpoint (default 168 = 7 days; Cohen used 2-3 years on outpatient data)")
    g.add_argument("--sensitivity", type=float, default=0.2,
                   help="matched sensitivity for the cutoff (Cohen: 0.2)")
    g = p.add_argument_group("lead_time")
    g.add_argument("--window", type=float, default=168.0, help="follow-up hours from the index (default 7 d)")
    g.add_argument("--early-sens", type=float, default=0.05,
                   help="anchor 'early_sens': every method thresholded so it flags this fraction "
                        "of eventually-abnormal pairs BEFORE the crossing")
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
    cls, methods = load_classified(ds)
    print(f"  {len(methods)} methods: {', '.join(methods)}")
    if "future_abnormal" in todo:
        print("=== future_abnormal ===")
        run_future_abnormal(ds, cls, methods, args, results_dir)
    if "lead_time" in todo:
        print("=== lead_time ===")
        run_lead_time(ds, cls, methods, args, results_dir)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from models import collapse_run_id


def _direction(df, which="toward_bound"):
    """Rows for one flag rule. Files written before the rule existed have no
    `direction` column and are treated as the directional result."""
    if df is None or "direction" not in df.columns:
        return df
    return df[df["direction"].astype(str) == which]


def _future_abnormal_frame(ds, direction="toward_bound"):
    """Pooled RR per method (median across analytes) with the IQR across analytes."""
    both = _direction(load_result(ds, "future_abnormal.csv"), direction)
    if both is None or len(both) == 0 or "rr" not in both.columns:
        return None
    both = pooled_rows(both, "age_band", "sex")
    is_med = both["analyte"].astype(str) == MEDIAN_ROW
    df, pa = both[is_med], both[~is_med]
    if not len(df):
        return None
    df = to_numeric(df.copy(), skip=("analyte", "method", "direction")).dropna(subset=["rr"])
    df["m"] = df["method"].map(collapse_run_id)
    keep = [m for m in _main_methods(set(df["m"])) if m != "PopRI"]         # Pop_RI is the endpoint
    df = df[df["m"].isin(keep)].drop_duplicates("m").set_index("m")
    if pa is not None and len(pa):
        pa = to_numeric(pa.copy(), skip=("analyte", "method", "direction"))
        pa["m"] = pa["method"].map(collapse_run_id)
        q = pa.groupby("m")["rr"].quantile([.25, .75]).unstack()
        df["q25"] = q[.25].reindex(df.index)
        df["q75"] = q[.75].reindex(df.index)
    else:
        df["q25"] = np.nan
        df["q75"] = np.nan
    return df


def fig_future_abnormal():
    """Cohen Fig. 5b, pooled: RR of a future Pop_RI-abnormal value per method
    (median across analytes, bar = IQR across analytes), one panel per cohort,
    methods on a shared y axis in a fixed order so the panels read across.
    Directional flag only (a deviation toward the nearer Pop_RI bound); the
    any-direction variant was dropped 2026-09-04 as one rule too many.
    Pop_RI is the endpoint, not a candidate. A cohort without results is a pending panel."""
    frames = {ds: _future_abnormal_frame(ds) for ds in VAL_COHORTS}
    frames = {ds: (f if f is not None and len(f) else None) for ds, f in frames.items()}
    if all(f is None for f in frames.values()):
        return {}
    present = set().union(*[set(f.index) for f in frames.values() if f is not None])
    order = [m for m in _main_methods(present) if m != "PopRI"]
    y = np.arange(len(order))
    lab_w = 0.055 * max(len(RI_LABELS.get(m, m)) for m in order)      # room for the y labels
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n + 0.6 + lab_w, 0.36 * len(order) + 1.2), squeeze=False)
    first = True
    for ax, (ds, f) in zip(axes[0], frames.items()):
        if f is None:
            pending_axis(ax, ds)
            ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, loc="center", pad=4)
            continue
        for i, m in enumerate(order):
            if m not in f.index:
                continue
            r = f.loc[m]
            c = _BM_COLORS.get(m, METHOD_COLORS.get(m, "#999"))
            ax.barh(i, r["rr"] - 1.0, left=1.0, height=0.6, color=c, alpha=0.9, linewidth=0)
            if np.isfinite(r["q25"]) and np.isfinite(r["q75"]):
                err = [[max(r["rr"] - r["q25"], 0)], [max(r["q75"] - r["rr"], 0)]]
                ax.errorbar(r["rr"], i, xerr=err, fmt="none", ecolor=DARK, elinewidth=0.8, capsize=2,
                            alpha=0.8, zorder=4)
        ax.axvline(1.0, color=DARK, lw=0.8, zorder=2)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, loc="center", pad=4)
        ax.tick_params(axis="x", labelsize=FONT_TICK)
        hide_spines(ax)
        ax.set_yticks(y)
        ax.set_yticklabels([RI_LABELS.get(m, m) for m in order] if first else [], fontsize=FONT_TICK)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("Risk ratio", fontsize=FONT_AXIS)
        first = False
    fig.tight_layout(w_pad=1.2)
    return {"": fig}


def _future_abnormal_age_frame(ds):
    """Per (analyte, method, age band) RR with a log-scale 95% CI, sexes pooled."""
    df = load_result(ds, "future_abnormal.csv")
    if df is not None and "age_band" in df.columns:
        df = df[df["age_band"].astype(str) != ALL_SPLIT]      # the age-band rows
    df = _direction(df)
    if df is None or len(df) == 0:
        return None
    df = to_numeric(df.copy(), skip=("analyte", "method", "sex", "direction"))
    df["m"] = df["method"].map(collapse_run_id)
    df = df[df["m"].isin([m for m in bm_methods() if m != "PopRI"])]  # Pop_RI is the endpoint
    agg = df.groupby(["analyte", "m", "age_band"], observed=True)[["tp", "fp", "fn", "tn"]].sum().reset_index()
    n = agg[["tp", "fp", "fn", "tn"]].sum(axis=1)
    agg["rr"] = (agg.tp / (agg.tp + agg.fp)) / ((agg.tp + agg.fn) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["se"] = np.sqrt(np.clip(1 / agg.tp - 1 / (agg.tp + agg.fp) + 1 / (agg.tp + agg.fn) - 1 / n, 0, None))
    return agg[(agg.tp >= 20) & np.isfinite(agg.rr)]                   # RR SE ~ 1/sqrt(tp)


def fig_future_abnormal_age(ds):
    """Cohen Fig. 5b proper for one cohort: RR of a future Pop_RI-abnormal value
    AGAINST AGE, one panel per analyte, one line per method, band = 95% CI.
    Pop_RI is the endpoint, not a series; cells need >= 20 true positives."""
    agg = _future_abnormal_age_frame(ds)
    if agg is None or not len(agg):
        return {}
    methods = _per_analyte_methods(set(agg["m"]))
    agg = agg[agg["m"].isin(methods)]
    ok = agg.groupby("analyte")["age_band"].nunique() >= 3
    analytes = [a for a in all_analytes() if ok.get(a, False)]
    if not analytes or not methods:
        return {}
    nc = 6
    nr = int(np.ceil(len(analytes) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(1.45 * nc + 0.4, 1.3 * nr + 0.7), squeeze=False, sharex=True)
    for ax in axes.flat[len(analytes):]:
        ax.set_axis_off()
    for ax, a in zip(axes.flat, analytes):
        sub = agg[agg["analyte"] == a]
        for m in methods:
            g = sub[sub["m"] == m].sort_values("age_band")
            if len(g) < 2:
                continue
            c = _BM_COLORS.get(m, METHOD_COLORS.get(m, "#999"))
            ax.fill_between(g["age_band"] + 5, g["rr"] * np.exp(-1.96 * g["se"]), g["rr"] * np.exp(1.96 * g["se"]),
                            color=c, alpha=0.10, lw=0)
            ax.plot(g["age_band"].to_numpy() + 5, g["rr"].to_numpy(), "-o", ms=2.5, lw=1.3, color=c, alpha=0.9)
        ax.axhline(1.0, color=DARK, lw=0.6, alpha=0.5, zorder=0)
        ax.set_title(a, fontsize=FONT_TICK, loc="left", pad=2)
        ax.tick_params(labelsize=FONT_TICK - 1)
        hide_spines(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Age (years)", fontsize=FONT_AXIS)
    for ax in axes[:, 0]:
        ax.set_ylabel("Risk ratio", fontsize=FONT_AXIS)
    handles = [Line2D([], [], color=_BM_COLORS.get(m, "#999"), lw=1.3, label=RI_LABELS.get(m, m)) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6), frameon=False, fontsize=FONT_LEGEND,
               handlelength=1.4, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return {"": fig}


# ── lead time before a value leaves Pop_RI (disease-agnostic) ────────────────


_PER_ANALYTE = ["PerRI", "Cohen_m4", "NORMA"]


def _per_analyte_methods(present):
    """Per-analyte grids carry only the three methods that make the argument (the
    NORMA arms in ablation mode); the summary figures keep the full benchmark set."""
    want = bm_methods() if in_ablation_mode() else _PER_ANALYTE
    return [m for m in want if m in present and m != "PopRI"]


def _main_methods(present):
    """Methods drawn on every figure in this folder: the configured benchmark set
    (lib/figlib.bm_methods -- all RI methods; the NORMA arms in ablation mode).
    Pop_RI is dropped by the callers where it is the endpoint."""
    return [m for m in bm_methods() if m in present]




def _cohort_row(draw, width=2.3, height=2.5, legend_ncol=6):
    """One subplot per cohort (DATASETS order); `draw(ax, ds)` returns the legend
    handles it drew, or None when that cohort has no results ("pending")."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(width * len(DATASETS) + 0.6, height), squeeze=False)
    handles, any_drawn = {}, False
    for ax, ds in zip(axes[0], DATASETS):
        h = draw(ax, ds)
        if h is None:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=FONT_LEGEND, color="#999", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            hide_spines(ax)
        else:
            any_drawn = True
            for hh in h:
                handles.setdefault(hh.get_label(), hh)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, pad=4)
    if not any_drawn:
        plt.close(fig)
        return None
    fig.legend(handles=list(handles.values()), loc="upper center", ncol=min(len(handles), legend_ncol), frameon=False,
               fontsize=FONT_LEGEND, handlelength=1.4, columnspacing=1.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / height))
    return fig


def _lead_frame(ds, direction="toward_bound"):
    df = _direction(pooled_rows(load_result(ds, "lead_time.csv"), "age_band", "target_rate"), direction)
    if df is None or len(df) == 0 or "anchor" not in df.columns:
        return None
    df = to_numeric(df.copy(), skip=("analyte", "method", "outcome", "anchor", "direction"))
    lead_cols = ["analyte", "method", "anchor", "direction", "alerts_per_100_patient_days",
                 "median_lead_h_early", "iqr25_lead_h_early", "iqr75_lead_h_early"]
    lead_cols += [c for c in df.columns if c.startswith("detected_") and c.endswith("h_before")]
    at_soc = df[df["anchor"].astype(str) == "soc_rate"]
    df["m"] = df["method"].map(lambda a: a if a == STANDARD_OF_CARE else collapse_run_id(a))
    return df


def _detect_cols(df):
    cols = sorted([c for c in df.columns if c.startswith("detected_") and c.endswith("h_before")],
                  key=lambda c: float(c.split("_")[1][:-1]))
    return cols, [float(c.split("_")[1][:-1]) for c in cols]


def _draw_lead_time(ax, ds):
    df = _lead_frame(ds)
    if df is None:
        return None
    cols, hours = _detect_cols(df)
    d = df[(df["anchor"].astype(str) == "soc_rate") & ~df["m"].isin([STANDARD_OF_CARE, "PopRI"])]
    grouped = d.groupby("m")[cols]
    med, lo, hi = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
    ms = [m for m in _main_methods(set(med.index)) if m != "PopRI"]
    if not cols or not ms:
        return None
    handles = []
    for m in ms:
        c = _BM_COLORS.get(m, METHOD_COLORS.get(m, "#999"))
        ax.fill_between(hours, lo.loc[m, cols].to_numpy(float), hi.loc[m, cols].to_numpy(float), color=c, alpha=0.12, lw=0)
        handles += ax.plot(hours, med.loc[m, cols].to_numpy(float), "-o", ms=3, lw=1.3, color=c, label=RI_LABELS.get(m, m))
    soc_b = df[(df["anchor"].astype(str) == "soc_rate") & (df["m"] == STANDARD_OF_CARE)]["alerts_per_100_patient_days"].median()
    if np.isfinite(soc_b):
        ax.text(0.98, 0.97, f"{soc_b:.1f} alerts / 100 pt-days", transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_LEGEND, color=DARK)
    ax.set_xlabel("Hours before crossing", fontsize=FONT_AXIS)
    ax.set_ylabel("Fraction already flagged", fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_ylim(0, None)
    hide_spines(ax)
    return handles


def fig_lead_time():
    """Outcome-agnostic lead time, one subplot per cohort: of the pairs whose value
    leaves Pop_RI within 7 d, the fraction each method had already flagged h hours
    earlier (directional flag), every method at the standard of care's alert rate.
    Median across analytes, band = IQR across analytes."""
    fig = _cohort_row(_draw_lead_time)
    return {"": fig} if fig is not None else {}


def _lead_analyte_frame(ds):
    """Per analyte and method, at the standard of care's alert rate: median lead of
    an early flag (hours) with its IQR across pairs.  (methods, median, q25, q75)."""
    df = _lead_frame(ds)
    if df is None:
        return None
    d = df[(df["anchor"].astype(str) == "soc_rate") & ~df["m"].isin([STANDARD_OF_CARE, "PopRI"])]
    methods = _per_analyte_methods(set(d["m"]))
    if not methods:
        return None
    pivot = lambda col: d.pivot_table(index="analyte", columns="m", values=col)
    return methods, pivot("median_lead_h_early"), pivot("iqr25_lead_h_early"), pivot("iqr75_lead_h_early")


def _draw_lead_time_analyte(ax, ds):
    frame = _lead_analyte_frame(ds)
    if frame is None:
        return None
    methods, lead, lo, hi = frame
    analytes = [a for a in all_analytes() if a in lead.index]
    y = np.arange(len(analytes))
    offsets = np.linspace(-0.22, 0.22, len(methods)) if len(methods) > 1 else [0.0]
    handles = []
    for k, (m, marker) in enumerate(zip(methods, ("D", "o", "s", "^"))):
        c = _BM_COLORS.get(m, METHOD_COLORS.get(m, "#999"))
        yy = y + offsets[k]
        med = lead.reindex(analytes)[m].to_numpy(float)
        q25 = lo.reindex(analytes)[m].to_numpy(float) if m in lo else med
        q75 = hi.reindex(analytes)[m].to_numpy(float) if m in hi else med
        ax.hlines(yy, q25, q75, color=c, lw=0.7, alpha=0.6)
        handles.append(ax.scatter(med, yy, marker=marker, s=11, color=c, edgecolors="none", zorder=3,
                                  label=RI_LABELS.get(m, m)))
    for yi in y[1:]:
        ax.axhline(yi - 0.5, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(analytes, fontsize=FONT_TICK - 1)
    ax.set_ylim(-0.7, len(analytes) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Lead time (h)", fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    hide_spines(ax)
    return handles


def fig_lead_time_analyte():
    """Lead time per analyte, one panel per cohort: how many hours before the value
    left Pop_RI the method had flagged it, when it flagged early, every method at
    the standard of care's alert rate.  Pop_RI is the endpoint, so not a series."""
    fig = _cohort_row(_draw_lead_time_analyte, width=2.4, height=6.0, legend_ncol=4)
    return {"": fig} if fig is not None else {}


FIGURES = [
    FigSpec("11_lead_time", "future_abnormal", fig_future_abnormal, False, (), None),
    FigSpec("11_lead_time", "future_abnormal_norma", ablation_variant(fig_future_abnormal), False, (), None),
    FigSpec("11_lead_time", "future_abnormal_age", fig_future_abnormal_age, True,
            ("future_abnormal.csv",), _one("future_abnormal_age")),
    FigSpec("11_lead_time", "future_abnormal_age_norma", ablation_variant(fig_future_abnormal_age), True,
            ("future_abnormal.csv",), _one("future_abnormal_age_norma")),
    FigSpec("11_lead_time", "lead_time", fig_lead_time, False, (), None),
    FigSpec("11_lead_time", "lead_time_analyte", fig_lead_time_analyte, False, (), None),
]


# ══════════════════════════════════════════════════════════════════════════
# Tables — 11_lead_time: table_* definitions and registry slice (disease-agnostic).
# ══════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from models import collapse_run_id, label as _label

# save_table()'s first argument is the folder the table is written into, so it
# must match this directory name.
_FOLDER = "11_lead_time"
_MAIN = ["PopRI", "PerRI", "Cohen_m4", "NORMA"]


def _canon(df):
    df = df.copy(); df["m"] = df["method"].map(lambda a: a if a == STANDARD_OF_CARE else collapse_run_id(a))
    if "direction" in df.columns:                       # directional rows are the result
        df = df[df["direction"].astype(str) == "toward_bound"]
    return df


def table_lead_time():
    """Per cohort x method: RR of a future Pop_RI-abnormal value (matched 20 %
    sensitivity), and lead time before the value leaves Pop_RI -- fraction of
    eventually-abnormal pairs flagged strictly earlier and the median lead of an
    early flag -- at the method's own interval and at the standard of care's
    alert rate. Medians over analytes. Pop_RI is the reference (its boundary
    defines both endpoints), so it carries the RR only."""
    rows = []
    for ds in DATASETS:
        fa = pooled_rows(load_result(ds, "future_abnormal.csv"), "age_band", "sex")
        lt = pooled_rows(load_result(ds, "lead_time.csv"), "age_band", "target_rate")
        if fa is None and lt is None:
            continue
        if fa is not None and "analyte" in fa.columns:
            fa = fa[fa["analyte"].astype(str) == MEDIAN_ROW]        # medians over analytes
        fa = _canon(to_numeric(fa.copy(), skip=("analyte", "method", "direction"))) if fa is not None else None
        lt = _canon(to_numeric(lt.copy(), skip=("analyte", "method", "outcome", "anchor"))) if lt is not None else None
        for m in _MAIN:
            r = {"Cohort": DATASET_DISPLAY.get(ds, ds), "Method": _label(m)}
            rr = fa[fa["m"] == m]["rr"] if fa is not None else pd.Series(dtype=float)
            r['RR future abnormal'] = f"{rr.iloc[0]:.2f}" if len(rr) else "---"
            for anchor, tag in (("native", "own"), ("soc_rate", "matched")):
                d = lt[(lt["m"] == m) & (lt["anchor"].astype(str) == anchor)] if lt is not None else None
                if m == "PopRI" or d is None or not len(d):
                    r[f"Early flags ({tag})"] = "---"; r[f"Median lead h ({tag})"] = "---"; r[f"Alerts/100 pt-d ({tag})"] = "---"
                    continue
                r[f"Early flags ({tag})"] = f"{100 * d['events_flagged_before_frac'].median():.1f}\\%"
                r[f"Median lead h ({tag})"] = f"{d['median_lead_h_early'].median():.0f}"
                r[f"Alerts/100 pt-d ({tag})"] = f"{d['alerts_per_100_patient_days'].median():.1f}"
            rows.append(r)
    if not rows:
        return []
    cols = list(rows[0].keys())
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(tex_escape(c) if "%" not in c else c for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table(_FOLDER, "lead_time", lines, pd.DataFrame(rows), landscape=True, font_size="scriptsize")
    return ["lead_time"]


TABLES = [
    TableSpec(_FOLDER, "lead_time", table_lead_time, False, (), None),
]


if __name__ == "__main__":
    main()
