#!/usr/bin/env python
"""Cohen et al. 2021 Fig. 5c-f on every reference-interval method: Kaplan-Meier
cumulative incidence of a clinical endpoint for flagged vs unflagged patients.

Cohort     one row per patient: the first Pop_RI-normal value of the analyte (the
           index), every method's deviation score there, and the time to the
           endpoint or censoring.  Prevalent cases (event or censoring at or before
           the landmark) are excluded by time, as 13_cox does.  Only time-to-event
           outcomes; a label defined by the follow-up time itself (prolonged stay)
           stays a binary outcome in 12_eval.
Flags      every method at Cohen's matched sensitivity (--sensitivity, cutoff per
           10-year age band x sex) and at the standard of care's alert rate.  The
           flags are NOT directional: for a clinical endpoint a deviation either way
           is a risk signal (the lab-value endpoints of 11_lead_time are directional).
Standard   the latest value AVAILABLE AT the landmark read against Pop_RI (no
of care    look-ahead).  It cannot flag at landmark 0, where the index value is its
           latest value and normal by construction.
Landmarks  --landmarks reproduces their Fig. 5e earliness design: risk is assessed
           at the index measurement but follow-up starts `landmark` hours later, so a
           model flag at t0 is compared against the standard of care as it would
           read at t0 + landmark.
Output     results/raw/<cohort>/17_incidence.csv: per outcome x analyte x method x landmark
           x anchor, cumulative incidence in both arms at every --horizons, their
           ratio, and the log-rank p-value.

Usage:
    python 17_outcomes.py --dataset eicu
    python 17_outcomes.py --dataset eicu --outcomes mortality --landmarks 0 24

Figures and tables
------------------
Every figure is one file for all cohorts: rows = eICU / INSPIRE / CHS (VAL_COHORTS),
columns = that cohort's outcomes; a cohort whose results are missing renders as a
pending row.
  incidence        RR of the endpoint for flagged vs unflagged patients at the index
                   measurement, every method at a matched sensitivity
  incidence_norma  the same with the NORMA covariate arms instead of the RI methods
  earliness        the same RR as follow-up starts later and later after the index
                   test, every method at the standard of care's alert rate
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from datasets import (already_done, add_dataset_args, get_dataset, save_csv, EXCLUDE_LABS,
                      NORMA_RUN_ID)
from constants import MATCHED_SENSITIVITY
from metrics import hours, jitter, matched_sensitivity_flags, strata_of, threshold_at_rate


def incidence_cohort(cls, outcome_cfg, analyte, age_range, unit):
    """(patients, observations) for one analyte: one row per patient at the first
    Pop_RI-normal measurement with the outcome and its times in hours, and every
    measurement's Pop_RI class over time for the standard-of-care arm."""
    lab = cls[cls["analyte"] == analyte].copy()
    lab["t"] = pd.to_numeric(lab["t_hours"], errors="coerce")
    lab = lab.dropna(subset=["t"]).sort_values(["patient_id", "t"])
    index = lab[lab["PopRI_class"] == 1].groupby("patient_id", observed=True).head(1).copy()
    index["age"] = pd.to_numeric(index["age"], errors="coerce")
    index["sex"] = pd.to_numeric(index["sex"], errors="coerce")
    index = index[(index["age"] >= age_range[0]) & (index["age"] < age_range[1])]
    index["event"] = pd.to_numeric(index[outcome_cfg["event_col"]], errors="coerce")
    index["t_event"] = hours(index[outcome_cfg["time_col"]], unit)
    index["t_censor"] = hours(index[outcome_cfg["censor_col"]], unit)
    index["t0"] = index["t"]
    patients = index.dropna(subset=["event", "age", "sex", "t0"])
    observations = lab[["patient_id", "t", "PopRI_class"]]
    return patients, observations


# ── per-chunk cohorts (chunked cohorts) ────────────────────────────────────
# Both steps need one row per patient at their first Pop_RI-normal measurement, plus
# what the standard of care would read at each landmark.  Both are chunk-local, so each
# chunk is reduced once and cached beside it; the landmarks are baked into the cache,
# which --force rebuilds.
COHORT_CACHE = "17_cohort"             # a directory: one parquet per analyte


def soc_column(landmark):
    return f"soc__{landmark:g}"


def chunk_cohorts(ds, args, unit, landmarks):
    """<chunk>/17_cohort.parquet: per (analyte, patient) the index row, every outcome's
    event and times, and the standard-of-care flag at each landmark."""
    import datasets as _ds
    methods = [m for m in ds.methods if f"{m}_z" in set(ds.classification_columns())]
    made = reused = 0
    for chunk_dir in ds._chunk_dirs():
        if _ds.analyte_cache_ready(chunk_dir, COHORT_CACHE) and not args.force:
            reused += 1
            continue
        i = int(os.path.basename(chunk_dir).rsplit("_", 1)[-1])
        sub_ds = _ds.DATASETS[ds.name](chunk=i)
        for attr in ("norma_alias", "run_ids", "no_norma", "cohen_models", "gaussian_models"):
            setattr(sub_ds, attr, getattr(ds, attr))
        cls = _ds.read_classification(chunk_dir)
        if cls is None:
            continue
        cls["analyte"] = cls["analyte"].replace("", "NA").fillna("NA")
        cls = cls[~cls["analyte"].isin(EXCLUDE_LABS)]
        outcomes = [o for o in ds.primary_outcomes if ds.outcomes[o].get("survival", True)]
        if any(ds.outcomes[o]["event_col"] not in cls.columns for o in outcomes):
            cls = sub_ds.attach_outcomes(cls)
        frames = []
        for analyte in sorted(cls["analyte"].unique()):
            base = None
            for outcome in outcomes:
                patients, observations = incidence_cohort(cls, ds.outcomes[outcome], analyte,
                                                          (0, 200), unit)
                if not len(patients):
                    continue
                cols = ["patient_id", "age", "sex", "t0"] + [f"{m}_z" for m in methods
                                                             if f"{m}_z" in patients.columns]
                t = patients[cols].copy()
                t[f"event__{outcome}"] = patients["event"].to_numpy()
                t[f"t_event__{outcome}"] = patients["t_event"].to_numpy()
                t[f"t_censor__{outcome}"] = patients["t_censor"].to_numpy()
                if base is None:
                    for landmark in landmarks:      # what Pop_RI says at each landmark
                        flag = soc_flag_at(patients.assign(start=patients["t0"] + landmark),
                                           observations)
                        t[soc_column(landmark)] = flag
                    base = t
                else:
                    base = base.merge(t[["patient_id", f"event__{outcome}", f"t_event__{outcome}",
                                         f"t_censor__{outcome}"]], on="patient_id", how="outer")
            if base is not None and len(base):
                frames.append(base.assign(analyte=analyte))
        del cls
        if frames:
            _ds.write_analyte_cache(pd.concat(frames, ignore_index=True), chunk_dir, COHORT_CACHE)
        made += 1
        print(f"    {os.path.basename(chunk_dir)}: cohort cached")
    print(f"  cohort cache: {made} chunk(s) computed, {reused} reused")
    return methods


def cached_analytes(ds):
    """Analytes present in the per-chunk cohort caches."""
    import datasets as _ds
    return [a for a in _ds.cached_analytes(ds._chunk_dirs(), COHORT_CACHE)
            if a not in set(EXCLUDE_LABS)]


def cached_cohort(ds, analyte, outcome):
    """One analyte's cohort for this outcome, pooled over the chunks."""
    import datasets as _ds
    frames = []
    for chunk_dir in ds._chunk_dirs():       # only this analyte's slice of each chunk
        d = _ds.read_analyte_cache(chunk_dir, COHORT_CACHE, analyte)
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    ren = {f"{c}__{outcome}": c for c in ("event", "t_event", "t_censor")}
    if not set(ren) <= set(t.columns):
        return None
    t = t.rename(columns=ren)
    return t.dropna(subset=["event", "age", "sex", "t0"])


def soc_flag_at(patients, observations):
    """Standard of care: Pop_RI class of each patient's latest value at or before its
    follow-up start.  A patient with no value yet is NOT flagged (NaN != 1 is True)."""
    starts = patients[["patient_id", "start"]].sort_values("start").rename(columns={"start": "t"})
    latest = pd.merge_asof(starts, observations.sort_values("t"), on="t", by="patient_id",
                           direction="backward")
    class_at_start = latest.drop_duplicates("patient_id").set_index("patient_id")["PopRI_class"]
    v = patients["patient_id"].map(class_at_start)
    return (v.notna() & (v != 1)).to_numpy(bool)


def incidence_rows(patients, flag, method, analyte, outcome, horizons):
    """Cumulative incidence at each horizon for flagged vs unflagged, plus the log-rank
    test between the two curves; None when either arm is too small."""
    d = patients.assign(flag=flag)
    # cast before the arithmetic: a parquet cohort can carry these as object, and
    # lifelines then coerces them itself with a warning per fit -- the sums above would
    # already have been done on objects by then
    for c in ("event", "t_event", "t_censor", "start"):
        if c in d.columns and d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    end = np.where(d["event"] == 1, d["t_event"], d["t_censor"])
    d = d.assign(duration=pd.to_numeric(end, errors="coerce") - d["start"])
    d = d[d["duration"] > 0]
    flagged, unflagged = d[d["flag"]], d[~d["flag"]]
    if len(flagged) < 20 or len(unflagged) < 20:
        return None
    test = logrank_test(flagged["duration"], unflagged["duration"], flagged["event"], unflagged["event"])
    row = {
        "analyte": analyte, "outcome": outcome, "method": method,
        "n": len(d), "n_flagged": len(flagged), "flag_rate": len(flagged) / len(d),
        "events_flagged": int(flagged["event"].sum()), "events_unflagged": int(unflagged["event"].sum()),
        "logrank_p": float(test.p_value),
    }
    for arm, grp in (("flag", flagged), ("unflag", unflagged)):
        km = KaplanMeierFitter().fit(grp["duration"], grp["event"])
        for h in horizons:
            try:
                row[f"inc_{arm}_{h:g}h"] = float(1.0 - km.predict(h))
            except Exception:
                row[f"inc_{arm}_{h:g}h"] = np.nan
    for h in horizons:
        inc_flagged, inc_unflagged = row[f"inc_flag_{h:g}h"], row[f"inc_unflag_{h:g}h"]
        if np.isfinite(inc_flagged) and np.isfinite(inc_unflagged) and inc_unflagged > 0:
            row[f"rr_{h:g}h"] = inc_flagged / inc_unflagged
        else:
            row[f"rr_{h:g}h"] = np.nan
    return row


# ── progression: Cohen et al. 2021 Fig. 5e (glucose -> T2D) and 5f (creatinine -> CKD)
# Their design: among people who are currently normal, split by what the model predicts
# for two years' time, and follow disease incidence from that point.  The comparison is
# against what the VALUE ITSELF says two years later -- the "observed" arms -- because
# the claim is identification earlier than waiting for the test to cross the line.
#
# Arms here, per analyte x method:
#   predicted_severe    flagged, and further than --severe_z from the method's centre
#   predicted_abnormal  flagged, but not severe
#   predicted_normal    not flagged
#   observed_abnormal   Pop_RI-abnormal at the landmark (the standard of care then)
#   observed_normal     Pop_RI-normal at the landmark
# The predicted arms are read at the INDEX measurement, the observed arms at the
# landmark, so the two differ by exactly the waiting time the analysis is about.
PROGRESSION = {"t2d": ["GLU", "A1C"], "ckd": ["CRE"]}       # Cohen's pairs
PROGRESSION_AGE = {"t2d": (50, 60), "ckd": (60, 70)}        # their bands
DEFAULT_AGE_RANGE = (0, 200)                                # --age_range's default: no band
YEAR_H = 8766.0                                             # hours in a year


def progression_arms(cohort, observations, method, severe_z, soc=None):
    """{arm: boolean mask} over the cohort, the five arms of Cohen Fig. 5e."""
    z = pd.to_numeric(cohort[f"{method}_z"], errors="coerce").to_numpy(float)
    flagged = np.isfinite(z) & (z > 1.0)          # outside the method's own interval
    severe = flagged & (z > severe_z)
    if soc is None:
        soc = soc_flag_at(cohort, observations)   # Pop_RI at the landmark, no look-ahead
    return {
        "predicted_severe": severe,
        "predicted_abnormal": flagged & ~severe,
        "predicted_normal": np.isfinite(z) & ~flagged,
        "observed_abnormal": soc,
        "observed_normal": ~soc,
    }


def progression_rows(cohort, arms, method, analyte, outcome, horizons, min_arm=20):
    """Cumulative incidence per arm at each horizon -- the curves of Fig. 5e."""
    end = np.where(cohort["event"] == 1, cohort["t_event"], cohort["t_censor"])
    duration = end - cohort["start"].to_numpy(float)
    event = cohort["event"].to_numpy(float)
    rows = []
    for arm, mask in arms.items():
        keep = mask & (duration > 0)
        if keep.sum() < min_arm:
            continue
        km = KaplanMeierFitter().fit(duration[keep], event[keep])
        row = {"analyte": analyte, "outcome": outcome, "method": method, "arm": arm,
               "n": int(keep.sum()), "n_events": int(event[keep].sum())}
        for h in horizons:
            try:
                row[f"inc_{h:g}h"] = float(1.0 - km.predict(h))
            except Exception:
                row[f"inc_{h:g}h"] = np.nan
        rows.append(row)
    return rows


def run_progression(ds, cls, methods, args, results_dir, outcome, unit):
    cfg = ds.outcomes[outcome]
    analytes = [a for a in PROGRESSION.get(outcome, []) if not ds._analytes or a in ds._analytes]
    if not analytes:
        print(f"  {outcome}: no analyte pairing (Cohen used {PROGRESSION.get(outcome, [])})")
        return []
    # their band unless one was asked for
    age_range = (tuple(args.age_range) if tuple(args.age_range) != DEFAULT_AGE_RANGE
                 else PROGRESSION_AGE.get(outcome, DEFAULT_AGE_RANGE))
    landmark = args.progression_landmark
    rows = []
    for analyte in analytes:
        if cls is None:                           # chunked: out of the per-chunk caches
            patients = cached_cohort(ds, analyte, outcome)
            observations = None
            if patients is None:
                continue
            patients = patients[(pd.to_numeric(patients["age"], errors="coerce") >= age_range[0])
                                & (pd.to_numeric(patients["age"], errors="coerce") < age_range[1])]
        else:
            patients, observations = incidence_cohort(cls, cfg, analyte, age_range, unit)
        if len(patients) < 100:
            print(f"  {analyte}: {len(patients)} patients in the age band, too few")
            continue
        cohort = patients.assign(start=patients["t0"] + landmark)
        prevalent = (cohort["event"] == 1) & (cohort["t_event"] <= cohort["start"])
        cohort = cohort[~prevalent & (cohort["t_censor"] > cohort["start"])]
        if len(cohort) < 100:
            continue
        print(f"  {analyte} age {age_range[0]}-{age_range[1]}, landmark {landmark:g}h: "
              f"n={len(cohort):,}, incident {int(cohort['event'].sum()):,}, "
              f"{int(prevalent.sum()):,} prevalent dropped")
        soc = cohort[soc_column(landmark)].to_numpy(bool) if observations is None else None
        for method in methods:
            arms = progression_arms(cohort, observations, method, args.severe_z, soc)
            rows += progression_rows(cohort, arms, method, analyte, outcome, args.horizons)
    for r in rows:
        r.update(landmark_hours=landmark, severe_z=args.severe_z,
                 age_lo=age_range[0], age_hi=age_range[1])
    return rows


def run_incidence(ds, cls, methods, args, results_dir, outcome, unit):
    cfg = ds.outcomes[outcome]
    analytes = (list(ds._analytes) if ds._analytes
                else sorted(cls["analyte"].dropna().unique()) if cls is not None
                else cached_analytes(ds))
    rows = []

    def add(row, landmark, anchor):
        if row is not None:
            rows.append({**row, "landmark_hours": landmark, "anchor": anchor})

    for analyte in analytes:
        if cls is None:                           # chunked: out of the per-chunk caches
            patients = cached_cohort(ds, analyte, outcome)
            observations = None
            if patients is None:
                continue
            age = pd.to_numeric(patients["age"], errors="coerce")
            patients = patients[(age >= args.age_range[0]) & (age < args.age_range[1])]
        else:
            patients, observations = incidence_cohort(cls, cfg, analyte, args.age_range, unit)
        if len(patients) < 100 or patients["event"].nunique() < 2:
            continue
        for landmark in args.landmarks:
            cohort = patients.assign(start=patients["t0"] + landmark)
            # prevalent / uninformative: the event already happened, or follow-up ended,
            # at or before the landmark
            prevalent = (cohort["event"] == 1) & (cohort["t_event"] <= cohort["start"])
            cohort = cohort[~prevalent & (cohort["t_censor"] > cohort["start"])]
            if len(cohort) < 100 or cohort["event"].nunique() < 2:
                continue
            print(f"  {analyte} landmark={landmark:g}h: n={len(cohort):,} incident events="
                  f"{int(cohort['event'].sum()):,} ({cohort['event'].mean():.3f}) | "
                  f"{int(prevalent.sum()):,} prevalent dropped")
            soc = (cohort[soc_column(landmark)].to_numpy(bool) if observations is None
                   else soc_flag_at(cohort, observations))
            soc_rate = float(soc.mean())
            strata = strata_of(cohort)
            y = cohort["event"].to_numpy(float)
            for method in methods:
                z = jitter(pd.to_numeric(cohort[f"{method}_z"], errors="coerce").to_numpy(float))
                # Cohen's protocol: matched sensitivity per age band x sex
                flag, _ = matched_sensitivity_flags(z, y, strata, args.sensitivity)
                add(incidence_rows(cohort, flag, method, analyte, outcome, args.horizons),
                    landmark, "sensitivity")
                # head-to-head with current practice: matched alert rate
                threshold = threshold_at_rate(z, soc_rate)
                if np.isfinite(threshold):
                    flag = np.isfinite(z) & (z >= threshold)
                    add(incidence_rows(cohort, flag, method, analyte, outcome, args.horizons),
                        landmark, "soc_rate")
            # the standard of care is a rule, not a score: the same row under both anchors
            soc_row = incidence_rows(cohort, soc, "StandardOfCare", analyte, outcome, args.horizons)
            add(soc_row, landmark, "sensitivity")
            add(soc_row, landmark, "soc_rate")

    if not rows:
        print("  nothing with enough data")
        return
    out = pd.DataFrame(rows)
    out["target_sensitivity"] = args.sensitivity
    save_csv(out, os.path.join(results_dir, "incidence.csv"),
             analytes=ds._analytes, keys=("outcome",))
    key = f"rr_{args.horizons[-1]:g}h"
    summary = out.groupby("method")[[key, "flag_rate", "logrank_p"]].median()
    print("\n" + summary.sort_values(key, ascending=False).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--outcomes", nargs="*", default=None,
                   help="clinical endpoints (default: every time-to-event primary outcome)")
    p.add_argument("--sensitivity", type=float, default=MATCHED_SENSITIVITY,
                   help="matched sensitivity for the cutoff (Cohen: 0.25 for Fig. 5e/f)")
    p.add_argument("--landmarks", type=float, nargs="*", default=[0.0, 24.0, 72.0, 168.0],
                   help="hours after the index measurement at which follow-up starts, swept in "
                        "one pass (0 = at the index measurement; Cohen's Fig. 5e used 2 years)")
    p.add_argument("--horizons", type=float, nargs="*", default=[24.0, 72.0, 168.0, 720.0],
                   help="hours at which to report cumulative incidence")
    p.add_argument("--age_range", type=float, nargs=2, default=list(DEFAULT_AGE_RANGE),
                   metavar=("LO", "HI"),
                   help="restrict the cohort (Cohen used a single 10-year band)")
    g = p.add_argument_group("progression (Cohen Fig. 5e/f)")
    g.add_argument("--progression", action="store_true",
                   help="disease progression by predicted vs observed status: glucose/A1C -> T2D "
                        "(ages 50-60) and creatinine -> CKD (60-70), as Cohen Fig. 5e/f")
    g.add_argument("--progression_landmark", type=float, default=2 * YEAR_H,
                   help="hours between the prediction and the start of follow-up (default: 2 years)")
    g.add_argument("--severe_z", type=float, default=2.0,
                   help="deviation above which a flag counts as the severe arm (their FG > 110)")
    args = p.parse_args()

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    prog = []
    out_file = "progression.csv" if args.progression else "incidence.csv"
    if already_done(args, results_dir, out_file, label=out_file.split(".")[0]):
        return
    unit = getattr(ds, "outcome_time_unit", None) or getattr(ds, "time_unit", None) or "minutes"
    landmarks = ([args.progression_landmark] if args.progression else list(args.landmarks))
    if ds.name == "chs":            # never one frame: reduce per chunk, then pool per analyte
        cls = None
        methods = chunk_cohorts(ds, args, unit, landmarks)
    else:
        cls = ds.load_classification()
        cls["analyte"] = cls["analyte"].replace("", "NA").fillna("NA")
        cls = cls[~cls["analyte"].isin(EXCLUDE_LABS)]
        methods = [m for m in ds.methods if f"{m}_z" in cls.columns]
    print(f"  {len(methods)} methods: {', '.join(methods)}")

    outcomes = args.outcomes
    if outcomes is None:
        outcomes = [o for o in ds.primary_outcomes if o in ds.outcomes]
    if cls is not None and any(ds.outcomes[o]["event_col"] not in cls.columns for o in outcomes):
        cls = ds.attach_outcomes(cls)
    for outcome in outcomes:
        if not ds.outcomes[outcome].get("survival", True):
            print(f"=== {outcome}: skipped, its label is defined by the follow-up time itself ===")
            continue
        print(f"=== {outcome} ===")
        if args.progression:
            prog += run_progression(ds, cls, methods, args, results_dir, outcome, unit)
        else:
            run_incidence(ds, cls, methods, args, results_dir, outcome, unit)
    if args.progression:
        if not prog:
            print("  nothing with enough data")
            return
        save_csv(pd.DataFrame(prog), os.path.join(results_dir, "progression.csv"),
                 analytes=ds._analytes, keys=("outcome",))
        last = f"inc_{args.horizons[-1]:g}h"
        shown = pd.DataFrame(prog)
        shown = shown[shown["method"].isin(["PopRI", "NORMA", f"NORMA_{NORMA_RUN_ID}"])]
        if len(shown):
            print("\n" + shown.pivot_table(index=["analyte", "arm"], columns="method",
                                           values=last).round(4).to_string())


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from models import collapse_run_id

COHORTS = VAL_COHORTS         # eicu, inspire, chs: one row each


# ─────────────────────────────────────────────────────────────────────────────
# shared
# ─────────────────────────────────────────────────────────────────────────────
def _incidence_frame(ds, outcome, anchor="sensitivity"):
    """anchor='sensitivity' is Cohen's protocol (every model at 25% sensitivity);
    anchor='soc_rate' puts every model at the standard of care's own alert rate,
    the only footing on which an RR can be compared against it."""
    df = load_result(ds, "incidence.csv")
    if df is None or len(df) == 0:
        return None, None
    df = df[df["outcome"].astype(str) == outcome]
    if not len(df):
        return None, None
    if "anchor" in df.columns:
        df = df[df["anchor"].astype(str) == anchor]
        if not len(df):
            return None, None
    rr = [c for c in df.columns if c.startswith("rr_")]
    if not rr:
        return None, None
    df = to_numeric(df.copy())
    # The LONGEST horizon that is actually reachable, not simply the last column:
    # follow-up does not extend to every horizon for every outcome (INSPIRE
    # mortality has rr_720h entirely NaN), and taking rr[-1] blindly made the
    # whole outcome disappear from the figure with no warning.
    usable = [c for c in rr if df[c].notna().sum() >= 5]
    if not usable:
        return None, None
    key = usable[-1]
    df = df.dropna(subset=[key])
    # the medians the figure draws, one pseudo-analyte row per method x landmark
    groups = [c for c in ("method", "anchor", "landmark_hours") if c in df.columns]
    summary = df.groupby(groups)[[key, "logrank_p"]].median().reset_index().assign(analyte="median")
    df["m"] = df["method"].map(lambda a: a if a == STANDARD_OF_CARE else collapse_run_id(a))
    return df, key


def _method_color(m):
    return _BM_COLORS.get(m, METHOD_COLORS.get(m, "#999"))


def _outcome_label(o):
    return OUTCOME_SHORT.get(o, OUTCOME_DISPLAY.get(o, o))


def _cohort_label(ax, ds):
    """Cohort name on the right of the row's last axis."""
    ax.text(1.04, 0.5, DATASET_DISPLAY.get(ds, ds), transform=ax.transAxes,
            rotation=270, ha="left", va="center", fontsize=FONT_AXIS, color=DARK)


def _pending_row(axes, ds):
    """First axis carries the notice, the rest of the row is blank."""
    pending_axis(axes[0], ds)
    for ax in axes[1:]:
        ax.set_axis_off()
    _cohort_label(axes[-1], ds)


def _top_legend(fig, handles, H):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / H),
               ncol=len(handles), frameon=False, fontsize=FONT_LEGEND,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)


def _grid(panels, row_height):
    """rows = cohorts, columns = the widest outcome set; returns fig, axes, H.
    panels: ds -> list of per-outcome payloads (None = cohort pending)."""
    n_cols = max([len(row) for row in panels.values() if row] + [1])
    heights = [row_height if panels.get(ds) else PENDING_H for ds in COHORTS]
    W, H = 7.2, sum(heights) + 0.5
    fig, axes = plt.subplots(len(COHORTS), n_cols, figsize=(W, H), squeeze=False,
                             gridspec_kw=dict(height_ratios=heights))
    return fig, axes, H


# ─────────────────────────────────────────────────────────────────────────────
# incidence: rows = cohorts, columns = outcomes, RR bars per method at landmark 0
# ─────────────────────────────────────────────────────────────────────────────
def _incidence_panels(ds):
    """[(outcome, med)] with med = median RR and log-rank p per method at landmark 0."""
    panels = []
    for outcome in OUTCOMES.get(ds, []):
        df, key = _incidence_frame(ds, outcome)
        if df is None:
            continue
        if "landmark_hours" in df.columns:
            df = df[df["landmark_hours"].fillna(0) == 0]
        med = df.groupby("m")[[key, "logrank_p"]].median().rename(columns={key: "rr"})
        if len(med) < 2:
            continue
        panels.append((outcome, med))
    return panels


def _rr_bars(ax, med, methods):
    """One bar per method in registry order, RR = 1 (no separation) as the reference
    line; a cross marks methods whose curves the log-rank test does not separate."""
    for i, m in enumerate(methods):
        if m not in med.index:
            continue
        rr, p = med.loc[m, "rr"], med.loc[m, "logrank_p"]
        ax.barh(i, rr - 1.0, left=1.0, height=0.68, color=_method_color(m), alpha=0.85,
                edgecolor="white", linewidth=0.3)
        if p >= 0.05:            # not separated: mark it, don't hide it
            ax.plot(rr, i, marker="x", ms=4, color=DARK, mew=1.0, zorder=3)
    ax.axvline(1.0, color=DARK, lw=0.8, zorder=2)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    hide_spines(ax)


def fig_incidence():
    """Per cohort (row) and outcome (column): RR of the endpoint for flagged vs
    unflagged patients at the index measurement (landmark 0), each method at a
    matched sensitivity.  The standard of care cannot flag at landmark 0 (the index
    value IS its latest value, normal by construction), so it appears only in
    fig_earliness."""
    panels = {}
    for ds in COHORTS:
        if load_result(ds, "incidence.csv") is not None:
            panels[ds] = _incidence_panels(ds)
    present = set()
    for row in panels.values():
        for _, med in row:
            present |= set(med.index)
    methods = [m for m in bm_methods() if m in present]
    if not methods:
        return {}

    fig, axes, H = _grid(panels, 0.24 * len(methods) + 0.9)
    for ri, ds in enumerate(COHORTS):
        row = panels.get(ds)
        if not row:
            _pending_row(axes[ri], ds)
            continue
        for ci, ax in enumerate(axes[ri]):
            if ci >= len(row):
                ax.set_axis_off()
                continue
            outcome, med = row[ci]
            _rr_bars(ax, med, methods)
            ax.set_title(_outcome_label(outcome), fontsize=FONT_TITLE, loc="left")
            ax.set_xlabel("Risk ratio", fontsize=FONT_AXIS)
            if ci == 0:
                ax.set_yticklabels([RI_LABELS.get(m, m) for m in methods], fontsize=FONT_TICK)
            else:
                ax.tick_params(axis="y", labelleft=False)
        _cohort_label(axes[ri, -1], ds)
    fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0, 0.98, 1))
    return {None: fig}


# ─────────────────────────────────────────────────────────────────────────────
# earliness: rows = cohorts, columns = outcomes, RR vs follow-up start (Cohen Fig. 5e)
# ─────────────────────────────────────────────────────────────────────────────
def _earliness_panels(ds):
    """[(outcome, med)] with med = median RR per method and landmark (> 0 hours)."""
    panels = []
    for outcome in OUTCOMES.get(ds, []):
        df, key = _incidence_frame(ds, outcome, anchor="soc_rate")
        if df is None or "landmark_hours" not in df.columns:
            continue
        # at landmark 0 the standard of care has (almost) no alerts, so matching its rate
        # is degenerate; the sweep starts at the first positive landmark
        df = df.dropna(subset=["landmark_hours"])
        df = df[df["landmark_hours"] > 0]
        if df["landmark_hours"].nunique() < 2:
            continue
        med = df.groupby(["m", "landmark_hours"])[key].median().rename("rr").reset_index()
        panels.append((outcome, med))
    return panels


def _rr_lines(ax, med, methods):
    for m in methods:
        g = med[med["m"] == m].sort_values("landmark_hours")
        ax.plot(g["landmark_hours"].to_numpy(), g["rr"].to_numpy(), "-o", ms=3, lw=1.2,
                color=_method_color(m))
    soc = med[med["m"] == STANDARD_OF_CARE].sort_values("landmark_hours")
    if len(soc):
        ax.plot(soc["landmark_hours"].to_numpy(), soc["rr"].to_numpy(), "--", lw=1.1, color=DARK)
    ax.axhline(1.0, color=DARK, lw=0.8, alpha=0.4, zorder=0)
    ax.tick_params(labelsize=FONT_TICK)
    hide_spines(ax)


def fig_earliness():
    """Risk is assessed at the index measurement, but follow-up starts `landmark`
    hours later, so each method is scored against the endpoint as the standard of
    care would read it that much later. A method whose line stays above the
    standard-of-care rule at landmark d has identified those patients d earlier."""
    panels = {}
    for ds in COHORTS:
        if load_result(ds, "incidence.csv") is not None:
            panels[ds] = _earliness_panels(ds)
    present = set()
    for row in panels.values():
        for _, med in row:
            present |= set(med["m"])
    methods = [m for m in bm_methods() if m in present]
    if not methods:
        return {}

    fig, axes, H = _grid(panels, 2.0)
    for ri, ds in enumerate(COHORTS):
        row = panels.get(ds)
        if not row:
            _pending_row(axes[ri], ds)
            continue
        for ci, ax in enumerate(axes[ri]):
            if ci >= len(row):
                ax.set_axis_off()
                continue
            outcome, med = row[ci]
            _rr_lines(ax, med, methods)
            ax.set_title(_outcome_label(outcome), fontsize=FONT_TITLE, loc="left")
            ax.set_xlabel("Hours after index test", fontsize=FONT_AXIS)
            if ci == 0:
                ax.set_ylabel("Risk ratio", fontsize=FONT_AXIS)
        _cohort_label(axes[ri, -1], ds)
    handles = [Line2D([], [], color=_method_color(m), lw=1.2, marker="o", ms=3, label=RI_LABELS.get(m, m))
               for m in methods]
    handles.append(Line2D([], [], color=DARK, lw=1.1, ls="--", label="Standard of care"))
    _top_legend(fig, handles, H)
    fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0, 0.98, 1 - 0.28 / H))
    return {None: fig}


FIGURES = [
    FigSpec("17_outcomes", "incidence", fig_incidence, False, (), None),
    FigSpec("17_outcomes", "incidence_norma", ablation_variant(fig_incidence), False, (), None),
    FigSpec("17_outcomes", "earliness", fig_earliness, False, (), None),
]

# ══════════════════════════════════════════════════════════════════════════
# Tables — 17_outcomes: one table per cohort x outcome, the numbers behind
# fig_incidence (RR per analyte at the index measurement, matched sensitivity).
# ══════════════════════════════════════════════════════════════════════════

from figlib import RI_LABELS, _BM_SUPP


def _count(row, col):
    return int(row[col]) if col in row and pd.notna(row[col]) else "---"


def _rr_cell(rr, p):
    """RR at the reported horizon, with the log-rank p that goes with it."""
    if pd.isna(rr) or not np.isfinite(rr):
        return "---"
    return f"{rr:.2f} ({fmt_pval(p)})"


def table_incidence(ds):
    written = []
    for outcome in OUTCOMES.get(ds, []):
        df, key = _incidence_frame(ds, outcome)
        if df is None:
            continue
        if "landmark_hours" in df.columns:      # the table is the landmark-0 figure
            df = df[df["landmark_hours"].fillna(0) == 0]
        df = df[df["analyte"].astype(str) != "median"]
        if not len(df):
            continue
        methods = [m for m in _BM_SUPP if m in set(df["m"])]
        if not methods:
            continue
        horizon = key.replace("rr_", "").replace("h", "")

        rows = []
        for analyte in all_analytes():
            cell = df[df["analyte"] == analyte]
            row = {"Analyte": analyte, "N": "---"}
            if len(cell):
                row["N"] = _count(cell.iloc[0], "n")
            for method in methods:
                match = cell[cell["m"] == method]
                row[f"{method} RR"] = "---"
                if len(match) == 1:
                    r = match.iloc[0]
                    row[f"{method} RR"] = _rr_cell(r[key], r["logrank_p"])
            rows.append(row)

        header = ("Analyte & N"
                  + "".join(f" & {RI_LABELS.get(m, m)} RR ($p$)" for m in methods))
        body = [" & ".join([r["Analyte"], str(r["N"])] + [r[f"{m} RR"] for m in methods]) + r" \\"
                for r in rows]
        lines = _table("lr" + "r" * len(methods), [header + r" \\"], body)
        lines.insert(2, r"\caption{Risk ratio of " + tex_escape(_outcome_label(outcome))
                     + f" at {horizon} h for flagged vs unflagged patients, each method at a "
                     + r"matched sensitivity; $p$ is the log-rank test.}")
        name = f"incidence_{ds}_{outcome}"
        save_table("17_outcomes", name, lines, pd.DataFrame(rows), landscape=True)
        written.append(name)
    return written


TABLES = [
    TableSpec("17_outcomes", "incidence", table_incidence, True, ("incidence.csv",),
              lambda ds: [f"incidence_{ds}_{o}" for o in OUTCOMES.get(ds, [])]),
]


if __name__ == "__main__":
    main()
