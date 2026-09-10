#!/usr/bin/env python
"""Landmark Cox PH models per (outcome, analyte, RI method), one row per stay.

  exposure   which index measurement represents the stay (lib/metrics.py):
               first          the first index measurement;   follow-up from it
               window_worst   the most Pop_RI-deviant one within --window_hours of it;
                              follow-up from the end of the window
               window_any     every measurement in that window (flag = any abnormal,
                              z = max, category = of the most deviant)
  encoding   binary (outside the interval), z (|x - centre| / halfwidth, clipped at
             10), category (low / high vs normal)
  follow-up  event or censoring time minus the landmark, in hours; stays whose event
             or censoring is at/before the landmark are excluded
  patients   restricted, per analyte, to stays every method can score
  subsets    all; pop_normal = the exposure measurement is inside Pop_RI
  fitting    HR and CI on all stays (age + sex + exposure), BH-FDR within (outcome,
             exposure, encoding) and globally.  Minimum 3 events.  Outcomes whose
             label is defined by the follow-up time (prolonged stay) are skipped --
             they stay binary outcomes in 12_eval.
  -> 13_cox.csv (long format: subset x exposure x encoding x level; subset "all"
     is every stay, "pop_normal" only those inside the population interval)

Usage:
    python 13_cox.py --dataset eicu --n_jobs 8

Figures and tables
------------------
Every figure is one file for all cohorts: rows = eICU / INSPIRE / CHS (VAL_COHORTS), and
a cohort whose results are missing renders as a pending row.  13_cox.csv is long format
(exposure x encoding x level, see 13_cox.py); cox and methods_all show the primary
definition (first index measurement, binary flag).
  cox            rows = cohort, columns = outcome: hazard ratio of the abnormal flag
                 (95% CI) per analyte, one marker per RI method; only FDR-significant
                 estimates are drawn, a row is an analyte where a majority of the drawn
                 methods have one, and each panel keeps the MAX_ANALYTES analytes with the
                 largest NORMA hazard ratio
  cox_mortality  the mortality column of cox, the three cohorts side by side
  sensitivity    median HR across analytes per method under every exposure definition (x)
                 and encoding (columns), one row per cohort x outcome
  methods_all    within Pop_RI-normal tests: fraction of analytes with a significant
                 HR > 1 per method (bar = pooled, markers = per outcome), one panel per cohort
  *_norma        the same with the NORMA covariate arms instead of the RI methods
"""
import bootstrap  # noqa: F401

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

from datasets import already_done, add_dataset_args, get_dataset, save_csv, EXCLUDE_LABS
from metrics import EXPOSURES, WINDOW_HOURS, exposure_rows, mark_exposures, stay_col, to_hours, method_prefix

warnings.filterwarnings("ignore")

ENCODINGS = ("binary", "z", "category")
ENCODING_COVARIATES = {"binary": ["abnormal"], "z": ["z"], "category": ["low", "high"]}
Z_CLIP = 10.0
MIN_COX_EVENTS = 3


# =============================================================================
# stay-level exposure table
# =============================================================================

def method_z(df, method):
    """The stored deviation score, or |x - centre| / halfwidth from the bounds."""
    if f"{method}_z" in df.columns:
        return pd.to_numeric(df[f"{method}_z"], errors="coerce")
    prefix = method_prefix(method)
    low = pd.to_numeric(df[f"{prefix}_low"], errors="coerce")
    high = pd.to_numeric(df[f"{prefix}_high"], errors="coerce")
    x = pd.to_numeric(df["value"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - (low + high) / 2).abs() / ((high - low) / 2)
    return z.replace([np.inf, -np.inf], np.nan)


def stay_table(rows, landmark, methods, outcome_cfg, unit):
    """One row per (stay, analyte): each method's class / z of the exposure, the
    landmark, and the outcome with follow-up measured from the landmark."""
    stay = stay_col(rows)
    keys = [stay, "analyte"]
    rows = rows.copy()
    rows["_landmark"] = landmark.to_numpy()
    rows["_event"] = pd.to_numeric(rows[outcome_cfg["event_col"]], errors="coerce")
    rows["_t_event"] = to_hours(rows[outcome_cfg["time_col"]], unit)
    rows["_t_censor"] = to_hours(rows[outcome_cfg["censor_col"]], unit)
    for m in methods:
        rows[f"z__{m}"] = method_z(rows, m)
        rows[f"cls__{m}"] = pd.to_numeric(rows[f"{m}_class"], errors="coerce")

    table = (rows.sort_values("t_hours").groupby(keys, sort=False)
             .agg(age=("age", "first"), sex=("sex", "first"), landmark=("_landmark", "first"),
                  event=("_event", "first"), t_event=("_t_event", "first"),
                  t_censor=("_t_censor", "first")))
    # per method the most deviant exposure row (for first / window_worst there is one row)
    for m in methods:
        sub = rows[keys + [f"z__{m}", f"cls__{m}"]].dropna(subset=[f"cls__{m}"])
        if sub.empty:
            table[f"cls__{m}"] = np.nan
            table[f"z__{m}"] = np.nan
            continue
        worst = sub[f"z__{m}"].fillna(-1).groupby([sub[stay], sub["analyte"]]).idxmax()
        picked = sub.loc[worst.to_numpy()].set_index(keys)
        table[f"cls__{m}"] = picked[f"cls__{m}"]
        table[f"z__{m}"] = picked[f"z__{m}"]

    table = table.reset_index()
    end = np.where(table["event"] == 1, table["t_event"], table["t_censor"])
    table["duration"] = end - table["landmark"]
    table["age"] = pd.to_numeric(table["age"], errors="coerce")
    table["sex"] = pd.to_numeric(table["sex"], errors="coerce")
    table = table.dropna(subset=["duration", "event", "age", "sex"])
    return table[table["duration"] > 0]


def analyte_table(df, analyte, methods, cfg, unit, exposure, window_hours, min_rows):
    """Stay table for one analyte, restricted to the stays every method can score."""
    lab = df[df["analyte"] == analyte]
    if len(lab) < 20:
        return None
    rows, landmark = exposure_rows(lab, exposure, window_hours)
    if rows.empty:
        return None
    table = stay_table(rows, landmark, methods, cfg, unit)
    table = table.dropna(subset=[f"cls__{m}" for m in methods])
    return table if len(table) >= min_rows else None


def subset_rows(table, subset):
    return table if subset == "all" else table[table["cls__PopRI"] == 1]


def fit_cox(df, covariates):
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(df[["duration", "event", *covariates]], duration_col="duration", event_col="event")
    return cph


def survival_outcomes(ds, df):
    """(key, cfg) of the primary outcomes that are time-to-event."""
    outcomes = []
    for key in ds.primary_outcomes:
        cfg = ds.outcomes.get(key)
        if cfg is None or cfg["event_col"] not in df.columns:
            continue
        if not cfg.get("survival", True):
            print(f"  {key}: skipped, not a time-to-event outcome (the label is defined by the "
                  f"follow-up time); it is scored as a binary outcome in 12_eval instead")
            continue
        outcomes.append((key, cfg))
    return outcomes


def encoded(table, method):
    """The method's exposure in every encoding, one column each."""
    d = table[["duration", "event", "age", "sex", f"cls__{method}", f"z__{method}"]].dropna().copy()
    cls = d[f"cls__{method}"]
    d["abnormal"] = (cls != 1).astype(float)
    d["z"] = d[f"z__{method}"].clip(upper=Z_CLIP)
    d["low"] = (cls == 0).astype(float)
    d["high"] = (cls == 2).astype(float)
    return d


def usable_covariates(d, encoding):
    """A level needs variation and >= MIN_COX_EVENTS events among the exposed, otherwise the
    partial likelihood is flat and lifelines fails or returns HR ~ 1e7."""
    out = []
    for cov in ENCODING_COVARIATES[encoding]:
        if d[cov].nunique() <= 1:
            continue
        if cov != "z" and (d[cov] * d["event"]).sum() < MIN_COX_EVENTS:
            continue
        out.append(cov)
    return out


def fit_one(table, method, encoding, analyte, outcome, exposure):
    """Rows (one per level) for one (analyte, method, encoding)."""
    d = encoded(table, method)
    if len(d) < 10 or d["event"].sum() < MIN_COX_EVENTS:
        return []
    covariates = usable_covariates(d, encoding)
    if not covariates:
        return []
    try:
        full = fit_cox(d, ["age", "sex", *covariates])
    except Exception as e:
        print(f"    [fit failed] {analyte}/{method}/{encoding}: {e}")
        return []
    common = {
        "analyte": analyte, "method": method, "outcome": outcome, "exposure": exposure,
        "encoding": encoding, "n": len(d), "n_events": int(d["event"].sum()),
    }
    rows = []
    for cov in covariates:
        ci = full.confidence_intervals_.loc[cov]
        rows.append({
            **common, "level": cov,
            "n_exposed": len(d) if cov == "z" else int((d[cov] > 0).sum()),
            "HR": float(full.hazard_ratios_[cov]),
            "HR_lower": float(np.exp(ci.iloc[0])), "HR_upper": float(np.exp(ci.iloc[1])),
            "p_value": float(full.summary.loc[cov, "p"]),
        })
    return rows


def models_for_analyte(df, analyte, methods, outcome, cfg, unit, exposure, window_hours,
                       encodings, subsets):
    """All (method, encoding, subset) models for one analyte; returns {subset: rows}."""
    table = analyte_table(df, analyte, methods, cfg, unit, exposure, window_hours, min_rows=10)
    if table is None:
        return {}
    out = {}
    for subset in subsets:
        sub = subset_rows(table, subset)
        out[subset] = []
        if len(sub) < 20:
            continue
        for method in methods:
            if subset == "pop_normal" and method == "PopRI":
                continue
            for encoding in encodings:
                out[subset] += fit_one(sub, method, encoding, analyte, outcome, exposure)
    return out


def apply_fdr(df):
    df = df.copy()
    df["p_fdr"] = np.nan
    for _, idx in df.groupby(["outcome", "exposure", "encoding"]).groups.items():
        df.loc[idx, "p_fdr"] = multipletests(df.loc[idx, "p_value"].to_numpy(), method="fdr_bh")[1]
    df["p_fdr_global"] = multipletests(df["p_value"].to_numpy(), method="fdr_bh")[1]
    return df


def run_models(ds, df, methods, analytes, unit, args, results_dir):
    print(f"  exposures {args.exposures}; encodings {args.encodings}")
    results = {s: [] for s in args.subsets}
    for exposure in args.exposures:
        for outcome, cfg in survival_outcomes(ds, df):
            per_analyte = Parallel(n_jobs=args.n_jobs)(
                delayed(models_for_analyte)(df, a, methods, outcome, cfg, unit, exposure,
                                            args.window_hours, args.encodings, args.subsets)
                for a in analytes)
            n_rows = 0
            for result in per_analyte:
                for subset, rows in result.items():
                    results[subset].extend(rows)
                    n_rows += len(rows)
            binary = pd.DataFrame([r for result in per_analyte for r in result.get("all", [])])
            if len(binary):
                binary = binary[binary.encoding == "binary"]
                median_hr = binary.groupby("method").HR.median().round(2).to_dict()
                print(f"  [{exposure}] {outcome}: {n_rows} rows; stays/analyte median "
                      f"{int(binary.n.median())}, events median {int(binary.n_events.median())}; "
                      f"median HR(binary) {median_hr}")
            else:
                print(f"  [{exposure}] {outcome}: no models (no stays with positive follow-up?)")

    decimals = {"HR": 3, "HR_lower": 3, "HR_upper": 3, "p_value": 5}
    for subset, rows in results.items():
        if not rows:
            print(f"  {subset}: nothing to save")
            continue
        out = apply_fdr(pd.DataFrame(rows)).round(decimals)
        out.insert(0, "subset", subset)        # one file, the subset is a column
        path = os.path.join(results_dir, "cox.csv")
        if args.dry_run:
            print(f"  [dry run] {subset}: {len(out)} rows -> {path}\n{out.head(8).to_string()}")
            continue
        save_csv(out, path, analytes=ds._analytes, keys=("subset",))
        print(f"  Saved {len(out)} rows -> {path}  (FDR<0.05: {(out.p_fdr < 0.05).sum()})")


# =============================================================================
# main
# =============================================================================

def load_classified(ds, window_hours):
    df = ds.load_classification()
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    if "exp_first" not in df.columns:
        print("  classification.parquet has no exposure markers — computing them now "
              "(rerun 07_classify to store them)")
        df = mark_exposures(df, ds.time_unit, window_hours)
    missing = [o for o in ds.primary_outcomes
               if o in ds.outcomes and ds.outcomes[o]["event_col"] not in df.columns]
    if missing:
        print(f"  Attaching outcomes: {missing}")
        df = ds.attach_outcomes(df)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--subsets", nargs="+", default=["all", "pop_normal"])
    p.add_argument("--window_hours", type=float, default=WINDOW_HOURS)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--exposures", nargs="+", default=list(EXPOSURES), choices=EXPOSURES)
    p.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=ENCODINGS)
    p.add_argument("--dry_run", action="store_true", help="print, do not write results")
    args = p.parse_args()

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    if already_done(args, results_dir, "cox.csv", label="landmark Cox models"):
        return
    unit = getattr(ds, "outcome_time_unit", None) or ds.time_unit or "minutes"
    df = load_classified(ds, args.window_hours)
    analytes = sorted(a for a in df["analyte"].unique() if a not in set(EXCLUDE_LABS))
    methods = [m for m in ds.methods if f"{m}_class" in df.columns]
    print(f"  {len(df):,} index measurements, {df[stay_col(df)].nunique():,} stays, {len(analytes)} analytes")
    print(f"  methods {methods}; outcome clock: {unit}")

    run_models(ds, df, methods, analytes, unit, args, results_dir)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from matplotlib.ticker import FixedLocator, NullFormatter

_RI_MARKER = FAMILY_MARKERS   # lib/models.py, via figlib
COHORTS = VAL_COHORTS         # eicu, inspire, chs: one row each
PRIMARY = dict(exposure="first", encoding="binary")   # what cox / methods_all / tables show
MAX_ANALYTES = 10             # per panel: the analytes with the largest reference-method HR
_EXPORT_COLS = ["subset", "analyte", "method", "outcome", "exposure", "encoding", "level", "n", "n_events",
                "HR", "HR_lower", "HR_upper", "p_value", "p_fdr"]   # the rows the figure draws
_ENC_LEVELS = [("binary", "abnormal", "Outside interval"), ("z", "z", "Per unit z"),
               ("category", "low", "Low vs normal"), ("category", "high", "High vs normal")]
_EXPOSURE_LABEL = {"first": "First\nindex", "window_worst": "Worst\nin 48 h", "window_any": "Any\nin 48 h"}


# ─────────────────────────────────────────────────────────────────────────────
# shared
# ─────────────────────────────────────────────────────────────────────────────
def _load_cox(ds, subset="all", primary=True):
    df = load_result(ds, "cox.csv")
    if df is None or len(df) == 0:
        return None
    if "subset" in df.columns:
        df = df[df["subset"].astype(str) == subset]
        if not len(df):
            return None
    df["method"] = df["method"].map(_bm_method)
    to_numeric(df)
    df = df[(~df.analyte.isin(EXCLUDE_ANALYTES)) & np.isfinite(df.HR) & (df.HR < 50) & (df.HR > 0.02)].copy()
    if primary and "exposure" in df.columns:
        df = df[(df.exposure == PRIMARY["exposure"]) & (df.encoding == PRIMARY["encoding"])]
    return df if len(df) else None


def _methods_in(frames, exclude=()):
    present = set()
    for df in frames:
        if df is not None:
            present |= set(df.method)
    return [m for m in bm_methods() if m in present and m not in exclude]


def _outcome_label(o):
    return OUTCOME_SHORT.get(o, OUTCOME_DISPLAY.get(o, o))


def _cohort_label(ax, ds):
    """Cohort name on the right of the row's last axis."""
    ax.text(1.04, 0.5, DATASET_DISPLAY.get(ds, ds), transform=ax.transAxes,
            rotation=270, ha="left", va="center", fontsize=FONT_AXIS, color=DARK)


def _pending_row(axes, ds, message=None):
    """First axis carries the notice, the rest of the row is blank."""
    if message is None:
        pending_axis(axes[0], ds)
    else:
        axes[0].text(0.5, 0.5, message, transform=axes[0].transAxes, ha="center", va="center",
                     fontsize=FONT_TICK, color="#999999")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    for ax in axes[1:]:
        ax.set_axis_off()
    _cohort_label(axes[-1], ds)


def _top_legend(fig, methods, H):
    handles = [Line2D([], [], marker=_RI_MARKER[m.split("_")[0]], ls="", color=_BM_COLORS[m], ms=4,
                      label=RI_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / H),
               ncol=len(handles), frameon=False, fontsize=FONT_LEGEND,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# cox: rows = cohorts, columns = outcomes, HR forest per analyte
# ─────────────────────────────────────────────────────────────────────────────
def _plain_log_ticks(axis, ticks):
    """Plain-number ticks (1, 2, 5) on a log axis instead of 2 x 10^0."""
    lo, hi = axis.get_view_interval()
    axis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _log_hr_axis(ax, sub):
    """Log x axis over the bulk of the intervals: one analyte with a degenerate CI must
    not squash every other row."""
    lo_all, hi_all = sub.HR_lower.dropna(), sub.HR_upper.dropna()
    if len(lo_all) and len(hi_all):
        lo = min(0.9, max(0.1, np.percentile(lo_all, 5) * 0.8))     # HR = 1 always in view
        hi = max(1.1, min(50, np.percentile(hi_all, 95) * 1.25))
        ax.set_xlim(lo, hi)
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis, (0.2, 0.5, 1, 2, 3, 5, 10, 20))


def _significant_analytes(sub, reference, methods):
    """Analytes where a MAJORITY of the drawn methods reach FDR < 0.05 (Aashna 2026-09-04:
    "if most of the labs are insignificant, dont show that lab"), ordered by the reference
    method's hazard ratio and capped at the MAX_ANALYTES largest (with ICU sample sizes
    nearly every lab is significant for mortality; the cap keeps the panel readable)."""
    drawn = sub[sub.method.isin(methods)].drop_duplicates(["analyte", "method"])
    present = drawn.method.nunique()          # CHS legacy rows carry three methods, not seven
    n_significant = drawn[drawn.p_fdr < FDR].groupby("analyte").size()
    majority = set(n_significant[n_significant > present / 2].index)
    analytes = [a for a in all_analytes() if a in majority]
    ref_hr = sub[sub.method == reference].set_index("analyte").HR
    ordered = sorted(analytes, key=lambda a: ref_hr.get(a, -np.inf))
    return ordered[-MAX_ANALYTES:]


def _forest(ax, sub, analytes, methods):
    """Only FDR-significant estimates are drawn (Aashna 2026-09-04); a row is an analyte
    where at least one method has one."""
    y = np.arange(len(analytes))
    step = 0.7 / max(len(methods) - 1, 1)
    sub = sub[sub.p_fdr < FDR]
    for k, m in enumerate(methods):
        offset = (k - (len(methods) - 1) / 2) * step
        rows = sub[sub.method == m].drop_duplicates("analyte").set_index("analyte").reindex(analytes)
        ax.hlines(y + offset, rows.HR_lower, rows.HR_upper, color=_BM_COLORS[m], lw=0.7, alpha=0.6)
        ax.scatter(rows.HR, y + offset, s=9, marker=_RI_MARKER[m.split("_")[0]], color=_BM_COLORS[m],
                   edgecolor="white", linewidth=0.3, zorder=3)
    for yi in y[1:]:
        ax.axhline(yi - 0.5, color="#EEEEEE", lw=0.4, zorder=0)
    ax.axvline(1, color=DARK, lw=0.5, ls="--")
    _log_hr_axis(ax, sub[sub.analyte.isin(analytes)])
    ax.set_xlabel("Hazard ratio", fontsize=FONT_AXIS)
    ax.set_yticks(y)
    ax.set_yticklabels(analytes, fontsize=FONT_TICK - 1)
    ax.set_ylim(-0.6, len(analytes) - 0.4)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    hide_spines(ax)


def fig_cox():
    data = {ds: _load_cox(ds) for ds in COHORTS}
    methods = _methods_in(data.values())
    if not methods:
        return {}
    reference = "NORMA" if "NORMA" in methods else methods[0]

    panels = {}   # ds -> [(outcome, rows, analytes)], only outcomes with a significant analyte
    for ds, df in data.items():
        if df is None:
            continue
        df = df[df.method.isin(methods)]
        panels[ds] = []
        for outcome in [o for o in OUTCOMES.get(ds, []) if o in set(df.outcome)]:
            sub = df[df.outcome == outcome]
            analytes = _significant_analytes(sub, reference, methods)
            if analytes:
                panels[ds].append((outcome, sub, analytes))

    n_cols = max([len(row) for row in panels.values()] + [1])
    heights = []
    for ds in COHORTS:
        n = max([len(analytes) for _, _, analytes in panels.get(ds, [])], default=0)
        heights.append(0.18 * n + 0.8 if n else PENDING_H)
    W, H = 7.2, sum(heights) + 0.4
    fig, axes = plt.subplots(len(COHORTS), n_cols, figsize=(W, H), squeeze=False,
                             gridspec_kw=dict(height_ratios=heights))
    for ri, ds in enumerate(COHORTS):
        row = panels.get(ds)
        if row is None:
            _pending_row(axes[ri], ds)
            continue
        if not row:
            _pending_row(axes[ri], ds, f"no analyte significant for a majority of methods")
            continue
        for ci, ax in enumerate(axes[ri]):
            if ci >= len(row):
                ax.set_axis_off()
                continue
            outcome, sub, analytes = row[ci]
            _forest(ax, sub, analytes, methods)
            ax.set_title(_outcome_label(outcome), fontsize=FONT_TITLE, loc="left")
        _cohort_label(axes[ri, -1], ds)
    _top_legend(fig, methods, H)
    fig.tight_layout(w_pad=1.0, h_pad=1.0, rect=(0, 0, 0.98, 1 - 0.28 / H))
    return {None: fig}


def fig_cox_mortality():
    """Mortality only, one panel per cohort side by side: the same forest as fig_cox
    (significant estimates, majority-significant rows, top MAX_ANALYTES by NORMA HR).
    A cohort without results is a pending panel."""
    data = {ds: _load_cox(ds) for ds in COHORTS}
    methods = _methods_in(data.values())
    if not methods:
        return {}
    reference = "NORMA" if "NORMA" in methods else methods[0]
    panels = {}   # ds -> (rows, analytes) or None when pending
    for ds in COHORTS:
        df = data[ds]
        if df is None or "mortality" not in set(df.outcome):
            panels[ds] = None
            continue
        sub = df[(df.method.isin(methods)) & (df.outcome == "mortality")]
        panels[ds] = (sub, _significant_analytes(sub, reference, methods))
    n_rows = max([len(a) for p in panels.values() if p for a in [p[1]]] + [1])
    W, H = 2.4 * len(COHORTS) + 0.6, 0.22 * n_rows + 1.3
    fig, axes = plt.subplots(1, len(COHORTS), figsize=(W, H), squeeze=False)
    for ax, ds in zip(axes[0], COHORTS):
        panel = panels[ds]
        if panel is None:
            pending_axis(ax, ds)
        elif not panel[1]:
            _pending_row([ax], ds, "no analyte significant for a majority of methods")
        else:
            _forest(ax, panel[0], panel[1], methods)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, loc="left")
    _top_legend(fig, methods, H)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 1 - 0.3 / H))
    return {None: fig}


# ─────────────────────────────────────────────────────────────────────────────
# sensitivity: rows = cohort x outcome, columns = encoding, x = exposure definition
# ─────────────────────────────────────────────────────────────────────────────
def _sensitivity_panel(ax, sub, exposures, methods):
    """Median HR across analytes per method and exposure; thin line = IQR across analytes."""
    step = 0.8 / max(len(methods) - 1, 1)
    for k, m in enumerate(methods):
        offset = (k - (len(methods) - 1) / 2) * step
        for x, e in enumerate(exposures):
            v = sub[(sub.method == m) & (sub.exposure == e)].HR
            if v.empty:
                continue
            q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.plot([x + offset] * 2, [q1, q3], color=_BM_COLORS[m], lw=0.6, alpha=0.5)
            ax.scatter([x + offset], [q2], s=10, marker=_RI_MARKER[m.split("_")[0]],
                       color=_BM_COLORS[m], edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(1, color=DARK, lw=0.5, ls="--")
    ax.set_yscale("log")
    _plain_log_ticks(ax.yaxis, (0.8, 1, 1.25, 1.5, 2, 3, 5))
    ax.set_xlim(-0.6, len(exposures) - 0.4)
    ax.set_xticks(range(len(exposures)))
    ax.set_xticklabels([_EXPOSURE_LABEL[e] for e in exposures], fontsize=FONT_TICK - 1)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    hide_spines(ax)


def fig_sensitivity():
    data = {ds: _load_cox(ds, primary=False) for ds in COHORTS}
    frames = [df for df in data.values() if df is not None and "exposure" in df.columns]
    methods = _methods_in(frames)
    if not methods:
        return {}
    exposures = [e for e in _EXPOSURE_LABEL if any(e in set(df.exposure) for df in frames)]

    rows = []   # (ds, outcome); outcome None = pending cohort
    for ds in COHORTS:
        df = data[ds]
        outcomes = [o for o in OUTCOMES.get(ds, []) if df is not None and o in set(df.outcome)]
        rows += [(ds, o) for o in outcomes] or [(ds, None)]
    heights = [1.4 if outcome else PENDING_H for _, outcome in rows]
    W, H = 7.2, sum(heights) + 0.5
    fig, axes = plt.subplots(len(rows), len(_ENC_LEVELS), figsize=(W, H), squeeze=False,
                             gridspec_kw=dict(height_ratios=heights))
    for ri, (ds, outcome) in enumerate(rows):
        if outcome is None:
            _pending_row(axes[ri], ds)
            continue
        df = data[ds]
        last_of_cohort = ri + 1 == len(rows) or rows[ri + 1][0] != ds
        for ci, (enc, level, title) in enumerate(_ENC_LEVELS):
            ax = axes[ri, ci]
            sub = df[(df.outcome == outcome) & (df.encoding == enc) & (df.level == level)]
            _sensitivity_panel(ax, sub, exposures, methods)
            ax.tick_params(axis="x", labelbottom=last_of_cohort)
            if ri == 0:
                ax.set_title(title, fontsize=FONT_TITLE, loc="left")
        axes[ri, 0].set_ylabel(f"{_outcome_label(outcome)}\nHazard ratio", fontsize=FONT_AXIS)
        _cohort_label(axes[ri, -1], ds)
    _top_legend(fig, methods, H)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0, 0.98, 1 - 0.28 / H))
    return {None: fig}


# ─────────────────────────────────────────────────────────────────────────────
# methods_all: one panel per cohort, fraction of analytes with a significant HR > 1
# ─────────────────────────────────────────────────────────────────────────────
def _significant_fraction(ax, cox, outcomes, methods):
    """Bar per method: fraction of analytes with FDR < 0.05 and HR > 1 (pooled), markers
    per outcome."""
    sig = cox.assign(sig=((cox.p_fdr < FDR) & (cox.HR > 1)).astype(float))
    _bm_bars(ax, sig, "sig", outcomes, "Fraction of analytes\nwith significant HR > 1", agg="mean",
             methods=methods, labels=RI_LABELS)
    ax.set_ylim(0, 1.0)


def fig_methods_all():
    """Every reference-interval method within Pop_RI-normal tests, one panel per cohort.
    Bars are pooled over outcomes, markers are per outcome, so the marker key is per
    panel: the cohorts do not share an outcome set."""
    data = {ds: _load_cox(ds, "pop_normal") for ds in COHORTS}
    methods = _methods_in(data.values(), exclude=("PopRI",))
    if len(methods) < 2:
        return {}
    n = len(COHORTS)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.5), squeeze=False)
    for ax, ds in zip(axes[0], COHORTS):
        cox = data[ds]
        name = DATASET_DISPLAY.get(ds, ds)
        if cox is None:
            pending_axis(ax, ds)
            ax.set_title(name, fontsize=FONT_TITLE, loc="left")
            continue
        outcomes = [o for o in OUTCOMES.get(ds, []) if o in set(cox.outcome)]
        _significant_fraction(ax, cox, outcomes, methods)
        ax.set_title(name, fontsize=FONT_TITLE, loc="left")
        _bm_legend(ax, outcomes)
    axes[0][0].set_title(f"{DATASET_DISPLAY.get(COHORTS[0], COHORTS[0])}: within Pop$_{{RI}}$-normal tests",
                         fontsize=FONT_TITLE, loc="left")
    fig.tight_layout(w_pad=0.8)
    return {None: fig}


FIGURES = [
    FigSpec("13_cox", "cox",         fig_cox,         False, (), None),
    FigSpec("13_cox", "cox_mortality", fig_cox_mortality, False, (), None),
    FigSpec("13_cox", "sensitivity", fig_sensitivity, False, (), None),
    FigSpec("13_cox", "sensitivity_norma", ablation_variant(fig_sensitivity), False, (), None),
    FigSpec("13_cox", "methods_all", fig_methods_all, False, (), None),
    FigSpec("13_cox", "methods_all_norma", ablation_variant(fig_methods_all), False, (), None),
]


# ══════════════════════════════════════════════════════════════════════════
# Tables — 13_cox: table_* definitions and registry slice.
# ══════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from figlib import RI_LABELS, _BM_SUPP, _bm_method


def _count(row, col):
    return int(row[col]) if col in row and pd.notna(row[col]) else "---"


def table_cox(ds):
    df = load_result(ds, "cox.csv")
    if df is None or len(df) == 0:
        return []
    if "subset" in df.columns:                 # the table reports every stay
        df = df[df["subset"].astype(str) == "all"]
    df["method"] = df["method"].map(_bm_method)
    df = to_numeric(df)
    if "exposure" in df.columns:   # long format: primary definition only
        df = df[(df.exposure == "first") & (df.encoding == "binary")]
    methods = [m for m in _BM_SUPP if m in set(df.method)]
    written = []
    for outcome in sorted(df["outcome"].unique()):
        sub = df[df["outcome"] == outcome]
        rows = []
        for analyte in all_analytes():
            cell = sub[sub["analyte"] == analyte]
            row = {"Analyte": analyte, "N (events)": "---"}
            if len(cell):
                row["N (events)"] = f'{_count(cell.iloc[0], "n")} ({_count(cell.iloc[0], "n_events")})'
            for method in methods:
                match = cell[cell["method"] == method]
                row[f"{method} HR"] = "---"
                if len(match) == 1:
                    r = match.iloc[0]
                    row[f"{method} HR"] = _fmt_hr(r["HR"], r["HR_lower"], r["HR_upper"])
            rows.append(row)

        header = "Analyte & N (events)" + "".join(f" & {RI_LABELS[m]} HR [95\\% CI]" for m in methods)
        lines = [r"\begin{table}[ht]", r"\centering",
                 r"\begin{tabular}{lr" + "r" * len(methods) + "}", r"\toprule",
                 header + r" \\", r"\midrule"]
        for row in rows:
            cells = [row["Analyte"], row["N (events)"]] + [row[f"{m} HR"] for m in methods]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        name = f"cox_{ds}_{outcome}"
        save_table("13_cox", name, lines, pd.DataFrame(rows), landscape=True)
        written.append(name)
    return written


TABLES = [
    TableSpec("13_cox", "cox", table_cox, True, ("cox.csv",),
              lambda ds: [f"cox_{ds}_{o}" for o in OUTCOMES.get(ds, [])]),
]


if __name__ == "__main__":
    main()
