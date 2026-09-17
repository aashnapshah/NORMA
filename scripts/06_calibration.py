#!/usr/bin/env python
"""Calibration of every reference-interval method on the same index measurements.

Both steps read the interval bounds 07_classify stored per measurement, so this
runs after 07; rows are restricted to measurements EVERY method has bounds for
(Cohen has no interval where the healthy history is too thin), so all methods
are scored on the same values.

  coverage   per method (Pop_RI, Per_RI, Gaussian mle/trunc/eb, Cohen m2/m3/m4,
             NORMA_RI) and every index measurement:
                 covered     value inside [low, high]   (nominal ~95% for every
                             method: Pop_RI 2.5-97.5th pct, Per_RI +/-2 SD,
                             Gaussian/Cohen z = 1.96, NORMA 95% PI)
                 width_rel   (high - low) / (Pop_RI high - Pop_RI low)
                 inside_pop  |[low, high] n Pop_RI| / (high - low)
             aggregated per (method, analyte, realised Pop_RI state of the value:
             low / normal / high / all).  NORMA_RI is the normal-conditioned
             interval (the deployment query), so its coverage of realised-abnormal
             values is expected to be low -- that is the interval doing its job.
             -> 06_calibration.csv (per analyte + analyte="median" rows)

  conformal  which interval is narrowest AT the coverage it claims (R1-5, R1-8)?
             Methods miss their nominal coverage by different amounts (NORMA 0.92
             on eICU, empirical Bayes 0.93-0.94), so comparing raw widths compares
             nothing.  Split conformal calibration removes that confound, per
             method and analyte, on Pop_RI-normal measurements:
               1. split PATIENTS 50/50 (seeded) so calibration and evaluation
                  share no patient
               2. conformity score on the calibration half: s = |x - mid| / half_width
               3. gamma = the split-conformal ceil((n+1)*level)/n quantile of s
               4. widen the evaluation half's intervals to mid +/- gamma * half_width
                  and measure achieved coverage and width
             gamma > 1 means the method was over-confident; after step 4 every
             method sits at the same coverage, so width relative to Pop_RI is a
             like-for-like comparison.
             -> 06_conformal.csv (per analyte + analyte="median" rows)

Usage:
    python 06_calibration.py --dataset eicu
    python 06_calibration.py --dataset eicu --only conformal --level 0.9

Figures and tables
------------------
Composite files, each one figure with subplots:

    calibration.pdf               every RI method on every cohort's index measurements
                                  (06_calibration.py, coverage step) in one row, colour = method and
                                  marker = cohort: a coverage of the next value when it
                                  is realised Pop_RI-normal, b width / Pop_RI width,
                                  c fraction inside Pop_RI
    calibration_dev.pdf           NORMA on the dev test split, by queried state:
                                  nominal-vs-empirical quantiles (3) + 95% coverage

The state-conditional densities (state_conditional.pdf) live with the
reference-interval code in the refs folder: they describe the interval, not
its calibration on a cohort.
"""
import bootstrap  # noqa: F401

import argparse
import os

import hashlib

import numpy as np
import pandas as pd

import datasets

from constants import MEDIAN_ROW
from datasets import already_done, EXCLUDE_LABS, add_dataset_args, get_dataset, save_csv
from metrics import method_prefix

# Reuse is keyed on these: a step whose files are all present is skipped
# unless --force (datasets.already_done).
STEP_OUTPUTS = {
    "coverage": ["calibration.csv"],
    "conformal": ["conformal.csv"],
}
STEPS = ("coverage", "conformal")
STATE_NAMES = {0: "low", 1: "normal", 2: "high"}
MIN_N = 20        # coverage: analytes with fewer rows per state are dropped
MIN_CAL = 50      # conformal: analytes with fewer calibration rows give a meaningless quantile


def load_bounds(ds, methods, state=None, all_rows=False):
    """Index measurements with every method's bounds.  state: keep this realised
    Pop_RI state only (None = all).  all_rows: keep rows some methods lack bounds
    for (each method is then scored on its own rows)."""
    cols = (["patient_id", "analyte", "value", "PopRI_class"]
            + [f"{method_prefix(m)}_{b}" for m in methods for b in ("low", "high")])
    df = ds.load_classification(usecols=cols)
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    df = df[~df["analyte"].isin(set(EXCLUDE_LABS))]
    df = df[df["PopRI_class"].isin(STATE_NAMES)].dropna(subset=["value"])
    if state is not None:
        df = df[df["PopRI_class"] == state]
    bound_cols = [c for c in cols if c.endswith(("_low", "_high"))]
    if not all_rows:
        n0 = len(df)
        df = df[df[bound_cols].notna().all(axis=1)]
        print(f"  {n0:,} index measurements -> {len(df):,} with bounds from every method "
              f"({len(df) / max(n0, 1):.1%})")
    df = df[df["pop_ri_high"] > df["pop_ri_low"]]
    df["state"] = df["PopRI_class"].map(STATE_NAMES)
    print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients, "
          f"{df['analyte'].nunique()} analytes; methods {methods}")
    return df


def bound_batches(ds, methods, args, state=None, all_rows=False, skip=()):
    """(label, frame) over the cohort: one frame for an unchunked cohort, a few analytes
    at a time for a chunked one.  Every metric here is computed per analyte, so the
    batching cannot change a number -- it only changes how much is held at once."""
    if ds.name != "chs":
        yield "all", load_bounds(ds, methods, state=state, all_rows=all_rows)
        return
    first = datasets.classification_paths(ds._chunk_dirs()[0])[0]
    analytes = (list(ds._analytes) if ds._analytes else
                sorted(set(pd.read_parquet(first, columns=["analyte"])["analyte"]
                           .replace("", "NA").dropna()) - set(EXCLUDE_LABS)))
    analytes = [a for a in analytes if a not in set(skip)]
    keep = ds._analytes
    for i in range(0, len(analytes), args.analyte_batch):
        batch = analytes[i:i + args.analyte_batch]
        print(f"  {', '.join(batch)}  ({i + 1}-{i + len(batch)} of {len(analytes)})")
        ds._analytes = batch                      # filters per chunk, before the concat
        try:
            yield ",".join(batch), load_bounds(ds, methods, state=state, all_rows=all_rows)
        finally:
            ds._analytes = keep


def calibration_split(patient_ids, seed):
    """Half the patients, chosen by a hash of the id rather than a draw over whoever is
    in memory: a patient lands in the same half however the analytes are batched."""
    ids = pd.Series(patient_ids).astype(str) + f"|{seed}"
    h = ids.map(lambda v: int(hashlib.md5(v.encode()).hexdigest()[:8], 16))
    return (h % 2 == 0).to_numpy()


def _methods_with_bounds(ds):
    """Methods whose interval columns the classification file carries: the
    covariate-ablation arms exist in ref_intervals long before 07_classify is rerun."""
    have = ds.classification_columns()
    methods = [m for m in ds.methods if {f"{method_prefix(m)}_low", f"{method_prefix(m)}_high"} <= have]
    dropped = [m for m in ds.methods if m not in methods]
    if dropped:
        print(f"  not in classification.parquet, skipped: {dropped}")
    if not methods:
        raise SystemExit("no method has interval columns in classification.parquet")
    return methods


# ----------------------------------------------------------------- coverage

def per_row_metrics(df, method):
    """covered / width_rel / inside_pop for one method, aligned with df.index."""
    p = method_prefix(method)
    lo, hi = df[f"{p}_low"].to_numpy(float), df[f"{p}_high"].to_numpy(float)
    plo, phi = df["pop_ri_low"].to_numpy(float), df["pop_ri_high"].to_numpy(float)
    x = df["value"].to_numpy(float)
    width, pop_width = hi - lo, phi - plo
    with np.errstate(divide="ignore", invalid="ignore"):
        width_rel = np.where(pop_width > 0, width / pop_width, np.nan)
        overlap = np.clip(np.minimum(hi, phi) - np.maximum(lo, plo), 0, None)
        inside_pop = np.where(width > 0, overlap / width, np.nan)
    return pd.DataFrame({"covered": (x >= lo) & (x <= hi),
                         "width_rel": width_rel, "inside_pop": inside_pop}, index=df.index)


def done_analytes(results_dir, name, force):
    """Analytes already in `name`: this step writes per batch, so a rerun resumes."""
    if force:
        return set()
    path = datasets.find_in(results_dir, name)
    if not os.path.exists(path) or not os.path.getsize(path):
        return set()
    d = pd.read_csv(path, usecols=lambda c: c == "analyte", keep_default_na=False)
    return set(d["analyte"].astype(str)) - {MEDIAN_ROW} if "analyte" in d.columns else set()


def run_coverage(ds, args, results_dir):
    methods = _methods_with_bounds(ds)
    rows = []
    already = done_analytes(results_dir, "calibration.csv", args.force)
    if already:
        print(f"  already in calibration.csv, skipped: {len(already)} analyte(s)")
    for _, df in bound_batches(ds, methods, args, all_rows=args.all_rows, skip=already):
        if df is None or not len(df):
            continue
        batch_rows = []
        for method in methods:
            m = per_row_metrics(df, method).dropna(subset=["covered"])
            m["analyte"], m["state"] = df.loc[m.index, "analyte"], df.loc[m.index, "state"]
            for state, part in list(m.groupby("state")) + [("all", m)]:
                g = part.groupby("analyte")
                out = pd.DataFrame({"n": g.size(), "coverage95": g["covered"].mean(),
                                    "width_rel": g["width_rel"].median(),
                                    "inside_pop": g["inside_pop"].median()}).reset_index()
                out = out[out["n"] >= MIN_N]
                out.insert(0, "state", state)
                out.insert(0, "method", method)
                rows.append(out)
                batch_rows.append(out)
        if batch_rows:                             # written as it goes, not at the end
            part = pd.concat(batch_rows, ignore_index=True)
            save_csv(part.round({m: 4 for m in ("coverage95", "width_rel", "inside_pop")}),
                     os.path.join(results_dir, "calibration.csv"),
                     analytes=sorted(part["analyte"].unique()))
    if not rows:
        print("  nothing with enough data")
        return
    metrics = ["coverage95", "width_rel", "inside_pop"]
    detail = pd.concat(rows, ignore_index=True)
    detail[metrics] = detail[metrics].round(4)
    summary = detail.groupby(["method", "state"])[metrics].median().reset_index()
    summary["n_analytes"] = detail.groupby(["method", "state"])["analyte"].nunique().to_numpy()
    summary["n"] = detail.groupby(["method", "state"])["n"].sum().to_numpy()
    # one file: per-analyte rows plus the across-analyte rows as analyte="median"
    summary["analyte"] = MEDIAN_ROW
    save_csv(pd.concat([detail, summary], ignore_index=True),
             os.path.join(results_dir, "calibration.csv"))

    print("\n  median across analytes (state=all / normal):")
    for method in methods:
        a = summary[(summary.method == method) & (summary.state == "all")]
        n = summary[(summary.method == method) & (summary.state == "normal")]
        if len(a) and len(n):
            print(f"    {method:<18s} coverage {a.coverage95.iloc[0]:.3f} / {n.coverage95.iloc[0]:.3f}   "
                  f"width/Pop {a.width_rel.iloc[0]:.2f}   inside Pop {a.inside_pop.iloc[0]:.2f}")
    print(f"\nWrote {results_dir}/calibration.csv ({len(detail)} rows + medians)")
    print(f"Wrote {results_dir}/calibration.csv ({len(summary)} rows)")


# ----------------------------------------------------------------- conformal

def conformal_gamma(scores, level):
    """Split-conformal quantile: the ceil((n+1)*level)/n empirical quantile."""
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    if n < MIN_CAL:
        return np.nan
    k = int(np.ceil((n + 1) * level))
    if k > n:                      # too few points to certify this level
        return np.nan
    return float(s[k - 1])


def run_conformal(ds, args, results_dir):
    methods = _methods_with_bounds(ds)
    rows = []
    already = done_analytes(results_dir, "conformal.csv", args.force)
    if already:
        print(f"  already in conformal.csv, skipped: {len(already)} analyte(s)")
    for _, df in bound_batches(ds, methods, args, state=1 if args.state == "normal" else None,
                               skip=already):
        if df is None or not len(df):
            continue
        batch = conformal_rows(df, methods, args)
        rows += batch
        if batch:                                  # written as it goes, not at the end
            save_csv(pd.DataFrame(batch).round(4), os.path.join(results_dir, "conformal.csv"),
                     analytes=sorted({r["analyte"] for r in batch}))
    detail = pd.DataFrame(rows)
    return _save_conformal(detail, ds, args, results_dir)


def conformal_rows(df, methods, args):
    # patient-level split: a patient's rows are all in calibration or all in evaluation
    is_cal = calibration_split(df["patient_id"].to_numpy(), args.seed)
    print(f"  calibration {is_cal.sum():,} rows / evaluation {(~is_cal).sum():,} rows")

    pop_w = (df["pop_ri_high"] - df["pop_ri_low"]).to_numpy(float)
    x = df["value"].to_numpy(float)
    analyte = df["analyte"].to_numpy()

    rows = []
    for m in methods:
        pre = method_prefix(m)
        lo, hi = df[f"{pre}_low"].to_numpy(float), df[f"{pre}_high"].to_numpy(float)
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        ok = np.isfinite(mid) & np.isfinite(half) & (half > 0)
        score = np.full(len(df), np.nan)
        score[ok] = np.abs(x[ok] - mid[ok]) / half[ok]
        for a in np.unique(analyte):
            sel = (analyte == a) & ok
            cal, ev = sel & is_cal, sel & ~is_cal
            if ev.sum() < MIN_CAL:
                continue
            gamma = conformal_gamma(score[cal], args.level)
            raw_cov = float(np.mean(score[ev] <= 1.0))
            raw_w = float(np.median(2 * half[ev] / pop_w[ev]))
            if np.isfinite(gamma):
                cal_cov = float(np.mean(score[ev] <= gamma))
                cal_w = float(np.median(2 * gamma * half[ev] / pop_w[ev]))
            else:
                cal_cov = cal_w = np.nan
            rows.append({"method": m, "analyte": a, "n_cal": int(cal.sum()), "n_eval": int(ev.sum()),
                         "gamma": gamma, "coverage_raw": raw_cov, "width_rel_raw": raw_w,
                         "coverage_cal": cal_cov, "width_rel_cal": cal_w})

    return rows


def _save_conformal(detail, ds, args, results_dir):
    if detail.empty:
        print("  no analyte had enough rows; nothing written")
        return
    metrics = ["gamma", "coverage_raw", "width_rel_raw", "coverage_cal", "width_rel_cal"]
    summary = detail.groupby("method")[metrics].median().reset_index()
    summary["n_analytes"] = detail.groupby("method").size().values
    summary = summary.sort_values("width_rel_cal")
    summary["analyte"] = MEDIAN_ROW      # across-analyte rows live in the same file
    save_csv(pd.concat([detail, summary], ignore_index=True).round(4),
             os.path.join(results_dir, "conformal.csv"), analytes=ds._analytes)
    print(f"\n  Median across analytes (target coverage {args.level}); "
          f"ranked by width at the corrected coverage:")
    print(summary.round(3).to_string(index=False))
    print("\n  gamma > 1 = the method's stated interval was too narrow and had to be widened.")


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--runs", nargs="+", default=None,
                   help="NORMA run ids to score (default: dataset.run_ids); lets the covariate-ablation "
                        "arms be scored without flipping NORMA_ABLATION_RUN_IDS for the whole pipeline")
    g = p.add_argument_group("coverage")
    g.add_argument("--all-rows", action="store_true",
                   help="score each method on every row it has bounds for, instead of "
                        "restricting to rows where all methods have bounds")
    g = p.add_argument_group("conformal")
    g.add_argument("--level", type=float, default=0.95, help="target coverage")
    g.add_argument("--seed", type=int, default=42)
    p.add_argument("--analyte_batch", type=int, default=4,
                   help="chunked cohorts: analytes held in memory per pass (default: %(default)s)")
    g.add_argument("--state", default="normal", choices=["normal", "all"],
                   help="which realised Pop_RI state to calibrate on")
    args = p.parse_args()

    ds = get_dataset(args)
    if args.runs:
        if ds.no_norma:
            raise SystemExit("--runs names NORMA arms, --no_norma drops them; pick one")
        ds.run_ids = list(args.runs)
    results_dir = ds.setup_output()
    # Reuse is the default: drop any step whose output is already written.
    # This runs BEFORE the classification is loaded, so a fully-cached run
    # costs nothing rather than paying the read and then skipping.
    todo = [s for s in args.only
            if not already_done(args, results_dir, *STEP_OUTPUTS[s], label=s)]
    if not todo:
        return
    if "coverage" in todo:
        print("=== coverage ===")
        run_coverage(ds, args, results_dir)
    if "conformal" in todo:
        print("=== conformal ===")
        run_conformal(ds, args, results_dir)


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

import os

from figlib import *  # noqa: F401,F403
from datasets import dev_results_dir


_CAL_STATES = [("low", "Below Pop$_{RI}$"), ("normal", "Within Pop$_{RI}$"),
               ("high", "Above Pop$_{RI}$")]
_CAL_QUANTILES = [(0.025, "q025"), (0.25, "q25"), (0.5, "q50"),
                  (0.75, "q75"), (0.975, "q975")]


def _load_calibration():
    """Per-analyte calibration rows for the published NORMA run."""
    path = find_in(dev_results_dir("06_calibration"), "calibration_by_analyte.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df[df["model"].astype(str).str.contains("Quantile")]
    df = df[~df["code"].isin(EXCLUDE_ANALYTES)].copy()
    return df if len(df) else None


def _plot_quantiles(ax, sub, title):
    """Empirical vs nominal quantile for every analyte; perfect calibration is the diagonal."""
    ax.plot([0, 1], [0, 1], color=DARK, lw=0.8, ls="--", zorder=1)
    for nominal, col in _CAL_QUANTILES:
        emp = pd.to_numeric(sub.get(f"{col}_emp"), errors="coerce").dropna()
        if emp.empty:
            continue
        ax.scatter(np.full(len(emp), nominal), emp, s=5, alpha=0.45,
                   color=METHOD_COLORS["NORMA"], linewidths=0, zorder=2)
        ax.scatter([nominal], [emp.median()], s=26, color=DARK, marker="_", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    style_axes(ax, "Nominal quantile", None, title)


def _plot_coverage(ax, df):
    """Realised 95% coverage per analyte, by state — the headline calibration number."""
    cov = pd.to_numeric(df["coverage95"], errors="coerce")
    for i, (state, _) in enumerate(_CAL_STATES):
        v = cov[df["state"] == state].dropna()
        if v.empty:
            continue
        ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-0.16, 0.16, len(v)),
                   v, s=7, alpha=0.5, color=METHOD_COLORS["NORMA"], linewidths=0)
        ax.scatter([i], [v.median()], s=40, marker="_", color=DARK, zorder=3)
    ax.axhline(0.95, color=DARK, lw=0.8, ls="--")
    ax.set_xticks(range(len(_CAL_STATES)))
    ax.set_xticklabels([s.capitalize() for s, _ in _CAL_STATES])
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    style_axes(ax, "Realised state", "Coverage of 95% interval", "95% coverage")


def fig_calibration_dev():
    """NORMA on the dev test split, one row: quantile calibration per queried state (a-c) + 95% coverage (d).

    Separating by state matters because the query state is what NORMA
    conditions on: the normal-state interval is the one used to flag, and it
    is the one that has to be honest.
    """
    df = _load_calibration()
    if df is None:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, (state, title) in zip(axes[:3], _CAL_STATES):
        _plot_quantiles(ax, df[df["state"] == state], title)
    axes[0].set_ylabel("Empirical quantile", fontsize=FONT_AXIS)
    for ax in axes[1:3]:
        ax.tick_params(labelleft=False)
    _plot_coverage(axes[3], df)
    fig.tight_layout(w_pad=0.8)
    return {None: fig}


# ── Calibration of every RI method, cohorts overlaid (Supp. Fig. 7 redo) ──────
def _quartiles(sub, col):
    """(q1, median, q3) of `col` across the rows, or the stored quartiles when the file is a
    transcribed summary (one 'median' pseudo-analyte row with <col>_q25 / <col>_q75)."""
    if f"{col}_q25" in sub.columns and len(sub) == 1:
        r = sub.iloc[0]
        return float(r[f"{col}_q25"]), float(r[col]), float(r[f"{col}_q75"])
    v = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
    if not len(v):
        return None
    return tuple(np.percentile(v, [25, 50, 75]))


def _median_and_iqr(ax, x, sub, col, color, marker):
    """Median across analytes as the cohort's marker, IQR as a whisker. Returns the drawn
    extent (q1, q3) so the caller can scale the panel to the data."""
    q = _quartiles(sub, col)
    if q is None:
        return None
    q1, q2, q3 = q
    ax.plot([x, x], [q1, q3], color=color, lw=0.8, alpha=0.55, zorder=2, solid_capstyle="butt")
    ax.scatter([x], [q2], s=18, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=4)
    return q1, q3


def _prop_ylim(ax, extents, ref):
    """Scale a proportion panel (coverage, fraction inside Pop_RI) to the drawn whiskers instead
    of pinning 0-1, for the same reason the sensitivity panels do not start at zero: when every
    method sits between 0.8 and 0.95, a full 0-1 axis spends nine tenths of the height on empty
    space. The reference line is always kept in view, and the range is clipped to [0, 1] because
    a proportion outside that is meaningless."""
    if not extents:
        ax.set_ylim(-0.02, 1.04)
        return
    lo = min(min(e) for e in extents)
    hi = max(max(e) for e in extents)
    lo, hi = min(lo, ref), max(hi, ref)
    pad = 0.06 * (hi - lo or 1)
    ax.set_ylim(max(lo - pad, 0.0), min(hi + pad, 1.0))


def _method_axis(ax, methods, labels, show_labels):
    ax.set_xticks(range(len(methods)))
    long = max((len(str(labels[m])) for m in methods), default=0) > 14
    ax.set_xticklabels([labels[m] for m in methods] if show_labels else [],
                       rotation=45 if long else 90, ha="right" if long else "center",
                       fontsize=5.5)
    ax.set_xlim(-0.6, len(methods) - 0.4)


def _cohort_offsets(cohorts):
    """x offsets so a method's cohorts sit side by side inside its slot."""
    n = len(cohorts)
    step = 0.5 / max(n, 1)
    return {ds: (k - (n - 1) / 2) * step for k, ds in enumerate(cohorts)}


def _panel_coverage(ax, frames, methods, offsets, colors):
    """a: coverage of the next value when it is realised Pop_RI-normal — the value every
    method's interval is meant to contain (Aashna 2026-08-28: low / high states not shown)."""
    extents = []
    for i, m in enumerate(methods):
        for ds, det in frames.items():
            extents.append(_median_and_iqr(ax, i + offsets[ds], det[(det.method == m) & (det.state == "normal")],
                                           "coverage95", colors[m], DATASET_MARKERS.get(ds, "o")))
    ax.axhline(0.95, color=DARK, lw=0.7, ls="--", zorder=1)
    _prop_ylim(ax, [e for e in extents if e], ref=0.95)
    style_axes(ax, None, "Coverage of 95% interval")


def _panel_method_metric(ax, frames, methods, offsets, colors, col, ylabel, ref=None,
                         log=False, proportion=False):
    """b/c: one value per method x analyte (interval geometry does not depend on the realised state)."""
    extents = []
    for i, m in enumerate(methods):
        for ds, det in frames.items():
            sub = det[det.state == "all"]
            extents.append(_median_and_iqr(ax, i + offsets[ds], sub[sub.method == m], col,
                                           colors[m], DATASET_MARKERS.get(ds, "o")))
    if ref is not None:
        ax.axhline(ref, color=DARK, lw=0.7, ls="--", zorder=1)
    if log:
        ax.set_yscale("log")
    elif proportion:
        _prop_ylim(ax, [e for e in extents if e], ref=ref if ref is not None else 1.0)
    style_axes(ax, None, ylabel)


def _load_calibration_per_analyte(ds):
    det = load_result(ds, "calibration.csv")
    if det is None:
        return None
    det = det[(det.analyte != MEDIAN_ROW) & ~det.analyte.isin(EXCLUDE_ANALYTES)].copy()
    for c in ("n", "coverage95", "width_rel", "inside_pop"):   # not to_numeric(): it would blank `state`
        det[c] = pd.to_numeric(det[c], errors="coerce")
    if len(det):
        # the medians the figure draws, one pseudo-analyte row per method x state
        g = det[det.state.isin(["normal", "all"])].groupby(["method", "state"])
        cols = ["coverage95", "width_rel", "inside_pop"]
        summary = g[cols].median()
        for col in cols:
            summary[f"{col}_q25"] = g[col].quantile(0.25)
            summary[f"{col}_q75"] = g[col].quantile(0.75)
        summary["n"] = g["n"].sum()
        summary = summary.reset_index().assign(analyte="median")
    return det if len(det) else None


_CAL_COLS = ["Coverage of realised-normal values", "Interval width", "Agreement with Pop$_{RI}$"]


def _calibration_fig(methods_for, labels, colors, all_arms=False):
    """One row, columns = a/b/c, colour = method, marker = cohort; marker = median across
    analytes, whisker = IQR. `methods_for(frames)` picks and orders the methods to show.

    all_arms=True keeps a slot for every NORMA arm even when it has nothing to
    plot, and labels the empty ones with why (figlib.arm_notes)."""
    frames = {ds: _load_calibration_per_analyte(ds) for ds in DATASETS}
    frames = {ds: d for ds, d in frames.items() if d is not None}
    if not frames:
        return None
    methods = methods_for(frames)
    if not methods:
        return None
    wide = pd.concat([d.loc[d.state == "all", "width_rel"] for d in frames.values()])
    log_width = bool(len(wide.dropna()) and wide.quantile(0.95) > 4)
    offsets = _cohort_offsets(list(frames))

    present = set().union(*(set(d.method) for d in frames.values()))
    notes = arm_notes(present, [m for m in methods if m.startswith("NORMA_")]) if all_arms else {}
    width = 7.2 + (0.30 * max(0, len(methods) - 8) if all_arms else 0)
    fig, axes = plt.subplots(1, 3, figsize=(width, 3.0))
    _panel_coverage(axes[0], frames, methods, offsets, colors)
    _panel_method_metric(axes[1], frames, methods, offsets, colors, "width_rel",
                         "Width / Pop$_{RI}$ width", ref=1.0, log=log_width)
    # no reference line here: 1.0 is Pop_RI's own value, kept in range but not drawn
    _panel_method_metric(axes[2], frames, methods, offsets, colors, "inside_pop",
                         "Fraction inside Pop$_{RI}$", proportion=True)
    for k, (ax, title) in enumerate(zip(axes, _CAL_COLS)):
        _method_axis(ax, methods, labels, show_labels=True)
        if notes and k == 0:
            lo, hi = ax.get_ylim()
            for i, m in enumerate(methods):
                if m in notes:
                    ax.text(i, lo + 0.5 * (hi - lo), notes[m], rotation=90, ha="center",
                            va="center", fontsize=5.0, color="#AAAAAA")
        ax.set_title(title, fontsize=FONT_TITLE, loc="left")
    # shape = cohort, in neutral ink: colour is already spent on the method
    handles = [Line2D([], [], ls="none", marker=DATASET_MARKERS.get(ds, "o"), color=DARK,
                      markersize=4, label=DATASET_DISPLAY.get(ds, ds)) for ds in frames]
    handles += pending_handles(frames)
    fig.legend(handles=handles, frameon=False, fontsize=FONT_LEGEND, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 1 - 0.24 / fig.get_figheight()))
    return {None: fig}


def fig_calibration():
    """Every reference-interval method on every cohort, except Pop_RI: it defines the
    realised-normal state, the width ratio and the agreement panel, so it scores exactly
    1.0 in all three by construction and only took up a slot (Aashna 2026-09-04). The
    dashed lines are where it sits.

    The NORMA quantile-calibration panel of the old 2x2 was the dev-test split,
    identical on every row; it stays in calibration_dev.pdf instead.
    """
    return _calibration_fig(
        lambda frames: [m for m in _BM_SUPP if m != "PopRI" and any(m in set(d.method) for d in frames.values())],
        RI_LABELS, _BM_COLORS)


def fig_calibration_norma():
    """The NORMA covariate-ablation arms only, on the real cohorts: does feeding age at draw /
    care setting / same-draw co-analytes change coverage, width or Pop_RI agreement?

    Empty until the arms have reference intervals for a cohort: set NORMA_ABLATION_RUN_IDS in
    lib/datasets.py, run 04_refs.py --only norma --runs <arms>, then 06_calibration.py, coverage step.
    """
    def arms(frames):
        present = set().union(*(set(d.method) for d in frames.values()))
        # Every arm gets a slot, plotted or not, so the figure shows what was
        # tried; arm_notes() says why each empty slot is empty.
        found = [m for m in ALL_ARM_METHODS if m in present]
        if len(found) < 2:                       # the baseline alone is not a comparison
            return []
        return ALL_ARM_METHODS
    # Twenty arms will not fit with the covariate labels the other figures use
    # ("NORMA | sex, age, setting" x 20 collides into an unreadable band), so the
    # axis carries the run id and 05_norma_arms carries what each one is.
    labels = {m: m.replace("NORMA_", "") for m in ALL_ARM_METHODS}
    return _calibration_fig(arms, labels, ALL_ARM_COLORS, all_arms=True)




def fig_conformal():
    """Width each method needs to actually cover 95% (06_calibration.py, conformal step).

    Answers R1-5 (are the intervals calibrated) and R1-8 (does the transformer beat
    simple fits to the same Pop_RI-normal history) on one axis. Comparing raw widths
    is meaningless while the methods miss their nominal coverage by different
    amounts, so every method is first widened by its own split-conformal factor
    until it truly covers 95%; only then is width comparable.

    Left: that widening factor (1 = the method was already honest, >1 = it was
    over-confident). Right: the resulting width relative to Pop_RI — below 1 means
    the method genuinely beats the population interval, above 1 means it buys
    nothing. Pop_RI is the reference line, not a competitor: the normal state is
    defined by Pop_RI, so its own coverage here is circular.
    """
    frames = []
    for ds in VAL_COHORTS:
        d = load_result(ds, "conformal.csv")
        if d is not None and "analyte" in d.columns:
            d = d[d["analyte"] == MEDIAN_ROW]      # one row per method
        if d is None or not len(d):
            frames.append((ds, None)); continue
        d = to_numeric(d)
        frames.append((ds, {r["method"]: {"gamma": (float(r["gamma"]), np.nan, np.nan),
                                          "width": (float(r["width_rel_cal"]), np.nan, np.nan)}
                            for _, r in d.iterrows()}))
    if not any(f[1] for f in frames):
        return {}
    have = set()
    for _, d in frames:
        if d:
            have |= set(d)
    methods = [m for m in bm_methods() if m in have]
    if not methods:
        return {}
    metrics = [("gamma", "Widening needed to cover 95%", None),
               ("width", r"Width at true 95% coverage (vs Pop$_{RI}$)", None)]
    fig = dot_blocks(frames, metrics, methods, _BM_COLORS, RI_LABELS, W=7.0, row_h=0.16,
                     label_rotation=90)
    for ax in fig.axes:
        ax.axvline(1.0, color=DARK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    fig.text(0.5, 0.995, "Dashed line = no widening needed (left) and no gain over the population "
                         "interval (right); left of it is better on the right panel",
             ha="center", va="top", fontsize=FONT_TICK, color=DARK)
    return {"": fig}

def _arm_key(method):
    """Method label -> the ALL_ARM_METHODS key, resolving the bare "NORMA" alias
    the result files use for whichever run is the main model."""
    return f"NORMA_{NORMA_RUN_ID}" if str(method) == "NORMA" else str(method)


def fig_conformal_norma():
    """The same conformal recalibration, for the NORMA arms rather than the
    benchmark methods: how much each arm has to be widened to truly cover 95%,
    and what its width is worth once it does.

    Every arm keeps a row whether or not it can be plotted, with the reason on
    the blank ones, so the figure shows what was tried (figlib.arm_notes).
    """
    frames = []
    for ds in VAL_COHORTS:
        d = load_result(ds, "conformal.csv")
        if d is not None and "analyte" in d.columns:
            d = d[d["analyte"] == MEDIAN_ROW]
        if d is None or not len(d):
            frames.append((ds, None)); continue
        d = to_numeric(d)
        # conformal.csv calls the main model "NORMA" where calibration.csv calls
        # it "NORMA_<run>"; without this the main arm reads as never run.
        frames.append((ds, {_arm_key(r["method"]): {"gamma": (float(r["gamma"]), np.nan, np.nan),
                                                    "width": (float(r["width_rel_cal"]), np.nan, np.nan)}
                            for _, r in d.iterrows()}))
    have = set()
    for _, d in frames:
        if d:
            have |= set(d)
    if len(have & set(ALL_ARM_METHODS)) < 2:
        return {}
    methods = list(ALL_ARM_METHODS)
    labels = {m: m.replace("NORMA_", "") for m in methods}
    metrics = [("gamma", "Widening needed to cover 95%", None),
               ("width", r"Width at true 95% coverage (vs Pop$_{RI}$)", None)]
    fig = dot_blocks(frames, metrics, methods, ALL_ARM_COLORS, labels, W=7.0, row_h=0.16,
                     label_rotation=90, row_labels=True, notes=arm_notes(have, methods))
    for ax in fig.axes:
        ax.axvline(1.0, color=DARK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    return {"": fig}


def _load_arm_calibration():
    """Per-analyte dev-test calibration for every arm that has one.

    model/logs/<arm>/calibration_test.csv is written per training run, so this
    needs no pipeline stage: it is the one view where an arm's coverage can be
    read without 04_refs having been run over a cohort. Arms trained under
    --split_by patient measure it on their own held-out patients rather than the
    sequence split, which the row labels carry.
    """
    from datasets import MODEL_LOG_DIR
    out = {}
    for run in ALL_ARMS:
        p = os.path.join(MODEL_LOG_DIR, run, "calibration_test.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, keep_default_na=False, na_values=[""])
        d = d[~d["code"].isin(EXCLUDE_ANALYTES)]
        if len(d):
            out[f"NORMA_{run}"] = to_numeric(d)
    return out


def fig_calibration_dev_norma():
    """Coverage, width and Pop_RI agreement of every arm on the development test
    split: median across analytes, whisker = IQR.

    The cohort-level calibration figure can only show the arms 04_refs was run
    over; this one is written by training itself, so it reaches every arm that
    finished -- the patient-split arms included, which no cohort figure has.
    """
    per_arm = _load_arm_calibration()
    if len(per_arm) < 2:
        return {}
    methods = list(ALL_ARM_METHODS)
    labels = {m: m.replace("NORMA_", "") for m in methods}

    def stat(d, col, state):
        sub = d[d["state"] == state] if state else d
        v = pd.to_numeric(sub[col], errors="coerce").dropna()
        if not len(v):
            return (np.nan, np.nan, np.nan)
        return (float(v.median()), float(v.quantile(0.25)), float(v.quantile(0.75)))

    block = {}
    for m, d in per_arm.items():
        block[m] = {"coverage95": stat(d, "coverage95", "normal"),
                    "width_rel": stat(d, "width_rel", "normal"),
                    "inside_pop": stat(d, "inside_pop", "normal")}
    metrics = [("coverage95", "Coverage of 95% interval", None),
               ("width_rel", r"Width / Pop$_{RI}$ width", None),
               ("inside_pop", r"Fraction inside Pop$_{RI}$", None)]
    fig = dot_blocks([("dev test split", block)], metrics, methods,
                     ALL_ARM_COLORS, labels, W=7.0, row_h=0.16,
                     row_labels=True, notes=arm_notes(set(per_arm), methods))
    for ax in fig.axes[:1]:
        ax.axvline(0.95, color=DARK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    return {"": fig}


FIGURES = [
    FigSpec("06_calibration", "calibration", fig_calibration, False, (), None),
    FigSpec("06_calibration", "calibration_dev_norma", fig_calibration_dev_norma, False, (), None),
    FigSpec("06_calibration", "conformal_norma", fig_conformal_norma, False, (), None),
    FigSpec("06_calibration", "calibration_norma", fig_calibration_norma, False, (), None),
    FigSpec("06_calibration", "calibration_dev", fig_calibration_dev, False, (), None),
    FigSpec("06_calibration", "conformal", fig_conformal, False, (), None),
]


if __name__ == "__main__":
    main()
