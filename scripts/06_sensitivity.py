#!/usr/bin/env python
"""Sensitivity of every reference-interval method to the history it is given.

Every method sees the SAME synthetic histories — a 50-year-old male with n
measurements BASELINE_SPACING days apart drawn from N(Pop_RI midpoint, sd) —
and returns its interval for the next value.  Three sweeps, the ones the
NORMA-only model/sensitivity_analysis.py reports:

    history_length   n = 2 ... 300 at sd = BASE_SD x Pop_RI width
    horizon          7 ... 3650 days ahead.  Only NORMA sees the horizon; the
                     other methods return the same interval whatever it is,
                     so their curves are flat by construction
    history_std      sd = 0 ... 0.3 x Pop_RI width (0 = perfectly flat history)

Methods, keyed as in the figure code (_BM_SUPP):
    PopRI           the population interval, constant
    PerRI           GMM setpoint +/- 2 SD of the whole history (lib/metrics.py,
                    the estimator 04_refs uses)
    Gaussian_mle / Gaussian_trunc / Gaussian_eb
                    fits to the Pop_RI-normal values of the history, EB with the
                    dev-cohort prior (model/baselines/gaussian.py); histories
                    with too few normal values fall back to Pop_RI, as in 04_refs
    Cohen_m4        dev-trained model applied through cohen.apply_dev_cohen on a
                    synthetic split_df.  build_pair_table's min_bl is lowered
                    from 5 to 1 here so histories shorter than the pipeline's
                    minimum are still scored
    NORMA           queried with state = normal (the interval used to flag);
                    covariate-ablation arms as NORMA_<arm> (--norma_runs)

Widths are the 95% interval as % of the Pop_RI width; the centre shift is the
interval centre relative to the Pop_RI midpoint the histories are drawn around.
N_DRAWS histories per (analyte, sweep value), averaged.

Needs the Cohen artifact and the dev EB prior in artifacts/, not ref_intervals.
Output: results/raw/dev/06_sensitivity_methods.csv, one row per
model x analyte x feature x value, with the column names of sensitivity.csv so
fig_sensitivity in figures.py can read either.

    python 06_sensitivity.py [--norma_runs q_age 334f7e21] [--analytes HGB NA]

Figures and tables
------------------
    sensitivity.pdf           one panel per sweep, every method overlaid: CI width vs
                              history length / horizon / within-person SD.  The
                              across-analyte spread is in the sensitivity_summary table.
    sensitivity_norma.pdf     the same three sweeps for the NORMA covariate-ablation arms
                              only (baseline vs + age / setting / co-analytes); empty
                              until the arms are swept with --norma_runs.

Stage 06 has two scripts and both write to the 06_calibration folder, so these land
in results/figures/<tag>/ beside the calibration figures (both 06_ prefixed).
"""
import bootstrap

import argparse
import functools
import os
import time

import numpy as np
import pandas as pd
import torch

def _load(name, path):
    """Import a model module by path (model/ is not a package)."""
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the model code and its baselines import each other by bare name, and
# `import config` has to reach process/config.py there
import importlib.util as _ilu
import sys as _sys
for _p in (os.path.join(bootstrap.MODEL_DIR, "baselines"), bootstrap.MODEL_DIR,
           os.path.join(bootstrap.SCRIPTS_DIR, "process")):
    if _p not in _sys.path:
        _sys.path.append(_p)

import datasets as _VCFG   # validation paths + run constants (lib/datasets.py)
from metrics import REFERENCE_INTERVALS as REF

EXCLUDE_LABS, MODEL_LOG_DIR = _VCFG.EXCLUDE_LABS, _VCFG.MODEL_LOG_DIR
NORMA_CHECKPOINT, NORMA_RUN_ID = _VCFG.NORMA_CHECKPOINT, _VCFG.NORMA_RUN_ID
NORMA_ABLATION_RUN_IDS = getattr(_VCFG, "NORMA_ABLATION_RUN_IDS", [])

BASE_SD = 0.10        # history noise sd as a fraction of the Pop_RI width (= history_std value 1.0)
N_DRAWS = 20
PER_N_STD = 2         # PerRI = setpoint +/- 2 SD (04_refs --gmm_n_std default)
Z = 1.96
SWEEP_METHODS = ["PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb", "Cohen_m4", "NORMA"]


def make_histories(labs, features, n_draws, seed):
    """One record per synthetic history.  Horizon sweeps reuse the baseline histories."""
    import sensitivity_analysis as SA
    recs = []
    for lab in labs:
        low, high, unit = REF[lab]["M"]
        mid, span = (low + high) / 2.0, high - low
        for feat, vals in features.items():
            for v in vals:
                n_hist, sd, horizon = SA.BASELINE_N_HIST, BASE_SD * span, SA.BASELINE_HORIZON
                if feat == "history_length":
                    n_hist = int(v)
                elif feat == "horizon":
                    horizon = float(v)
                elif feat == "history_std":
                    sd = float(v) * span / 10.0
                for i in range(n_draws):
                    rng = np.random.default_rng([seed, i, labs.index(lab), n_hist])
                    # identical draws for the horizon sweep and the n=10 / sd=base points
                    if sd > 0:
                        t_h, x_h = SA.make_noisy_history(mid, n_hist, SA.BASELINE_SPACING, sd, rng)
                    else:
                        t_h, x_h = SA.make_flat_history(mid, n_hist, SA.BASELINE_SPACING)
                    recs.append(dict(lab=lab, feature=feat, value=float(v), draw=i, t=t_h, x=x_h,
                                     horizon=horizon, low=low, high=high, mid=mid, span=span,
                                     key=f"{lab}|{feat}|{float(v)}|{i}",
                                     # the n=10 history-length record with the same draw: identical
                                     # history, which the horizon sweep reuses for horizon-blind methods
                                     base_key=f"{lab}|history_length|{float(SA.BASELINE_N_HIST)}|{i}"))
    return recs


def norma_intervals(recs, run_id, checkpoint):
    """One NORMA run's intervals. Covariate arms are fed the per-measurement inputs they were
    trained with (see SA.predict(covariates=True)); the baseline run gets none."""
    import sensitivity_analysis as SA
    from utils import load_checkpoint, create_model, uses_covariates
    from data import TEST_VOCAB
    device = torch.device("cpu")
    ckpt, hparams = load_checkpoint(MODEL_LOG_DIR, run_id, best=(checkpoint == "best"),
                                    device=device, quiet=True)
    model = create_model(hparams, ncodes=len(TEST_VOCAB), checkpoint=ckpt).to(device).eval()
    SA.init_model(model=model, device=device, hparams=hparams)
    cov = bool(uses_covariates(model))
    flags = [n for n in ("use_age_t", "use_setting", "use_coanalytes") if getattr(model, n, False)]
    print(f"  NORMA {run_id}: checkpoint_{checkpoint} (epoch {ckpt.get('epoch')}), "
          f"{'quantile' if SA.IS_QUANTILE else 'gaussian'} head"
          f"{', covariates: ' + '+'.join(flags) if flags else ''}")
    out = {}
    t0 = time.time()
    for k, r in enumerate(recs):
        try:
            mu, second = SA.predict(r["lab"], 0, SA.BASELINE_AGE, r["t"].tolist(), r["x"].tolist(),
                                    r["t"][-1] + r["horizon"], state=1, covariates=cov)
            width = float(second) if SA.IS_QUANTILE else 2 * Z * float(second)
            out[r["key"]] = (float(mu) - width / 2, float(mu) + width / 2)
        except Exception:   # analyte missing from the vocabulary etc.
            out[r["key"]] = (np.nan, np.nan)
        if (k + 1) % 5000 == 0:
            print(f"    {run_id} {k + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    return out


def simple_intervals(recs, prior):
    """PopRI, PerRI and the three Gaussian fits, straight from the estimators."""
    import gaussian as G
    out = {m: {} for m in ("PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb")}
    for r in recs:
        x, low, high = r["x"], r["low"], r["high"]
        out["PopRI"][r["key"]] = (low, high)
        m, s = G.gmm_setpoint(x)
        out["PerRI"][r["key"]] = (m - PER_N_STD * s, m + PER_N_STD * s)
        normal = x[(x >= low) & (x <= high)]
        for meth in ("mle", "trunc", "eb"):
            key = (r["lab"], "M")
            if len(normal) < G.MIN_N[meth] or (meth == "eb" and prior.get(key) is None):
                out[f"Gaussian_{meth}"][r["key"]] = (low, high)          # Pop_RI fallback, as in 04_refs
                continue
            if meth == "mle":
                mu, sig = G.fit_mle(normal)
            elif meth == "trunc":
                mu, sig = G.fit_trunc(normal, low, high)
            else:
                mu, sig = G.eb_posterior(normal, prior[key])
            out[f"Gaussian_{meth}"][r["key"]] = (mu - Z * sig, mu + Z * sig)
    return out


def cohen_intervals(recs):
    """Cohen m4 through the pipeline's own apply_dev_cohen on a synthetic split_df."""
    import pickle
    import cohen
    with open(cohen.DEFAULT_ARTIFACT, "rb") as f:
        artifact = pickle.load(f)
    # _VAL_DIR was the validation/ tree removed on 2026-09-08; the repo root is
    # what makes this path readable now.
    print(f"  Cohen artifact {os.path.relpath(cohen.DEFAULT_ARTIFACT, bootstrap.BASE_DIR)}: {artifact['meta']}")
    rows = []
    for r in recs:
        if r["feature"] == "horizon":
            continue   # Cohen never sees the horizon: the sweep copies the n=10 history-length result
        for t, x in zip(r["t"], r["x"]):
            rows.append((r["key"], r["lab"], float(t), float(x), "baseline"))
        rows.append((r["key"], r["lab"], float(r["t"][-1] + r["horizon"]), r["mid"], "index"))
    split_df = pd.DataFrame(rows, columns=["patient_id", "analyte", "timestamp", "value", "split"])
    split_df["sex"], split_df["age"] = "M", 50.0
    split_df.attrs["time_unit"] = "days"
    orig = cohen.build_pair_table
    cohen.build_pair_table = functools.partial(orig, min_bl=1)     # histories of 2-4 values too
    try:
        ref = cohen.apply_dev_cohen(split_df, artifact, z=Z)
    finally:
        cohen.build_pair_table = orig
    ref = ref[ref.method == "cohen_m4"]
    got = {pid: (lo, hi) for pid, lo, hi in zip(ref.patient_id, ref.ri_low, ref.ri_high)}
    nan = (np.nan, np.nan)
    return {r["key"]: got.get(r["base_key"] if r["feature"] == "horizon" else r["key"], nan) for r in recs}


def summarise(recs, intervals):
    rows = []
    fields = ("key", "lab", "feature", "value", "draw", "mid", "span")
    df = pd.DataFrame([{k: r[k] for k in fields} for r in recs])
    for model, iv in intervals.items():
        lo = df.key.map(lambda k: (iv.get(k) or (np.nan, np.nan))[0]).astype(float)
        hi = df.key.map(lambda k: (iv.get(k) or (np.nan, np.nan))[1]).astype(float)
        d = df.assign(width=hi - lo, centre=(lo + hi) / 2)
        d["ci_norm"] = d.width / d.span * 100
        d["dev"] = (d.centre - d.mid) / d.mid * 100
        for (lab, feat, v), g in d.groupby(["lab", "feature", "value"]):
            w, dv = g.ci_norm.dropna(), g.dev.dropna()
            if not len(w):
                continue
            rows.append(dict(model=model, test_name=lab, feature=feat, value=v, ref_span=g.span.iloc[0],
                             midpoint=g.mid.iloc[0], n_draws=len(w),
                             ci_mean=g.width.mean(), ci_norm_mean=w.mean(),
                             ci_norm_lo=np.percentile(w, 2.5), ci_norm_hi=np.percentile(w, 97.5),
                             mu_mean=g.centre.mean(), mu_pct_dev=dv.mean() if len(dv) else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Sensitivity of every RI method on shared synthetic histories.")
    ap.add_argument("--run_id", default=NORMA_RUN_ID)
    ap.add_argument("--norma_runs", nargs="*", default=None,
                    help="extra NORMA run_ids to sweep alongside --run_id (covariate-ablation "
                         "arms). Default: NORMA_ABLATION_RUN_IDS from lib/datasets.py")
    ap.add_argument("--checkpoint", default=NORMA_CHECKPOINT, choices=["latest", "best"])
    ap.add_argument("--n_draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analytes", nargs="*", help="restrict (smoke test)")
    ap.add_argument("--methods", nargs="*", default=SWEEP_METHODS, choices=SWEEP_METHODS)
    ap.add_argument("--out", default=os.path.join(_VCFG.dev_results_dir("06_calibration"), "sensitivity_methods.csv"))
    args = ap.parse_args()

    import sensitivity_analysis as SA
    features = {f: list(SA.SWEEPS[f]) for f in ("history_length", "horizon", "history_std")}
    labs = sorted(k for k in REF if k not in set(EXCLUDE_LABS) | set(SA.EXCLUDE_LABS))
    if args.analytes:
        labs = [l for l in labs if l in set(args.analytes)]
    recs = make_histories(labs, features, args.n_draws, args.seed)
    print(f"  {len(labs)} analytes x {sum(len(v) for v in features.values())} sweep values x "
          f"{args.n_draws} draws = {len(recs)} histories")

    intervals = {}
    simple = [m for m in args.methods if m.startswith(("PopRI", "PerRI", "Gaussian"))]
    if simple:
        prior = {}
        if "Gaussian_eb" in simple:
            import gaussian as G
            prior = G.load_or_build_prior("dev")["prior"]
            print(f"  EB prior: {len(prior)} (analyte, sex) entries")
        intervals.update({m: v for m, v in simple_intervals(recs, prior).items() if m in simple})
    if "Cohen_m4" in args.methods:
        intervals["Cohen_m4"] = cohen_intervals(recs)
    if "NORMA" in args.methods:
        # baseline keeps the bare "NORMA" label; arms are NORMA_<arm> so the figure code
        # can pick them out without a second registry
        arms = args.norma_runs if args.norma_runs is not None else list(NORMA_ABLATION_RUN_IDS)
        for run in [args.run_id] + [a for a in arms if a != args.run_id]:
            label = "NORMA" if run == args.run_id else f"NORMA_{run}"
            intervals[label] = norma_intervals(recs, run, args.checkpoint)

    df = summarise(recs, intervals)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    base = df[(df.feature == "history_length") & (df.value == SA.BASELINE_N_HIST)]
    print("\n  median CI width (% of Pop_RI width) at n=10, sd=0.1 x width:")
    for m, g in base.groupby("model"):
        print(f"    {m:<15s} {g.ci_norm_mean.median():6.1f}   centre shift {g.mu_pct_dev.median():+.2f}%")
    print(f"Wrote {args.out} ({len(df)} rows)")


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from datasets import dev_results_dir


# ── Sensitivity: one panel per sweep, methods overlaid ──────────────────────
_SENS_PANELS = [  # (feature, xlabel, ycol, log-x)
    ("history_length", "Number of measurements", "ci_norm_mean", True),
    ("horizon", "Prediction horizon (days)", "ci_norm_mean", True),
    ("history_std", "Within-person SD\n(x Pop$_{RI}$ width / 10)", "ci_norm_mean", False)]
# no panel titles: each x label already names its sweep
_SENS_YLABEL = "CI width (% of Pop$_{RI}$ width)"   # same quantity in every panel, labelled once


def _load_sensitivity():
    """Every RI method on shared synthetic histories (sensitivity.py); falls back to
    the NORMA-only sweep from model/sensitivity_analysis.py (flat histories) if that is absent."""
    path = find_in(dev_results_dir("06_calibration"), "sensitivity_methods.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        # Pop_RI is constant across every sweep, so its row is omitted; the dashed
        # line at 100% marks its width in the remaining rows.
        methods = [m for m in _BM_SUPP if m != "PopRI" and m in set(df["model"])]
    else:
        df = load_prediction("sensitivity.csv")
        if df is None:
            return None, []
        df = df[df["model"] == NORMA_MODEL].assign(model="NORMA")
        methods = ["NORMA"]
    df = to_numeric(df[~df["test_name"].isin(EXCLUDE_ANALYTES)].copy())
    return (df, methods) if len(df) else (None, [])


# The three Gaussian variants are steps on one indigo ramp (OKLab separation 12-14, under the
# 15 floor), so linestyle carries the variant and hue carries the family.
_SENS_LINESTYLE = METHOD_LINESTYLE   # lib/models.py, via figlib


def _sens_panel(ax, fsub, ycol, use_log, color, linestyle="-", label=None, lw=1.6):
    """One method's median curve across analytes. The across-analyte IQR is deliberately not
    drawn: with every method in one panel the bands overlap into mush, so the spread is
    reported numerically in the sensitivity_summary table instead. Returns the curve so the
    caller can scale the panel to the data."""
    g = fsub.groupby("value")[ycol].median().sort_index()
    ax.plot(g.index.values, g.values, color=color, lw=lw, ls=linestyle, label=label,
            solid_capstyle="round")
    if use_log:
        ax.set_xscale("log")
    return g


def _sens_ylim(ax, curves):
    """Scale to the plotted curves rather than anchoring at zero.

    These are ratios on a line chart, not bars: the meaningful anchor is the Pop_RI width at
    100%, and forcing a zero floor squeezes a 37-62% band into a quarter of the panel. The
    100% line is drawn only when it falls inside the data range -- the y label already gives
    the unit, so a reference line off in empty space would just cost height. The within-person
    SD sweep does start at zero (a flat history has no spread), and that comes out of the data.
    """
    lo = min(float(c.min()) for c in curves)
    hi = max(float(c.max()) for c in curves)
    pad = 0.08 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    if lo <= 100 <= hi:
        ax.axhline(100, color="gray", ls="--", lw=0.5, zorder=0)   # the Pop_RI width
    ax.set_ylim(lo, hi)


def fig_sensitivity():
    """One row, one panel per sweep, every method's median curve overlaid. Each method sees the
    same synthetic histories, so a flat curve means the method does not use that input: only
    NORMA responds to the horizon, and Cohen's width is one residual SD per analyte, so it is
    flat against all three. Spread across analytes is not drawn -- six bands would be mush --
    and is reported as an IQR in the sensitivity_summary table instead."""
    df, methods = _load_sensitivity()
    if df is None:
        return {}
    fig, axes = plt.subplots(1, len(_SENS_PANELS), figsize=(7.2, 2.3), squeeze=False)
    for c, (feat, xlabel, ycol, use_log) in enumerate(_SENS_PANELS):
        ax = axes[0, c]
        curves = []
        for m in methods:
            fsub = df[(df["model"] == m) & (df["feature"] == feat)]
            if fsub.empty:
                continue
            # the palest ramp step needs a little more weight to read as a dotted line
            curves.append(_sens_panel(ax, fsub, ycol, use_log, _BM_COLORS[m],
                                      linestyle=_SENS_LINESTYLE.get(m, "-"), label=RI_LABELS[m],
                                      lw=1.7 if m == "Gaussian_mle" else 1.4))
        if curves:
            _sens_ylim(ax, curves)
        # no panel title: the x label already names the sweep
        style_axes(ax, xlabel, None)
    # on the left axes rather than fig.supylabel, which sits too far left and centres
    # on the figure (including the legend strip) instead of on the plot area
    axes[0, 0].set_ylabel(_SENS_YLABEL, fontsize=FONT_AXIS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT_LEGEND, ncol=len(labels),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 1 - 0.26 / fig.get_figheight()))
    return {None: fig}


def fig_sensitivity_norma():
    """The same three sweeps for the NORMA covariate-ablation arms only (baseline vs + age at
    draw / care setting / same-draw co-analytes). Each arm is queried on a history of the one
    analyte, so the setting is 'unknown' and the co-analyte panel is empty -- the arms are being
    asked what they do with a lone analyte, not what their extra inputs buy on real panels."""
    df, _ = _load_sensitivity()
    if df is None:
        return {}
    present = set(df["model"])
    arms = [m for m in ALL_ARM_METHODS if m in present]
    if len(arms) < 2:          # nothing to compare until the arms have been swept
        return {}
    # A curve cannot be drawn for an arm with no sweep, so the arms that are
    # absent are named underneath instead of being left unmentioned.
    notes = arm_notes(present)
    fig, axes = plt.subplots(1, len(_SENS_PANELS), figsize=(7.2, 2.3), squeeze=False)
    for c, (feat, xlabel, ycol, use_log) in enumerate(_SENS_PANELS):
        ax = axes[0, c]
        curves = []
        for m in arms:
            fsub = df[(df["model"] == m) & (df["feature"] == feat)]
            if fsub.empty:
                continue
            curves.append(_sens_panel(ax, fsub, ycol, use_log, ALL_ARM_COLORS.get(m, "#999999"),
                                      label=ALL_ARM_LABELS.get(m, m),
                                      lw=1.9 if m.endswith(NORMA_RUN_ID) else 1.3))
        if curves:
            _sens_ylim(ax, curves)
        style_axes(ax, xlabel, None)
    axes[0, 0].set_ylabel(_SENS_YLABEL, fontsize=FONT_AXIS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT_LEGEND, ncol=len(labels),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 1 - 0.26 / fig.get_figheight()))
    if notes:
        by_reason = {}
        for m, why in notes.items():
            by_reason.setdefault(why, []).append(m.replace("NORMA_", ""))
        line = "   ".join(f"{why}: {', '.join(sorted(v))}"
                          for why, v in sorted(by_reason.items()))
        fig.text(0.5, 0.005, line, ha="center", va="bottom",
                 fontsize=FONT_TICK - 0.5, color="#AAAAAA")
    return {None: fig}


# ── NORMA dev-test calibration by queried state (Referee 1.5) ────────────────

FIGURES = [
    FigSpec("06_calibration", "sensitivity", fig_sensitivity, False, (), None),
    FigSpec("06_calibration", "sensitivity_norma", fig_sensitivity_norma, False, (), None),
]


# ══════════════════════════════════════════════════════════════════════════
# Tables — sensitivity: table_* definitions and registry slice.
# ══════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403
from figlib import RI_LABELS, _BM_SUPP, ABLATION_LABELS


def table_sensitivity_summary():
    """Endpoint contrast per method x sweep: interval width at each end of the sweep and the
    paired change, as median [IQR] across analytes.

    A paired per-analyte difference, not a fitted slope: the curves saturate (history length)
    or are flat by construction (every method but NORMA against the horizon), so a linear fit
    would misdescribe them, and the spread across analytes is real heterogeneity rather than
    sampling noise -- hence an IQR and no standard error.
    """
    path = find_in(dev_results_dir("06_calibration"), "sensitivity_methods.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = to_numeric(df[~df["test_name"].isin(EXCLUDE_ANALYTES)].copy())
    # Pop_RI is constant by construction and omitted, as in sensitivity.pdf; the
    # covariate-ablation arms follow the canonical methods when they have been swept
    present = set(df["model"])
    methods = ([m for m in _BM_SUPP if m != "PopRI" and m in present]
               + [m for m in ABLATION_LABELS if m != "NORMA" and m in present])
    if not methods or not len(df):
        return []

    def med_iqr(v, sign=False):
        v = v.dropna()
        if not len(v):
            return "--"
        fmt = "{:+.1f}" if sign else "{:.1f}"
        return (fmt + r" [{:.1f}, {:.1f}]").format(v.median(), v.quantile(0.25), v.quantile(0.75))

    units = {"history_length": "measurements", "horizon": "days", "history_std": r"$\times$ width/10"}
    rows = []
    for feat in SENSITIVITY_FEATURES:
        f = df[df["feature"] == feat]
        if not len(f):
            continue
        lo_v, hi_v = f["value"].min(), f["value"].max()
        # the endpoints ride in the sweep label, so every row shares one set of columns
        sweep = f"{FEATURE_LABELS[feat]} ({lo_v:g} $\\to$ {hi_v:g} {units[feat]})"
        for k, m in enumerate(methods):
            sub = f[f["model"] == m]
            start = sub[sub["value"] == lo_v].set_index("test_name")["ci_norm_mean"]
            end = sub[sub["value"] == hi_v].set_index("test_name")["ci_norm_mean"]
            if not len(start) or not len(end):
                continue
            label = RI_LABELS.get(m) or ABLATION_LABELS[m]
            rows.append({"Sweep": sweep if k == 0 else "", "Method": label,
                         "Start": med_iqr(start), "End": med_iqr(end),
                         "Change": med_iqr((end - start).dropna(), sign=True)})
    if not rows:
        return []
    header = [r"\multicolumn{2}{l}{} & \multicolumn{3}{c}{CI width (\% of Pop$_{RI}$ width), median [IQR]} \\",
              r"\cmidrule(lr){3-5}",
              r"Sweep & Method & At sweep start & At sweep end & Change \\"]
    body = []
    for i, r in enumerate(rows):
        if i and r["Sweep"]:
            body.append(r"\addlinespace")
        body.append(" & ".join(r.values()) + r" \\")
    save_table("06_calibration", "sensitivity_summary", _table("llrrr", header, body), pd.DataFrame(rows))
    return ["sensitivity_summary"]

TABLES = [
    TableSpec("06_calibration",   "sensitivity_summary",     table_sensitivity_summary,     False, (), None),
]


if __name__ == "__main__":
    main()
