#!/usr/bin/env python
"""Compute mean RI bounds by age stratum for each analyte and method."""
import bootstrap  # noqa: F401

import argparse
import os
import sys

import pandas as pd
import numpy as np

from datasets import already_done, NORMA_RUN_ID, add_dataset_args, get_dataset, save_csv


NORMA_METHOD = f'norma_{NORMA_RUN_ID}'


def compute_age_ri(ref_df, n_sample=2000, seed=42, exclude_analytes=None):
    """Subsample per analyte and compute mean RI bounds by integer age."""
    exclude = exclude_analytes or set()
    ref_df['analyte'] = ref_df['analyte'].replace('', 'NA').fillna('NA')
    ref_df['age'] = pd.to_numeric(ref_df['age'], errors='coerce')
    ref_df = ref_df.dropna(subset=['age'])

    # Only use methods that have ri_low/ri_high (not base)
    methods = [m for m in ref_df['method'].unique() if m != 'base']

    analytes = sorted(a for a in ref_df['analyte'].unique() if a not in exclude)
    rng = np.random.RandomState(seed)

    all_rows = []
    for analyte in analytes:
        lab = ref_df[ref_df['analyte'] == analyte].copy()

        # Get unique patient-analyte pairs for sampling
        pairs = lab[['patient_id', 'analyte']].drop_duplicates()
        if len(pairs) < 50:
            continue
        if len(pairs) > n_sample:
            pairs = pairs.sample(n=n_sample, random_state=rng)
            lab = lab.merge(pairs, on=['patient_id', 'analyte'], how='inner')

        for method in methods:
            method_df = lab[lab['method'] == method].copy()
            if len(method_df) == 0:
                continue

            method_df['age_bin'] = method_df['age'].astype(int)

            for age_bin, grp in method_df.groupby('age_bin'):
                if len(grp) < 5:
                    continue
                lo = grp['ri_low'].dropna().mean()
                hi = grp['ri_high'].dropna().mean()
                if np.isnan(lo) or np.isnan(hi):
                    continue

                # Map method names for output
                # 'norma_<primary>' is the canonical NORMA; the covariate-ablation arms
                # keep their own names so they stay separable downstream (collapsing
                # every norma_* to "NORMA" silently stacked all arms into one series).
                if method.startswith('norma_'):
                    rid = method[len('norma_'):]
                    method_label = 'NORMA' if rid == NORMA_RUN_ID else f'NORMA_{rid}'
                else:
                    method_label = {'pop': 'PopRI', 'per': 'PerRI'}.get(method, method)
                all_rows.append({
                    'analyte': analyte, 'age': age_bin, 'method': method_label,
                    'ri_low': lo, 'ri_high': hi,
                    'ri_mid': (lo + hi) / 2, 'n': len(grp),
                })

        print(f"    {analyte}: {len(pairs)} samples")

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument('--n_sample', type=int, default=2000)
    args = parser.parse_args()

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    if already_done(args, results_dir, "age_ri.csv", label="age-stratified reference intervals"):
        return

    ref_df = ds.load_ref_intervals()
    print(f"  {len(ref_df)} rows in ref_intervals")

    age_df = compute_age_ri(ref_df, n_sample=args.n_sample,
                            exclude_analytes=set(ds.exclude_labs))

    age_df = age_df.round({'ri_low': 2, 'ri_high': 2, 'ri_mid': 2})
    out_path = os.path.join(results_dir, 'age_ri.csv')
    save_csv(age_df, out_path, analytes=ds._analytes)
    print(f"\n  Saved {len(age_df)} rows to {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403


def fig_age_ri():
    data = {}
    for ds in available_datasets("age_ri.csv"):
        df = load_result(ds, "age_ri.csv")
        if df is not None and len(df):
            data[ds] = df[df["method"] == "NORMA"]
            age = pd.to_numeric(data[ds]["age"], errors="coerce")
    if not data:
        return {}
    all_a = all_analytes(); ncols = 6; nrows = int(np.ceil(len(all_a) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 1.6), sharex=True)
    axes = axes.flatten()
    for i, analyte in enumerate(all_a):
        ax = axes[i]
        for ds, df in data.items():
            if analyte not in df["analyte"].unique():
                continue
            sub = df[df["analyte"] == analyte].sort_values("age")
            if len(sub) < 3:
                continue
            ages = sub["age"].to_numpy(float); mids = pd.to_numeric(sub["ri_mid"], errors="coerce").to_numpy(float)
            valid = ~np.isnan(mids)
            ax.scatter(ages[valid], mids[valid], s=6, color=DATASET_COLORS[ds], alpha=0.15, edgecolors="none", zorder=2)
            if valid.sum() >= 6:
                coeffs = np.polyfit(ages[valid], mids[valid], 3)
                xs = np.linspace(ages[valid].min(), ages[valid].max(), 100)
                ax.plot(xs, np.polyval(coeffs, xs), color=DATASET_COLORS[ds], lw=1.5, alpha=0.9, zorder=3)
        ax.set_title(analyte, fontsize=FONT_TITLE, pad=3)
        ax.tick_params(axis="both", labelsize=FONT_TICK); hide_spines(ax)
    for j in range(len(all_a), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([0], [0], color=DATASET_COLORS[ds], lw=2, label=DATASET_DISPLAY[ds]) for ds in data]
    handles += pending_handles(data)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(handles),
               frameon=False, fontsize=FONT_LEGEND)
    fig.text(0.5, -0.01, "Age (years)", ha="center", fontsize=FONT_AXIS)
    fig.text(-0.02, 0.5, "NORMA Midpoint Prediction", va="center", rotation="vertical", fontsize=FONT_AXIS)
    fig.tight_layout()
    return {"": fig}

def fig_age_ri_norma(ds):
    """Same age-vs-midpoint grid as fig_age_ri, one line per NORMA covariate arm.

    Per cohort rather than per dataset-overlay: colour is spent on the ablation
    arms here, so each cohort needs its own file."""
    df = load_result(ds, "age_ri.csv")
    if df is None or len(df) == 0 or "method" not in df.columns:
        return {}
    present = set(df["method"].astype(str).unique())
    arms = [m for m in ABLATION_METHODS if m in present]
    # Only worth drawing when at least one arm sits alongside the baseline.
    if not [m for m in arms if m != "NORMA"]:
        return {}
    data = {m: df[df["method"] == m] for m in arms}
    all_a = all_analytes(); ncols = 6; nrows = int(np.ceil(len(all_a) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 1.6), sharex=True)
    axes = axes.flatten()
    for i, analyte in enumerate(all_a):
        ax = axes[i]
        for arm, arm_df in data.items():
            sub = arm_df[arm_df["analyte"] == analyte].sort_values("age")
            if len(sub) < 3:
                continue
            ages = pd.to_numeric(sub["age"], errors="coerce").to_numpy(float)
            mids = pd.to_numeric(sub["ri_mid"], errors="coerce").to_numpy(float)
            valid = ~(np.isnan(ages) | np.isnan(mids))
            color = _BM_COLORS.get(arm, "#999")
            ax.scatter(ages[valid], mids[valid], s=6, color=color, alpha=0.15, edgecolors="none", zorder=2)
            if valid.sum() >= 6 and len(np.unique(ages[valid])) >= 4:
                coeffs = np.polyfit(ages[valid], mids[valid], 3)
                xs = np.linspace(ages[valid].min(), ages[valid].max(), 100)
                ax.plot(xs, np.polyval(coeffs, xs), color=color, lw=1.5, alpha=0.9, zorder=3)
        ax.set_title(analyte, fontsize=FONT_TITLE, pad=3)
        ax.tick_params(axis="both", labelsize=FONT_TICK); hide_spines(ax)
    for j in range(len(all_a), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([0], [0], color=_BM_COLORS.get(m, "#999"), lw=2, label=RI_LABELS.get(m, m)) for m in arms]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(arms),
               frameon=False, fontsize=FONT_LEGEND)
    fig.text(0.0, 1.03, DATASET_DISPLAY.get(ds, ds), ha="left", va="bottom", fontsize=FONT_TITLE)
    fig.text(0.5, -0.01, "Age (years)", ha="center", fontsize=FONT_AXIS)
    fig.text(-0.02, 0.5, "NORMA Midpoint Prediction", va="center", rotation="vertical", fontsize=FONT_AXIS)
    fig.tight_layout()
    return {"": fig}

FIGURES = [
    FigSpec("09_age_ri",        "age_ri",              fig_age_ri,                 False, (),                                            None),
    FigSpec("09_age_ri",        "age_ri_norma",     fig_age_ri_norma,        True,  ("age_ri.csv",),                               _one("age_ri_norma")),
]


if __name__ == "__main__":
    main()
