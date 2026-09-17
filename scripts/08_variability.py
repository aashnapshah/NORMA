#!/usr/bin/env python
"""Compute intra/inter-individual CV and index of individuality per analyte."""
import bootstrap  # noqa: F401

import argparse
import os

import pandas as pd
import numpy as np

from datasets import already_done, add_dataset_args, get_dataset, save_csv, CODE_TO_PANEL


def bootstrap_variability(means, stds, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CIs for CV_intra, CV_inter, and individuality index."""
    rng = np.random.RandomState(seed)
    n = len(means)
    cv_intra_vals = stds / means

    boot_intra = np.empty(n_bootstrap)
    boot_inter = np.empty(n_bootstrap)
    boot_ii = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_means = means[idx]
        b_stds = stds[idx]
        b_cv_intra = (b_stds / b_means).mean()
        b_cv_inter = b_means.std() / b_means.mean()
        boot_intra[b] = b_cv_intra
        boot_inter[b] = b_cv_inter
        boot_ii[b] = b_cv_intra / b_cv_inter if b_cv_inter > 0 else np.nan

    def ci(arr):
        return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))

    return {
        'cv_intra': cv_intra_vals.mean(),
        'cv_intra_ci_lower': ci(boot_intra)[0],
        'cv_intra_ci_upper': ci(boot_intra)[1],
        'cv_inter': means.std() / means.mean(),
        'cv_inter_ci_lower': ci(boot_inter)[0],
        'cv_inter_ci_upper': ci(boot_inter)[1],
        'ii': cv_intra_vals.mean() / (means.std() / means.mean()),
        'ii_ci_lower': ci(boot_ii)[0],
        'ii_ci_upper': ci(boot_ii)[1],
    }


def compute_variability(ref_df, n_bootstrap=1000, exclude_analytes=None):
    # Filter to per (GMM setpoint) method rows
    ref_df = ref_df[ref_df["method"] == "per"].copy()
    exclude = exclude_analytes or set()

    rows = []
    for analyte in sorted(c for c in ref_df['analyte'].unique() if c not in exclude):
        lab_df = ref_df[ref_df['analyte'] == analyte].copy()
        lab_df = lab_df[(lab_df["ri_mean"] > 0) & (lab_df["ri_std"] >= 0)]
        if len(lab_df) < 5:
            continue

        means = lab_df["ri_mean"].values
        stds = lab_df["ri_std"].values
        stats = bootstrap_variability(means, stds, n_bootstrap)

        rows.append({
            "Panel": CODE_TO_PANEL.get(analyte, ""),
            "analyte": analyte,
            "n_patients": len(lab_df),
            "cv_intra": stats['cv_intra'],
            "cv_intra_ci_lower": stats['cv_intra_ci_lower'],
            "cv_intra_ci_upper": stats['cv_intra_ci_upper'],
            "cv_inter": stats['cv_inter'],
            "cv_inter_ci_lower": stats['cv_inter_ci_lower'],
            "cv_inter_ci_upper": stats['cv_inter_ci_upper'],
            "individuality_index": stats['ii'],
            "ii_ci_lower": stats['ii_ci_lower'],
            "ii_ci_upper": stats['ii_ci_upper'],
        })

    result = pd.DataFrame(rows)
    panel_order = {"CBC": 0, "BMP": 1, "HFP": 2, "Lipid": 3, "": 4}
    result["_sort"] = result["Panel"].map(panel_order).fillna(4)
    return result.sort_values(["_sort", "analyte"]).drop(columns="_sort").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap replicates")
    args = parser.parse_args()

    ds = get_dataset(args)
    exclude = set(ds.exclude_labs)
    results_dir = ds.setup_output()
    if already_done(args, results_dir, "variability.csv", label="variability"):
        return

    ref_df = ds.load_ref_intervals()
    print(f"  {len(ref_df)} rows in ref_intervals")
    print(f"  Bootstrap replicates: {args.bootstrap}")

    var_df = compute_variability(ref_df, n_bootstrap=args.bootstrap,
                                  exclude_analytes=exclude)

    # Convert CVs and their CIs to percentages
    for col in ['cv_intra', 'cv_intra_ci_lower', 'cv_intra_ci_upper',
                'cv_inter', 'cv_inter_ci_lower', 'cv_inter_ci_upper']:
        var_df[col] = (var_df[col] * 100).round(1)
    var_df = var_df.round({'individuality_index': 2, 'ii_ci_lower': 2, 'ii_ci_upper': 2})
    out_path = os.path.join(results_dir, "variability.csv")
    save_csv(var_df, out_path, analytes=ds._analytes)
    print(f"  Saved {len(var_df)} analytes to {out_path}")


# Figures and tables

from figlib import *  # noqa: F401,F403
from models import lighten

COHORTS = VAL_COHORTS          # eicu, inspire, chs: one row each
INTRA_TINT = 0.55              # intra-individual bar = the cohort colour blended towards white
II_THRESHOLD = 0.6             # below this the population interval is a poor reference


def _cohort_data():
    data = {ds: _load_variability(ds) for ds in COHORTS}
    return {ds: (df.set_index("analyte") if df is not None and not df.empty else None) for ds, df in data.items()}


def _order_by_ii(data):
    """Analytes ordered by individuality index averaged over the cohorts that have them."""
    present = [d for d in data.values() if d is not None]
    all_a = [a for a in all_analytes() if a != "DBIL" and any(a in d.index for d in present)]

    def mean_ii(a):
        vals = [d.loc[a, "individuality_index"] for d in present if a in d.index]
        return np.nanmean(vals) if vals and not np.all(np.isnan(vals)) else -np.inf
    return sorted(all_a, key=mean_ii, reverse=True)


def _cohort_label(ax, ds):
    """Cohort name on the right of the row."""
    ax.text(1.01, 0.5, DATASET_DISPLAY.get(ds, ds), transform=ax.transAxes,
            rotation=270, ha="left", va="center", fontsize=FONT_AXIS, color=DARK)


def _errors(d, idx, kind):
    vals = d.loc[idx, kind].to_numpy(float)
    if f"{kind}_ci_lower" not in d.columns:
        return vals, np.zeros((2, len(idx)))
    lo = np.clip(vals - d.loc[idx, f"{kind}_ci_lower"].to_numpy(float), 0, None)
    hi = np.clip(d.loc[idx, f"{kind}_ci_upper"].to_numpy(float) - vals, 0, None)
    return vals, np.nan_to_num(np.vstack([lo, hi]))


def _cv_row(ax, ds, d, all_a):
    """Inter- (cohort colour) and intra-individual (tint) CV per analyte for one cohort."""
    idx = [a for a in all_a if a in d.index]
    xs = np.array([all_a.index(a) for a in idx], float)
    colors = {"cv_inter": DATASET_COLORS[ds], "cv_intra": lighten(DATASET_COLORS[ds], INTRA_TINT)}
    for k, kind in enumerate(("cv_inter", "cv_intra")):
        vals, err = _errors(d, idx, kind)
        x = xs + (k - 0.5) * 0.4
        ax.bar(x, vals, 0.4, color=colors[kind], edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=DARK, elinewidth=0.4, capsize=1, alpha=0.5)
    ax.set_ylabel("CV (%)", fontsize=FONT_AXIS)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    hide_spines(ax)


def _ii_row(ax, data, all_a):
    """Individuality index per analyte, one dot per cohort."""
    for ds, d in data.items():
        if d is None:
            continue
        idx = [a for a in all_a if a in d.index and pd.notna(d.loc[a, "individuality_index"])]
        xs = [all_a.index(a) for a in idx]
        vals, err = _errors(d, idx, "individuality_index") if idx else (np.array([]), np.zeros((2, 0)))
        if "ii_ci_lower" in d.columns and idx:
            lo = np.clip(vals - d.loc[idx, "ii_ci_lower"].to_numpy(float), 0, None)
            hi = np.clip(d.loc[idx, "ii_ci_upper"].to_numpy(float) - vals, 0, None)
            err = np.nan_to_num(np.vstack([lo, hi]))
        ax.errorbar(xs, vals, yerr=err, fmt=DATASET_MARKERS.get(ds, "o"), color=DATASET_COLORS[ds],
                    markersize=3.5, elinewidth=0.5, capsize=1.5, alpha=0.9, label=DATASET_DISPLAY.get(ds, ds))
    ax.axhline(II_THRESHOLD, color="gray", ls="--", lw=0.5, alpha=0.6)
    ax.set_ylabel("Individuality index", fontsize=FONT_AXIS)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    hide_spines(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles += pending_handles({ds for ds, d in data.items() if d is not None})
    ax.legend(handles=handles, frameon=False, fontsize=FONT_LEGEND, ncol=len(handles), loc="upper right",
              handletextpad=0.4, columnspacing=1.2)


def fig_variability():
    data = _cohort_data()
    if all(d is None for d in data.values()):
        return {}
    all_a = _order_by_ii(data)
    x = np.arange(len(all_a))
    n_rows = len(COHORTS) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 1.55 * n_rows + 0.5), sharex=True)
    for ax, ds in zip(axes, COHORTS):
        d = data[ds]
        if d is None:
            pending_axis(ax, ds)
        else:
            _cv_row(ax, ds, d, all_a)
        _cohort_label(ax, ds)
    _ii_row(axes[-1], data, all_a)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(all_a, rotation=90, fontsize=FONT_TICK)
    axes[-1].set_xlim(-0.6, len(all_a) - 0.4)
    handles = [Patch(facecolor=DARK, label="Inter-individual"),
               Patch(facecolor=lighten(DARK, INTRA_TINT), label="Intra-individual")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=FONT_LEGEND, handlelength=1.2, handleheight=0.9, columnspacing=1.2)
    fig.tight_layout(h_pad=0.8, rect=(0, 0, 0.985, 0.97))
    return {None: fig}


FIGURES = [
    FigSpec("08_variability", "variability", fig_variability, False, (), None),
]


# Tables — 08_variability: table_* definitions and registry slice.

from figlib import *  # noqa: F401,F403

# save_table()'s first argument is the folder the table is written into, so it must match this
# directory name.

def table_variability():
    data = {ds: _load_variability(ds) for ds in DATASETS}
    data = {ds: (df.set_index("analyte") if df is not None else None) for ds, df in data.items()}
    if all(v is None for v in data.values()):
        return []

    def ci(r, v, lo, hi):
        return f"{r[v]:.1f} [{r[lo]:.1f}, {r[hi]:.1f}]" if pd.notna(r[v]) else "---"
    csv_rows = []
    for analyte in all_analytes():
        row = {"Analyte": analyte}
        for ds in DATASETS:
            label = DATASET_DISPLAY[ds]; present = data[ds]
            if present is not None and analyte in present.index:
                r = present.loc[analyte]
                row[f"{label} CV_intra"] = ci(r, "cv_intra", "cv_intra_ci_lower", "cv_intra_ci_upper")
                row[f"{label} CV_inter"] = ci(r, "cv_inter", "cv_inter_ci_lower", "cv_inter_ci_upper")
                row[f"{label} II"] = ci(r, "individuality_index", "ii_ci_lower", "ii_ci_upper")
            else:
                row[f"{label} CV_intra"] = row[f"{label} CV_inter"] = row[f"{label} II"] = "---"
        csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)

    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule"]
    h1, cmid = " ", []
    for i, ds in enumerate(DATASETS):
        h1 += r" & \multicolumn{3}{c}{" + DATASET_DISPLAY[ds] + "}"; cmid.append(f"\\cmidrule(lr){{{2 + i * 3}-{4 + i * 3}}}")
    lines += [h1 + r" \\", " ".join(cmid), "Analyte" + r" & CV$_{\text{intra}}$ & CV$_{\text{inter}}$ & II" * len(DATASETS) + r" \\", r"\midrule"]
    for row in csv_rows:
        cells = [row["Analyte"]]
        for ds in DATASETS:
            label = DATASET_DISPLAY[ds]
            cells += [row[f"{label} CV_intra"], row[f"{label} CV_inter"], row[f"{label} II"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("08_variability", "variability", lines, csv_df, landscape=True, font_size="tiny")
    return ["variability"]

TABLES = [
    TableSpec("08_variability",   "variability",             table_variability,             False, (), None),
]


if __name__ == "__main__":
    main()
