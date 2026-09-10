#!/usr/bin/env python
"""Compute mortality rate by value quintile and by deviation from baseline."""
import bootstrap  # noqa: F401

import argparse
import os

import pandas as pd
import numpy as np

from constants import BASELINE_Z
import datasets
from datasets import add_dataset_args, get_dataset, save_csv



def compute_quintile_mortality(df, analyte_col, value_col, event_col, exclude_analytes=None):
    """Mortality rate with Wilson 95% CI by value quintile, per analyte."""
    exclude = exclude_analytes or set()
    df[analyte_col] = df[analyte_col].replace('', 'NA').fillna('NA')
    analytes = sorted(a for a in df[analyte_col].unique() if a not in exclude)

    all_rows = []
    for analyte in analytes:
        lab_df = df[df[analyte_col] == analyte].dropna(subset=[value_col, event_col]).copy()
        lab_df[event_col] = pd.to_numeric(lab_df[event_col], errors='coerce')
        lab_df = lab_df.dropna(subset=[event_col])
        if len(lab_df) < 20:
            continue

        lab_df['quintile'] = pd.qcut(lab_df[value_col], 5, labels=False, duplicates='drop')
        mort = lab_df.groupby('quintile').agg(
            mortality_rate=(event_col, 'mean'),
            n=(event_col, 'count'),
        ).reset_index()
        mort['q_label'] = mort['quintile'] + 1

        # Wilson score 95% CI
        z = 1.96
        p = mort['mortality_rate']
        n_q = mort['n']
        denom = 1 + z**2 / n_q
        center = (p + z**2 / (2 * n_q)) / denom
        halfwidth = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_q)) / n_q) / denom
        mort['ci_lo'] = (center - halfwidth) * 100
        mort['ci_hi'] = (center + halfwidth) * 100
        mort['mortality_pct'] = mort['mortality_rate'] * 100
        mort['analyte'] = analyte

        all_rows.append(mort[['analyte', 'q_label', 'mortality_pct', 'ci_lo', 'ci_hi', 'n']])

    if not all_rows:
        return pd.DataFrame(columns=['analyte', 'q_label', 'mortality_pct', 'ci_lo', 'ci_hi', 'n'])
    return pd.concat(all_rows, ignore_index=True)


def compute_deviation_mortality(df, event_col, n_bins=10, exclude_analytes=None):
    """Mortality rate by decile of |z-score| from personal baseline."""
    exclude = exclude_analytes or set()
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    analytes = sorted(a for a in df['analyte'].unique() if a not in exclude)

    all_rows = []
    for analyte in analytes:
        lab_df = df[df['analyte'] == analyte].dropna(
            subset=['value', event_col, 'baseline_mean', 'baseline_std']
        ).copy()
        lab_df[event_col] = pd.to_numeric(lab_df[event_col], errors='coerce')
        lab_df = lab_df.dropna(subset=[event_col])
        lab_df = lab_df[lab_df['baseline_std'] > 0]
        if len(lab_df) < 20:
            continue

        lab_df['z_score'] = (lab_df['value'] - lab_df['baseline_mean']).abs() / lab_df['baseline_std']

        lab_df['decile'] = pd.qcut(lab_df['z_score'], n_bins, labels=False, duplicates='drop')
        mort = lab_df.groupby('decile').agg(
            mortality_rate=(event_col, 'mean'),
            n=(event_col, 'count'),
            z_median=('z_score', 'median'),
        ).reset_index()
        mort['decile'] = mort['decile'] + 1

        # Wilson score 95% CI
        z = 1.96
        p = mort['mortality_rate']
        n_q = mort['n']
        denom = 1 + z**2 / n_q
        center = (p + z**2 / (2 * n_q)) / denom
        halfwidth = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_q)) / n_q) / denom
        mort['ci_lo'] = (center - halfwidth) * 100
        mort['ci_hi'] = (center + halfwidth) * 100
        mort['mortality_pct'] = mort['mortality_rate'] * 100
        mort['analyte'] = analyte

        all_rows.append(mort[['analyte', 'decile', 'z_median', 'mortality_pct', 'ci_lo', 'ci_hi', 'n']])

    if not all_rows:
        return pd.DataFrame(columns=['analyte', 'decile', 'z_median', 'mortality_pct', 'ci_lo', 'ci_hi', 'n'])
    return pd.concat(all_rows, ignore_index=True)


def _wilson(mort, event_col):
    """Wilson score 95% CI on a per-bin mortality rate, in percent."""
    z = 1.96
    p, n_q = mort['mortality_rate'], mort['n']
    denom = 1 + z**2 / n_q
    center = (p + z**2 / (2 * n_q)) / denom
    halfwidth = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_q)) / n_q) / denom
    mort['ci_lo'] = (center - halfwidth) * 100
    mort['ci_hi'] = (center + halfwidth) * 100
    mort['mortality_pct'] = mort['mortality_rate'] * 100
    return mort


def compute_deviation_mortality_by_method(cls, event_col, methods, n_bins=10, exclude_analytes=None):
    """Deviation-mortality curve binned on EACH method's own deviation score.

    compute_deviation_mortality() bins on |value - baseline_mean| / baseline_std,
    which comes from the `base` rows and is therefore identical for every method —
    so it cannot compare methods at all.  07_classify.py already stores `<method>_z`,
    the distance from that method's own interval centre in units of its own
    half-width (z = 1 is exactly that method's flag boundary), which is the
    per-method analogue.  Binning on it lets the reference-interval methods and the
    NORMA covariate-ablation arms be compared on the same curve.

    Returns the same columns as compute_deviation_mortality() plus `method`.
    """
    exclude = exclude_analytes or set()
    cls = cls.copy()
    cls['analyte'] = cls['analyte'].replace('', 'NA').fillna('NA')
    cls[event_col] = pd.to_numeric(cls[event_col], errors='coerce')
    analytes = sorted(a for a in cls['analyte'].unique() if a not in exclude)

    all_rows = []
    for method in methods:
        zcol = f'{method}_z'
        if zcol not in cls.columns:
            continue
        for analyte in analytes:
            lab_df = cls[cls['analyte'] == analyte].dropna(subset=[zcol, event_col])
            lab_df = lab_df[np.isfinite(lab_df[zcol].to_numpy(float))]
            if len(lab_df) < 20:
                continue
            decile = pd.qcut(lab_df[zcol], n_bins, labels=False, duplicates='drop')
            mort = lab_df.assign(decile=decile).groupby('decile').agg(
                mortality_rate=(event_col, 'mean'),
                n=(event_col, 'count'),
                z_median=(zcol, 'median'),
            ).reset_index()
            mort['decile'] = mort['decile'] + 1
            mort = _wilson(mort, event_col)
            mort['analyte'] = analyte
            mort['method'] = method
            all_rows.append(mort[['method', 'analyte', 'decile', 'z_median',
                                  'mortality_pct', 'ci_lo', 'ci_hi', 'n']])
    if not all_rows:
        return pd.DataFrame(columns=['method', 'analyte', 'decile', 'z_median',
                                     'mortality_pct', 'ci_lo', 'ci_hi', 'n'])
    return pd.concat(all_rows, ignore_index=True)


def load_chs_patient_level(ds):
    """Extract first index measurement per patient-analyte with baseline stats (cached)."""
    target_analytes = ds._analytes

    chunk_dirs = ds._chunk_dirs()
    print(f"  Using {len(chunk_dirs)} chunks")

    parts = []
    for chunk_dir in chunk_dirs:
        chunk_name = os.path.basename(chunk_dir)
        chunk_cache = os.path.join(chunk_dir, 'mortality_extract.parquet')
        cached_df = pd.read_parquet(chunk_cache) if os.path.exists(chunk_cache) else None

        # Full cache hit
        if cached_df is not None and target_analytes is None:
            parts.append(cached_df)
            continue

        # Recompute target analytes from source
        sp_path = os.path.join(chunk_dir, 'index_labs.parquet')
        ref_df = datasets.read_ref_intervals(chunk_dir)
        if not os.path.exists(sp_path) or ref_df is None:
            if cached_df is not None:
                parts.append(cached_df)
            continue

        index_labs = ds._standardize(pd.read_parquet(sp_path))

        if target_analytes is not None:
            index_labs = index_labs[index_labs['analyte'].isin(target_analytes)]
            ref_df = ref_df[ref_df['analyte'].isin(target_analytes)]

        idx = index_labs[index_labs['split'] == 'index'].copy()
        del index_labs
        idx = idx.sort_values('timestamp').groupby(['patient_id', 'analyte']).first().reset_index()

        base = ref_df[ref_df['method'] == 'base'][['patient_id', 'analyte', 'ri_mean', 'ri_std']]
        del ref_df
        base = base.rename(columns={"ri_mean": "baseline_mean", "ri_std": "baseline_std"})

        chunk_extract = idx.merge(base, on=['patient_id', 'analyte'], how='inner')
        del idx, base

        # Patch into cache
        if cached_df is not None and target_analytes is not None:
            kept = cached_df[~cached_df['analyte'].isin(target_analytes)]
            chunk_extract = pd.concat([kept, chunk_extract], ignore_index=True)

        chunk_extract.to_parquet(chunk_cache, index=False)
        print(f"    {chunk_name}: {'updated' if target_analytes else 'computed'}")
        parts.append(chunk_extract)

    df = pd.concat(parts, ignore_index=True)
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    print(f"  {len(df):,} patient-analyte rows collected")
    return df


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    args = parser.parse_args()

    ds = get_dataset(args)
    results_dir = ds.setup_output()
    exclude = set(ds.exclude_labs)

    if args.dataset == 'chs':
        merged = load_chs_patient_level(ds)
        mort_key = ds.mortality_outcome
        event_col = ds.outcomes[mort_key]['event_col']
        if event_col not in merged.columns:
            print(f"  Attaching outcomes...")
            merged = ds.attach_outcomes(merged)
        if event_col not in merged.columns:
            print(f"  Warning: {event_col} not in data, skipping mortality analysis")
            return
    else:
        index_labs = ds.load_index_labs()
        index_labs = index_labs[index_labs['split'] == 'index'].copy()

        mort_key = ds.mortality_outcome
        event_col = ds.outcomes[mort_key]['event_col']

        if event_col not in index_labs.columns:
            print(f"  Attaching mortality...")
            index_labs = ds.attach_outcomes(index_labs)

        ref_df = ds.load_ref_intervals()
        base_df = ref_df[ref_df["method"] == "base"].copy()
        base_ref = base_df.rename(columns={"ri_mean": "baseline_mean", "ri_std": "baseline_std"})
        base_ref = base_ref[['patient_id', 'analyte', 'baseline_mean', 'baseline_std']]
        merged = index_labs.merge(base_ref, on=['patient_id', 'analyte'], how='inner')

    print(f"  {len(merged):,} patient-analyte rows, event_col={event_col}")
    print(f"  Event rate: {merged[event_col].mean():.3f}")

    # 1. Decile mortality (raw values)
    print("\n  Computing decile mortality...")
    mort_df = compute_quintile_mortality(merged, 'analyte', 'value', event_col,
                                          exclude_analytes=exclude)
    mort_df = mort_df.round({'mortality_pct': 1, 'ci_lo': 1, 'ci_hi': 1})
    out_path = os.path.join(results_dir, 'mortality_quintile.csv')
    save_csv(mort_df, out_path, analytes=ds._analytes)
    print(f"  Saved {len(mort_df)} rows to {out_path}")

    # 2. Deviation mortality (z-score from baseline)
    print("\n  Computing deviation mortality...")
    dev_df = compute_deviation_mortality(merged, event_col)
    dev_df = dev_df.round({'z_median': 2, 'mortality_pct': 1, 'ci_lo': 1, 'ci_hi': 1})
    # binned on the deviation from the patient's own baseline period, not on any
    # RI method's z -- that is a method level of the same table (BASELINE_Z), so the
    # per-method rows below go into the same file.
    dev_df.insert(0, 'method', BASELINE_Z)
    dev_path = os.path.join(results_dir, 'mortality_deviation.csv')
    save_csv(dev_df, dev_path, analytes=ds._analytes, keys=('method',))
    print(f"  Saved {len(dev_df)} rows to {dev_path}")

    # 3. Deviation mortality per METHOD (each method's own z), so the RI methods and
    #    the NORMA ablation arms can be compared on the same curve.
    print("\n  Computing per-method deviation mortality...")
    try:
        cls = ds.load_classification()
    except FileNotFoundError:
        cls = None
    if cls is None:
        print("  No classification.parquet; skipping per-method deviation mortality")
    else:
        if event_col not in cls.columns:
            cls = ds.attach_outcomes(cls)
        if event_col not in cls.columns:
            print(f"  {event_col} unavailable after attach_outcomes; skipping")
        else:
            methods = [m for m in ds.methods if f'{m}_z' in cls.columns]
            print(f"  methods with a deviation score: {methods}")
            bym = compute_deviation_mortality_by_method(cls, event_col, methods,
                                                        exclude_analytes=exclude)
            bym = bym.round({'z_median': 2, 'mortality_pct': 1, 'ci_lo': 1, 'ci_hi': 1})
            save_csv(bym, dev_path, analytes=ds._analytes, keys=('method',))
            print(f"  Saved {len(bym)} rows to {dev_path}")


# ═════════════════════════════════════════════════════════════════════════
# Figures and tables
# ═════════════════════════════════════════════════════════════════════════

from figlib import *  # noqa: F401,F403


def fig_mortality_quintile():
    datasets = available_datasets("mortality_quintile.csv")
    data = {ds: load_result(ds, "mortality_quintile.csv") for ds in datasets}
    data = {ds: df for ds, df in data.items() if df is not None}
    if not data:
        return {}
    all_a = all_analytes(); ncols = 4; nrows = int(np.ceil(len(all_a) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.4), sharex=True)
    axes = axes.flatten()
    for i, analyte in enumerate(all_a):
        ax = axes[i]; has_any = False
        for ds, df in data.items():
            sub = df[df["analyte"] == analyte].sort_values("q_label")
            if len(sub) == 0 or sub["n"].min() < 100:
                continue
            has_any = True
            ax.plot(sub["q_label"].values, sub["mortality_pct"].values, "-o", color=DATASET_COLORS[ds],
                    markersize=3, linewidth=1)
            if "ci_lo" in sub.columns:
                ax.fill_between(sub["q_label"].values, sub["ci_lo"].values, sub["ci_hi"].values,
                                color=DATASET_COLORS[ds], alpha=0.15)
        ax.set_title(analyte, fontsize=FONT_TITLE, pad=4, color="black" if has_any else "lightgray")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"] if i >= (nrows - 1) * ncols else [], fontsize=FONT_TICK)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); hide_spines(ax)
    for j in range(len(all_a), len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, -0.01, "Quintile", ha="center", fontsize=FONT_AXIS)
    fig.text(-0.02, 0.5, "Mortality Rate (%)", va="center", rotation="vertical", fontsize=FONT_AXIS)
    handles = [Line2D([0], [0], color=DATASET_COLORS[ds], lw=2, label=DATASET_DISPLAY[ds]) for ds in data]
    handles += pending_handles(data)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(handles),
               frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    return {"": fig}

def fig_mortality_deviation():
    data = {ds: _load_mortality_deviation(ds) for ds in available_datasets("mortality_deviation.csv")}
    data = {ds: df for ds, df in data.items() if df is not None and len(df)}
    if not data:
        return {}
    all_a = all_analytes(); ncols = 6; nrows = int(np.ceil(len(all_a) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6), sharex=True)
    axes = axes.flatten()
    for i, analyte in enumerate(all_a):
        ax = axes[i]
        for ds, df in data.items():
            sub = df[df["analyte"] == analyte].sort_values("decile")
            if len(sub) == 0:
                continue
            ax.plot(sub["z_median"].values, sub["mortality_pct"].values, "-o", color=DATASET_COLORS[ds],
                    markersize=3, linewidth=1)
            if "ci_lo" in sub.columns:
                ax.fill_between(sub["z_median"].values, sub["ci_lo"].values, sub["ci_hi"].values,
                                color=DATASET_COLORS[ds], alpha=0.15)
        ax.set_xlim(0, 5); ax.set_title(analyte, fontsize=FONT_TITLE, pad=4)
        ax.tick_params(axis="both", labelsize=FONT_TICK); hide_spines(ax)
    for j in range(len(all_a), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([0], [0], color=DATASET_COLORS[ds], lw=2, label=DATASET_DISPLAY[ds]) for ds in data]
    handles += pending_handles(data)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(handles),
               frameon=False, fontsize=FONT_LEGEND)
    fig.text(0.5, -0.01, "Standardized Distance from Baseline (|z|)", ha="center", fontsize=FONT_AXIS)
    fig.text(-0.02, 0.5, "Mortality Rate (%)", va="center", rotation="vertical", fontsize=FONT_AXIS)
    fig.tight_layout()
    return {"": fig}

def fig_mortality_deviation_norma(ds):
    """fig_mortality_deviation for one cohort, one line per NORMA ablation arm.

    The main figure's z is |value - baseline_mean| / baseline_std, which comes from
    the `base` rows and is identical for every method, so it cannot separate them.
    This one reads the per-method rows of mortality_deviation.csv, each binned on
    that method's OWN deviation score (10_mortality.py:compute_deviation_mortality_by_method).
    """
    df = load_result(ds, "mortality_deviation.csv")
    if df is not None and "method" in df.columns:
        df = df[df["method"].astype(str) != BASELINE_Z]      # each method's own z
    if df is None or not len(df) or "method" not in df.columns:
        return {}
    df = to_numeric(df)
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    if "n" in df.columns:
        df = df[df["n"] >= 100]
    methods = [m for m in bm_methods() if m in set(df["method"])]
    if not methods:
        return {}
    all_a = all_analytes(); ncols = 6; nrows = int(np.ceil(len(all_a) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6), sharex=True)
    axes = axes.flatten()
    for i, analyte in enumerate(all_a):
        ax = axes[i]
        for m in methods:
            sub = df[(df["analyte"] == analyte) & (df["method"] == m)].sort_values("decile")
            if len(sub) == 0:
                continue
            ax.plot(sub["z_median"].values, sub["mortality_pct"].values, "-o",
                    color=_BM_COLORS[m], markersize=3, linewidth=1)
        ax.set_xlim(0, 5); ax.set_title(analyte, fontsize=FONT_TITLE, pad=4)
        ax.tick_params(axis="both", labelsize=FONT_TICK); hide_spines(ax)
    for j in range(len(all_a), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([0], [0], color=_BM_COLORS[m], lw=2, label=RI_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(len(methods), 6),
               frameon=False, fontsize=FONT_LEGEND)
    fig.text(0.5, -0.01, f"{DATASET_DISPLAY.get(ds, ds)}: distance from each method's own interval centre (|z|)",
             ha="center", fontsize=FONT_AXIS)
    fig.text(-0.02, 0.5, "Mortality Rate (%)", va="center", rotation="vertical", fontsize=FONT_AXIS)
    fig.tight_layout()
    return {"": fig}


FIGURES = [
    FigSpec("10_mortality",     "mortality_quintile",  fig_mortality_quintile,     False, (),                                            None),
    FigSpec("10_mortality",     "mortality_deviation", fig_mortality_deviation,    False, (),                                            None),
    FigSpec("10_mortality", "mortality_deviation_norma", ablation_variant(fig_mortality_deviation_norma), True,
            ("mortality_deviation.csv",), _one("mortality_deviation_norma")),
]


if __name__ == "__main__":
    main()
