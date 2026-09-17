#!/usr/bin/env python
"""Classify every index measurement under each reference-interval method, then
count what each method flags.

Usage:
    python 07_classify.py --dataset eicu --force
    python 07_classify.py --dataset eicu --only prevalence
"""
import bootstrap  # noqa: F401

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import datasets
from datasets import already_done, cached_chunk_frames, read_chunk_classification, EXCLUDE_LABS, add_dataset_args, get_dataset, save_csv
from metrics import WINDOW_HOURS, mark_exposures, pop_side, signed_z

STEPS = ("classify", "prevalence")


def _fix_analyte(df):
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df


def _atomic_parquet(df, path):
    """Write to a temp file and rename, so a reader never sees a partial file."""
    tmp = f"{path}.tmp-{os.getpid()}"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# classify

def classify_values(values, lows, highs):
    """0 = low, 1 = normal, 2 = high, NaN where the value or a bound is missing."""
    values = pd.to_numeric(values, errors="coerce")
    lows = pd.to_numeric(lows, errors="coerce")
    highs = pd.to_numeric(highs, errors="coerce")
    result = np.where(values < lows, 0, np.where(values > highs, 2, 1)).astype(float)
    result[values.isna() | lows.isna() | highs.isna()] = np.nan
    return result


def pivot_ref_wide(ref_df):
    """Long ref_intervals -> one row per (patient, analyte) with <method>_ri_{low,high,mean}."""
    keys = ["patient_id", "analyte", "method"]
    sub = ref_df[keys + ["ri_low", "ri_high", "ri_mean"]]
    # legacy files carry duplicate rows for sodium, some without bounds: keep a bounded one
    has_bounds = sub["ri_low"].notna() & sub["ri_high"].notna()
    sub = pd.concat([sub[has_bounds], sub[~has_bounds]]).drop_duplicates(subset=keys)
    wide = sub.set_index(keys).unstack("method")
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()
    shared = [c for c in ["patient_id", "analyte", "sex", "age", "n_bl", "t_span"] if c in ref_df.columns]
    pair_info = ref_df[shared].drop_duplicates(subset=["patient_id", "analyte"])
    return wide.merge(pair_info, on=["patient_id", "analyte"], how="left")


# ref_intervals method prefix -> the `<Method>_class` column it produces
CLASS_FAMILIES = [("pop_ri", "PopRI"), ("per_ri", "PerRI"), ("cohen_", "Cohen_"),
                  ("gaussian_", "Gaussian_"), ("norma_", "NORMA_")]


def class_column(low_col):
    """'cohen_m4_ri_low' -> 'Cohen_m4_class', 'pop_ri_low' -> 'PopRI_class', else None."""
    if not low_col.endswith("_ri_low"):
        return None
    for prefix, label in CLASS_FAMILIES:
        if not low_col.startswith(prefix):
            continue
        if prefix in ("pop_ri", "per_ri"):
            return f"{label}_class"
        variant = low_col[len(prefix):-len("_ri_low")]
        return f"{label}{variant}_class"
    return None


def propagate_pop_abnormal(result):
    """A Pop_RI-abnormal call is inherited by every personalised method: a value outside
    the population range is out of range full stop; a wider personalised interval
    should not be able to wave it through."""
    if "PopRI_class" not in result.columns:
        return
    pop_abnormal = (result["PopRI_class"] != 1) & result["PopRI_class"].notna()
    for col in result.columns:
        if col.endswith("_class") and col != "PopRI_class":
            still_normal = pop_abnormal & (result[col] == 1)
            result.loc[still_normal, col] = result.loc[still_normal, "PopRI_class"]


def add_deviation_scores(result):
    """`<method>_z`, `<method>_zs` and `pop_side` (lib/ri_metrics).  These are the raw
    geometric deviation from the method's own interval and do NOT carry the Pop_RI
    override: a measurement can have NORMA_class == 2 (inherited) with NORMA_z < 1.
    The override is a binary safety rule, not a statement about distance, so `_class`
    and `z > 1` agree only inside the Pop_RI-normal subset."""
    for cls_col in [c for c in result.columns if c.endswith("_class")]:
        method = cls_col[:-len("_class")]
        prefix = method.lower().replace("popri", "pop").replace("perri", "per")
        if f"{prefix}_ri_low" not in result.columns or f"{prefix}_ri_high" not in result.columns:
            continue
        zs = signed_z(result.drop(columns=[f"{method}_zs"], errors="ignore"), method)
        result[f"{method}_zs"] = zs
        result[f"{method}_z"] = zs.abs()
    if "pop_ri_low" in result.columns and "pop_ri_high" in result.columns:
        result["pop_side"] = pop_side(result.drop(columns=["pop_side"], errors="ignore"))


def build_classification(index_labs, ref_df, time_unit, window_hours):
    """Index measurements x wide intervals -> classes, deviation scores, exposure markers."""
    _fix_analyte(index_labs)
    _fix_analyte(ref_df)
    index_df = index_labs[index_labs["split"] == "index"].copy()
    print(f"  {len(index_df)} index measurements")
    result = index_df.merge(pivot_ref_wide(ref_df), on=["patient_id", "analyte"], suffixes=("", "_ref"))
    result = result.drop(columns=[c for c in result.columns if c.endswith("_ref")])
    print(f"  {len(result)} matched to ref intervals")

    for low_col in [c for c in result.columns if c.endswith("_ri_low")]:
        high_col = low_col.replace("_ri_low", "_ri_high")
        cls_col = class_column(low_col)
        if cls_col and high_col in result.columns:
            result[cls_col] = classify_values(result["value"], result[low_col], result[high_col])

    # the override rewrites `_class`; the deviation scores are deliberately not overridden
    propagate_pop_abnormal(result)
    add_deviation_scores(result)
    return mark_exposures(result, time_unit, window_hours)


# Caches other stages derived from a chunk's classification.
DERIVED_CACHES = ["prevalence_counts.parquet", "eval_counts.parquet", "mortality_extract.parquet",
                  "mortality_zbins.parquet", "11_future_pairs", "11_lead_measurements",
                  "11_lead_patients", "12_patient_scores", "13_stay_tables", "17_cohort"]


def expected_columns(ref_df):
    """What build_classification will write for the methods this chunk has intervals for."""
    want = {"pop_side", "t_hours", "t0_hours", "exp_first", "exp_window", "exp_window_worst"}
    for method in ref_df["method"].astype(str).unique():
        cls_col = class_column(f"{method}_ri_low")
        if cls_col:
            name = cls_col[:-len("_class")]
            want |= {cls_col, f"{name}_z", f"{name}_zs"}
    return want


def stale_caches(chunk_dir):
    """Move this chunk's derived caches aside; the stages rebuild them from the new rows."""
    moved = 0
    for name in DERIVED_CACHES:
        path = datasets.find_in(chunk_dir, name)
        if os.path.exists(path):
            dest = path + ".bak-reclassified"
            if os.path.exists(dest):
                import shutil
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            else:
                os.replace(path, dest)
            moved += 1
    return moved


def classify_chunks(ds, args):
    """CHS: one pair of classification files per chunk."""
    analytes = ds._analytes
    n_rows = n_patients = n_skipped = n_refreshed = 0
    for i, chunk_dir, index_labs, ref_df in ds.iter_chunks():
        out_path = datasets.classification_paths(chunk_dir)[0]
        if not args.force and analytes is None and os.path.exists(out_path):
            have = datasets.classification_column_names(chunk_dir) or set()
            missing = sorted(expected_columns(ref_df) - have) if ref_df is not None else []
            if not missing:
                n_skipped += 1
                continue
            print(f"    chunk_{i}: reclassifying, it has no {', '.join(missing[:4])}"
                  f"{f' (+{len(missing) - 4} more)' if len(missing) > 4 else ''}")
            n_refreshed += 1
        if index_labs is None or ref_df is None:
            continue
        result = build_classification(index_labs, ref_df, ds.time_unit, args.window_hours)
        if analytes is not None and os.path.exists(out_path):
            existing = datasets.read_classification(chunk_dir)
            kept = existing[~existing["analyte"].isin(analytes)]
            result = pd.concat([kept, result], ignore_index=True)
        datasets.write_classification(result, chunk_dir, atomic=_atomic_parquet)
        stale_caches(chunk_dir)             # what other stages derived from the old rows
        n_rows += len(result)
        n_patients += result["patient_id"].nunique()
        print(f"    chunk_{i}: {len(result)} rows")
    if n_skipped:
        print(f"  Skipped {n_skipped} chunks (already classified, with every column)")
    if n_refreshed:
        print(f"  Reclassified {n_refreshed} chunk(s) written by an older version; their "
              f"derived caches were set aside, so the stages below rebuild those chunks")
    print(f"  Classified {n_rows} new rows, {n_patients} patients")


def run_classify(ds, args):
    if args.dataset == "chs":
        classify_chunks(ds, args)
        return
    out_dir = ds.classification_dir()
    out_path = datasets.classification_paths(out_dir)[0]
    if not args.force and os.path.exists(out_path):
        print(f"  Classification already exists at {out_path}, skipping (use --force to redo)")
        return
    result = build_classification(ds.load_index_labs(), ds.load_ref_intervals(),
                                  ds.time_unit, args.window_hours)
    print(f"  Exposure markers: {int(result['exp_first'].sum()):,} first-index rows, "
          f"{int(result['exp_window'].sum()):,} within {args.window_hours:g}h, "
          f"{int(result['exp_window_worst'].sum()):,} window-worst rows")
    datasets.write_classification(result, out_dir, atomic=_atomic_parquet)
    print(f"  {result['patient_id'].nunique()} patients, {result['analyte'].nunique()} analytes")


# prevalence

def per_normal_mask(df):
    """Per_RI setpoint inside Pop_RI, or None when the columns are absent."""
    cols = ["per_ri_mean", "pop_ri_low", "pop_ri_high"]
    if not set(cols) <= set(df.columns):
        return None
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (df["per_ri_mean"] >= df["pop_ri_low"]) & (df["per_ri_mean"] <= df["pop_ri_high"])


def count_flags(df, methods):
    """Per-analyte counts: measurements, patients, Pop_RI-normal measurements, and per
    method the valid / abnormal calls overall and among Pop_RI-normal (reclassification)."""
    counts = defaultdict(lambda: defaultdict(int))
    cls_cols = {m: f"{m}_class" for m in methods if f"{m}_class" in df.columns}
    for analyte, grp in df.groupby("analyte"):
        c = counts[analyte]
        c["n"] += len(grp)
        c["n_patients"] += int(grp["patient_id"].nunique())   # patients do not span CHS chunks
        pop_normal = None
        if "PopRI_class" in grp.columns:
            pop_normal = grp[grp["PopRI_class"] == 1]
            c["n_popri_normal"] += len(pop_normal)
        for method, cls_col in cls_cols.items():
            valid = grp[cls_col].dropna()
            c[f"{method}_valid"] += len(valid)
            c[f"{method}_abn"] += int((valid != 1).sum())
            if method != "PopRI" and pop_normal is not None:
                valid_normal = pop_normal[cls_col].dropna()
                c[f"{method}_reclass_valid"] += len(valid_normal)
                c[f"{method}_reclass_abn"] += int((valid_normal != 1).sum())
    return counts


def counts_frame(df, methods):
    """count_flags on all rows and on the per_normal subset, as one long frame
    (columns analyte, subset, <count columns>)."""
    df = _fix_analyte(df.copy())
    df = df[~df["analyte"].isin(EXCLUDE_LABS)]
    frames = {"all": count_flags(df, methods)}
    ns = per_normal_mask(df)
    if ns is not None:
        frames["per_normal"] = count_flags(df[ns], methods)
    rows = []
    for subset, counts in frames.items():
        for analyte, c in counts.items():
            rows.append({"analyte": analyte, "subset": subset, **c})
    return pd.DataFrame(rows)


def add_counts(frame, totals):
    """Accumulate a counts frame into {subset: {analyte: {column: count}}}."""
    for row in frame.to_dict("records"):
        target = totals[row.pop("subset")][row.pop("analyte")]
        for col, value in row.items():
            if pd.notna(value):
                target[col] += int(value)


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion, in percent."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def bootstrap_cis(df, methods, n_boot=200, seed=42):
    """Patient-cluster bootstrap 95% intervals of every rate, per analyte."""
    rng = np.random.default_rng(seed)
    df = _fix_analyte(df.copy())
    df = df[~df["analyte"].isin(EXCLUDE_LABS)]
    pop_normal = (df["PopRI_class"] == 1) if "PopRI_class" in df.columns else None
    cols, rates = {}, []
    for m in methods:
        cls = df.get(f"{m}_class")
        if cls is None:
            continue
        valid = cls.notna()
        abnormal = valid & (cls != 1)
        cols[f"{m}_valid"] = valid.astype(np.int16)
        cols[f"{m}_abn"] = abnormal.astype(np.int16)
        rates.append(f"{m}_pct")
        if m != "PopRI" and pop_normal is not None:
            cols[f"{m}_reclass_valid"] = (valid & pop_normal).astype(np.int16)
            cols[f"{m}_reclass_abn"] = (abnormal & pop_normal).astype(np.int16)
            rates.append(f"{m}_reclass_pct")
    per_patient = (pd.DataFrame(cols, index=df.index)
                   .assign(analyte=df["analyte"].to_numpy(), patient_id=df["patient_id"].to_numpy())
                   .groupby(["analyte", "patient_id"]).sum())

    out = {}
    for analyte, g in per_patient.groupby(level="analyte"):
        counts = g.to_numpy(dtype=np.float64)                 # patients x count columns
        n_pat = len(counts)
        if n_pat < 2:
            continue
        weights = rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=n_boot).astype(np.float64)
        sums = weights @ counts                               # n_boot x count columns
        intervals = {}
        for rate in rates:
            stem = rate[:-len("_pct")]
            i_valid = g.columns.get_loc(f"{stem}_valid")
            i_abn = g.columns.get_loc(f"{stem}_abn")
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(sums[:, i_valid] > 0, sums[:, i_abn] / sums[:, i_valid] * 100, np.nan)
            if np.isfinite(r).sum() >= n_boot // 2:
                intervals[rate] = (float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5)))
        out[analyte] = intervals
    return out


def prevalence_table(counts, methods, cis):
    """Counts -> prevalence percentages per analyte, with 95% intervals from the
    patient-cluster bootstrap when available, else Wilson on the measurement counts."""
    rows = []
    for analyte in sorted(counts):
        c = counts[analyte]
        if c["n"] == 0:
            continue
        boot = (cis or {}).get(analyte, {})
        row = {"analyte": analyte, "n": c["n"], "n_patients": c.get("n_patients", 0)}
        for m in methods:
            valid, abnormal = c.get(f"{m}_valid", 0), c.get(f"{m}_abn", 0)
            row[f"{m}_pct"] = abnormal / valid * 100 if valid > 0 else np.nan
            row[f"{m}_valid"] = valid
            row[f"{m}_abn"] = abnormal
            row[f"{m}_pct_lo"], row[f"{m}_pct_hi"] = boot.get(f"{m}_pct", wilson(abnormal, valid))
            r_valid, r_abnormal = c.get(f"{m}_reclass_valid", 0), c.get(f"{m}_reclass_abn", 0)
            if m != "PopRI" and r_valid > 0:
                row[f"{m}_reclass_pct"] = r_abnormal / r_valid * 100
                row[f"{m}_reclass_valid"] = r_valid
                row[f"{m}_reclass_abn"] = r_abnormal
                row[f"{m}_reclass_pct_lo"], row[f"{m}_reclass_pct_hi"] = \
                    boot.get(f"{m}_reclass_pct", wilson(r_abnormal, r_valid))
        row["n_popri_normal"] = c.get("n_popri_normal", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def overall_row(df):
    """Counts summed; rates = mean across analytes, plus median / quartiles of the
    percentages for the overall figure; per-analyte intervals are meaningless here."""
    overall = {"analyte": "Overall"}
    for col in df.columns:
        if col == "analyte":
            continue
        if col.startswith("n") or col.endswith(("_valid", "_abn")):
            overall[col] = df[col].sum()
        elif col.endswith(("_lo", "_hi")):
            overall[col] = np.nan
        else:
            overall[col] = df[col].mean()
            if col.endswith("_pct"):
                overall[f"{col}_median"] = df[col].median()
                overall[f"{col}_q25"] = df[col].quantile(0.25)
                overall[f"{col}_q75"] = df[col].quantile(0.75)
    return overall


def save_prevalence(df, methods, results_dir, subset, analytes=None):
    for m in methods:
        if f"{m}_pct" in df.columns:
            per_1000 = (df[f"{m}_pct"] / 100 * 1000).replace([np.inf, -np.inf], np.nan)
            df[f"{m}_per_1000"] = per_1000.fillna(0).round(0).astype(int)
    df = pd.concat([df, pd.DataFrame([overall_row(df)])], ignore_index=True)
    decimals = {c: 2 for c in df.columns if "_pct" in c}
    decimals.update({c: 0 for c in df.columns if c.endswith("_per_1000")})
    df = df.round(decimals)

    df.insert(0, "subset", subset)      # one file; the subset is a column
    out_path = os.path.join(results_dir, "prevalence.csv")
    save_csv(df, out_path, analytes=analytes, keys=("subset",))
    print(f"  Saved {len(df)} rows to {out_path}")
    overall = df[df["analyte"] == "Overall"].iloc[0]
    print("\n  Per 1,000 tests screened (mean across analytes):")
    for m in methods:
        col = f"{m}_per_1000"
        if col in overall.index and pd.notna(overall[col]):
            print(f"    {m}: {int(overall[col])} flagged abnormal")


def run_prevalence(ds, args):
    results_dir = ds.setup_output()
    if already_done(args, results_dir, "prevalence.csv", label="prevalence"):
        return
    methods = ds.methods
    print(f"  Methods: {methods}")
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))   # subset -> analyte -> counts
    cis = {"all": None, "per_normal": None}

    if args.dataset == "chs":
        def compute(chunk_dir):
            chunk = read_chunk_classification(chunk_dir)
            if chunk is None:
                return None
            if ds._analytes is not None:
                chunk = chunk[_fix_analyte(chunk)["analyte"].isin(ds._analytes)]
            return counts_frame(chunk, methods)

        for frame in cached_chunk_frames(ds, "prevalence_counts.parquet", compute, force=args.force):
            add_counts(frame, totals)
    else:
        need = ["patient_id", "analyte"] + [f"{m}_class" for m in methods]
        need += ["per_ri_mean", "pop_ri_low", "pop_ri_high"]
        classified = ds.load_classification(usecols=lambda c: c in need)
        print(f"  {len(classified)} classified measurements")
        add_counts(counts_frame(classified, methods), totals)
        if args.n_boot > 0:
            print(f"  patient-cluster bootstrap ({args.n_boot} replicates)...")
            cis["all"] = bootstrap_cis(classified, methods, n_boot=args.n_boot)
            ns = per_normal_mask(classified)
            if ns is not None:
                cis["per_normal"] = bootstrap_cis(classified[ns], methods, n_boot=args.n_boot)

    for subset in ("all", "per_normal"):
        counts = totals.get(subset)
        if not counts or not any(c["n"] > 0 for c in counts.values()):
            continue
        print(f"\n  --- {subset} ---")
        table = prevalence_table(counts, methods, cis[subset])
        save_prevalence(table, methods, results_dir, subset, analytes=ds._analytes)


# main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--window_hours", type=float, default=WINDOW_HOURS,
                   help="exposure window after the first index measurement (lib/metrics.py)")
    p.add_argument("--n_boot", type=int, default=200,
                   help="prevalence: patient-cluster bootstrap replicates (0 = Wilson only)")
    args = p.parse_args()

    ds = get_dataset(args)
    if "classify" in args.only:
        print("=== classify ===")
        run_classify(ds, args)
    if "prevalence" in args.only:
        print("=== prevalence ===")
        run_prevalence(ds, args)


# Figures and tables

from matplotlib.transforms import blended_transform_factory

from figlib import *  # noqa: F401,F403


_RATE_COLS = ["pct", "pct_lo", "pct_hi", "reclass_pct", "reclass_pct_lo", "reclass_pct_hi"]


def _prevalence_long(prev):
    """Screenshot export: one row per analyte x method with the rates and their intervals
    (the wide file has ~50 columns, too wide to photograph)."""
    methods = [m for m in dict.fromkeys([*_BM_SUPP, "NORMA"]) if f"{m}_pct" in prev.columns]
    keys = [c for c in ("analyte", "n", "n_patients") if c in prev.columns]
    rows = []
    for m in methods:
        block = prev[keys].copy()
        block["method"] = m
        for col in _RATE_COLS:
            if f"{m}_{col}" in prev.columns:
                block[col] = prev[f"{m}_{col}"]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def _prevalence_wide(long):
    """A transcribed long export back to the wide prevalence.csv layout."""
    keys = [c for c in ("analyte", "n", "n_patients") if c in long.columns]
    wide = long.drop_duplicates("analyte")[keys].set_index("analyte")
    for m, block in long.groupby("method"):
        block = block.set_index("analyte")
        for col in _RATE_COLS:
            if col in block.columns:
                wide[f"{m}_{col}"] = block[col]
    return wide.reset_index()


def _load_prevalence(ds, subset="all"):
    """One subset of prevalence.csv: "all" tests, or "per_normal" (within each
    method's own normal range)."""
    prev = load_result(ds, "prevalence.csv")
    if prev is None:
        return None
    if "subset" in prev.columns:
        prev = prev[prev["subset"].astype(str) == subset]
        if not len(prev):
            return None
    if "method" in prev.columns and "pct" in prev.columns:     # a transcribed long export
        prev = _prevalence_wide(prev)
    prev = to_numeric(prev)
    prev["analyte"] = prev["analyte"].replace("", "NA").fillna("NA")
    return prev


_PANEL_GAP = 0.8      # extra slots of space between analyte panels
# MIN_PATIENTS (lib/constants.py): bar figures omit analytes with fewer patients, the
# supplementary heatmap keeps them with grey values

def _panel_layout(analytes):
    """x position per analyte with a gap between panels; returns (x dict, [(panel, x0, x1)])."""
    x, spans, pos = {}, [], 0.0
    groups = [(name, [a for a in analytes if a in members]) for name, members in ANALYTE_PANELS.items()]
    rest = [a for a in analytes if not any(a in m for m in ANALYTE_PANELS.values())]
    if rest:
        groups.append(("Other", rest))
    for name, members in groups:
        if not members:
            continue
        start = pos
        for a in members:
            x[a] = pos; pos += 1
        spans.append((name, start, pos - 1))
        pos += _PANEL_GAP
    return x, spans


def _analyte_bars(suffix, ylabel, title, methods):
    """One row per cohort; grouped bars per analyte (one bar per method) with 95% whiskers."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return None
    shown = set()
    for d in frames.values():
        if d is None:
            continue
        d = d[d.analyte != "Overall"]
        if "n_patients" in d.columns:
            d = d[d.n_patients >= MIN_PATIENTS]
        shown |= set(d.analyte)
    analytes = analyte_panel_order(shown - EXCLUDE_ANALYTES)
    xpos, spans = _panel_layout(analytes)
    methods = [m for m in methods if any(d is not None and f"{m}_{suffix}" in d.columns for d in frames.values())]
    n_m = len(methods)
    w = 0.8 / n_m
    x_all = np.array([xpos[a] for a in analytes])

    fig, axes = plt.subplots(len(VAL_COHORTS), 1, figsize=(7.2, 1.75 * len(VAL_COHORTS) + 0.7), sharex=True, squeeze=False)
    axes = axes[:, 0]                       # one cohort (--dataset X) must still give a row array
    for ax, ds in zip(axes, VAL_COHORTS):
        d = frames[ds]
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, loc="left", pad=(12 if ax is axes[0] else 3))
        if d is None:
            pending_axis(ax, ds); continue
        d = d[d.analyte != "Overall"].set_index("analyte").reindex(analytes)
        if "n_patients" in d.columns:
            small = d["n_patients"].fillna(0) < MIN_PATIENTS
            d.loc[small, [c for c in d.columns if c != "n_patients"]] = np.nan
        for k, m in enumerate(methods):
            col = f"{m}_{suffix}"
            if col not in d.columns:
                continue
            v = d[col].to_numpy(float)
            lo = d[f"{col}_lo"].to_numpy(float) if f"{col}_lo" in d.columns else np.full_like(v, np.nan)
            hi = d[f"{col}_hi"].to_numpy(float) if f"{col}_hi" in d.columns else np.full_like(v, np.nan)
            err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
            err = np.where(np.isfinite(err), err, 0)
            ax.bar(x_all + (k - (n_m - 1) / 2) * w, v, width=w * 0.92, color=_BM_COLORS[m], linewidth=0,
                   yerr=err, error_kw=dict(lw=0.5, ecolor=DARK, capsize=0), zorder=2, label=_BM_LABELS[m])
        for _, s0, _ in spans[1:]:
            ax.axvline(s0 - _PANEL_GAP / 2, color="#E3E5E8", lw=0.6, zorder=0)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
        ax.set_ylim(0, 100 if suffix == "pct" else None)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.grid(axis="y", color="#F0F1F3", lw=0.5, zorder=0)
        hide_spines(ax)
    axes[-1].set_xticks(x_all); axes[-1].set_xticklabels(analytes, rotation=90, fontsize=FONT_TICK)
    axes[-1].set_xlim(-0.7, max(xpos.values()) + 0.7)
    for ax in axes:
        ax.tick_params(axis="x", length=0)
    tr = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for name, s0, s1 in spans:
        axes[0].text((s0 + s1) / 2, 1.02, name, transform=tr, ha="center", va="bottom",
                     fontsize=FONT_TICK, color=DARK, clip_on=False)
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(hl.values(), hl.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(hl), 7),
               frameon=False, fontsize=FONT_LEGEND, handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(h_pad=1.0, rect=(0, 0, 1, 0.96))
    return fig


def _analyte_heatmap(suffix, cbar_label, methods):
    """Cohort blocks stacked; rows = methods, columns = analytes, cell = rate (%)."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return None
    analytes = analyte_panel_order(set().union(*[set(d.analyte) - {"Overall"} for d in frames.values() if d is not None])
                                   - EXCLUDE_ANALYTES)
    methods = [m for m in methods if any(d is not None and f"{m}_{suffix}" in d.columns for d in frames.values())]
    blocks = []
    for ds in VAL_COHORTS:
        d = frames[ds]
        if d is None:
            blocks.append((ds, None)); continue
        d = d[d.analyte != "Overall"].set_index("analyte").reindex(analytes)
        mat = np.array([[float(d.loc[a, f"{m}_{suffix}"]) if f"{m}_{suffix}" in d.columns else np.nan
                         for a in analytes] for m in methods])
        ncol = "n_patients" if "n_patients" in d.columns else "n"
        nmat = np.array([[float(d.loc[a, ncol]) for a in analytes] for _ in methods])
        blocks.append((ds, {"mat": mat, "rows": [RI_LABELS[m] for m in methods], "nmat": nmat}))
    col_groups = {name: [a for a in members if a in analytes] for name, members in ANALYTE_PANELS.items()}
    vmax = 100 if suffix == "pct" else float(np.nanmax([np.nanmax(b["mat"]) for _, b in blocks if b]))
    return heatmap_blocks(blocks, analytes, CORAL_CMAP, 0, vmax, lambda v: f"{v:.0f}", cbar_label,
                          col_groups=col_groups, row_label_fontsize=FONT_TICK)


def fig_prevalence():
    out = {}
    fig = _analyte_bars("pct", "Abnormal (%)", "Abnormality prevalence per analyte",
                        bm_methods(_BM_MAIN))
    if fig is not None:
        out[None] = fig
    hm = _analyte_heatmap("pct", "Abnormality prevalence (%)", bm_methods())
    if hm is not None:
        out["supp"] = hm
    return out


def fig_reclassification():
    out = {}
    fig = _analyte_bars("reclass_pct", "Reclassified (%)",
                        "Reclassification of Pop$_{RI}$-normal measurements per analyte",
                        [m for m in bm_methods(_BM_MAIN) if m != "PopRI"])
    if fig is not None:
        out[None] = fig
    hm = _analyte_heatmap("reclass_pct", "Pop$_{RI}$-normal measurements flagged (%)",
                          [m for m in bm_methods() if m != "PopRI"])
    if hm is not None:
        out["supp"] = hm
    return out


def fig_prevalence_overall():
    """Bars = median across analytes per method and cohort, whisker = interquartile range."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return {}
    methods = [m for m in bm_methods() if any(d is not None and f"{m}_pct" in d.columns for d in frames.values())]
    n_m = len(methods); w = 0.8 / n_m
    panels = [("pct", "Abnormal (%)"), ("reclass_pct", "Reclassified (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, (suffix, ylabel) in zip(axes, panels):
        for i, ds in enumerate(VAL_COHORTS):
            d = frames[ds]
            if d is None or not (d.analyte == "Overall").any():
                ax.text(i, 2, "pending", ha="center", va="bottom", fontsize=FONT_TICK, color="#AAAAAA")
                continue
            row = d[d.analyte == "Overall"].iloc[0]
            for k, m in enumerate(methods):
                col = f"{m}_{suffix}"
                if col not in d.columns or not np.isfinite(float(row[col])):
                    continue
                if f"{col}_median" in d.columns and np.isfinite(float(row[f"{col}_median"])):
                    v, lo, hi = (float(row[f"{col}_median"]), float(row[f"{col}_q25"]), float(row[f"{col}_q75"]))
                else:
                    v, lo, hi = float(row[col]), np.nan, np.nan
                err = np.array([[max(v - lo, 0)], [max(hi - v, 0)]]) if np.isfinite(lo) else None
                ax.bar(i + (k - (n_m - 1) / 2) * w, v, width=w * 0.92, color=_BM_COLORS[m], linewidth=0,
                       yerr=err, error_kw=dict(lw=0.6, ecolor=DARK, capsize=0), zorder=2, label=_BM_LABELS[m])
        ax.set_xticks(range(len(VAL_COHORTS)))
        ax.set_xticklabels([DATASET_DISPLAY.get(c, c) for c in VAL_COHORTS], fontsize=FONT_AXIS)
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
        ax.set_ylim(0, 100 if suffix == "pct" else None)
        ax.grid(axis="y", color="#F0F1F3", lw=0.5, zorder=0)
        hide_spines(ax)
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(hl.values(), hl.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(hl), 7),
               frameon=False, fontsize=FONT_LEGEND, handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    # The y labels are short, so what the bar and the whisker are goes here instead.
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
    return {"": fig}


# ── Redesign candidates (2026-08-31) ──────────────────────────────────────────── The grouped-
# bar figures above answer "how often does each method flag" with one bar per method per analyte.

_GAUSS_FAMILY = ["Gaussian_mle", "Gaussian_trunc", "Gaussian_eb"]


def _overall_stats(d, method, suffix):
    """(median, q25, q75) across analytes from the Overall row 07_classify.py (prevalence step) writes."""
    row = d[d.analyte == "Overall"]
    col = f"{method}_{suffix}"
    if not len(row) or col not in d.columns:
        return (np.nan, np.nan, np.nan)
    row = row.iloc[0]
    if f"{col}_median" in d.columns and np.isfinite(float(row[f"{col}_median"])):
        return (float(row[f"{col}_median"]), float(row[f"{col}_q25"]), float(row[f"{col}_q75"]))
    v = float(row[col])
    return (v, np.nan, np.nan)


def fig_prevalence_summary():
    """Main-text candidate: median rate across analytes per method, cohorts stacked."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return {}
    methods = [m for m in bm_methods()
               if any(d is not None and f"{m}_pct" in d.columns for d in frames.values())]
    rows = []
    for ds in VAL_COHORTS:
        d = frames[ds]
        if d is None or not (d.analyte == "Overall").any():
            rows.append((ds, None)); continue
        rows.append((ds, {m: {"pct": _overall_stats(d, m, "pct"),
                              "reclass_pct": _overall_stats(d, m, "reclass_pct")}
                          for m in methods}))
    metrics = [("pct", "Abnormal (%)", (0, 100)),
               ("reclass_pct", "Reclassified (%)", (0, None))]
    fig = dot_blocks(rows, metrics, methods, _BM_COLORS, RI_LABELS, label_rotation=90, row_h=0.155)
    return {"": fig}


def fig_prevalence_delta():
    """Supplementary candidate: excess flagging over Pop_RI per analyte, as a heatmap."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return {}
    analytes = analyte_panel_order(
        set().union(*[set(d.analyte) - {"Overall"} for d in frames.values() if d is not None])
        - EXCLUDE_ANALYTES)
    methods = [m for m in bm_methods() if m != "PopRI"
               and any(d is not None and f"{m}_pct" in d.columns for d in frames.values())]
    blocks, pooled = [], []
    for ds in VAL_COHORTS:
        d = frames[ds]
        if d is None:
            blocks.append((ds, None)); continue
        d = d[d.analyte != "Overall"].set_index("analyte").reindex(analytes)
        base = d["PopRI_pct"].to_numpy(float)
        mat = np.array([[(float(d.loc[a, f"{m}_pct"]) - float(d.loc[a, "PopRI_pct"]))
                         if f"{m}_pct" in d.columns else np.nan for a in analytes] for m in methods])
        ncol = "n_patients" if "n_patients" in d.columns else "n"
        nmat = np.array([[float(d.loc[a, ncol]) for a in analytes] for _ in methods])
        pooled.append(mat)
        blocks.append((ds, {"mat": mat, "rows": [RI_LABELS[m] for m in methods],
                            "nmat": nmat, "row_groups": [METHOD_FAMILY.get(m, m) for m in methods]}))
    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in pooled]), 95))
    vmax = max(5.0, np.ceil(vmax / 5) * 5)
    col_groups = {name: [a for a in members if a in analytes] for name, members in ANALYTE_PANELS.items()}
    fig = heatmap_blocks(blocks, analytes, CORAL_CMAP, 0, vmax, lambda v: f"{v:.0f}",
                         r"Excess over Pop$_{RI}$ (pp)", extend="max",
                         col_groups=col_groups, row_label_fontsize=FONT_TICK)
    return {"": fig}


def fig_prevalence_excess():
    """Third candidate: per-analyte excess over Pop_RI as dots on a zero line."""
    frames = {ds: _load_prevalence(ds) for ds in VAL_COHORTS}
    if not any(d is not None for d in frames.values()):
        return {}
    shown = set()
    for d in frames.values():
        if d is None:
            continue
        d = d[d.analyte != "Overall"]
        if "n_patients" in d.columns:
            d = d[d.n_patients >= MIN_PATIENTS]
        shown |= set(d.analyte)
    analytes = analyte_panel_order(shown - EXCLUDE_ANALYTES)
    xpos, spans = _panel_layout(analytes)
    x_all = np.array([xpos[a] for a in analytes])
    dots = [m for m in ["PerRI", "Cohen_m4", "NORMA"] if any(
        d is not None and f"{m}_pct" in d.columns for d in frames.values())]
    off = {m: (k - (len(dots) - 1) / 2) * 0.20 for k, m in enumerate(dots)}

    fig, axes = plt.subplots(len(VAL_COHORTS), 1, figsize=(7.2, 1.75 * len(VAL_COHORTS) + 0.7), sharex=True, squeeze=False)
    axes = axes[:, 0]                       # one cohort (--dataset X) must still give a row array
    for ax, ds in zip(axes, VAL_COHORTS):
        d = frames[ds]
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=FONT_TITLE, loc="left", pad=(12 if ax is axes[0] else 3))
        if d is None:
            pending_axis(ax, ds); continue
        d = d[d.analyte != "Overall"].set_index("analyte").reindex(analytes)
        if "n_patients" in d.columns:
            d.loc[d["n_patients"].fillna(0) < MIN_PATIENTS,
                  [c for c in d.columns if c != "n_patients"]] = np.nan
        base = d["PopRI_pct"].to_numpy(float)
        gcols = [f"{m}_pct" for m in _GAUSS_FAMILY if f"{m}_pct" in d.columns]
        if gcols:
            g = d[gcols].to_numpy(float) - base[:, None]
            ok = np.isfinite(g).any(1)          # analytes with no Gaussian fit draw no band
            if ok.any():
                ax.vlines(x_all[ok], np.nanmin(g[ok], 1), np.nanmax(g[ok], 1),
                          color=_BM_COLORS["Gaussian_trunc"], lw=2.6, alpha=0.45, zorder=1,
                          label=f"{RI_LABELS['Gaussian_mle']} family "
                                f"(MLE / truncated / {RI_LABELS['Gaussian_eb']})")
        for m in dots:
            if f"{m}_pct" not in d.columns:
                continue
            v = d[f"{m}_pct"].to_numpy(float) - base
            ax.plot(x_all + off[m], v, ls="", marker="o", ms=2.8, mfc=_BM_COLORS[m],
                    mec="white", mew=0.4, zorder=3, label=RI_LABELS[m])
        ax.axhline(0, color=DARK, lw=0.7, zorder=2)
        for _, s0, _ in spans[1:]:
            ax.axvline(s0 - _PANEL_GAP / 2, color="#E3E5E8", lw=0.6, zorder=0)
        ax.set_ylabel("Excess over Pop$_{RI}$ (pp)", fontsize=FONT_AXIS)
        ax.set_ylim(bottom=-2)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.grid(axis="y", color="#F0F1F3", lw=0.5, zorder=0)
        hide_spines(ax)
    axes[-1].set_xticks(x_all); axes[-1].set_xticklabels(analytes, rotation=90, fontsize=FONT_TICK)
    axes[-1].set_xlim(-0.7, max(xpos.values()) + 0.7)
    for ax in axes:
        ax.tick_params(axis="x", length=0)
    tr = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for name, s0, s1 in spans:
        axes[0].text((s0 + s1) / 2, 1.02, name, transform=tr, ha="center", va="bottom",
                     fontsize=FONT_TICK, color=DARK, clip_on=False)
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(hl.values(), hl.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(hl), 5),
               frameon=False, fontsize=FONT_LEGEND, handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(h_pad=1.0, rect=(0, 0, 1, 0.96))
    return {"": fig}


FIGURES = [
    FigSpec("07_classify", "prevalence",         fig_prevalence,         False, (), None),
    FigSpec("07_classify", "reclassification",   fig_reclassification,   False, (), None),
    FigSpec("07_classify", "prevalence_overall", fig_prevalence_overall, False, (), None),
    FigSpec("07_classify", "prevalence_norma",       ablation_variant(fig_prevalence),       False, (), None),
    FigSpec("07_classify", "reclassification_norma", ablation_variant(fig_reclassification), False, (), None),
    FigSpec("07_classify", "prevalence_overall_norma", ablation_variant(fig_prevalence_overall), False, (), None),
    FigSpec("07_classify", "prevalence_summary", fig_prevalence_summary, False, (), None),
    FigSpec("07_classify", "prevalence_delta",   fig_prevalence_delta,   False, (), None),
    FigSpec("07_classify", "prevalence_excess",  fig_prevalence_excess,  False, (), None),
]


# Tables — 07_classify: table_* definitions and registry slice.

from figlib import *  # noqa: F401,F403

# save_table()'s first argument is the folder the table is written into, so it must match this
# directory name.

def table_prevalence():
    rows = []
    for ds in DATASETS:
        df = _load_prevalence(ds)
        if df is None:
            rows.append({"Dataset": DATASET_DISPLAY[ds], "N analytes": "---", "PopRI (%)": "---", "PerRI (%)": "---",
                         "NORMA (%)": "---", "PerRI RR (%)": "---", "NORMA RR (%)": "---"})
            continue
        p = to_numeric(df[df["analyte"] != "Overall"].set_index("analyte").copy())
        rows.append({"Dataset": DATASET_DISPLAY[ds], "N analytes": len(p),
                     "PopRI (%)": f'{p["PopRI_pct"].mean():.1f}', "PerRI (%)": f'{p["PerRI_pct"].mean():.1f}',
                     "NORMA (%)": f'{p["NORMA_pct"].mean():.1f}',
                     "PerRI RR (%)": f'{p["PerRI_reclass_pct"].mean():.1f}' if "PerRI_reclass_pct" in p.columns else "---",
                     "NORMA RR (%)": f'{p["NORMA_reclass_pct"].mean():.1f}' if "NORMA_reclass_pct" in p.columns else "---"})
    csv_df = pd.DataFrame(rows)
    header = [r"Dataset & N & Pop$_{RI}$ (\%) & Per$_{RI}$ (\%) & NORMA$_{RI}$ (\%) & Per$_{RI}$ RR (\%) & NORMA$_{RI}$ RR (\%) \\"]
    body = [" & ".join(str(r[k]) for k in ("Dataset", "N analytes", "PopRI (%)", "PerRI (%)", "NORMA (%)", "PerRI RR (%)", "NORMA RR (%)")) + r" \\"
            for r in rows]
    save_table("07_classify", "prevalence", _table("lrrrrrr", header, body), csv_df)
    return ["prevalence"]

def table_prevalence_detail(ds):
    df = _load_prevalence(ds)
    if df is None:
        return []
    present = to_numeric(df.set_index("analyte").copy())
    rows = []
    for a in all_analytes():
        if a in present.index:
            r = present.loc[a]
            g = lambda c: f"{r[c]:.1f}" if c in r and pd.notna(r[c]) else "---"
            rows.append({"Analyte": a, "N": f'{int(r["n"]):,}' if pd.notna(r["n"]) else "---", "PopRI (%)": g("PopRI_pct"),
                         "PerRI (%)": g("PerRI_pct"), "PerRI RR (%)": g("PerRI_reclass_pct"),
                         "NORMA (%)": g("NORMA_pct"), "NORMA RR (%)": g("NORMA_reclass_pct")})
        else:
            rows.append({"Analyte": a, "N": "---", "PopRI (%)": "---", "PerRI (%)": "---", "PerRI RR (%)": "---",
                         "NORMA (%)": "---", "NORMA RR (%)": "---"})
    header = [r" & & \multicolumn{1}{c}{Pop$_{RI}$} & \multicolumn{2}{c}{Per$_{RI}$} & \multicolumn{2}{c}{NORMA$_{RI}$} \\",
              r"\cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
              r"Analyte & N & Abn (\%) & Abn (\%) & RR (\%) & Abn (\%) & RR (\%) \\"]
    body = [" & ".join(str(r[k]) for k in ("Analyte", "N", "PopRI (%)", "PerRI (%)", "PerRI RR (%)", "NORMA (%)", "NORMA RR (%)")) + r" \\"
            for r in rows]
    name = f"prevalence_detail_{ds}"
    save_table("07_classify", name, _table("lrrrrrr", header, body), pd.DataFrame(rows))
    return [name]

def table_prevalence_interpretable():
    """Per 1,000 screened -> how many flagged abnormal, per dataset x method."""
    ds_data = {}
    for ds in DATASETS:
        df = _load_prevalence(ds, "per_normal")
        if df is None:
            df = _load_prevalence(ds)
        if df is None or len(df) == 0:
            ds_data[ds] = None; continue
        for method in METHODS:
            per1k, pct = f"{method}_per_1000", f"{method}_pct"
            if pct in df.columns:
                df[pct] = pd.to_numeric(df[pct], errors="coerce")
                if per1k not in df.columns:
                    df[per1k] = (df[pct] / 100 * 1000).round(0)
                else:
                    df[per1k] = pd.to_numeric(df[per1k], errors="coerce")
                    mask = df[per1k].isna()
                    df.loc[mask, per1k] = (df.loc[mask, pct] / 100 * 1000).round(0)
        ds_data[ds] = df.set_index("analyte")
    if all(v is None for v in ds_data.values()):
        return []
    rows = []
    for a in all_analytes():
        row = {"Analyte": a}
        for ds in DATASETS:
            df = ds_data[ds]
            for method in METHODS:
                col = f"{method}_per_1000"
                if df is not None and a in df.index and col in df.columns and pd.notna(df.loc[a, col]) and df.loc[a, col] > 0:
                    row[f"{ds}_{method}"] = f"{int(df.loc[a, col]):,}"
                else:
                    row[f"{ds}_{method}"] = "---"
        rows.append(row)
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule"]
    h1, cmid = "", []
    for i, ds in enumerate(DATASETS):
        h1 += r" & \multicolumn{3}{c}{" + DATASET_DISPLAY[ds] + "}"; cmid.append(f"\\cmidrule(lr){{{2 + i * 3}-{4 + i * 3}}}")
    lines += [h1 + r" \\", " ".join(cmid), "Analyte" + "".join(" & " + _ri(m) for _ in DATASETS for m in METHODS) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join([row["Analyte"]] + [row[f"{ds}_{m}"] for ds in DATASETS for m in METHODS]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("07_classify", "prevalence_interpretable", lines, pd.DataFrame(rows))
    return ["prevalence_interpretable"]

TABLES = [
    TableSpec("07_classify",    "prevalence",              table_prevalence,              False, (), None),
    TableSpec("07_classify",    "prevalence_detail",       table_prevalence_detail,       True,  ("prevalence.csv",), lambda ds: [f"prevalence_detail_{ds}"]),
    TableSpec("07_classify",    "prevalence_interpretable", table_prevalence_interpretable, False, (), None),
]


if __name__ == "__main__":
    main()
