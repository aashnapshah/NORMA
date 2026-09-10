#!/usr/bin/env python
"""Write results/processed/<cohort>/ -- the figure data -- from the raw results.

    python export.py --dataset chs                 # every stage that has raw results
    python export.py --dataset chs --only 'cox*'   # some files
    python export.py --dataset chs --txt           # + fixed-width .txt copies for screenshots

A stage writes its full results to results/raw/<cohort>/; this derives
the slice the figures and tables read (SPEC below: columns, rows, numbers rounded
to SIG significant figures) so that one small folder is all that has to leave
Clalit.  jobs/run_clalit.py calls it after every stage.  Here,
`make_figures.py --dataset eicu --from processed` checks that it is enough.
"""
import bootstrap  # noqa: F401

import argparse
import glob
import os
import re
from collections import namedtuple
from fnmatch import fnmatch

import pandas as pd
from datasets import NORMA_RUN_ID, results_dir



# ═══════════════════════════════════════════════════════════════════════════
# What leaves a cohort
# ═══════════════════════════════════════════════════════════════════════════

# What leaves a cohort: the slice of every raw result file that the figures and
# tables read (results/processed/<cohort>/, written by scripts/export.py).
#
# One entry per raw result filename (or glob).  `cols` = the columns to keep (None
# = all), `patterns` = regexes for further columns to keep, `rows` = a row filter
# (a callable frame -> boolean mask).  A raw file with no entry is not exported.
#
# The spec is the only definition of what the plotting code needs from a cohort,
# so it is checked here rather than trusted: `make_figures.py --dataset eicu
# --from processed` and the matching make_tables call must draw exactly what the
# raw results draw (compare results/figures/eicu/*.pdf and
# results/tables/eicu/csv/*.csv between the two builds).  Add a column here when
# a figure starts reading one; the check catches the omission.

Spec = namedtuple("Spec", "cols patterns rows")
ALL = Spec(None, (), None)


def _cols(*cols, patterns=(), rows=None):
    return Spec(list(cols), tuple(patterns), rows)


def _norma_runs(df):
    """Every NORMA run: the main model and the covariate-ablation arms (age_ri_norma)."""
    return df["method"].astype(str).str.startswith("NORMA")


# 13_cox: the drawn / tabulated columns, every exposure x encoding row
_COX = _cols("subset", "analyte", "method", "outcome", "exposure", "encoding", "level", "n", "n_events",
             "HR", "HR_lower", "HR_upper", "p_value", "p_fdr")

SPEC = {
    # 03_cohort_summary (tables)
    "cohort.csv": ALL, "demographics.csv": ALL,   # both carry the split column ("all" = whole cohort)
    # 05_forecasting (summary + by_analyte figures, prediction / analyte performance tables)
    "forecast.csv": ALL,   # per analyte + analyte="pooled"/"median"; target_state = normal | all_states
    # 06_calibration: per analyte x method x state, plus the analyte="median" rows
    "calibration.csv": ALL,
    # the conformal figure draws one row per method, so only the median rows travel
    "conformal.csv": Spec(None, (), lambda d: d["analyte"].astype(str) == "median"),
    # 07_classify (prevalence / reclassification figures and tables)
    "prevalence.csv": ALL,   # subset column: all | per_normal
    # 08_variability
    "variability.csv": ALL,
    # 09_age_ri: the figures draw NORMA only (main run, and the ablation arms in age_ri_norma)
    "age_ri.csv": Spec(None, (), _norma_runs),
    # 10_mortality
    "mortality_quintile.csv": ALL,
    "mortality_deviation.csv": ALL,   # method column: baseline_z + one per RI method
    # 11_lead_time: future_abnormal (per analyte + the analyte="median" rows, per
    # age band), lead time at every anchor
    "future_abnormal.csv": ALL,   # per analyte, analyte="median", and the age_band rows
    # only the pooled rows are drawn; the age-band and sweep rows stay in raw/
    "lead_time.csv": Spec(None, (), lambda d: (d["age_band"].astype(str) == "all")
                                              & (d["target_rate"].astype(str) == "all")),
    # 12_eval: per analyte x method x outcome metrics, AUROC, and the pooled ('all') deviation-score rows
    "eval.csv": ALL,   # subset column: all | per_normal | pop_normal | per_normal_pop_normal
    "auroc.csv": ALL,
    "deviation_score.csv": Spec(None, (), lambda d: d["analyte"].astype(str) == "all"),
    # 13_cox
    "cox.csv": _COX,   # subset column: all | pop_normal
    # 14_patient_level
    "concordance.csv": _cols("method", "model_type", "panel", "subset", "outcome", "time_window",
                                       "n_test", "n_events_test",
                                       "concordance_test", "concordance_lower", "concordance_upper"),
    "nri.csv": ALL,   # design column: refit | swap
    # 16_benchmark: the matched circos = Per_RI at its native threshold vs NORMA at Per_RI's flag rate
    "matched_operating_point.csv": Spec(
        ["subset", "anchor", "outcome", "analyte", "method", "n_patients", "n_events", "n_flagged",
         "ppv", "sensitivity", "specificity"], (),
        lambda d: (d["analyte"].astype(str) != "median")
                  & (d["subset"].astype(str) == "pop_normal")
                  & d["anchor"].astype(str).isin(["native", "rate_of:PerRI"])
                  & d["method"].astype(str).isin(["PerRI", "NORMA", f"NORMA_{NORMA_RUN_ID}"])),
    # 17_outcomes: every outcome in one file (the `outcome` column), RR at every horizon
    "incidence.csv": _cols("outcome", "analyte", "method", "anchor", "landmark_hours",
                           "target_sensitivity", "n", "n_flagged", "flag_rate", "logrank_p",
                           patterns=(r"^rr_\d+h$",)),
}


def spec_for(filename):
    """The Spec for a raw result filename, or None if it is not exported.

    SPEC is keyed by the bare name; files on disk carry their stage prefix
    (12_eval.csv), so strip it before looking up."""
    filename = re.sub(r"^\d\d_", "", os.path.basename(filename))
    if filename in SPEC:
        return SPEC[filename]
    for pattern, spec in SPEC.items():
        if "*" in pattern and fnmatch(filename, pattern):
            return spec
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

SIG = None   # no rounding: even 8 significant figures moved a few values across a 3-s.f. display boundary


def _round(df):
    if SIG is None:
        return df
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: v if pd.isna(v) else float(f"{v:.{SIG}g}"))
    return df


def process_frame(df, spec):
    if spec.rows is not None:
        df = df[spec.rows(df)]
    if spec.cols is not None:
        keep = [c for c in df.columns
                if c in spec.cols or any(re.match(p, c) for p in spec.patterns)]
        df = df[keep]
    return _round(df.copy())


def _fmt(v):
    if pd.isna(v):
        return ""
    if float(v).is_integer() and abs(v) < 1e7:
        return f"{int(v)}"
    return f"{v:.3g}"


def fixed_width(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt)
    return out.to_string(index=False)


def export_dataset(dataset, only=None, txt=False, quiet=False):
    """results/raw/<cohort>/*.csv -> results/processed/<cohort>/*.csv, one pass over
    the cohort's flat result folder.  `only` = filename globs to restrict to.
    Returns the files written."""
    sub = dataset
    src, dst = results_dir(sub, kind="raw"), results_dir(sub, kind="processed")
    written = []
    for path in sorted(glob.glob(os.path.join(src, "*.csv"))):
        name = os.path.basename(path)
        if only and not any(fnmatch(name, pat) for pat in only):
            continue
        spec = spec_for(name)
        if spec is None:
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
        except pd.errors.EmptyDataError:
            continue                      # a stage that found nothing writes an empty file
        out = process_frame(df, spec)
        os.makedirs(dst, exist_ok=True)
        out.to_csv(os.path.join(dst, name), index=False)
        if txt:
            with open(os.path.join(dst, name[:-4] + ".txt"), "w") as f:
                f.write(f"# results/processed/{sub}/{name} | {len(out)} rows\n" + fixed_width(out) + "\n")
        written.append(os.path.join(dst, name))
        if not quiet:
            kb = os.path.getsize(path) / 1024, os.path.getsize(written[-1]) / 1024
            print(f"  {name:40s} {len(df):7d} -> {len(out):6d} rows   {kb[0]:8.0f} -> {kb[1]:6.0f} KB")
    if not quiet:
        total = sum(os.path.getsize(pth) for pth in written) / 1024
        print(f"  {len(written)} files, {total:.0f} KB -> {dst}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="cohort key: eicu, chs, inspire, sandbox, ehrshot, mimiciv")
    ap.add_argument("--only", nargs="+", metavar="GLOB", help="filename glob(s) to export, e.g. 'eval*' cox.csv")
    ap.add_argument("--txt", action="store_true", help="also write fixed-width .txt copies")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    export_dataset(args.dataset, args.only, txt=args.txt, quiet=args.quiet)


if __name__ == "__main__":
    main()
