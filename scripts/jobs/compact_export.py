#!/usr/bin/env python
"""Shrink a cohort's exported results to the numbers the figures actually draw.

    python jobs/compact_export.py --dataset chs
    python jobs/compact_export.py --dataset chs --n_chunks 10    # -> chs_10chunks
    python jobs/compact_export.py --dataset chs_10chunks         # the same folder
    python jobs/compact_export.py --dataset chs --out results/compact/chs

results/processed/<cohort>/ is already the slice the plotting code reads, but it is still
thousands of rows -- fine to copy as files, impossible to photograph.  This writes one
dense table per figure instead: only the methods that are drawn, values rounded to three
significant figures, analytes down the side and methods across the top.  Each table is a
screen or two, so a cohort that can only leave as screenshots still can.

It reads results/processed/<cohort>/ and writes .txt tables plus the same numbers as
.csv, so whoever retypes them has a machine-readable target.  A subset run writes to
`chs_<N>chunks`, and that folder works here like any other cohort -- name it directly or
give --n_chunks.  If export.py has not been run for it yet, results/raw/<cohort>/ is read
instead: the compact tables only ever use columns export keeps, so the numbers are the
same, and it saves running export first just to shrink it again.

Where the raw file carries confidence bounds, they come too, as a companion `<name>_ci`
table of "lo-hi" cells -- an interval beside every estimate would treble the width of a
ten-column table and cost the readability this whole file is for.  The mortality tables
also get an `_n` companion, since a rate without its denominator cannot be pooled.

WHAT IS LOST: the per-analyte detail of anything the figures summarise, the intervals of
figures whose raw file has none (calibration, forecasting, eval, incidence -- the stages
never computed them), and every row a figure filters out.  This is a transcription
channel, not an archive -- keep results/processed/ inside Clalit.
"""
import argparse
import os
import sys
import traceback

import pandas as pd
from math import exp

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS_DIR)
SIG = 3

# The seven method rows the main figures draw (poster_figures.METHODS).  The ablation
# arms stay behind: a table with every NORMA variant does not fit on a screen, and the
# supplementary heatmaps are not the thing being transcribed.
MAIN = ["PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb", "Cohen_m4",
        "NORMA"]


# The forecasting scores are per state query, and the figures name them: the realized
# state, the leak-free marginal, and the normal-state query the reference interval uses.
# CHS's marginal is the frequency-mixed approximation (import_old_norma.py), same role.
NORMA_VARIANTS = (("_oracle", "NORMA_oracle"), ("_marginal_freq", "NORMA_marginal"),
                  ("_marginal", "NORMA_marginal"), ("_normal", "NORMA_normal"))


def collapse_norma(name):
    """Any NORMA_<run> -> NORMA, but keep the state-query variant.

    The run id is not always a hex string -- a cohort scored through the alias carries
    the configured run's name (NORMA_q_age_set) -- so this cannot key off the id's shape.
    Matching on the variant suffix instead keeps _normal / _oracle / _marginal apart
    while everything else folds onto the one name the figures' legends use.
    """
    s = str(name)
    if not s.startswith("NORMA"):
        return s
    for suffix, label in NORMA_VARIANTS:
        if s.endswith(suffix):
            return label
    return "NORMA"


MAIN_RUN = None       # --norma_run: which NORMA_<run> is this cohort's main model


def norma_cols(df, col="method"):
    """NORMA_<run> -> NORMA, so the table has a column named like the figure's legend.

    Only when that is unambiguous.  A cohort carrying the covariate-ablation arms has
    several NORMA_<arm> methods, and folding them all onto one name silently medianed
    four different models into a single row.  There, the names are left alone unless
    --norma_run says which is the main one.
    """
    if col not in df.columns:
        return df
    names = df[col].astype(str)
    runs = {n for n in names.unique() if str(n).startswith("NORMA")}
    bases = {collapse_norma(n) for n in runs}
    if MAIN_RUN:
        df[col] = names.map(lambda n: collapse_norma(n) if n == MAIN_RUN or not
                            str(n).startswith("NORMA") else n)
    elif len(runs) <= len(bases):          # nothing collapses onto anything else
        df[col] = names.map(collapse_norma)
    else:
        print(f"  note: {len(runs)} NORMA runs ({', '.join(sorted(runs))}); leaving the "
              f"names as they are -- pass --norma_run <name> to mark the main one")
    return df


def pivot(df, index, columns, values, keep=MAIN, expect_dup=False):
    if not {index, columns, values} <= set(df.columns):
        return None
    df = norma_cols(df.copy(), columns)
    if keep:
        df = df[df[columns].isin(keep)]
    if not len(df):
        return None
    dup = 0 if expect_dup else df.duplicated([index, columns]).sum()
    if dup:
        print(f"  note: {values} has {dup} duplicate {index} x {columns} rows; taking the median")
    out = df.pivot_table(index=index, columns=columns, values=values, aggfunc="median")
    cols = [c for c in keep if c in out.columns] if keep else _ordered(out.columns)
    return out[cols].apply(lambda s: s.map(lambda v: signif(v)))


def _ordered(cols):
    """Deciles are 1..10, not "1", "10", "2" -- sort numerically when they are numbers."""
    try:
        return sorted(cols, key=lambda c: float(str(c).lstrip("Q").lstrip("D")))
    except ValueError:
        return list(cols)


def ci_pivot(df, index, columns, lo, hi, keep=MAIN, expect_dup=False):
    """The same table as pivot(), each cell "lo-hi" instead of the estimate.

    The interval goes in its own table rather than beside the estimate: putting
    "0.734 (0.71-0.76)" in ten decile columns triples the width and the screenshot
    stops being readable, which is the one thing this file exists to protect.
    """
    if not {lo, hi} <= set(df.columns):
        return None
    a = pivot(df, index, columns, lo, keep, expect_dup)
    b = pivot(df, index, columns, hi, keep, expect_dup)
    return _pair(a, b)


def _pair(a, b):
    if a is None or b is None:
        return None
    out = a.astype(object).copy()
    for c in a.columns:
        out[c] = [fmt_ci(x, y) for x, y in zip(a[c], b[c] if c in b.columns else [None] * len(a))]
    return out


def fmt_ci(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{signif(lo)}-{signif(hi)}"


def signif(v, digits=SIG):
    if pd.isna(v):
        return v
    try:
        return float(f"{float(v):.{digits}g}")
    except (TypeError, ValueError):
        return v


# Which stage writes each source file, so a missing one names the command to run.
STAGE = {"prevalence.csv": "07_classify", "calibration.csv": "06_calibration",
         "conformal.csv": "06_calibration", "forecast.csv": "05_forecasting",
         "variability.csv": "08_variability", "mortality_quintile.csv": "10_mortality",
         "mortality_deviation.csv": "10_mortality", "eval.csv": "12_eval",
         "auroc.csv": "12_eval", "cox.csv": "13_cox", "incidence.csv": "17_outcomes",
         "concordance.csv": "14_patient_level", "nri.csv": "14_patient_level",
         "future_abnormal.csv": "11_lead_time --only future_abnormal",
         "lead_time.csv": "11_lead_time --only lead_time",
         "deviation_score.csv": "12_eval --only deviation",
         "progression.csv": "17_outcomes --progression"}


def read(src, name, missing=None):
    """The one file named `*<name>` in src, or None -- recording which ones were absent.

    `src` is a list of folders in priority order (processed, then raw), searched PER
    FILE.  Taking whole folders in order instead meant a processed/ exported halfway
    through a run hid every result written after it: the stage had run, its file was in
    raw/, and the table still came out missing.

    Every stage writes independently, so a half-run cohort is the normal case, not an
    error.  The names collected here are printed at the end: without that, a stage that
    has not run and a stage whose table came out empty look exactly the same.
    """
    for f, path in sorted(candidates(src)):
        if f.endswith(name):
            try:
                # a stage that wrote a unit string ("mg/dL", a degree sign) leaves a
                # latin-1 byte in an otherwise utf-8 file; that is a bad encoding, not a
                # missing table, so fall back rather than skipping the whole file
                try:
                    return pd.read_csv(path, keep_default_na=False, na_values=[""])
                except UnicodeDecodeError:
                    return pd.read_csv(path, keep_default_na=False, na_values=[""],
                                       encoding="latin-1")
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                # a stage killed mid-write leaves a header-less or half-written file;
                # that is the same situation as not having run, not a reason to stop
                print(f"  unreadable, skipping: {f} ({e.__class__.__name__})")
                break
    if missing is not None:
        missing.append(name)
    return None


import contextlib


@contextlib.contextmanager
def guard(label):
    """One unbuildable table must not cost the other forty.

    Each source file is read and reshaped independently, so a column a cohort never
    wrote, or a shape only one cohort produces, ends as a named skip in the output
    rather than an exception that loses the whole export.
    """
    try:
        yield
    except Exception as e:
        print(f"  skipped {label}: {e.__class__.__name__}: {e}")


def tables(src, missing=None):
    """{name: frame} -- one entry per figure, each small enough to photograph."""
    out = {}
    read_ = lambda name: read(src, name, missing)

    with guard("prevalence"):
        prev = read_("prevalence.csv")
        if prev is not None:
            p = prev[prev["subset"].astype(str) == "all"]
            for metric, label in (("pct", "prevalence"), ("reclass_pct", "reclassification")):
                long = melt_wide(p, metric)
                if long is not None:
                    out[label] = long
                    # patient-cluster bootstrap bounds, so they cannot be recomputed from n.
                    # NOT the legacy import's _q25/_q75: those are the spread ACROSS
                    # analytes on a pooled row, which is not this row's uncertainty.
                    out[f"{label}_ci"] = _pair(melt_wide(p, f"{metric}_lo"),
                                               melt_wide(p, f"{metric}_hi"))

    with guard("calibration"):
        cal = read_("calibration.csv")
        if cal is not None:
            med = cal[cal["analyte"].astype(str) == "median"]
            out["calibration"] = pivot(med, "state", "method", "coverage95")
            out["interval_width"] = pivot(med, "state", "method", "width_rel")
            # 06_calibration writes no bounds; n is what a Wilson interval needs
            out["calibration_n"] = pivot(med, "state", "method", "n")
            # the figure drops the median row and takes an IQR across analytes, so the
            # per-analyte rows are the ones it actually draws -- three states, three tables
            per_a = cal[cal["analyte"].astype(str) != "median"]
            for col, state, label in (("coverage95", "normal", "calibration_analyte"),
                                      ("width_rel", "all", "interval_width_analyte"),
                                      ("inside_pop", "all", "inside_pop_analyte")):
                if col in per_a.columns:
                    out[label] = pivot(per_a[per_a["state"].astype(str) == state],
                                       "analyte", "method", col)

    with guard("conformal"):
        conf = read_("conformal.csv")
        if conf is not None:
            c = norma_cols(conf[conf["analyte"].astype(str) == "median"].copy())
            out["conformal"] = c.set_index("method")[
                [c_ for c_ in ("gamma", "coverage_raw", "width_rel_raw", "coverage_cal", "width_rel_cal")
                 if c_ in c.columns]].apply(lambda s: s.map(signif))

    with guard("forecast"):
        fc = read_("forecast.csv")
        if fc is not None:
            f = fc[(fc["analyte"].astype(str) == "median")]
            if "target_state" in f.columns:
                f = f[f["target_state"].astype(str) == "all_states"]
            cols = [c for c in ("mae", "mape", "r2", "n") if c in f.columns]
            out["forecasting"] = norma_cols(f.copy()).set_index("method")[cols].apply(
                lambda s: s.map(signif))
            # 05_forecasting computes no intervals, and the figure's spread is the weighted
            # SD across analytes -- so the per-analyte rows are the only way to show one
            per = norma_cols(fc[fc["analyte"].astype(str) != "median"].copy())
            if "target_state" in per.columns:
                per = per[per["target_state"].astype(str) == "all_states"]
            for metric in ("mape", "r2", "n"):
                if metric in per.columns:
                    out[f"forecast_{metric}_analyte"] = pivot(per, "analyte", "method", metric,
                                                              keep=None)

    with guard("variability"):
        var = read_("variability.csv")
        if var is not None:
            cols = [c for c in ("cv_intra", "cv_inter", "individuality") if c in var.columns]
            if cols:
                out["variability"] = var.set_index("analyte")[cols].apply(lambda s: s.map(signif))
                bounds = {"cv_intra": ("cv_intra_ci_lower", "cv_intra_ci_upper"),
                          "cv_inter": ("cv_inter_ci_lower", "cv_inter_ci_upper"),
                          "individuality": ("ii_ci_lower", "ii_ci_upper")}
                v = var.set_index("analyte")
                ci = pd.DataFrame({c: [fmt_ci(l, h) for l, h in zip(v[bounds[c][0]], v[bounds[c][1]])]
                                   for c in cols if set(bounds[c]) <= set(v.columns)}, index=v.index)
                out["variability_ci"] = ci if len(ci.columns) else None

    with guard("mortality_quintile"):
        mq = read_("mortality_quintile.csv")
        if mq is not None:
            q = mq.assign(method="PopRI")
            out["mortality_quintile"] = pivot(q, "analyte", "q_label", "mortality_pct", keep=None)
            out["mortality_quintile_n"] = pivot(q, "analyte", "q_label", "n", keep=None)
            out["mortality_quintile_ci"] = ci_pivot(q, "analyte", "q_label", "ci_lo", "ci_hi",
                                                    keep=None)

    with guard("mortality_deviation"):
        md = read_("mortality_deviation.csv")
        if md is not None:
            base = md[md["method"].astype(str) == "baseline_z"].assign(method="baseline_z")
            out["mortality_deviation"] = pivot(base, "analyte", "decile", "mortality_pct", keep=None)
            out["mortality_deviation_n"] = pivot(base, "analyte", "decile", "n", keep=None)
            out["mortality_deviation_ci"] = ci_pivot(base, "analyte", "decile", "ci_lo", "ci_hi",
                                                     keep=None)

    with guard("eval"):
        ev = read_("eval.csv")
        if ev is not None and "outcome" in ev.columns:
            e = ev[ev["analyte"].astype(str).isin(["median", "pooled"])]
            for metric in ("ppv", "npv", "sensitivity", "specificity", "auroc"):
                if metric in e.columns:
                    t = pivot(e, "outcome", "method", metric)
                    if t is not None:
                        out[f"eval_{metric}"] = t
            for count, label in (("n", "eval_n"), ("n_events", "eval_events")):
                if count in e.columns:
                    out[label] = pivot(e, "outcome", "method", count)

    with guard("auroc"):
        au = read_("auroc.csv")
        if au is not None and "analyte" in au.columns:
            a = au[(au["analyte"].astype(str) == "Overall")]
            if "subset" in a.columns:
                a = a[a["subset"].astype(str) == "all"]
            out["auroc"] = pivot(a, "outcome", "method", "auc")
            out["auroc_ci"] = ci_pivot(a, "outcome", "method", "auc_lo", "auc_hi")

    with guard("cox"):
        cox = read_("cox.csv")
        if cox is not None and {"HR", "outcome", "analyte", "method"} <= set(cox.columns):
            c = cox
            for col, val in (("subset", "all"), ("encoding", "binary"), ("exposure", "first"),
                             # raw carries every level (abnormal / high / low / z); the
                             # binary figure draws "abnormal", and medianing the four
                             # together is not any model's hazard ratio
                             ("level", "abnormal")):
                if col in c.columns:
                    c = c[c[col].astype(str) == val]
            c = norma_cols(c.copy())
            c = c[c["method"].isin(MAIN)]
            # the figure draws the strongest analytes, not all thirty
            top = (c[c["method"] == "NORMA"].groupby(["outcome", "analyte"])["HR"].median()
                   .reset_index().sort_values("HR", ascending=False).groupby("outcome").head(10))
            keep = set(map(tuple, top[["outcome", "analyte"]].to_numpy()))
            c = c[[tuple(x) in keep for x in c[["outcome", "analyte"]].to_numpy()]]
            for outcome, g in c.groupby("outcome"):
                t = pivot(g, "analyte", "method", "HR")
                if t is not None:
                    out[f"cox_{outcome}"] = t
                    out[f"cox_{outcome}_ci"] = ci_pivot(g, "analyte", "method",
                                                        "HR_lower", "HR_upper")

    with guard("future_abnormal"):
        fa = read_("future_abnormal.csv")
        if fa is not None and "rr" in fa.columns:
            band = fa.get("age_band", pd.Series("all", index=fa.index)).astype(str)
            sex = fa.get("sex", pd.Series("all", index=fa.index)).astype(str)
            # Every row is one endpoint: direction high / low per analyte, and "all" on
            # the analyte="median" row that pools them.  An earlier version filtered on
            # direction == "toward_bound" -- a value 11_lead_time stopped writing when
            # the design changed -- which silently emptied every table below.
            flat = fa[(band == "all") & (sex == "all")]
            per = norma_cols(flat[flat["analyte"].astype(str) != "median"].copy())
            # the figures take the median across analytes with an IQR band, so the pooled
            # table has to carry the quartiles -- they cannot be rebuilt from a median
            metrics = [c for c in ("rr", "auc", "auc_within_position", "auprc_norm") if c in per.columns]
            if metrics:
                g = per.groupby("method")
                summary = g[metrics].median()
                for m in metrics:
                    summary[f"{m}_q25"] = g[m].quantile(.25)
                    summary[f"{m}_q75"] = g[m].quantile(.75)
                summary = summary.reindex([m for m in MAIN if m in summary.index])
                # long, not wide: four metrics x three columns is 170 characters across,
                # which is exactly the screenshot this file exists to avoid
                out["future_abnormal"] = pd.concat(
                    {m: summary[[m, f"{m}_q25", f"{m}_q75"]].set_axis(["median", "q25", "q75"], axis=1)
                     for m in metrics}, names=["metric"]).apply(lambda s_: s_.map(signif))
            out["future_abnormal_rr"] = pivot(per, "analyte", "method", "rr", expect_dup=True)
            if "auc" in per.columns:
                out["future_abnormal_auc"] = pivot(per, "analyte", "method", "auc", expect_dup=True)
                out["future_abnormal_auc_ci"] = ci_pivot(per, "analyte", "method", "auc_lo", "auc_hi",
                                                         expect_dup=True)
            # the age rows are crossed with sex in some cohorts, so the band curve is the
            # median over analytes AND sexes, which is what the figure plots
            aged = norma_cols(fa[band != "all"].copy())
            # the age figure is per analyte AND endpoint -- its four panels are
            # "K low", "NA low", "BUN high", "WBC high" -- so the pooled curve above
            # cannot fill it and each panel needs its own table
            for a, d in (("K", "low"), ("NA", "low"), ("BUN", "high"), ("WBC", "high")):
                one = aged[(aged["analyte"].astype(str) == a)
                           & (aged["direction"].astype(str) == d)]
                if len(one):
                    out[f"future_age_{a}_{d}"] = pivot(one, "age_band", "method", "rr",
                                                       expect_dup=True)
            # The age curve wants a band, and a median of per-analyte RRs has no interval
            # attached to it -- which is why CHS was the one age panel on the poster drawn
            # as a bare line. The interval comes from the per-band counts, pooled across
            # analyte x endpoint, the same way 11_lead_time._future_abnormal_age_frame
            # builds it for the hospital cohorts.
            #
            # Two groups of columns, and they are needed for different things, so a file
            # that has one and not the other still gets what it can:
            #   tp/fp/fn/tn                      -> the log-scale SE, and so the interval
            #   tp_norm/fp_norm/pop_pos/pop_neg  -> the population-weighted point estimate
            per_age = aged[aged["analyte"].astype(str) != "median"]
            if not len(per_age):
                # a cohort that only ever shipped the pooled age curve (a re-export of an
                # imported cohort is the usual case) has nothing but analyte="median" rows;
                # dropping them left the age tables missing with nothing said
                per_age = aged
            se_counts = ["tp", "fp", "fn", "tn"]
            wt_counts = ["tp_norm", "fp_norm", "pop_pos", "pop_neg"]
            have = set(per_age.columns)
            missing = [c for c in se_counts if c not in have]
            if len(per_age) and not missing:
                cols = se_counts + [c for c in wt_counts if c in have]
                g = per_age.groupby(["method", "age_band"], observed=True)[cols].sum().reset_index()
                n = g[se_counts].sum(axis=1)
                if not set(wt_counts) - have:
                    g["rr"] = ((g.tp_norm / (g.tp_norm + g.fp_norm))
                               / (g.pop_pos / (g.pop_pos + g.pop_neg)))
                else:
                    # no population weights in this file: keep the curve the cohort already
                    # shipped (the median over analytes) and hang the counts-based interval
                    # off it, rather than dropping the band for want of a better centre
                    print("  note: future_abnormal_age has no population weights "
                          f"({', '.join(sorted(set(wt_counts) - have))}); "
                          "the curve stays the median over analytes")
                    med = per_age.groupby(["method", "age_band"], observed=True)["rr"].median()
                    g["rr"] = g.set_index(["method", "age_band"]).index.map(med)
                se = ((1 / g.tp - 1 / (g.tp + g.fp) + 1 / (g.tp + g.fn) - 1 / n)
                      .clip(lower=0) ** 0.5)
                g["rr_lo"], g["rr_hi"] = g.rr * (-1.96 * se).apply(exp), g.rr * (1.96 * se).apply(exp)
                g = g[(g.tp >= 20) & g.rr.notna()]          # RR SE ~ 1/sqrt(tp)
                out["future_abnormal_age"] = pivot(g, "age_band", "method", "rr", expect_dup=True)
                out["future_abnormal_age_ci"] = ci_pivot(g, "age_band", "method", "rr_lo", "rr_hi",
                                                         expect_dup=True)
            elif len(per_age):
                # Say so. A band that quietly fails to appear is indistinguishable from a
                # cohort that has no age data, and that is how it went unnoticed before.
                print(f"  note: future_abnormal_age has no interval: {', '.join(missing)} "
                      "not in this cohort's raw file; re-run 11_lead_time --only future_abnormal")
                out["future_abnormal_age"] = pivot(per_age, "age_band", "method", "rr",
                                                   expect_dup=True)
            else:
                print("  note: no age_band rows in future_abnormal.csv, so no age tables")

    with guard("lead_time"):
        lt = read_("lead_time.csv")
        if lt is not None and "median_lead_h_early" in lt.columns:
            d = norma_cols(lt.copy())
            if "anchor" in d.columns:        # every method at the standard of care's alert rate
                d = d[d["anchor"].astype(str) == "soc_rate"]
            out["lead_time"] = pivot(d, "analyte", "method", "median_lead_h_early")
            out["lead_time_iqr"] = ci_pivot(d, "analyte", "method",
                                            "iqr25_lead_h_early", "iqr75_lead_h_early")
            for col, label in (("detected_24h_before", "lead_time_detected_24h"),
                               ("n_events", "lead_time_events")):
                if col in d.columns:
                    out[label] = pivot(d, "analyte", "method", col)

    with guard("deviation_score"):
        dev = read_("deviation_score.csv")
        if dev is not None:
            d = norma_cols(dev.copy())
            # raw holds every landmark x analyte group x subset x score; the figure draws
            # one cell of that -- the pooled group, the toward-bound score, at the
            # largest landmark, among patients normal by Pop_RI
            if "analyte" in d.columns:
                d = d[d["analyte"].astype(str) == "all"]
            if "score" in d.columns:
                d = d[d["score"].astype(str) == "toward"]
            if "subset" in d.columns:
                keep = "pop_normal" if (d["subset"].astype(str) == "pop_normal").any() else "all"
                d = d[d["subset"].astype(str) == keep]
            if "landmark_h" in d.columns and d["landmark_h"].notna().any():
                d = d[d["landmark_h"] == pd.to_numeric(d["landmark_h"], errors="coerce").max()]
            # the figures' operating point is the top 10%: rr_at_10, not whichever
            # rr_at_* column happens to come first (rr_at_05)
            rr_cols = [c for c in d.columns
                       if c.startswith("rr_at_") and not c.endswith(("_lo", "_hi"))]
            rr = "rr_at_10" if "rr_at_10" in rr_cols else (rr_cols[0] if rr_cols else None)
            if rr and "outcome" in d.columns:
                out["outcome_rr"] = pivot(d, "outcome", "method", rr)          # top 10% flagged
                out["outcome_rr_ci"] = ci_pivot(d, "outcome", "method", f"{rr}_lo", f"{rr}_hi")
            if "auc" in d.columns:
                out["outcome_auc"] = pivot(d, "outcome", "method", "auc")
                out["outcome_auc_ci"] = ci_pivot(d, "outcome", "method", "auc_lo", "auc_hi")

    with guard("incidence"):
        inc = read_("incidence.csv")
        if inc is not None:
            rr = [c for c in inc.columns if c.startswith("rr_")]
            if rr and "analyte" in inc.columns:
                i = inc
                if "anchor" in i.columns:
                    i = i[i["anchor"].astype(str) == "sensitivity"]
                out["incidence"] = pivot(i, "outcome", "method", rr[-1])

    with guard("concordance"):
        con = read_("concordance.csv")
        if con is not None and "concordance_test" in con.columns:
            c = con
            if "subset" in c.columns:
                c = c[c["subset"].astype(str) == "all"]
            c = norma_cols(c.copy())
            out["concordance"] = pivot(c, "outcome", "method", "concordance_test")
            out["concordance_ci"] = ci_pivot(c, "outcome", "method",
                                             "concordance_lower", "concordance_upper")

    with guard("nri"):
        nri = read_("nri.csv")
        if nri is not None and "NRI" in nri.columns:
            n_ = norma_cols(nri.copy(), "new_method")
            keep = [c for c in ("outcome", "new_method", "NRI", "NRI_se", "NRI_p") if c in n_.columns]
            out["nri"] = (n_[n_["new_method"].isin(MAIN)][keep]
                          .set_index([c for c in ("outcome", "new_method") if c in keep])
                          .apply(lambda s: s.map(signif)))

    with guard("progression"):
        prog = read_("progression.csv")
        if prog is not None and "arm" in prog.columns:
            inc_cols = [c for c in prog.columns if c.startswith("inc_")]
            if inc_cols:
                p = norma_cols(prog.copy())
                p = p[p["method"].isin(["NORMA", "PopRI"])]
                out["progression"] = (p.pivot_table(index=["analyte", "arm"], columns="method",
                                                    values=inc_cols[-1], aggfunc="median")
                                      .apply(lambda s: s.map(signif)))

    # an all-blank CI table means the raw file has the columns but no values in the rows
    # the figure draws (the legacy CHS prevalence import put its quartiles on a pooled
    # row only) -- writing it would look like a result
    return {k: v for k, v in out.items()
            if v is not None and len(v) and not _blank(v)}


def _blank(t):
    return bool(((t == "") | t.isna()).all().all())     # no DataFrame.map before pandas 2.1


def melt_wide(df, metric):
    """prevalence.csv is wide (<method>_pct); make it analytes x methods.

    "<method>_reclass_pct" also ends in "_pct", so asking for the prevalence columns
    picks up the reclassification ones as a second, wrongly-named column of the same
    method -- hence the explicit reject.  Likewise only the first NORMA run is kept: the
    ablation arms all rename to "NORMA" and duplicate labels break the pairing below.
    """
    cols = {c[:-len(f"_{metric}")]: c for c in df.columns if c.endswith(f"_{metric}")}
    keep, seen = {}, set()
    for m, c in cols.items():
        if m.endswith("_reclass") and not metric.startswith("reclass"):
            continue
        label = "NORMA" if m.startswith("NORMA") else m
        if label not in MAIN or label in seen:
            continue
        seen.add(label)
        keep[label] = c
    if not keep:
        return None
    out = df.set_index("analyte")[[keep[m] for m in keep]]
    out.columns = list(keep)
    return out[[c for c in MAIN if c in out.columns]].apply(lambda s: s.map(signif))


def candidates(dirs):
    """[(filename, full path)] over every source folder, first folder winning a tie."""
    seen, out = set(), []
    for d in dirs:
        for f in os.listdir(d):
            # "._11_future_abnormal.csv" is the AppleDouble sidecar macOS writes beside a
            # file on a non-native mount. It sorts BEFORE the real table and ends with the
            # same name, so read_() matched it first and every table from that stage went
            # missing -- silently, because a binary sidecar reads as a frame with no
            # columns rather than as an error.
            if f.startswith("."):
                continue
            if f not in seen:
                seen.add(f)
                out.append((f, os.path.join(d, f)))
    return out


def resolve(dataset, n_chunks=None):
    """(folder name, [source dirs]) -- processed first, then raw, both searched.

    A subset run lives in `<cohort>_<N>chunks`; accept either the full name or
    --n_chunks, so this matches whatever jobs/run_chs.py wrote.
    """
    name = dataset
    if n_chunks and not name.endswith("chunks"):
        name = f"{dataset}_{n_chunks}chunks"
    dirs = [os.path.join(ROOT, "results", kind, name) for kind in ("processed", "raw")]
    dirs = [d for d in dirs if os.path.isdir(d) and any(f.endswith(".csv") for f in os.listdir(d))]
    if dirs:
        return name, dirs
    have = []
    for kind in ("processed", "raw"):
        d = os.path.join(ROOT, "results", kind)
        if os.path.isdir(d):
            have += [f"{kind}/{f}" for f in sorted(os.listdir(d))
                     if os.path.isdir(os.path.join(d, f))]
    raise SystemExit(f"No results for '{name}'. Found: {', '.join(have) or 'nothing'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="chs", help="cohort folder, e.g. chs or chs_10chunks")
    p.add_argument("--n_chunks", type=int, default=None,
                   help="subset run: read <dataset>_<N>chunks")
    p.add_argument("--norma_run", default=None,
                   help="which NORMA_<run> is the main model (only needed when a cohort "
                        "carries several NORMA arms)")
    p.add_argument("--src", default=None, help="override the source folder")
    p.add_argument("--out", default=None, help="where to write (default results/compact/<dataset>)")
    args = p.parse_args()
    global MAIN_RUN
    MAIN_RUN = args.norma_run

    if args.src:
        name, src = os.path.basename(args.src.rstrip("/")), [args.src]
        if not os.path.isdir(src[0]):
            raise SystemExit(f"No {src[0]}")
    else:
        name, src = resolve(args.dataset, args.n_chunks)
    out_dir = args.out or os.path.join(ROOT, "results", "compact", name)
    os.makedirs(out_dir, exist_ok=True)
    for d in src:
        print(f"  reading {d}")
    print(f"  writing {out_dir}")

    # BUILD FIRST, then clear.  Clearing up front meant one failing table wiped a whole
    # export that had already been screenshotted, with nothing to fall back on.
    missing = []
    try:
        built = tables(src, missing)
    except Exception as e:
        traceback.print_exc()
        raise SystemExit(f"\n  Build failed ({e.__class__.__name__}: {e}).\n"
                         f"  {out_dir} is untouched -- whatever was there is still there.")
    if not built:
        raise SystemExit(f"Nothing to write from {', '.join(src)} -- no result file this "
                         f"reads yet ({', '.join(missing)})")

    # only now is it safe to drop the previous run's tables; a table this run no longer
    # builds would otherwise sit there contradicting its neighbours
    for f in os.listdir(out_dir):
        if f.endswith((".txt", ".csv")):
            os.remove(os.path.join(out_dir, f))
    total = 0
    for table_name, table in built.items():
        text = table.to_string()
        lines = text.count("\n") + 1
        width = max(len(l) for l in text.split("\n"))
        total += lines
        with open(os.path.join(out_dir, f"{table_name}.txt"), "w") as f:
            f.write(f"{name}  {table_name}\n{text}\n")
        table.to_csv(os.path.join(out_dir, f"{table_name}.csv"))
        print(f"  {table_name:22s} {lines:4d} lines x {width:3d} chars")
    print(f"\n  {len(built)} tables, {total} lines total -> {out_dir}")
    if missing:
        print(f"  not written yet ({len(missing)}): " +
              ", ".join(f"{m[:-4]} [{STAGE.get(m, '?')}]" for m in missing))
        print("  run those stages and this again -- it rewrites whatever is there now")
    print("  one screenshot per table at a readable font; the .csv beside each is the "
          "same numbers for retyping")


if __name__ == "__main__":
    main()
