#!/usr/bin/env python
"""Turn compact tables back into the long files the figures read.

    python jobs/compact_import.py --dataset chs_10chunks
    python jobs/compact_import.py --dataset chs_10chunks --dest chs     # draw it as CHS

jobs/compact_export.py shrinks a cohort's results to tables small enough to photograph;
this is the other end of that channel, for when the numbers came back as screenshots and
were retyped into results/compact/<name>/.  It writes results/processed/<dest>/, so the
plotting code can read them like any other cohort's results.

WHAT THESE FILES ARE NOT: a pipeline output.  They carry three significant figures, only
the rows the figures draw, and only the columns the compact tables kept -- everything
else is absent, not zero.  A PROVENANCE.txt saying so is written beside them.  Anything
that needs the full results has to run inside the cohort.
"""
import argparse
import os

import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS_DIR)


def read(src, name):
    path = os.path.join(src, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df.set_index(df.columns[0])


def split_ci(frame):
    """A frame of "lo-hi" strings -> (lo, hi) frames of numbers."""
    if frame is None:
        return None, None
    def half(which):
        return frame.apply(lambda s: s.map(
            lambda v: float(str(v).split("-")[which]) if isinstance(v, str) and "-" in str(v)
            else float("nan")))
    return half(0), half(1)


def long(frame, index_name, value, methods=None):
    """analyte x method table -> one row per (index, method)."""
    if frame is None:
        return None
    d = frame.reset_index().melt(id_vars=frame.index.name or "index",
                                 var_name="method", value_name=value).dropna(subset=[value])
    d.columns = [index_name, "method", value]
    return d[d["method"].isin(methods)] if methods else d


def merge(*frames):
    """Join the long frames that share (index, method), keeping every value column."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=list(out.columns[:2]), how="outer")
    return out


def prevalence(src):
    est, ci = read(src, "prevalence"), read(src, "prevalence_ci")
    if est is None:
        return None
    lo, hi = split_ci(ci)
    rec, rec_ci = read(src, "reclassification"), read(src, "reclassification_ci")
    rlo, rhi = split_ci(rec_ci)
    out = pd.DataFrame(index=est.index)
    for m in est.columns:
        out[f"{m}_pct"] = est[m]
        if lo is not None and m in lo:
            out[f"{m}_pct_lo"], out[f"{m}_pct_hi"] = lo[m], hi[m]
    if rec is not None:
        for m in rec.columns:
            out[f"{m}_reclass_pct"] = rec[m]
            if rlo is not None and m in rlo:
                out[f"{m}_reclass_pct_lo"], out[f"{m}_reclass_pct_hi"] = rlo[m], rhi[m]
    out = out.reset_index().rename(columns={out.index.name or "index": "analyte"})
    out.insert(0, "subset", "all")

    # the figures draw the Overall row's median with an IQR ACROSS analytes; that spread
    # is not in the compact tables but it is exactly the per-analyte column's quartiles,
    # so recompute it rather than leave every cohort marker whiskerless
    per = out[out["analyte"].astype(str) != "Overall"]
    row = out["analyte"].astype(str) == "Overall"
    if row.any():
        for c in [c for c in out.columns if c.endswith("_pct")]:
            v = pd.to_numeric(per[c], errors="coerce").dropna()
            if len(v):
                out.loc[row, [f"{c}_median", f"{c}_q25", f"{c}_q75"]] = [
                    v.median(), v.quantile(.25), v.quantile(.75)]
    return out


def calibration(src):
    cov, width, n = read(src, "calibration"), read(src, "interval_width"), read(src, "calibration_n")
    m = merge(long(cov, "state", "coverage95"), long(width, "state", "width_rel"),
              long(n, "state", "n"))
    rows = []
    if m is not None:
        m["analyte"] = "median"
        rows.append(m)
    # the figure draws an IQR across analytes and ignores the median row, so without
    # these the cohort's panel is empty however good the pooled numbers are
    for label, col, state in (("calibration_analyte", "coverage95", "normal"),
                              ("interval_width_analyte", "width_rel", "all"),
                              ("inside_pop_analyte", "inside_pop", "all")):
        a = long(read(src, label), "analyte", col)
        if a is not None:
            rows.append(a.assign(state=state))
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    # one row per method x state x analyte, whichever table each column came from
    out = out.groupby(["method", "state", "analyte"], as_index=False).first()
    for c in ("n", "coverage95", "width_rel", "inside_pop"):
        if c not in out.columns:
            out[c] = float("nan")        # the loader coerces these by name
    return out


def conformal(src):
    c = read(src, "conformal")
    if c is None:
        return None
    out = c.reset_index().rename(columns={c.index.name or "index": "method"})
    out["analyte"] = "median"
    return out


def future_abnormal(src):
    rr, auc = read(src, "future_abnormal_rr"), read(src, "future_abnormal_auc")
    alo, ahi = split_ci(read(src, "future_abnormal_auc_ci"))
    per = merge(long(rr, "analyte", "rr"), long(auc, "analyte", "auc"),
                long(alo, "analyte", "auc_lo"), long(ahi, "analyte", "auc_hi"))
    rows = []
    if per is not None:
        # the figures select on this; compact_export keeps only these rows
        per["age_band"], per["sex"], per["direction"] = "all", "all", "endpoints"
        rows.append(per)

    # the pooled row: the medians ACROSS analyte x endpoint, which is what the figures
    # label as the cohort's value.  It is not recoverable from the per-analyte table
    # above (that one is already collapsed over direction), so it travels separately.
    pooled = read(src, "future_abnormal")
    if pooled is not None:
        p = pooled.reset_index()
        p.columns = ["metric"] + list(p.columns[1:])
        wide = p.pivot_table(index="method", columns="metric",
                             values=[c for c in ("median", "q25", "q75") if c in p.columns])
        wide.columns = [m if stat == "median" else f"{m}_{stat}" for stat, m in wide.columns]
        wide = wide.reset_index().assign(analyte="median", age_band="all", sex="all",
                                         direction="all")
        rows.append(wide)

    age = read(src, "future_abnormal_age")
    if age is not None:
        # the band on the age curve: present once the cohort's raw file carried the
        # per-band counts, absent on an older export, and the figure checks for it
        lo, hi = split_ci(read(src, "future_abnormal_age_ci"))
        a = merge(long(age, "age_band", "rr"), long(lo, "age_band", "rr_lo"),
                  long(hi, "age_band", "rr_hi"))
        a["analyte"], a["sex"], a["direction"] = "median", "all", "endpoints"
        rows.append(a)
    return pd.concat(rows, ignore_index=True) if rows else None


def lead_time(src):
    med = read(src, "lead_time")
    lo, hi = split_ci(read(src, "lead_time_iqr"))
    out = merge(long(med, "analyte", "median_lead_h_early"),
                long(lo, "analyte", "iqr25_lead_h_early"),
                long(hi, "analyte", "iqr75_lead_h_early"),
                long(read(src, "lead_time_detected_24h"), "analyte", "detected_24h_before"),
                long(read(src, "lead_time_events"), "analyte", "n_events"))
    if out is None:
        return None
    out["outcome"], out["anchor"] = "pop_abnormal", "soc_rate"
    out["age_band"], out["target_rate"] = "all", "all"
    return out


def forecast(src):
    """05_forecast.csv: the per-analyte rows the figure weights by n, plus the pooled row."""
    rows = []
    per = merge(long(read(src, "forecast_mape_analyte"), "analyte", "mape"),
                long(read(src, "forecast_r2_analyte"), "analyte", "r2"),
                long(read(src, "forecast_n_analyte"), "analyte", "n"))
    if per is not None:
        rows.append(per[per["analyte"].astype(str) != "pooled"])
    pooled = read(src, "forecasting")
    if pooled is not None:
        p = pooled.reset_index()
        p.columns = ["method"] + list(p.columns[1:])
        rows.append(p.assign(analyte="median"))
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out["target_state"] = "all_states"      # the figure selects on this
    return out


def auroc(src):
    est = read(src, "auroc")
    lo, hi = split_ci(read(src, "auroc_ci"))
    out = merge(long(est, "outcome", "auc"), long(lo, "outcome", "auc_lo"),
                long(hi, "outcome", "auc_hi"))
    if out is None:
        return None
    out["subset"], out["analyte"] = "all", "Overall"
    return out


BUILDERS = {"05_forecast.csv": forecast, "07_prevalence.csv": prevalence, "06_calibration.csv": calibration,
            "06_conformal.csv": conformal, "11_future_abnormal.csv": future_abnormal,
            "11_lead_time.csv": lead_time, "12_auroc.csv": auroc}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="chs_10chunks", help="folder under results/compact/")
    p.add_argument("--dest", default=None, help="cohort to write as (default: same name)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    src = os.path.join(ROOT, "results", "compact", args.dataset)
    if not os.path.isdir(src):
        raise SystemExit(f"No {src}")
    dest = os.path.join(ROOT, "results", "processed", args.dest or args.dataset)
    os.makedirs(dest, exist_ok=True)
    print(f"  {src} -> {dest}")

    wrote = []
    for name, build in BUILDERS.items():
        frame = build(src)
        if frame is None or not len(frame):
            print(f"  {name:26s} skipped (no compact tables for it)")
            continue
        path = os.path.join(dest, name)
        # figlib.find_result reads results/raw/ BEFORE results/processed/, so a stale
        # file in raw silently wins over everything written here.  Move it aside, or
        # the figures keep drawing the old numbers and nothing says why.
        raw = os.path.join(ROOT, "results", "raw", args.dest or args.dataset, name)
        if os.path.exists(raw) and not args.dry_run:
            os.rename(raw, raw + ".bak-superseded")
            print(f"  {'':26s}        moved results/raw/.../{name} aside (it would shadow this)")

        note = ""
        if os.path.exists(path) and not os.path.exists(path + ".bak-pipeline"):
            # keep the real pipeline output the FIRST time this replaces it;;a second
            # run would otherwise back up the retyped file over the genuine one
            os.rename(path, path + ".bak-pipeline")
            note = "  (pipeline file kept as .bak-pipeline)"
        elif os.path.exists(path):
            note = "  (overwriting an earlier import)"
        print(f"  {name:26s} {len(frame):5d} rows{note}")
        if not args.dry_run:
            frame.to_csv(path, index=False)
        wrote.append(name)

    if wrote and not args.dry_run:
        with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
            f.write(f"Rebuilt by jobs/compact_import.py from results/compact/{args.dataset}/.\n"
                    f"Those tables were transcribed from screenshots of the cohort's own run, so:\n"
                    f"  - every number is rounded to three significant figures\n"
                    f"  - only the rows and columns the figures draw are here\n"
                    f"  - a missing column is missing, not zero\n"
                    f"Files: {', '.join(wrote)}\n")
    print(f"\n  {len(wrote)} file(s) written. Anything needing full results must run in the cohort.")


if __name__ == "__main__":
    main()
