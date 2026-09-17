#!/usr/bin/env python
"""Inside Clalit: put the old NORMA intervals back into each chunk from the server's edits files.

The norma_<run> rows used to sit inside chunk_*/ref_intervals.parquet.  Rebuilding the
baselines there rewrote the file with the baseline rows only, so they were lost
(datasets.write_ref_intervals now moves such rows aside instead).  The model's
predictions survive on the server, one file per chunk:

    <edits_dir>/norma_<i>.parquet    pid, cid, code, x_next, t_next, s_next, state, mu, std

This turns them back into ref_intervals rows and writes them to
chunk_<i>/ref_intervals_norma.parquet, the file every CHS reader already looks in.
ref_intervals.parquet (the baselines) is never opened for writing.

This file is self-contained: copy just this one file into Clalit (anywhere -- it finds
data/clalit next to itself, or takes --data_root) and run it with pandas available.

    python import_old_norma.py --n_chunks 2 --dry_run          # report only, writes nothing
    python import_old_norma.py --n_chunks 2
    python import_old_norma.py                                # every chunk

State: model/edit.py (get_predictions_cf) scored every target twice, once with s_next
forced to ones (state=True, the NORMAL state) and once forced to zeros (state=False).
Only the state=True rows are kept, so the interval is the normal-state prediction.
Each chunk is checked for that layout, and a pair with several normal-state targets
keeps its earliest (t_next), as 04_refs keeps the first index target.

Interval: mu +/- z * std with z = 1.96, the half-width the Cohen and Gaussian arms use;
ri_mean = mu and ri_std = std, so z-score analyses work.  --z 2 reproduces the rows
01_process built.  sex / age / n_bl / t_span come from the chunk's pop rows, and only
pairs that have a pop row get a NORMA row.

Chunk numbers must line up with the edits file numbers.  The report prints the share of
each chunk's pop pairs that found a prediction.  A chunk below --min_match (default
0.5) is not written, because a low share usually means the numbering does not line up.

Afterwards run the pipeline with `jobs/run_clalit.py --reuse_refs`, which scores these
rows as NORMA and writes the run id into results/processed/<cohort>/NORMA_RUN.txt.
"""
import argparse
import glob
import os
import re
import shutil

import numpy as np
import pandas as pd

# Self-contained on purpose: this is the one file to copy into Clalit, so it repeats
# the few names it needs from scripts/lib/datasets.py rather than importing it.
EDITS_DIR = r"\\10.100.117.220\Projects$\R01-MainResearch\R01-Aashna\temp\NORMA_v2\model\logs\87345aff\edits"
RUN_ID = "87345aff"
REF_FILE = "ref_intervals.parquet"
REF_NORMA_FILE = "ref_intervals_norma.parquet"
REF_COLUMNS = ["patient_id", "analyte", "sex", "age", "n_bl", "t_span",
               "method", "ri_mean", "ri_std", "ri_low", "ri_high"]
EDIT_COLUMNS = ["pid", "code", "t_next", "state", "mu", "std"]
PRED_COLUMNS = EDIT_COLUMNS + ["x_next", "s_next"]
MAIN_RUN = "q_age_set"                    # datasets.NORMA_RUN_ID: what 05_forecasting asks for
PRED_FILE = "04_norma_predictions.parquet"
NORMAL_STATE = 1                          # s_next == 1 is the normal state, in both codings
SHARED = ["patient_id", "analyte", "sex", "age", "n_bl", "t_span"]


def find_in(directory, name):
    """The file, whichever form it is in: bare, or with a <nn>_ stage prefix."""
    p = os.path.join(directory, name)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(directory, f"[0-9][0-9]_{name}")))
    return hits[0] if hits else p


def sex_as_int(sex):
    """One encoding per file: 1 = female, 0 = male (the legacy CHS chunks wrote 'F'/'M',
    and mixing the two in one parquet breaks the write)."""
    if sex.dtype == object:
        mapped = sex.astype(str).str.upper().str[0].map({"F": 1, "M": 0})
        sex = mapped.where(mapped.notna(), pd.to_numeric(sex, errors="coerce"))
    return pd.to_numeric(sex, errors="coerce").astype("Int64")


def default_data_root():
    """data/clalit, whether this file sits in scripts/jobs/ or was copied somewhere else."""
    here = os.path.dirname(os.path.realpath(__file__))
    for base in (os.path.dirname(os.path.dirname(here)), here, os.getcwd(),
                 os.path.dirname(here), os.path.dirname(os.getcwd())):
        cand = os.path.join(base, "data", "clalit")
        if glob.glob(os.path.join(cand, "chunk_*")):
            return cand
        if glob.glob(os.path.join(base, "chunk_*")):
            return base
    return None


def chunk_dirs(root, n_chunks=None, only=None):
    dirs = sorted(glob.glob(os.path.join(root, "chunk_*")),
                  key=lambda d: int(re.search(r"chunk_(\d+)$", d).group(1)))
    out = [(int(re.search(r"chunk_(\d+)$", d).group(1)), d) for d in dirs]
    if only:
        out = [(i, d) for i, d in out if i in set(only)]
    return out[:n_chunks] if n_chunks else out


def same_id_type(a, b):
    """patient_id and pid as one comparable dtype.  A float id has already lost digits
    (these ids are 19-digit int64), so that is an error, not something to round."""
    for name, s in (("ref_intervals patient_id", a), ("edits pid", b)):
        if pd.api.types.is_float_dtype(s):
            raise SystemExit(f"{name} is float ({s.dtype}); 19-digit ids do not survive a float, "
                             f"so the pairs cannot be matched reliably")
    if pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
        return a.astype("int64"), b.astype("int64")
    return a.astype(str), b.astype(str)


def normal_state_rows(edits, name):
    """The state=True (normal) prediction, one per (pid, code), plus a report."""
    missing = [c for c in EDIT_COLUMNS if c not in edits.columns]
    if missing:
        raise SystemExit(f"{name}: columns {missing} missing (has {list(edits.columns)})")
    state = edits["state"]
    if state.dtype != bool:
        # stored as 0/1 or "True"/"False" somewhere along the way
        state = state.astype(str).str.lower().map({"true": True, "1": True, "false": False, "0": False})
        if state.isna().any():
            raise SystemExit(f"{name}: unreadable state values {edits['state'].unique()[:5]}")
    counts = state.value_counts().to_dict()
    n_true, n_false = counts.get(True, 0), counts.get(False, 0)
    print(f"    state rows: True (normal) {n_true:,}   False (abnormal) {n_false:,}")
    if n_true == 0:
        raise SystemExit(f"{name}: no state=True rows, so there is no normal-state prediction")
    if n_false and n_true != n_false:
        print(f"    NOTE: unequal True/False counts; model/edit.py writes one of each per target")

    normal = edits[state.to_numpy()].copy()
    normal["analyte"] = normal["code"].replace({"TG": "TGL"}).replace("", "NA").fillna("NA")
    per_pair = normal.groupby(["pid", "analyte"]).size()
    if (per_pair > 1).any():
        print(f"    {int((per_pair > 1).sum()):,} pairs have several normal-state targets; "
              f"keeping each pair's earliest t_next")
    normal = normal.sort_values("t_next").drop_duplicates(["pid", "analyte"], keep="first")
    return normal[["pid", "analyte", "mu", "std"]]


# ── the per-state point forecasts 05_forecasting scores ─────────────────────

def history_stats(index_labs):
    """Per pair: how many baseline measurements, and the last one's time in days from
    the pair's first -- the origin the model's t_next is measured from."""
    if index_labs is None:
        return pd.DataFrame(columns=["patient_id", "analyte", "n_hist", "t_last_hist"])
    df = index_labs.copy()
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    first = df.groupby(["patient_id", "analyte"])["timestamp"].transform("min")
    df["t_rel"] = df["timestamp"] - first
    hist = df[df["split"] == "baseline"] if "split" in df.columns else df
    g = hist.groupby(["patient_id", "analyte"])
    return pd.DataFrame({"n_hist": g["timestamp"].size(),
                         "t_last_hist": g["t_rel"].max()}).reset_index()


def prediction_rows(edits, index_labs, run, z):
    """One row per target with NORMA's point forecast under each state.

    model/edit.py scored every target twice: state=True with s_next forced to the
    normal state, state=False with it forced to abnormal.  So

        <run>_normal          the state=True prediction        (exact)
        <run>_oracle          the row matching the REALISED s_next of that target
        <run>_marginal_freq   the two mixed by how often the next state is normal in
                              this chunk -- an approximation: the pipeline mixes the
                              full predictive distributions and takes their median,
                              which two point forecasts cannot reproduce
        <run>_marginal        left empty: it weights by the state TRANSITION prior,
                              which needs s_last and the prior file, neither of which
                              is in the edits.  05_forecasting drops a method that
                              predicts nothing, so this costs only that arm.
    """
    e = edits.copy()
    e["analyte"] = e["code"].replace({"TG": "TGL"}).replace("", "NA").fillna("NA")
    e["state"] = e["state"].astype(bool)
    keys = ["pid", "analyte", "t_next"]
    normal = e[e["state"]].drop_duplicates(keys)
    abnormal = e[~e["state"]].drop_duplicates(keys)
    out = normal.merge(abnormal[keys + ["mu", "std"]], on=keys, how="left", suffixes=("", "_abn"))
    print(f"    {len(out):,} targets; {int(out['mu_abn'].isna().sum()):,} with no state=False "
          f"row (their oracle falls back to the normal-state prediction)")

    s_next = pd.to_numeric(out["s_next"], errors="coerce")
    if not np.isclose(s_next.dropna() % 1, 0).all():
        print(f"    WARNING: s_next is not whole numbers (e.g. {s_next.dropna().iloc[0]:.4g}); it "
              f"should be the realised state (1 = normal).  The oracle arm will be wrong.")
    realised_normal = s_next == NORMAL_STATE
    if not realised_normal.any():
        print(f"    WARNING: no target has s_next == {NORMAL_STATE}, so every oracle takes the "
              f"abnormal-state prediction.  Check the state coding before using the oracle arm.")
    out["oracle"] = np.where(realised_normal | out["mu_abn"].isna(), out["mu"], out["mu_abn"])
    p_normal = float(realised_normal.mean())   # the frequency weight, per chunk
    out["marginal_freq"] = np.where(out["mu_abn"].isna(), out["mu"],
                                    p_normal * out["mu"] + (1 - p_normal) * out["mu_abn"])
    print(f"    {p_normal:.1%} of targets land in the normal state (the marginal_freq weight)")

    out = out.sort_values(keys)
    out["target_idx"] = out.groupby(["pid", "analyte"]).cumcount()   # a pair's targets in time order
    out = out.rename(columns={"pid": "patient_id"})
    out = out.merge(history_stats(index_labs), on=["patient_id", "analyte"], how="left")
    out["horizon_days"] = out["t_next"] - out["t_last_hist"]
    bad = out["horizon_days"] < 0
    if bad.any():
        print(f"    {int(bad.sum()):,} targets have a negative horizon (the edits' t_next and "
              f"index_labs disagree on the time origin); left empty")
        out.loc[bad, "horizon_days"] = np.nan

    return pd.DataFrame({
        "patient_id": out["patient_id"], "analyte": out["analyte"],
        "target_idx": out["target_idx"], "x_next": out["x_next"],
        "s_next": pd.to_numeric(out["s_next"], errors="coerce"),
        "n_hist": out["n_hist"], "horizon_days": out["horizon_days"], "cohort": "chs",
        f"{run}_normal": out["mu"],
        f"{run}_normal_lo": out["mu"] - z * out["std"],
        f"{run}_normal_hi": out["mu"] + z * out["std"],
        f"{run}_oracle": out["oracle"],
        f"{run}_marginal_freq": out["marginal_freq"],
        f"{run}_marginal": np.nan,
    })


def import_predictions(chunk_dir, edits_path, args):
    """chunk_<i>/04_norma_predictions.parquet, the file 05_forecasting reads."""
    out_path = find_in(chunk_dir, PRED_FILE)
    if os.path.exists(out_path) and not args.force:
        print(f"    {os.path.basename(out_path)} exists; skipped (--force to replace)")
        return "already imported"
    il_path = find_in(chunk_dir, "index_labs.parquet")
    index_labs = pd.read_parquet(il_path) if os.path.exists(il_path) else None
    if index_labs is None:
        print("    no index_labs.parquet; n_hist and horizon_days left empty")
    rows = prediction_rows(pd.read_parquet(edits_path, columns=PRED_COLUMNS),
                           index_labs, args.column_run, args.z)
    if args.dry_run:
        print(f"    dry run: would write {len(rows):,} prediction rows to {out_path}")
        return "dry run"
    if os.path.exists(out_path):
        bak = out_path + ".bak-import"
        if not os.path.exists(bak):
            shutil.copy2(out_path, bak)
    rows.to_parquet(out_path, index=False)
    print(f"    wrote {len(rows):,} {args.column_run} prediction rows -> {out_path}")
    return "written"


def unit_check(rows, pop):
    """Analytes whose median NORMA centre is far from the median Pop_RI midpoint."""
    mid = pop.assign(mid=(pop["ri_low"] + pop["ri_high"]) / 2).groupby("analyte")["mid"].median()
    mu = rows.groupby("analyte")["ri_mean"].median()
    ratio = (mu / mid).dropna()
    off = ratio[(ratio < 1 / 3) | (ratio > 3)]
    if len(off):
        print("    NOTE: median NORMA centre / median Pop_RI midpoint outside 1/3-3x "
              "(unit mismatch?): " + ", ".join(f"{a} {r:.2g}x" for a, r in off.items()))


def import_chunk(i, chunk_dir, args, label=""):
    name = os.path.basename(chunk_dir)
    edits_path = os.path.join(args.edits_dir, f"norma_{i}.parquet")
    base_p = find_in(chunk_dir, REF_FILE)
    norma_p = find_in(chunk_dir, REF_NORMA_FILE)
    method = f"norma_{args.run_id}"
    print(f"\n{name}{label}")

    if not os.path.exists(edits_path):
        print(f"    no {edits_path}; skipped")
        return "no edits file"
    if "intervals" not in args.only:
        return import_predictions(chunk_dir, edits_path, args)
    if not os.path.exists(base_p):
        print(f"    no {REF_FILE}; skipped (the NORMA rows take sex/age/n_bl/t_span from its pop rows)")
        return "no ref_intervals"
    existing = pd.read_parquet(norma_p) if os.path.exists(norma_p) else None
    if existing is not None and (existing["method"] == method).any() and not args.force:
        print(f"    {method} already in {os.path.basename(norma_p)}; skipped (--force to replace)")
        return "already imported"

    base = pd.read_parquet(base_p)
    base["analyte"] = base["analyte"].replace("", "NA").fillna("NA")
    pop = base[base["method"] == "pop"]
    shared = pop[SHARED].drop_duplicates(["patient_id", "analyte"])

    edits = pd.read_parquet(edits_path, columns=EDIT_COLUMNS)
    print(f"    {edits_path}: {len(edits):,} rows")
    normal = normal_state_rows(edits, os.path.basename(edits_path))

    shared = shared.copy()
    shared["_id"], normal["_id"] = same_id_type(shared["patient_id"], normal["pid"])
    rows = shared.merge(normal.drop(columns="pid"), on=["_id", "analyte"], how="inner")
    match = len(rows) / max(len(shared), 1)
    print(f"    matched {len(rows):,} of {len(shared):,} pop pairs ({match:.1%}); "
          f"{len(normal) - len(rows):,} of {len(normal):,} normal-state predictions had no pop pair")

    rows = rows.assign(method=method, ri_mean=rows["mu"], ri_std=rows["std"],
                       ri_low=rows["mu"] - args.z * rows["std"],
                       ri_high=rows["mu"] + args.z * rows["std"])
    ok = np.isfinite(rows["ri_low"]) & np.isfinite(rows["ri_high"]) & (rows["std"] > 0)
    if (~ok).any():
        print(f"    dropped {int((~ok).sum()):,} rows with a missing or non-positive std")
    rows = rows[ok][REF_COLUMNS].reset_index(drop=True)
    rows["sex"] = sex_as_int(rows["sex"])
    unit_check(rows, pop)
    if args.verbose:
        print("    per analyte: " + ", ".join(
            f"{a} {n:,}" for a, n in rows["analyte"].value_counts().sort_index().items()))

    if match < args.min_match:
        print(f"    NOT written: match {match:.1%} < --min_match {args.min_match:.0%} "
              f"(check that norma_{i}.parquet belongs to {name})")
        return "low match"
    if args.dry_run:
        print(f"    dry run: would write {len(rows):,} {method} rows to {norma_p}")
        return "dry run"

    if existing is not None:
        bak = norma_p + ".bak-import"
        if not os.path.exists(bak):
            shutil.copy2(norma_p, bak)
        out = pd.concat([existing[existing["method"] != method], rows], ignore_index=True)
        out["sex"] = sex_as_int(out["sex"])
    else:
        out = rows
    out.to_parquet(norma_p, index=False)
    print(f"    wrote {len(rows):,} {method} rows -> {norma_p}")
    if "predictions" in args.only:
        return import_predictions(chunk_dir, edits_path, args)
    return "written"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edits_dir", default=EDITS_DIR, help="folder holding norma_<i>.parquet")
    p.add_argument("--run_id", default=RUN_ID, help="method is norma_<run_id> (default: %(default)s)")
    p.add_argument("--data_root", default=None,
                   help="folder holding chunk_*/ (default: the data/clalit next to this file)")
    p.add_argument("--n_chunks", type=int, default=None, help="first N chunks only")
    p.add_argument("--chunks", type=int, nargs="+", help="these chunk numbers only")
    p.add_argument("--z", type=float, default=1.96, help="interval half-width in std (default: 1.96)")
    p.add_argument("--only", nargs="+", choices=["intervals", "predictions"],
                   default=["intervals", "predictions"],
                   help="intervals = ref_intervals_norma.parquet (07 and below); "
                        "predictions = 04_norma_predictions.parquet (05_forecasting)")
    p.add_argument("--column_run", default=MAIN_RUN,
                   help="run id the prediction COLUMNS are named for; 05_forecasting asks for "
                        "the configured run, so leave it (default: %(default)s)")
    p.add_argument("--min_match", type=float, default=0.5,
                   help="skip a chunk where fewer than this share of pop pairs found a prediction")
    p.add_argument("--force", action="store_true", help="replace norma_<run_id> rows already imported")
    p.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    p.add_argument("--verbose", action="store_true", help="also list the rows written per analyte")
    p.add_argument("--stop_on_error", action="store_true",
                   help="stop at the first unreadable chunk (default: report it and carry on)")
    args = p.parse_args()

    root = args.data_root or default_data_root()
    if not root:
        raise SystemExit("Could not find a folder holding chunk_*/; pass --data_root")
    dirs = chunk_dirs(root, args.n_chunks, args.chunks)
    if not dirs:
        raise SystemExit(f"No chunk_* under {root}")
    if not os.path.isdir(args.edits_dir):
        raise SystemExit(f"No edits folder at {args.edits_dir} (--edits_dir)")
    print(f"{len(dirs)} chunk(s) under {root}\nedits: {args.edits_dir}\n"
          f"steps {', '.join(args.only)}; intervals as norma_{args.run_id} (normal state, "
          f"mu +/- {args.z} std); prediction columns named {args.column_run}"
          + ("   [dry run]" if args.dry_run else ""))

    # One unreadable chunk must not lose the hundreds that follow it, so the run
    # carries on and every outcome is reported at the end (--stop_on_error to stop).
    outcome = {}
    for n, (i, d) in enumerate(dirs, 1):
        label = f"   ({n}/{len(dirs)})"
        try:
            outcome[os.path.basename(d)] = import_chunk(i, d, args, label)
        except SystemExit as e:
            print(f"    ERROR: {e}")
            if args.stop_on_error:
                raise
            outcome[os.path.basename(d)] = "error"
        except Exception as e:                      # a corrupt parquet, a bad dtype
            print(f"    ERROR: {type(e).__name__}: {e}")
            if args.stop_on_error:
                raise
            outcome[os.path.basename(d)] = "error"

    print(f"\nSummary of {len(dirs)} chunk(s)")
    for status in sorted(set(outcome.values())):
        names = [n for n, s in outcome.items() if s == status]
        print(f"  {status:18s} {len(names):4d}  {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    if any(s in ("error", "low match", "no edits file") for s in outcome.values()):
        print("  re-run to retry those; chunks already written are skipped unless --force")


if __name__ == "__main__":
    main()
