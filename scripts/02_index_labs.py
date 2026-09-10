#!/usr/bin/env python
"""Cut every patient-analyte series into history (baseline) and index measurements.

    python 02_index_labs.py --dataset eicu
    python 02_index_labs.py --dataset inspire
    python 02_index_labs.py --dataset mimiciv      # after process/dev_cohort.py --source mimiciv
    python 02_index_labs.py --dataset ehrshot      # after process/dev_cohort.py --source ehrshot
    python 02_index_labs.py --dataset chs [--n_chunks N]   # per chunk, on the Clalit server

Reads <data_dir>/processed.parquet (01_process output) through the dataset's
loader, which standardises the columns and puts `timestamp` in DAYS re-anchored
to 0 per series — the unit NORMA was trained on — and writes
<data_dir>/index_labs.parquet with a `split` column:

    baseline   the first `baseline_pct` of the series (default 75 %): the history
               every reference-interval method is fitted on (PerRI, Gaussian,
               Cohen, NORMA context)
    index      the remaining measurements: classified / forecast against that RI

Series with fewer than `min_baseline_count` baseline measurements (or a baseline
spanning fewer than `min_baseline_days`) are dropped entirely. This is not a
training split — no model is trained on these cohorts.

CHS starts from the labs and uses a calendar rule instead of a fraction. Each
chunk's raw `labs_{i}_flag.parquet` already carries the per-lab flag
`meets_2015_gap90_tests5` (baseline = the pre-2015 window with ≥5 tests spanning
≥90 days, matching how the Clalit Bayes/setpoint intervals were fitted), so
flagged rows are baseline, later rows index, and the same ≥ min_baseline_count
filter applies. The columns are standardised in memory with process/clalit.py's
process_labs, so no processed.parquet is read or written: the split is the only
thing those columns are needed for, and 250 chunks of intermediate that can go
stale is worse than reading the labs twice. Timestamps stay datetimes on disk
and CHSDataset._standardize turns them into days.
"""

import bootstrap  # noqa: F401

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from datasets import DATA_DIR, DATASETS

SPLITTABLE = ["eicu", "inspire", "mimiciv", "ehrshot", "chs"]
CHS_FLAG = "meets_2015_gap90_tests5"
# Days an index draw must sit past the pair's last baseline draw to count.
# 0 = a pair only needs some index measurement after its baseline window, which
# is what the calendar cut already guarantees; raise it to require a lead-in gap
# (--min_index_gap_days).  Pairs with no index measurement at all are excluded
# where it matters, by 04_refs._eligible_pair_keys.
CHS_INDEX_GAP_DAYS = 30
CHS_BASELINE_GAP_DAYS = 90
CHS_AGE_RANGE = (18, 100)
# The server extract carries three eligibility columns and they are three
# different kinds of filter:
#   alive_2015               per patient -- who can have post-2015 follow-up at all
#   meets_2015_gap90_tests5  per lab     -- the qualifying pre-2015 baseline window
#   inpatient                per lab     -- hospital draw, not a stable outpatient one
# Only the second is a split; the other two decide who and what is in the cohort.
CHS_ALIVE_COL = "alive_2015"
CHS_INPATIENT_COL = "inpatient"


def chronological_split(df, baseline_pct=0.75, min_baseline_count=5, min_baseline_days=0.0):
    """Add `split` (baseline/index) per series and drop series with too little baseline.

    Expects the standardised frame (patient_id, analyte, timestamp in days).
    Rows keep every other column they came with.
    """
    df = (df.sort_values(["patient_id", "analyte", "timestamp"], kind="stable")
            .reset_index(drop=True))
    grp = df.groupby(["patient_id", "analyte"], sort=False)
    n = grp["timestamp"].transform("size").to_numpy()
    n_bl = np.maximum((n * baseline_pct).astype(int), 1)
    df["split"] = np.where(grp.cumcount().to_numpy() < n_bl, "baseline", "index")

    bl = (df[df["split"] == "baseline"]
          .groupby(["patient_id", "analyte"])["timestamp"].agg(["size", "min", "max"]))
    span_days = bl["max"] - bl["min"]
    keep = bl.index[(bl["size"] >= min_baseline_count) & (span_days >= min_baseline_days)]
    series = pd.MultiIndex.from_frame(df[["patient_id", "analyte"]])
    df = df[series.isin(keep)].reset_index(drop=True)
    print(f"  Baseline filter: {len(keep):,}/{len(bl):,} patient-analyte series kept "
          f"(≥{min_baseline_count} baseline meas, ≥{min_baseline_days:g}d span)")
    n_bl_rows = int((df["split"] == "baseline").sum())
    print(f"  Split: {len(df):,} rows ({n_bl_rows:,} baseline / {len(df) - n_bl_rows:,} index), "
          f"{df['patient_id'].nunique():,} patients")
    return df


_TRUE = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE = {"false", "f", "no", "n", "0", "0.0", "", "nan", "none", "<na>"}


def as_bool(series, col):
    """A true/false column as real booleans, whatever the extract stored.

    `.astype(bool)` is not safe here: on an object column every non-empty string
    is truthy, so a column of "False" strings comes back all True and silently
    removes the entire history.  Anything not recognisably boolean raises rather
    than being guessed at.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    text = series.astype(str).str.strip().str.lower()
    unknown = set(text.unique()) - _TRUE - _FALSE
    if unknown:
        raise SystemExit(
            f"\n  `{col}` is not a true/false column: dtype={series.dtype}, "
            f"unrecognised values {sorted(unknown)[:8]}\n"
            f"  Refusing to guess — it decides which rows stay in the cohort.")
    return text.isin(_TRUE)


def _empty_check(out, df, col, what):
    """A filter that removes everything is a data problem, not a result."""
    if len(out):
        return out
    raw = df[col]
    counts = raw.value_counts(dropna=False).head(6).to_dict()
    raise SystemExit(
        f"\n  {what} removed every row.\n"
        f"  `{col}` dtype={raw.dtype}, values={counts}, nulls={int(raw.isna().sum()):,}\n"
        f"  If the column is not a plain true/false flag it is being read wrong — "
        f"check the extract before rerunning, or disable the filter.")


def alive_filter(df, col=CHS_ALIVE_COL):
    """Keep only patients alive at the 2015 index date.

    Patient-level, so whole patients go: someone who died before 2015 has no
    post-2015 measurement to be scored on, and their pre-2015 history would
    otherwise sit in the cohort contributing baseline rows and nothing else.
    """
    if col not in df.columns:
        print(f"  {col} absent — no alive filter applied")
        return df
    keep = as_bool(df[col], col)
    n_pat = df["patient_id"].nunique()
    out = _empty_check(df[keep].reset_index(drop=True), df, col,
                       f"--keep_dead off: the {col} filter")
    print(f"  Alive at 2015: {out['patient_id'].nunique():,}/{n_pat:,} patients kept "
          f"({len(df):,} -> {len(out):,} rows)")
    return out


def inpatient_filter(df, scope, col=CHS_INPATIENT_COL):
    """`scope` decides what the baseline history is made of.

    require   baseline is inpatient draws only (the cohort definition)
    baseline  baseline is outpatient draws only
    all       inpatient draws dropped everywhere
    keep      no filtering

    Index measurements are never touched by `require` or `baseline`: the split
    is about what the reference interval is fitted on, and excluding index draws
    would remove the measurements the classification exists to judge.
    """
    if scope == "keep" or col not in df.columns:
        return df
    inpat = as_bool(df[col], col)
    is_bl = df["split"] == "baseline"
    if scope == "require":
        drop = is_bl & ~inpat
    elif scope == "all":
        drop = inpat
    else:
        drop = is_bl & inpat
    out = _empty_check(df[~drop].reset_index(drop=True), df, col,
                       f"--inpatient_scope {scope}: the {col} filter")
    print(f"  Inpatient ({scope}): dropped {int(drop.sum()):,} rows "
          f"({len(df):,} -> {len(out):,}); {int(inpat.sum()):,}/{len(df):,} rows are inpatient")
    return out


def age_filter(df, lo, hi):
    """Keep patients whose age at their FIRST INDEX measurement is in [lo, hi].

    Patient level and taken from one draw, not per row: CHS `age` is age_at_test
    over ~15 years, so filtering rows would keep the middle of a patient's record
    and cut its ends, leaving a series whose history and index measurements come
    from different eligibility windows.  Anchored to the first index draw because
    that is the measurement being classified -- a patient who was 16 during their
    pre-2015 history but 19 when first scored is in the adult cohort.

    Needs `split`, so this runs after the flag has been turned into baseline/index.
    A patient with no index draw has no age to test and nothing to be scored on.
    """
    if lo is None and hi is None:
        return df
    idx = df[df["split"] == "index"]
    first = (idx.sort_values(["patient_id", "timestamp"], kind="stable")
                .drop_duplicates("patient_id").set_index("patient_id")["age"])
    n_pat = df["patient_id"].nunique()
    no_index = n_pat - len(first)
    keep = pd.Series(True, index=first.index)
    if lo is not None:
        keep &= first >= lo
    if hi is not None:
        keep &= first <= hi
    out = df[df["patient_id"].isin(set(keep.index[keep]))].reset_index(drop=True)
    print(f"  Age {lo:g}-{hi:g} at first index draw: {int(keep.sum()):,}/{n_pat:,} patients kept "
          f"({no_index:,} had no index measurement; {len(df):,} -> {len(out):,} rows)")
    if n_pat and not keep.any():
        q = {f"p{p}": float(first.quantile(p / 100)) for p in (0, 50, 100)} if len(first) else {}
        raise SystemExit(f"\n  No patient is aged {lo:g}-{hi:g} at their first index draw.\n"
                         f"  Age at first index draw: {q}; {no_index:,} patients had none.")
    return out


def patient_gate(df, n_tests, gap_days, inpatient_only=True,
                 flag_col=CHS_FLAG, col=CHS_INPATIENT_COL):
    """Keep patients with >= n_tests qualifying baseline-window labs >= gap_days apart.

    Patient level, across all analytes: the criterion is about who is in the
    cohort, not about which analyte can be fitted -- a per-analyte reading is
    far stricter and empties the cohort.  The per-pair requirement is separate
    and still applies below.

    Counted on the baseline-window rows (the pre-2015 flag), restricted to
    inpatient draws when inpatient_only.  It is re-derived here rather than
    taken from the flag because the flag was computed over all labs: a patient
    can satisfy it on outpatient draws and have only a couple of spaced
    inpatient ones.
    """
    elig = df[as_bool(df[flag_col], flag_col)]
    if inpatient_only:
        if col not in df.columns:
            print(f"  {col} absent — patient gate counts all baseline-window labs")
        else:
            elig = elig[as_bool(elig[col], col)]
    per_pat = elig.groupby("patient_id")["timestamp"].apply(
        lambda t: spaced_count(t.to_numpy(), gap_days))
    keep = set(per_pat.index[per_pat >= n_tests])
    n_pat = df["patient_id"].nunique()
    out = df[df["patient_id"].isin(keep)].reset_index(drop=True)
    what = "inpatient " if inpatient_only else ""
    print(f"  Patient gate (≥{n_tests} {what}labs ≥{gap_days:g}d apart, pre-2015): "
          f"{len(keep):,}/{n_pat:,} patients kept ({len(df):,} -> {len(out):,} rows)")
    if n_pat and not len(keep):
        dist = {f"p{q}": int(per_pat.quantile(q / 100)) for q in (50, 95, 100)} if len(per_pat) else {}
        raise SystemExit(
            f"\n  No patient has {n_tests} {what}labs ≥{gap_days:g} days apart before 2015.\n"
            f"  Spaced count per patient: {dist}\n"
            f"  {len(elig):,} of {len(df):,} rows were eligible to count. "
            f"Try --no_inpatient_gate, or lower --gate_tests / --min_baseline_gap_days.")
    return out


def spaced_count(times, min_gap_days):
    """How many draws survive a greedy >= min_gap_days spacing rule.

    Sorted ascending, keep the first, then each draw at least min_gap_days after
    the last kept one -- the standard reading of "5 tests 90 days apart", and
    the same rule process/measurements.py uses.  Counting raw draws instead
    would let five same-week results qualify.
    """
    t = np.sort(np.asarray(times))
    if len(t) == 0:
        return 0
    day = np.timedelta64(1, "D") if np.issubdtype(t.dtype, np.datetime64) else 1
    kept, last = 1, t[0]
    for x in t[1:]:
        if (x - last) >= min_gap_days * day:
            kept += 1
            last = x
    return kept


def _gap_days(later, earlier):
    """later - earlier in days, for datetime or already-numeric-days timestamps."""
    d = later - earlier
    if pd.api.types.is_timedelta64_dtype(d):
        return d.dt.days
    if d.dtype == object:                     # object array of Timedeltas
        return pd.to_timedelta(d, errors="coerce").dt.days
    return d


def index_gap_filter(df, min_gap_days):
    """Drop index rows within `min_gap_days` of their pair's last baseline draw.

    Only the rows: the pair keeps its baseline history even if that leaves it
    with no index measurement, because those values are still co-analyte context
    for the other analytes of the same patient.  A pair with nothing left to
    score is already excluded where it matters -- 04_refs._eligible_pair_keys
    requires an index measurement before computing a reference interval.

    The last baseline draw is joined on rather than mapped through the index:
    Index.map returns an object array when any pair has no baseline row left,
    and the resulting object-dtype subtraction compares wrongly against a
    number instead of raising.
    """
    if not min_gap_days:
        return df
    keys = ["patient_id", "analyte"]
    last_bl = (df[df["split"] == "baseline"].groupby(keys)["timestamp"]
                 .max().rename("_last_bl"))
    joined = df.join(last_bl, on=keys)
    gap = _gap_days(joined["timestamp"], joined["_last_bl"])
    is_index = df["split"].to_numpy() == "index"
    too_soon = is_index & gap.lt(min_gap_days).fillna(False).to_numpy()
    before = df[is_index].groupby(keys).ngroups
    out = df[~too_soon].reset_index(drop=True)
    after = out[out["split"] == "index"].groupby(keys).ngroups
    print(f"  Index gap ≥{min_gap_days:g}d: dropped {int(too_soon.sum()):,} early index rows; "
          f"pairs with an index measurement {before:,} -> {after:,}")
    if before and not after:
        g = gap[is_index].dropna()
        q = {f"p{p}": (float(g.quantile(p / 100)) if len(g) else float("nan")) for p in (0, 25, 50, 100)}
        raise SystemExit(
            f"\n  The ≥{min_gap_days:g}d index gap removed every index measurement.\n"
            f"  Gap in days across {len(g):,} index rows: {q}\n"
            f"  If those are all near zero the baseline and index windows overlap — "
            f"check `{CHS_FLAG}` before rerunning, or pass --min_index_gap_days 0.")
    return out


def flag_split(df, flag_col=CHS_FLAG, min_baseline_count=5,
               min_index_gap_days=CHS_INDEX_GAP_DAYS, inpatient_scope="keep",
               require_alive=True, min_baseline_gap_days=CHS_BASELINE_GAP_DAYS,
               gate_tests=5, inpatient_gate=True, age_range=CHS_AGE_RANGE):
    """CHS: `split` from the server's baseline-window flag, then the ≥ min_baseline_count filter."""
    if flag_col not in df.columns:
        raise KeyError(f"{flag_col} missing from the chunk's labs — the server extract "
                       f"should carry it; check --flag_col and the labs file name")
    if require_alive:
        df = alive_filter(df)
    if gate_tests:
        df = patient_gate(df, gate_tests, min_baseline_gap_days, inpatient_gate, flag_col)
    # Same ordering chronological_split leaves behind.  Not cosmetic: the
    # cohort summary takes age and sex from the first row per patient
    # (03_cohort_summary._chunk_demo_stats), and CHS `age` is age_at_test over
    # ~15 years of follow-up, so an unsorted frame reports the age of whichever
    # row the raw labs happened to list first.
    df = (df.sort_values(["patient_id", "analyte", "timestamp"], kind="stable")
            .reset_index(drop=True))
    df["split"] = np.where(as_bool(df[flag_col], flag_col), "baseline", "index")
    df = age_filter(df, *age_range)
    keys = ["patient_id", "analyte"]
    bl_before = df[df["split"] == "baseline"].groupby(keys).size()
    df = inpatient_filter(df, inpatient_scope)
    bl = df[df["split"] == "baseline"].groupby(keys).size()
    series = pd.MultiIndex.from_frame(df[keys])
    keep = bl.index[bl >= min_baseline_count]
    n_series = series.nunique()
    df = df[series.isin(keep)].reset_index(drop=True)
    print(f"  Baseline filter: {len(keep):,}/{n_series:,} patient-analyte series kept "
          f"(≥{min_baseline_count} baseline meas)")
    if n_series and not len(keep):
        raise SystemExit(
            f"\n  No series has {min_baseline_count} baseline measurements.\n"
            f"  Baseline draws per pair: {dict((f'p{q}', int(bl.quantile(q / 100))) for q in (50, 95, 100)) if len(bl) else {}}\n"
            f"  Raw baseline draws per pair before --inpatient_scope {inpatient_scope}: "
            f"{dict((f'p{q}', int(bl_before.quantile(q / 100))) for q in (50, 95, 100)) if len(bl_before) else {}}\n"
            f"  Try --inpatient_scope keep, or lower --min_baseline_gap_days.")
    df = index_gap_filter(df, min_index_gap_days)
    n_bl_rows = int((df["split"] == "baseline").sum())
    print(f"  Split: {len(df):,} rows ({n_bl_rows:,} baseline / {len(df) - n_bl_rows:,} index), "
          f"{df['patient_id'].nunique():,} patients")
    return df


def _labs_template(labs_root, sandbox):
    """chunk_{i}/labs_{i}_flag.parquet under labs_root, or the server path itself."""
    from process.clalit import get_paths

    template = get_paths(sandbox)["labs"]
    if labs_root:
        template = os.path.join(labs_root, "chunk_{i}", os.path.basename(template))
    return template


def _labs_chunks(template, n_chunks=None):
    """Chunk indices that actually have a labs file, in numeric order.

    Discovery is driven by the labs, not by the output tree: on the Clalit box
    the chunk directories only exist under the server extract, so globbing the
    output root (which is where index_labs is written, and starts empty) finds
    nothing.
    """
    root = os.path.dirname(os.path.dirname(template))
    idx = sorted(int(m.group(1)) for m in
                 (re.search(r"chunk_(\d+)$", d) for d in glob.glob(os.path.join(root, "chunk_*")))
                 if m)
    if not idx:
        raise FileNotFoundError(
            f"No chunk_* directories under {root}; pass --labs_root pointing at the "
            f"server extract (the one holding chunk_{{i}}/labs_{{i}}_flag.parquet)")
    return idx[:n_chunks] if n_chunks else idx


def _chunk_labs(labs_path):
    """One chunk's labs, standardised in memory the way process/clalit.py does.

    Always the raw `labs_{i}_flag.parquet`: it is what carries the
    baseline-window flag, and process_labs (a rename, a column select,
    TG -> TGL, the unit fixes, the care setting) is all that stands between it
    and the split.  A processed.parquet in the output tree is deliberately NOT
    preferred -- a stale one would shadow the labs and the split would silently
    be cut from older values.
    """
    from process.clalit import process_labs

    print(f"  Reading {labs_path}")
    return process_labs(pd.read_parquet(labs_path))


def run_chs(args):
    """Per chunk: labs_{i}_flag.parquet -> chunk_{i}/index_labs.parquet.

    Same output name and shape as every other cohort's index_labs.parquet; CHS
    just carries the chunk index in the directory, since the cohort is far too
    large to split in one frame.
    """
    from process.clalit import resolve_path

    # Not CHSDataset's root resolution: that one picks data/clalit only when it
    # already holds chunk_* directories, and this is the stage that creates them
    # -- on a fresh box it would fall through to the sandbox and write there.
    out_root = args.data_root or os.path.join(DATA_DIR, "clalit")
    sandbox = os.path.basename(os.path.normpath(out_root)) == "sandbox"
    template = _labs_template(args.labs_root, sandbox)
    chunks = _labs_chunks(template, args.n_chunks)
    print(f"{len(chunks)} chunks with labs under {os.path.dirname(os.path.dirname(template))}")
    print(f"Writing index_labs.parquet under {out_root}")

    done = skipped = 0
    for i in chunks:
        chunk_dir = os.path.join(out_root, f"chunk_{i}")
        split_path = os.path.join(chunk_dir, "index_labs.parquet")
        print(f"\nchunk_{i}:")
        if os.path.exists(split_path) and not args.force:
            print(f"  {split_path} exists; use --force to redo the split")
            skipped += 1
            continue
        labs_path = resolve_path(template, i)
        if not os.path.exists(labs_path):
            print(f"  no labs at {labs_path}, skipping")
            continue
        index_labs = flag_split(_chunk_labs(labs_path), args.flag_col,
                                args.min_baseline_count, args.min_index_gap_days,
                                args.inpatient_scope, not args.keep_dead,
                                args.min_baseline_gap_days, args.gate_tests,
                                not args.no_inpatient_gate, (args.min_age, args.max_age))
        os.makedirs(chunk_dir, exist_ok=True)
        index_labs.to_parquet(split_path, index=False)
        print(f"  Saved {split_path}")
        done += 1
    print(f"\n{done} chunks split, {skipped} already there ({len(chunks)} with labs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=SPLITTABLE)
    ap.add_argument("--baseline_pct", type=float, default=0.75,
                    help="fraction of each series used as baseline/history (default 0.75)")
    ap.add_argument("--min_baseline_count", type=int, default=5,
                    help="drop series with fewer baseline measurements (default 5)")
    ap.add_argument("--min_baseline_days", type=float, default=0.0,
                    help="drop series whose baseline spans fewer days (default 0)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even where index_labs.parquet is already written")
    ap.add_argument("--n_chunks", type=int, default=None, help="CHS: only the first N chunks")
    ap.add_argument("--data_root", default=None, help="CHS: chunk directory root (default: data/clalit, else the sandbox)")
    ap.add_argument("--labs_root", default=None,
                    help="CHS: root holding chunk_{i}/labs_{i}_flag.parquet "
                         "(default: process/clalit.py's server path)")
    ap.add_argument("--inpatient_scope", choices=["require", "baseline", "all", "keep"],
                    default="keep",
                    help="CHS baseline history: no filtering (default -- cohort membership "
                         "is the patient gate's job), inpatient only, outpatient only, or "
                         "inpatient dropped everywhere")
    ap.add_argument("--min_age", type=float, default=CHS_AGE_RANGE[0],
                    help=f"CHS: drop patients under this age at their first baseline draw "
                         f"(default {CHS_AGE_RANGE[0]:g})")
    ap.add_argument("--max_age", type=float, default=CHS_AGE_RANGE[1],
                    help=f"CHS: drop patients over this age (default {CHS_AGE_RANGE[1]:g})")
    ap.add_argument("--gate_tests", type=int, default=5,
                    help="CHS: patients need this many spaced pre-2015 labs (0 disables)")
    ap.add_argument("--no_inpatient_gate", action="store_true",
                    help="CHS: count all pre-2015 labs for the patient gate, not just inpatient")
    ap.add_argument("--min_baseline_gap_days", type=float, default=CHS_BASELINE_GAP_DAYS,
                    help=f"CHS: baseline measurements must be this many days apart to count "
                         f"(default {CHS_BASELINE_GAP_DAYS:g})")
    ap.add_argument("--keep_dead", action="store_true",
                    help=f"CHS: keep patients with {CHS_ALIVE_COL} false (default: drop them)")
    ap.add_argument("--min_index_gap_days", type=float, default=CHS_INDEX_GAP_DAYS,
                    help=f"CHS: drop index measurements within this many days of the "
                         f"last baseline draw (default {CHS_INDEX_GAP_DAYS:g}; 0 disables)")
    ap.add_argument("--flag_col", default=CHS_FLAG, help="CHS: per-lab baseline-window flag column")
    args = ap.parse_args()

    if args.dataset == "chs":
        return run_chs(args)

    ds = DATASETS[args.dataset]()
    split_path = os.path.join(ds.data_dir, "index_labs.parquet")
    if os.path.exists(split_path) and not args.force:
        print(f"  {split_path} exists; use --force to redo the split")
        return

    print(f"Loading {ds.name} processed data...")
    df = ds.load_processed()
    print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients "
          f"(timestamp in {ds.time_unit})")

    print("Splitting into baseline/index...")
    index_labs = chronological_split(df, args.baseline_pct, args.min_baseline_count, args.min_baseline_days)
    index_labs.to_parquet(split_path, index=False)
    print(f"  Saved {split_path}")


if __name__ == "__main__":
    main()
