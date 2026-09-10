#!/usr/bin/env python
"""One-time, inside Clalit: chunk_*/split_df.parquet -> chunk_*/index_labs.parquet.

02_index_labs.py used to write each chunk's baseline/index split as
`split_df.parquet`; it now writes `index_labs.parquet`, and every CHS reader
(CHSDataset.load_index_labs, iter_chunks, 07_classify, 10_mortality) looks for
the new name only.  Recutting the split on the real chunks is slow, so this
renames what is already there instead.

    python jobs/migrate_chs_index_labs.py --report     # inventory the chunks first
    python jobs/migrate_chs_index_labs.py --dry_run    # what it would rename
    python jobs/migrate_chs_index_labs.py              # do it
    python jobs/migrate_chs_index_labs.py --undo       # rename back

On the Clalit box the chunk root is not where this script guesses, so pass it:
    python jobs/migrate_chs_index_labs.py --report --data_root Z:\\NORMA\\data\\clalit

The contents do not change: the file is the same split, timestamps stay
datetimes on disk, and CHSDataset._standardize turns them into days at load
time exactly as before.  os.replace is an atomic same-directory rename, so no
data is copied however large the chunk is, and --undo reverses it.

Safe to re-run: a chunk that already has index_labs.parquet is left alone, and
a chunk holding both files is reported rather than touched, since which one is
current is not this script's call to make.
"""
import argparse
import glob
import os
import re
import sys

# legacy name -> current name, applied inside each chunk_* directory.  Add a row
# here rather than writing another script; --report lists what is actually present
# so an unknown legacy name shows up before anything is renamed.
RENAMES = {
    "split_df.parquet": "index_labs.parquet",
}

BACKUP_SUFFIX = ".bak-rebuilt"   # where --force parks the file it replaces

OLD_NAME = "split_df.parquet"      # kept for the messages below
NEW_NAME = "index_labs.parquet"

# Files a chunk is expected to carry once migrated, so --report can say what is
# missing as well as what is stale.  Not every chunk has every one: the server
# extracts (bayes_*, setpoint_*, norma_*, labs_*_flagged) are per-chunk inputs and
# the rest are written as the pipeline advances.
EXPECTED = [
    "processed.parquet", "index_labs.parquet", "diagnosis.parquet",
    "ref_intervals.parquet", "ref_intervals_norma.parquet",
    "classification.parquet", "classification_norma.parquet",
    "norma_predictions.parquet",
]

# What a chunk's index_labs must carry for the CHS readers to work.  `split`
# is the point of the file; the rest is what build_pair_table and the
# classification join need.  CHS extras (inpatient, the 2015 flag) ride along
# untouched and are not required here.
REQUIRED = {"patient_id", "analyte", "timestamp", "value", "split"}


def natural_key(path):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", path)]


def chunk_dirs(root, n_chunks=None):
    dirs = sorted(glob.glob(os.path.join(root, "chunk_*")), key=natural_key)
    return dirs[:n_chunks] if n_chunks else dirs


def columns_of(path):
    """Column names without reading any row groups."""
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(path).schema_arrow.names)


def report(root, n_chunks=None):
    """Inventory every chunk: what it holds, what is stale, what is missing.

    Run this first inside Clalit.  Nothing here is guessed from the sandbox --
    it lists what is actually on disk, so an unexpected legacy name is visible
    before any rename touches it.
    """
    dirs = chunk_dirs(root, n_chunks)
    if not dirs:
        print(f"No chunk_* under {root}")
        return 1
    legacy_seen, extras = {}, {}
    for d in dirs:
        name = os.path.basename(d)
        files = sorted(os.listdir(d))
        stale = [f for f in files if f in RENAMES]
        missing = [f for f in EXPECTED if f not in files]
        other = [f for f in files if f not in EXPECTED and f not in RENAMES]
        for f in stale:
            legacy_seen[f] = legacy_seen.get(f, 0) + 1
        for f in other:
            extras[f] = extras.get(f, 0) + 1
        if d is dirs[0]:
            print(f"\n  {name}: {len(files)} files")
            for f in files:
                tag = ("  <- LEGACY, renamed to " + RENAMES[f]) if f in RENAMES else ""
                size = os.path.getsize(os.path.join(d, f)) / 1e6
                print(f"      {f:<30} {size:9.1f} MB{tag}")
                # Schema of anything we do not already recognise, so an unknown
                # file can be identified by its columns instead of by its name:
                # index_labs is processed + a `split` column, which is exactly
                # what tells the two apart.
                if f.endswith(".parquet") and f not in EXPECTED:
                    try:
                        cols = sorted(columns_of(os.path.join(d, f)))
                    except Exception as exc:
                        print(f"          (unreadable: {exc})")
                        continue
                    verdict = ""
                    if REQUIRED <= set(cols):
                        verdict = "  == looks like index_labs (has `split`)"
                    elif REQUIRED - {"split"} <= set(cols):
                        verdict = "  == looks like processed (no `split`)"
                    print(f"          columns ({len(cols)}):{verdict}")
                    print(f"          {', '.join(cols)}")
            if missing:
                print(f"      (absent: {', '.join(missing)})")
    print(f"\n  Across {len(dirs)} chunks:")
    for f, n in sorted(legacy_seen.items()):
        print(f"    LEGACY  {f:<34} in {n}/{len(dirs)} chunks -> {RENAMES[f]}")
    if not legacy_seen:
        print("    no legacy names found - nothing for this script to rename")
    unknown = {f: n for f, n in extras.items() if not f.endswith((".pkl", ".csv"))}
    if unknown:
        print("    other files present (left alone):")
        for f, n in sorted(unknown.items()):
            print(f"      {f:<36} in {n}/{len(dirs)} chunks")
    return 0


def migrate(root, n_chunks=None, dry_run=False, undo=False, force=False):
    pairs = [(v, k) for k, v in RENAMES.items()] if undo else list(RENAMES.items())
    dirs = chunk_dirs(root, n_chunks)
    if not dirs:
        print(f"No chunk_* under {root}")
        return 1

    renamed = skipped = both = missing = bad = 0
    for d in dirs:
        name = os.path.basename(d)
        for src_name, dst_name in pairs:
            src, dst = os.path.join(d, src_name), os.path.join(d, dst_name)

            if os.path.exists(src) and os.path.exists(dst):
                if not force:
                    print(f"  {name}: BOTH {src_name} and {dst_name} present - left alone "
                          f"(--force to let {src_name} win)")
                    both += 1
                    continue
                # Never delete the file being replaced: park it beside itself, once,
                # so a --force run is as reversible as a plain one.
                bak = dst + BACKUP_SUFFIX
                if os.path.exists(bak):
                    print(f"  {name}: {os.path.basename(bak)} already exists - left alone")
                    both += 1
                    continue
                if dry_run:
                    print(f"  {name}: would move {dst_name} -> {os.path.basename(bak)}, "
                          f"then {src_name} -> {dst_name}")
                    renamed += 1
                    continue
                os.replace(dst, bak)
                print(f"  {name}: {dst_name} -> {os.path.basename(bak)}")
            if not os.path.exists(src):
                if os.path.exists(dst):
                    skipped += 1                  # already migrated
                else:
                    print(f"  {name}: neither {src_name} nor {dst_name}")
                    missing += 1
                continue

            try:
                cols = columns_of(src)
            except Exception as exc:              # unreadable parquet: never rename it
                print(f"  {name}: cannot read {src_name} ({exc}) - left alone")
                bad += 1
                continue
            lacking = REQUIRED - cols if src_name == OLD_NAME or dst_name == OLD_NAME else set()
            if lacking and not undo:
                print(f"  {name}: {src_name} lacks {sorted(lacking)} - left alone")
                bad += 1
                continue

            if dry_run:
                print(f"  {name}: would rename {src_name} -> {dst_name} ({len(cols)} columns)")
            else:
                os.replace(src, dst)
                print(f"  {name}: {src_name} -> {dst_name}")
            renamed += 1

    verb = "would rename" if dry_run else "renamed"
    print(f"\n  {len(dirs)} chunks x {len(pairs)} name(s): {verb} {renamed}, "
          f"already done {skipped}, both present {both}, neither {missing}, left alone {bad}")
    if bad or both:
        print("  Resolve the chunks above by hand before running the pipeline.")
        return 1
    if renamed and not dry_run and not undo:
        print(f"  Undo with: python {os.path.relpath(__file__)} --undo"
              f" --data_root {root}")
    return 0


def default_root():
    """data/clalit next to this checkout, or the sandbox when that is all there is."""
    scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.dirname(scripts)
    real = os.path.join(base, "data", "clalit")
    if glob.glob(os.path.join(real, "chunk_*")):
        return real
    return os.path.join(real, "sandbox")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_root", default=None,
                   help="chunk directory root (default: data/clalit, else its sandbox)")
    p.add_argument("--n_chunks", type=int, default=None, help="only the first N chunks")
    p.add_argument("--dry_run", action="store_true", help="report, change nothing")
    p.add_argument("--undo", action="store_true",
                   help="rename the current names back to the legacy ones")
    p.add_argument("--force", action="store_true",
                   help="a chunk holding both files: park the current index_labs.parquet "
                        "as .bak-rebuilt and let the legacy file win")
    p.add_argument("--report", action="store_true",
                   help="inventory the chunks and stop: what is there, stale, missing")
    args = p.parse_args()

    root = args.data_root or default_root()
    print(f"{'UNDO: ' if args.undo else ''}{root}")
    if args.report:
        sys.exit(report(root, args.n_chunks))
    sys.exit(migrate(root, args.n_chunks, args.dry_run, args.undo, args.force))


if __name__ == "__main__":
    main()
