#!/usr/bin/env python
"""Inside Clalit: what each chunk's ref_intervals actually covers, per method.

04_refs decides what to recompute from `pop` rows alone (_missing_pairs), so a
chunk whose file came from 01_process -- whose pairs come from the server's
bayes extract, not from index_labs -- reports a large "missing" count that is
real work, not repeated work.  This shows the pair counts side by side so the
two cases are distinguishable before a long run.

    python jobs/chs_ref_coverage.py --n_chunks 10
"""
import os as _os, sys as _sys
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
for _p in (_os.path.join(_SCRIPTS_DIR, "lib"), _SCRIPTS_DIR, _os.path.dirname(_SCRIPTS_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import glob
import os
import re

import pandas as pd

KEYS = ["patient_id", "analyte"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_root", default=None)
    p.add_argument("--n_chunks", type=int, default=None)
    p.add_argument("--which_file", action="store_true",
                   help="say which ref_intervals file 04_refs would actually read")
    args = p.parse_args()

    from datasets import DATASETS
    root = args.data_root or DATASETS["chs"]().data_root
    dirs = sorted(glob.glob(os.path.join(root, "chunk_*")),
                  key=lambda d: int(re.search(r"(\d+)$", d).group(1)))
    dirs = dirs[:args.n_chunks] if args.n_chunks else dirs
    print(f"{len(dirs)} chunks under {root}\n")

    rows = []
    for d in dirs:
        name = os.path.basename(d)
        il = os.path.join(d, "index_labs.parquet")
        rp = os.path.join(d, "ref_intervals.parquet")
        rec = {"chunk": name}
        rec["split_pairs"] = (pd.read_parquet(il, columns=KEYS).drop_duplicates().shape[0]
                              if os.path.exists(il) else 0)
        if os.path.exists(rp):
            r = pd.read_parquet(rp, columns=KEYS + ["method"])
            for m, g in r.groupby("method"):
                rec[m] = g[KEYS].drop_duplicates().shape[0]
        rows.append(rec)

    df = pd.DataFrame(rows).fillna(0).set_index("chunk")
    for c in df.columns:
        df[c] = df[c].astype(int)
    if "pop" in df.columns:
        df["missing"] = df["split_pairs"] - df["pop"]
    print(df.to_string())
    if args.which_file:
        which_file(dirs[0])
    print("\n  missing = split_pairs - pop pairs: what 04_refs will compute.")
    print("  A chunk with pop == split_pairs is done; one with pop far below it")
    print("  still holds the 01_process (server bayes) pair set.")


def which_file(chunk_dir):
    """Which ref_intervals 04_refs --chunk would read, and what the alternative holds.

    The chunk-scoped path only exists once lib/datasets.py carries _ref_dir; an
    older copy sends both the read and the write to the cohort-level file, so
    every chunk is compared against the wrong pair set and recomputed each run.
    """
    from datasets import DATASETS, ref_paths, results_dir, REF_FILE, find_in
    import re as _re

    i = int(_re.search(r"(\d+)$", chunk_dir).group(1))
    ds = DATASETS["chs"](chunk=i)
    resolved = ref_paths(ds)[0]
    chunk_p = os.path.join(chunk_dir, "ref_intervals.parquet")
    cohort_p = find_in(results_dir(ds.output_sub()), REF_FILE)

    print(f"\n  04_refs --chunk {i} would read/write:\n    {resolved}")
    scoped = os.path.realpath(resolved) == os.path.realpath(chunk_p)
    print(f"    -> {'chunk-scoped (correct)' if scoped else 'COHORT-LEVEL: lib/datasets.py is the old copy'}")
    for label, path in (("chunk file ", chunk_p), ("cohort file", cohort_p)):
        if os.path.exists(path):
            r = pd.read_parquet(path, columns=KEYS + ["method"])
            n = r.loc[r.method == "pop", KEYS].drop_duplicates().shape[0]
            print(f"    {label}: {n:,} pop pairs, methods {sorted(r.method.unique())}")
        else:
            print(f"    {label}: absent")


if __name__ == "__main__":
    main()
