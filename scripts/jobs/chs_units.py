#!/usr/bin/env python
"""Inside Clalit: what units the raw labs actually report, per analyte per chunk.

process/clalit.py's fix_values infers units from the values (RBC median > 10 is
10^3/uL, A1C p95 > 20 is IFCC) because processed.parquet drops the `unit`
column.  That inference is per chunk, so a chunk with few RBC rows -- or with
both units inside it -- can be scaled the wrong way with nothing downstream to
catch it.  This reads `unit` straight off the server extract and reports it.

    python jobs/chs_units.py --n_chunks 250

Reads nothing but labs_{i}_flag.parquet, writes nothing.  Two columns to look at:
`units` per analyte (more than one means the fix cannot be a per-chunk constant)
and `chunks_flipped`, the chunks where the value heuristic and the `unit` string
disagree -- those are the ones fix_values gets wrong today.
"""
# Run from anywhere: scripts/ and scripts/lib go on sys.path, as in process/clalit.py.
import os as _os, sys as _sys
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
for _p in (_os.path.join(_SCRIPTS_DIR, "lib"), _SCRIPTS_DIR, _os.path.dirname(_SCRIPTS_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import os
import re

import pandas as pd

from process.clalit import get_paths, resolve_path  # noqa: E402

# analyte -> (unit strings meaning "needs the fix", the heuristic fix_values uses)
SUSPECT = {
    "RBC": lambda s: s.median() > 10,
    "A1C": lambda s: s.quantile(0.95) > 20,
}


def chunk_indices(root, n_chunks):
    idx = sorted(int(m.group(1)) for m in
                 (re.search(r"chunk_(\d+)$", d) for d in os.listdir(root)) if m)
    return idx[:n_chunks] if n_chunks else idx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labs_root", default=None, help="root holding chunk_{i}/ (default: the server path)")
    ap.add_argument("--n_chunks", type=int, default=None)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--out", default=None, help="write the per-analyte table here as CSV")
    args = ap.parse_args()

    template = get_paths(args.sandbox)["labs"]
    root = args.labs_root or os.path.dirname(os.path.dirname(template))
    if args.labs_root:
        template = os.path.join(root, "chunk_{i}", os.path.basename(template))

    rows = []
    idx = chunk_indices(root, args.n_chunks)
    print(f"{len(idx)} chunks under {root}")
    for n, i in enumerate(idx, 1):
        path = resolve_path(template, i)
        if not os.path.exists(path):
            continue
        labs = pd.read_parquet(path, columns=["code", "unit", "numeric_value"])
        labs["code"] = labs["code"].replace({"TG": "TGL"})
        for (code, unit), g in labs.groupby(["code", "unit"], dropna=False):
            v = g["numeric_value"]
            rows.append({"chunk": i, "analyte": code, "unit": unit, "n": len(g),
                         "median": v.median(), "p95": v.quantile(0.95), "max": v.max()})
        if n % 25 == 0 or n == len(idx):
            print(f"  {n}/{len(idx)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No labs found")
        return

    print("\nPer analyte, across chunks:")
    per = (df.groupby("analyte")
             .apply(lambda g: pd.Series({
                 "units": ", ".join(sorted(map(str, g["unit"].unique()))),
                 "n_units": g["unit"].nunique(dropna=False),
                 "chunks": g["chunk"].nunique(),
                 "n": int(g["n"].sum()),
             })).reset_index())
    print(per.to_string(index=False))

    # Where the value heuristic and the unit string would disagree: for each
    # analyte fix_values touches, a chunk is "flipped" if the heuristic fires on
    # some chunks and not others while the unit string is constant, or vice versa.
    print("\nChunks where the value heuristic disagrees with `unit`:")
    for analyte, test in SUSPECT.items():
        sub = df[df["analyte"] == analyte]
        if sub.empty:
            print(f"  {analyte}: absent")
            continue
        per_chunk = sub.groupby("chunk").agg(
            unit=("unit", lambda s: ", ".join(sorted(map(str, s.unique())))),
            fires=("median" if analyte == "RBC" else "p95",
                   lambda s: bool((s > (10 if analyte == "RBC" else 20)).any())))
        by_unit = per_chunk.groupby("unit")["fires"].agg(["mean", "size"])
        bad = by_unit[(by_unit["mean"] > 0) & (by_unit["mean"] < 1)]
        print(f"  {analyte}: {per_chunk['fires'].sum()}/{len(per_chunk)} chunks fire the fix; "
              f"units seen = {sorted(per_chunk['unit'].unique())}")
        if len(bad):
            print(f"    MIXED -- same unit string, different verdict:\n{bad.to_string()}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
