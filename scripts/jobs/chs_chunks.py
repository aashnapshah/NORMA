"""Per-chunk sanity check for the CHS (Clalit) data: value distribution of one analyte
across chunks, and the reference-interval scale per method, to catch unit shifts
between chunks (the RBC 10^6/uL vs 10^3/uL and the A1C % vs mmol/mol mix-ups).

    python jobs/chs_chunks.py --analyte A1C                # values per chunk
    python jobs/chs_chunks.py --analyte RBC --refs         # + ref_intervals per method
    python jobs/chs_chunks.py --analyte RBC --n_chunks 5 --data_root data/clalit/sandbox
"""
import argparse
import glob
import os as _os, sys as _sys
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))   # scripts/
for _p in (_os.path.join(_SCRIPTS_DIR, "lib"), _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import pandas as pd

from datasets import DATASETS


def value_stats(chunk_dir, analyte):
    """Summary of the analyte's values in the chunk (index_labs, else the raw labs file)."""
    split_path = _os.path.join(chunk_dir, "index_labs.parquet")
    if _os.path.exists(split_path):
        df = pd.read_parquet(split_path, columns=["analyte", "value"])
        col, val = "analyte", "value"
    else:
        raw = glob.glob(_os.path.join(chunk_dir, "labs_*_flag*.parquet"))
        if not raw:
            return None
        df = pd.read_parquet(raw[0])
        col = "code" if "code" in df.columns else "analyte"
        val = "numeric_value" if "numeric_value" in df.columns else "value"
    v = df.loc[df[col] == analyte, val].dropna()
    if len(v) == 0:
        return pd.Series(dtype=float)
    return v.describe(percentiles=[0.05, 0.5, 0.95, 0.99])


def ref_stats(chunk_dir, analyte):
    """Median ri_mean / ri_std / ri_low / ri_high per method for the analyte."""
    path = _os.path.join(chunk_dir, "ref_intervals.parquet")
    if not _os.path.exists(path):
        return None
    ref = pd.read_parquet(path)
    ref = ref[ref["analyte"] == analyte]
    if len(ref) == 0:
        return None
    cols = [c for c in ["ri_mean", "ri_std", "ri_low", "ri_high"] if c in ref.columns]
    out = ref.groupby("method")[cols].median()
    out.insert(0, "n", ref.groupby("method").size())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyte", required=True)
    ap.add_argument("--refs", action="store_true", help="also print ref_intervals per method")
    ap.add_argument("--n_chunks", type=int, default=None)
    ap.add_argument("--data_root", default=None, help="chunk directory root (default: data/clalit, else the sandbox)")
    args = ap.parse_args()

    ds = DATASETS["chs"](n_chunks=args.n_chunks, data_root=args.data_root)
    dirs = list(ds._chunk_dirs())
    print(f"{len(dirs)} chunks under {ds.data_root}\n")

    rows = {}
    for d in dirs:
        s = value_stats(d, args.analyte)
        rows[_os.path.basename(d)] = s if s is not None else pd.Series({"count": float("nan")})
    table = pd.DataFrame(rows).T
    print(f"{args.analyte} values per chunk")
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160):
        print(table.to_string())

    if args.refs:
        for d in dirs:
            r = ref_stats(d, args.analyte)
            if r is None:
                continue
            print(f"\n{_os.path.basename(d)}: {args.analyte} ref_intervals per method")
            with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 160):
                print(r.to_string())


if __name__ == "__main__":
    main()
