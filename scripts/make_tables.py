#!/usr/bin/env python
"""Build every validation table under results/tables/<tag>/{tex,csv,pdf}/<nn>_<name>.<ext>.

Usage:
    python make_tables.py                      # all tables, all cohorts -> results/tables/all/
    python make_tables.py --only 12_eval cox   # by stage or table base name
    python make_tables.py --dataset chs        # one cohort only -> results/tables/chs/
    python make_tables.py --dataset eicu --from processed   # from results/processed/eicu/ alone
    python make_tables.py --no-pdf             # skip the tectonic compile
    python make_tables.py --check              # list missing result inputs per dataset
    python make_tables.py --list
"""

import bootstrap  # noqa: F401

import argparse
import os
import sys
import traceback

import figlib as P
from datasets import load_registries

# each scripts/<stage>.py contributes its slice of the registry
TABLES = load_registries("TABLES")

if not TABLES:
    raise SystemExit(
        "No specs discovered: every scripts/<stage>.py with tables defines a TABLES list, so an "
        "empty registry means none were found (wrong working directory, a renamed "
        "registry variable, or a folder excluded by discover.SKIP). Failing rather "
        "than reporting success having built nothing."
    )


def _call(fn, *args):
    """Run one table function; a failure becomes a printed traceback and an empty result
    (-> placeholder), so one broken table never stops the build."""
    try:
        return fn(*args) or []
    except Exception:
        traceback.print_exc()
        FAILED.append(f"{fn.__module__}.{fn.__name__}")
        return []


FAILED = []


def build_table(spec):
    print(f"\n[{spec.analysis}/{spec.base}]")
    if spec.per_dataset:
        for ds in P.DATASETS:
            missing = P.missing_results(ds, *spec.requires)
            if missing:
                for name in spec.expected(ds):
                    P.save_placeholder_table(spec.analysis, name, P.pending_message(ds, missing))
                continue
            written = _call(spec.fn, ds)
            if not written:
                for name in spec.expected(ds):
                    P.save_placeholder_table(spec.analysis, name, f"{P.DATASET_DISPLAY.get(ds, ds)}: no usable rows in inputs")
    else:
        written = _call(spec.fn)
        if not written:
            P.save_placeholder_table(spec.analysis, spec.base, "pending: no inputs found")


def check_inputs(specs):
    ok = True
    required = sorted({f for s in specs if s.per_dataset for f in s.requires})
    for ds in P.DATASETS:
        missing = P.missing_results(ds, *required)
        print(f"{P.DATASET_DISPLAY.get(ds, ds):10s} results/raw/{ds}/  {'ok' if not missing else f'missing {len(missing)}/{len(required)}'}")
        for f in missing:
            print(f"    - {f}")
        ok &= not missing
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--from", dest="kinds", nargs="+", choices=list(P.RESULT_KINDS), metavar="KIND",
                    help="read only these result kinds (raw / processed); default raw then processed")
    ap.add_argument("--dataset", nargs="+", choices=list(P.DATASETS))
    ap.add_argument("--no-pdf", action="store_true", help="write csv + tex only")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.kinds:
        P.RESULT_KINDS[:] = args.kinds
    if args.dataset:
        P.set_datasets(args.dataset)
    if args.no_pdf:
        P.PDF_TABLES = False
    specs = [s for s in TABLES if not args.only or s.analysis in args.only or s.base in args.only]
    if args.only and not specs:
        sys.exit(f"nothing matches {args.only}; known: " + ", ".join(sorted({s.analysis for s in TABLES} | {s.base for s in TABLES})))

    if args.list:
        for s in specs:
            print(f"{s.analysis:14s} {s.base:26s} {'per-dataset' if s.per_dataset else 'pooled':12s} requires: {', '.join(s.requires) or '-'}")
        return
    if args.check:
        sys.exit(0 if check_inputs(specs) else 1)

    for s in specs:
        build_table(s)
    print(f"\nTables written under {os.path.relpath(os.path.join(P.TABLES_DIR, P.OUTPUT_TAG), os.getcwd())}/")
    if FAILED:
        sys.exit(f"{len(FAILED)} table function(s) raised (placeholders written): " + ", ".join(FAILED))


if __name__ == "__main__":
    main()
