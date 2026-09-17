#!/usr/bin/env python
"""Build every validation figure as an individual PDF: results/figures/<tag>/<nn>_<name>.pdf.

Usage:
    python make_figures.py                       # all figures, all cohorts -> results/figures/all/
    python make_figures.py --only 12_eval cox    # by stage or figure base name
    python make_figures.py --dataset chs         # one cohort only -> results/figures/chs/ (pooled figures get that row alone)
    python make_figures.py --dataset eicu --from processed   # prove results/processed/eicu/ alone can draw everything
    python make_figures.py --check               # list missing result inputs per cohort; no plotting
    python make_figures.py --list                # show the registry
"""

import bootstrap  # noqa: F401

import argparse
import os
import sys
import traceback

import figlib as P
from figlib import PREDICTION_INPUTS, output_name
from datasets import find_in
from datasets import load_registries

# each scripts/<stage>.py contributes its slice of the registry; the outputs
# go to results/figures/<tag>/.
FIGURES = load_registries("FIGURES")

if not FIGURES:
    raise SystemExit(
        "No specs discovered: every scripts/<stage>.py with figures defines a FIGURES list, so an "
        "empty registry means none were found (wrong working directory, a renamed "
        "registry variable, or a folder excluded by discover.SKIP). Failing rather "
        "than reporting success having built nothing."
    )


def _call(fn, *args):
    """Run one figure function; a failure becomes a printed traceback and an empty result
    (-> placeholder), so one broken figure never stops the build."""
    try:
        return fn(*args) or {}
    except Exception:
        traceback.print_exc()
        FAILED.append(f"{fn.__module__}.{fn.__name__}")
        return {}


FAILED = []


def build_figure(spec):
    print(f"\n[{spec.analysis}/{spec.base}]")
    if spec.per_dataset:
        for ds in P.DATASETS:
            missing = P.missing_results(ds, *spec.requires)
            if missing:
                for name in spec.expected(ds):
                    P.save_placeholder(spec.analysis, name, P.pending_message(ds, missing))
                continue
            out = _call(spec.fn, ds)
            if not out:
                for name in spec.expected(ds):
                    P.save_placeholder(spec.analysis, name, f"{P.DATASET_DISPLAY.get(ds, ds)}: no usable rows in inputs")
            for suffix, fig in out.items():
                P.save_fig(fig, spec.analysis, output_name(spec.base, ds, suffix))
    else:
        out = _call(spec.fn)
        if not out:
            P.save_placeholder(spec.analysis, spec.base, "pending: no inputs found")
        for suffix, fig in out.items():
            P.save_fig(fig, spec.analysis, output_name(spec.base, None, suffix))


def check_inputs(specs):
    """Report which result files each dataset still lacks (i.e. what to copy in from Clalit)."""
    ok = True
    required = sorted({f for s in specs if s.per_dataset for f in s.requires})
    for ds in P.DATASETS:
        missing = P.missing_results(ds, *required)
        status = "ok" if not missing else f"missing {len(missing)}/{len(required)}"
        print(f"{P.DATASET_DISPLAY.get(ds, ds):10s} results/raw/{ds}/  {status}")
        for f in missing:
            print(f"    - {f}")
        ok &= not missing
    missing_pred = [f for f in PREDICTION_INPUTS if not os.path.exists(find_in(P.PREDICTION_DIR, f))]
    print(f"{'prediction':10s} results/raw/dev/  {'ok' if not missing_pred else f'missing {len(missing_pred)}/{len(PREDICTION_INPUTS)}'}")
    for f in missing_pred:
        print(f"    - {f}")
    return ok and not missing_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="analysis folder(s) or figure base name(s)")
    ap.add_argument("--from", dest="kinds", nargs="+", choices=list(P.RESULT_KINDS), metavar="KIND",
                    help="read only these result kinds (raw / processed); default raw then processed")
    ap.add_argument("--dataset", nargs="+", choices=list(P.DATASETS), help="restrict per-dataset figures")
    ap.add_argument("--check", action="store_true", help="only report missing inputs")
    ap.add_argument("--list", action="store_true", help="print the figure registry")
    args = ap.parse_args()

    if args.kinds:
        P.RESULT_KINDS[:] = args.kinds
    if args.dataset:
        P.set_datasets(args.dataset)
    specs = [s for s in FIGURES if not args.only or s.analysis in args.only or s.base in args.only]
    if args.only and not specs:
        sys.exit(f"nothing matches {args.only}; known: " + ", ".join(sorted({s.analysis for s in FIGURES} | {s.base for s in FIGURES})))

    if args.list:
        for s in specs:
            print(f"{s.analysis:14s} {s.base:28s} {'per-dataset' if s.per_dataset else 'pooled':12s} requires: {', '.join(s.requires) or '-'}")
        return
    if args.check:
        sys.exit(0 if check_inputs(specs) else 1)

    P.setup_style()
    for s in specs:
        build_figure(s)
    print(f"\nFigures written under {os.path.relpath(os.path.join(P.FIGURES_DIR, P.OUTPUT_TAG), os.getcwd())}/")
    if FAILED:
        sys.exit(f"{len(FAILED)} figure function(s) raised (placeholders written): " + ", ".join(FAILED))


if __name__ == "__main__":
    main()
