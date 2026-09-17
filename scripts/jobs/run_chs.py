#!/usr/bin/env python
"""Run the CHS pipeline, skipping whatever is already done.

    python jobs/run_chs.py                 # every stage, in order
    python jobs/run_chs.py --jobs 3        # three at a time (they are independent)
    python jobs/run_chs.py --n_chunks 2    # a small test first
    python jobs/run_chs.py --dry_run       # print the commands, run nothing

Nothing here decides what is finished: every stage checks its own outputs first and
returns immediately if they are written, before reading any data.  The per-chunk caches
work the same way, so an interrupted run resumes at the chunk it reached, and widening
--n_chunks later only costs the new chunks.  `--force` is passed straight through and
turns all of that off.

Stages below 07_classify read the classification and write their own result file, so
they are independent of each other: --jobs N runs N at once, each with its own log in
logs/chs/.  07_classify must finish first, and `export` runs last.

Not run here: 04_refs (the reference intervals are reused as they are -- see
jobs/import_old_norma.py) and 01_process / 02_index_labs (the chunks already carry
index_labs.parquet).
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "logs", "chs")

# (stage, [argv]).  The order is the pipeline's; everything after 07_classify may run
# in any order, which is what --jobs exploits.
def stages(args):
    ds = ["--dataset", "chs"] + (["--n_chunks", str(args.n_chunks)] if args.n_chunks else []) \
         + (["--force"] if args.force else []) \
         + (["--analytes", args.analytes] if args.analytes else [])
    return [
        ("07_classify",    ["07_classify.py"] + ds),
        ("06_calibration", ["06_calibration.py"] + ds),
        ("05_forecasting", ["05_forecasting.py"] + ds + ["--workers", str(args.workers)]),
        ("08_variability", ["08_variability.py"] + ds),
        ("09_age_ri",      ["09_age_ri.py"] + ds),
        ("10_mortality",   ["10_mortality.py"] + ds),
        ("12_eval",        ["12_eval.py", "--only", "metrics", "auroc"] + ds),
        ("13_cox",         ["13_cox.py"] + ds),
        ("14_patient",     ["14_patient_level.py"] + ds + ["--min-coverage", "0.75"]),
        ("16_benchmark",   ["16_benchmark.py"] + ds),
        ("17_outcomes",    ["17_outcomes.py"] + ds),
        # 11 feeds the poster's future-RR, AUC and age panels and the lead-time figure.
        # The windows are outpatient ones: a 30-day gap after the baseline draw, and a
        # year to look ahead in -- CHS index rows are months apart, not hours.
        ("11_future",      ["11_lead_time.py", "--only", "future_abnormal"] + ds
                           + ["--gap", "720", "--horizon", "8760"]),
        ("11_lead",        ["11_lead_time.py", "--only", "lead_time"] + ds
                           + ["--window", "8760"]),
        # NOT here: 12_eval --only deviation (the poster's top-10% outcome RR).  That
        # step still loads the whole classification at once, which is what ran the node
        # out of memory on 11 before it was chunked.  It needs the same treatment.
        ("export",         ["export.py", "--dataset", sub(args), "--txt"]),
        ("compact",        ["jobs/compact_export.py", "--dataset", sub(args)]),
    ]


def sub(args):
    """results/{raw,processed}/<this>/ -- chs, or chs_<N>chunks for a subset run."""
    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    return datasets.CHSDataset(n_chunks=args.n_chunks).output_sub()


def launch(stage, cmd):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{stage}.log")
    log = open(path, "w")
    proc = subprocess.Popen([sys.executable] + cmd, cwd=SCRIPTS_DIR,
                            stdout=log, stderr=subprocess.STDOUT)
    return proc, log, path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=1, help="stages at once after 07_classify")
    p.add_argument("--n_chunks", type=int, default=None, help="first N chunks only")
    p.add_argument("--analytes", default=None, help="comma-separated analytes")
    p.add_argument("--workers", type=int, default=4, help="processes for 05_forecasting")
    p.add_argument("--only", nargs="+", help="run just these stages")
    p.add_argument("--skip", nargs="+", default=[], help="stages to leave out")
    p.add_argument("--force", action="store_true", help="recompute everything, ignoring what is done")
    p.add_argument("--dry_run", action="store_true", help="print the commands, run nothing")
    args = p.parse_args()

    plan = [(s, c) for s, c in stages(args)
            if (not args.only or s in args.only) and s not in args.skip]
    print(f"CHS pipeline: {len(plan)} stage(s) — {', '.join(s for s, _ in plan)}")
    print(f"  results -> results/raw/{sub(args)}/   logs -> {LOG_DIR}")
    if args.dry_run:
        for _, cmd in plan:
            print("  python " + " ".join(cmd))
        return

    t0, failed = time.time(), []
    first = [x for x in plan if x[0] == "07_classify"]          # everything waits on it
    rest = [x for x in plan if x[0] not in ("07_classify", "export", "compact")]
    last = [x for x in plan if x[0] in ("export", "compact")]   # compact reads export's output

    for group, jobs in ((first, 1), (rest, max(1, args.jobs)), (last, 1)):
        running = {}
        queue = list(group)
        while queue or running:
            while queue and len(running) < jobs:
                stage, cmd = queue.pop(0)
                print(f"### {stage} started")
                running[stage] = launch(stage, cmd) + (time.time(),)
            if not running:
                break
            time.sleep(2)
            for stage, (proc, log, path, start) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                running.pop(stage)
                mins = (time.time() - start) / 60
                if proc.returncode == 0:
                    print(f"### {stage} done in {mins:.1f} min   ({path})")
                else:
                    failed.append(stage)
                    print(f"### {stage} FAILED ({proc.returncode}) after {mins:.1f} min "
                          f"— see {path}")
        if failed and group is first:
            break                                   # nothing below can run without it

    print(f"\n{'=' * 64}\n  finished in {(time.time() - t0) / 60:.0f} min")
    if failed:
        print(f"  failed: {', '.join(failed)}   (rerun to pick up where they stopped)")
    print(f"  copy out: results/processed/{sub(args)}/")
    print(f"  screenshots: results/compact/{sub(args)}/\n{'=' * 64}")


if __name__ == "__main__":
    main()
