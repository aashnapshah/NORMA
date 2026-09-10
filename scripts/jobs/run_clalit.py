#!/usr/bin/env python
"""Run the whole CHS (Clalit) validation pipeline inside the Clalit environment.

    python jobs/run_clalit.py --check                 # preflight only, changes nothing
    python jobs/run_clalit.py                         # everything, all chunks
    python jobs/run_clalit.py --from 04_norma         # resume from a stage   (run from scripts/)
    python jobs/run_clalit.py --only 13_cox 14_patient
    python jobs/run_clalit.py --n_chunks 5 --dry_run  # print the commands
    python jobs/run_clalit.py --no_norma              # baselines only, no model

--no_norma runs the whole pipeline with the NORMA arms dropped: the 04_norma stage
disappears, every stage below it gets --no_norma (which empties dataset.run_ids, so
each one derives its method list without NORMA), and no model weights are needed --
nothing but Cohen / Gaussian / Pop_RI / Per_RI is computed or scored.  05_forecasting
builds its target rows from index_labs instead of from the norma step's predictions,
and 16_benchmark's DeLong reference falls back from NORMA to Per_RI.  Use it to get
the CHS baseline half out before the bundle is carried in, or when the model is not
part of the question.  It computes no NORMA numbers but deletes none either: the
result files are upserted per analyte, so NORMA rows an earlier run already wrote
stay where they are.

The heavy per-pair steps of 04_refs.py (norma, baselines) run one chunk
at a time with `--chunk i`, which scopes the whole cohort to `chunk_i/`: they hold
one chunk in memory and write `chunk_i/ref_intervals{,_norma}.parquet` and
`chunk_i/norma_predictions.parquet`, where every CHS reader already looks.  Each step
owns one ref_intervals file (baselines vs norma), so neither can clobber the other's
rows and the two may run at the same time on a chunk.  A chunk
whose output is already there is skipped, so an interrupted run resumes for free
(`--force` recomputes).  Everything downstream of 04 runs cohort-wide, as it does
for eICU and INSPIRE.

04_refs owns every reference interval on CHS as well (decided 2026-09-01), so
01_process runs with --no_refs: pop/per come from the same GMM code as the other
cohorts and Cohen / Gaussian arrive with them, which is what makes the CHS
head-to-head against Cohen et al. possible.  The server's Bayes/setpoint
intervals are no longer merged in; the first 04_refs write of a chunk backs up the
existing ref_intervals.parquet as ref_intervals.parquet.bak-01process (the server's
pop/per rows are baselines, so only that half is ever backed up).

Carry in first, on the cluster side:
    python jobs/run_clalit.py --pack_bundle /path/to/usb/bundle
The bundle mirrors repo-relative paths (model/logs/<run_id>/, model/logs/baselines/,
model/predictions/), so inside Clalit you copy its contents over the repo root and
every reader -- datasets.MODEL_LOG_DIR, datasets.artifact(), model/baselines/
{cohen,gaussian}.py, model/states.py -- finds its file where it already looks.

Bring results back out by copying ONE folder:
    results/processed/chs/     the figure data, a few hundred KB (export.py writes it
                                          after every stage; --txt adds a fixed-width copy for screenshots)
and dropping it at the same path here; `make_figures.py --dataset chs` then draws the
CHS row of every figure into results/figures/chs/, and the plain build fills it into results/figures/all/.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

# this file lives in scripts/jobs/; the commands below are relative to scripts/
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.dirname(SCRIPTS_DIR)
LOG_DIR = os.path.join(VAL_DIR, "logs", "clalit")
STATE_FILE = os.path.join(LOG_DIR, "progress.json")
DATA_ROOT = os.path.join(VAL_DIR, "data", "clalit")
SANDBOX_ROOT = os.path.join(DATA_ROOT, "sandbox")

# what a finished 04_refs chunk carries, so a resumed run knows to skip it
REFS_METHODS = {"pop", "per", "cohen_m2", "cohen_m3", "cohen_m4",
                "gaussian_mle", "gaussian_trunc", "gaussian_eb"}

# files to carry into Clalit, as paths relative to the repo root: the bundle is an
# overlay of the repo, so a file lands inside Clalit exactly where it sits here.
BUNDLE = [
    "model/logs/{run_id}/checkpoint_{ckpt}.pth",
    "model/logs/{run_id}/checkpoint_{ckpt}.json",
    "model/logs/baselines/cohen_dev_models.pkl",
    "model/logs/baselines/gaussian_eb_prior_dev.pkl",
    "model/predictions/state_priors_combined.json",
]


# ────────────────────────────────────────────────────────────── stages

def stages(args):
    """(id, description, [argv...] or a callable taking (args) -> list of argv)."""
    ds = ["--dataset", "chs"]
    # every stage from 04 down shares add_dataset_args, so it takes --no_norma;
    # 02_index_labs and 03_cohort_summary have their own parsers and never look at a
    # method, so they run unchanged either way.
    dsn = ds + (["--no_norma"] if args.no_norma else [])
    nc = ["--n_chunks", str(args.n_chunks)] if args.n_chunks else []
    an = ["--analytes", args.analytes] if args.analytes else []

    stage_list = [
        ("01_process", "process/clalit.py: raw chunks -> processed.parquet + diagnosis.parquet",
         [[os.path.join(SCRIPTS_DIR, "process", "clalit.py"), "--n_chunks", str(args.n_chunks or count_chunks(raw=True)),
           *(["--sandbox"] if data_root() == SANDBOX_ROOT else []),   # never touch the real paths from here
           "--no_refs"] + (["--force"] if args.force else []) + an]),
        ("02_index_labs", "baseline / index split (the server's pre-2015 window flag)",
         [["02_index_labs.py"] + ds + nc + an + (["--force"] if args.force else [])]),
        ("03_cohort", "cohort + demographics tables, overall and by split",
         [["03_cohort_summary.py"] + ds + nc + an]),
        ("04_norma", "NORMA at the actual first-index time, all three states (per chunk)",
         per_chunk(args, "04_refs.py", ["--only", "norma", "--device", args.device], norma_done)),
        ("04_baselines", "PopRI / PerRI / Cohen m2-m4 / Gaussian mle-trunc-eb (per chunk)",
         per_chunk(args, "04_refs.py", ["--only", "baselines"]
                   + (["--no_norma"] if args.no_norma else []), baselines_done)),
        ("07_classify", "classify every index measurement per method (+ exposure markers), prevalence",
         [["07_classify.py"] + dsn + nc + an + (["--force"] if args.force else [])]),
        ("06_calibration", "coverage / width of every method, conformal widths (reads classification)",
         [["06_calibration.py"] + dsn + nc + an]),
        ("05_forecasting", "NORMA vs Last / Mean / ARIMA and the interval centres",
         [["05_forecasting.py"] + dsn + nc + an + ["--workers", str(args.workers)]]),
        ("08_variability", "intra / inter-individual CV, individuality index",
         [["08_variability.py"] + dsn + nc + an]),
        ("09_age_ri", "age-stratified reference intervals",
         [["09_age_ri.py"] + dsn + nc + an]),
        ("10_mortality", "mortality by value quintile and by z-score deviation",
         [["10_mortality.py"] + dsn + nc + an]),
        ("11_lead_time", "future Pop_RI abnormality from a Pop_RI-normal index (Cohen Fig. 5a/b)",
         [["11_lead_time.py", "--only", "future_abnormal"] + dsn + nc + an]),
        ("12_eval", "PPV / sensitivity / specificity, AUROC and the matched-rate deviation score per analyte, method, outcome",
         [["12_eval.py", "--only", "metrics", "auroc", "deviation"] + dsn + nc + an
          + (["--force"] if args.force else [])]),
        ("13_cox", "landmark Cox per (outcome, analyte, method), one row per patient",
         [["13_cox.py"] + dsn + nc + an]),
        ("14_patient", "patient-level multi-analyte survival models and NRI (refit + swap)",
         [["14_patient_level.py"] + dsn + nc + an + ["--min-coverage", "0.75"]]),
        ("16_benchmark", "AUROC, matched operating points, abnormal burden, DeLong tests",
         [["16_benchmark.py"] + dsn + nc + an]),
        ("11_lead", "lead time before a value leaves Pop_RI (disease-agnostic)",
         [["11_lead_time.py", "--only", "lead_time"] + dsn + nc + an]),
        ("17_outcomes", "clinical endpoints: incidence at matched sensitivity, earliness",
         [["17_outcomes.py"] + dsn + nc + an]),
        ("export", "figure data: results/processed/chs/ (the folder to copy out; "
                   "also refreshed after every stage above)",
         [["export.py", "--dataset", "chs", "--txt"]]),
    ]
    if args.no_norma:
        stage_list = [s for s in stage_list if s[0] != "04_norma"]
    return stage_list


def per_chunk(args, script, extra, done_fn):
    """One invocation per chunk, skipping the chunks that already have output."""
    cmds = []
    for i, chunk_dir in chunk_dirs(args.n_chunks):
        if not args.force and done_fn(chunk_dir):
            continue
        cmds.append([script, "--dataset", "chs", "--chunk", str(i)] + extra
                    + (["--analytes", args.analytes] if args.analytes else []))
    return cmds


def norma_done(chunk_dir):
    return os.path.exists(os.path.join(chunk_dir, "norma_predictions.parquet"))


def baselines_done(chunk_dir):
    # every method in REFS_METHODS is a baseline, so only that half is read
    p = os.path.join(chunk_dir, "ref_intervals.parquet")
    if not os.path.exists(p):
        return False
    try:
        import pandas as pd
        return REFS_METHODS <= set(pd.read_parquet(p, columns=["method"])["method"].unique())
    except Exception:
        return False


# ────────────────────────────────────────────────────────────── chunks

def data_root():
    return DATA_ROOT if glob.glob(os.path.join(DATA_ROOT, "chunk_*")) else SANDBOX_ROOT


def chunk_dirs(n_chunks=None):
    """[(index, path)] of the chunk directories, in numeric order."""
    dirs = sorted(glob.glob(os.path.join(data_root(), "chunk_*")),
                  key=lambda p: int(re.search(r"chunk_(\d+)$", p).group(1)))
    out = [(int(re.search(r"chunk_(\d+)$", d).group(1)), d) for d in dirs]
    return out[:n_chunks] if n_chunks else out


def count_chunks(raw=False):
    """How many chunks exist. `raw=True` counts the source chunks 01_process reads,
    which on the first run is the only place they exist yet."""
    n = len(chunk_dirs())
    if n or not raw:
        return n
    sys.path.insert(0, os.path.join(os.path.dirname(VAL_DIR), "process"))
    try:
        import clalit
        src = os.path.dirname(os.path.dirname(clalit.REAL_PATHS["labs"]))
        return len(glob.glob(os.path.join(src, "chunk_*")))
    except Exception:
        return 0
    finally:
        sys.path.pop(0)


def backup_server_refs(n_chunks=None):
    """Keep the 01_process ref_intervals (server Bayes/setpoint) before 04_refs
    overwrites pop/per — once per chunk, never overwriting an existing backup."""
    kept = 0
    for _, d in chunk_dirs(n_chunks):
        p = os.path.join(d, "ref_intervals.parquet")
        bak = p + ".bak-01process"
        if os.path.exists(p) and not os.path.exists(bak) and not baselines_done(d):
            shutil.copy2(p, bak)
            kept += 1
    if kept:
        print(f"  backed up {kept} chunk ref_intervals.parquet as .bak-01process")


# ────────────────────────────────────────────────────────────── preflight

def preflight(args):
    ok = True
    print("Preflight")
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")

    missing = []
    # matplotlib is on the list because a stage script now holds its own figure
    # code and imports figlib at module level; nothing here draws, but the import
    # has to succeed for the stage to run at all.
    for mod in (["numpy", "pandas", "pyarrow", "scipy", "sklearn", "joblib", "tqdm",
                 "lifelines", "statsmodels", "sksurv", "matplotlib"]
                + ([] if args.no_norma else ["torch"])):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print(f"  packages      {'all present' if not missing else 'MISSING: ' + ', '.join(missing)}")
    ok &= not missing

    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    if args.no_norma:
        print("  NORMA         not used (--no_norma), no checkpoint needed")
    else:
        run_id, ckpt = datasets.NORMA_RUN_ID, datasets.NORMA_CHECKPOINT
        weights = os.path.join(datasets.MODEL_LOG_DIR, run_id, f"checkpoint_{ckpt}.pth")
        print(f"  NORMA {run_id}  checkpoint_{ckpt}: "
              f"{weights if os.path.exists(weights) else 'NOT FOUND at ' + weights}")
        ok &= os.path.exists(weights)

    cohen = datasets.artifact("cohen_dev_models.pkl")
    print(f"  Cohen models  {cohen if os.path.exists(cohen) else 'NOT FOUND at ' + cohen}")
    ok &= os.path.exists(cohen)

    root = data_root()
    n_out, n_raw = len(chunk_dirs()), count_chunks(raw=True)
    print(f"  chunks        {n_out} under {root}"
          + (f" (source has {n_raw})" if n_raw and not n_out else ""))
    if n_out:
        done_b = sum(baselines_done(d) for _, d in chunk_dirs())
        if args.no_norma:
            print(f"  04_refs done  baselines {done_b}/{n_out} chunks")
        else:
            done_n = sum(norma_done(d) for _, d in chunk_dirs())
            print(f"  04_refs done  NORMA {done_n}/{n_out} chunks, baselines {done_b}/{n_out}")
    ok &= bool(n_out or n_raw)
    if root == SANDBOX_ROOT:
        print("  NOTE: no real chunks found - this would run on the fake sandbox data")

    print("  " + ("ready" if ok else "NOT ready - fix the items above"))
    return ok


def pack_bundle(out_dir):
    """Assemble the inbound bundle on the cluster side."""
    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    root = VAL_DIR
    fmt = {"run_id": datasets.NORMA_RUN_ID, "ckpt": datasets.NORMA_CHECKPOINT}
    total = 0
    for tpl in BUNDLE:
        rel = tpl.format(**fmt)
        src = os.path.join(root, rel)
        dst = os.path.join(out_dir, rel)
        if not os.path.exists(src):
            print(f"  MISSING  {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        mb = os.path.getsize(dst) / 1e6
        total += mb
        print(f"  {mb:8.1f} MB  {dst}")
    print(f"  {total:8.1f} MB  total -> copy the contents of this directory over the repo root inside Clalit")


# ────────────────────────────────────────────────────────────── driver

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": []}


def save_state(state):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run(cmd, log_path, dry_run=False):
    printable = " ".join(["python"] + cmd)
    print(f"\n{'=' * 72}\n  {printable}\n{'=' * 72}", flush=True)
    if dry_run:
        return 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable] + cmd, cwd=SCRIPTS_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    mins = (time.time() - t0) / 60
    print(f"  -- {'done' if proc.returncode == 0 else 'FAILED (%d)' % proc.returncode} "
          f"in {mins:.1f} min, log: {log_path} --", flush=True)
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list the stages and exit")
    p.add_argument("--check", action="store_true", help="preflight only, run nothing")
    p.add_argument("--pack_bundle", metavar="DIR",
                   help="cluster side: assemble the artifacts to carry into Clalit")
    p.add_argument("--from", dest="from_stage", help="start at this stage id")
    p.add_argument("--to", dest="to_stage", help="stop after this stage id")
    p.add_argument("--only", nargs="+", help="run just these stage ids")
    p.add_argument("--skip", nargs="+", default=[], help="stage ids to leave out")
    p.add_argument("--n_chunks", type=int, help="first N chunks only (default: all)")
    p.add_argument("--analytes", help="comma-separated analytes (default: all)")
    p.add_argument("--device", default="cpu", help="NORMA inference device (default: cpu)")
    p.add_argument("--no_norma", action="store_true",
                   help="baselines only: drop the 04_norma stage and every NORMA arm "
                        "downstream (no model weights needed)")
    p.add_argument("--workers", type=int, default=4, help="processes for 05_forecasting")
    p.add_argument("--force", action="store_true",
                   help="recompute finished chunks and re-derive cached per-chunk results")
    p.add_argument("--resume", action="store_true",
                   help="skip whole stages recorded complete in logs/clalit/progress.json")
    p.add_argument("--keep_going", action="store_true", help="carry on after a failing stage")
    p.add_argument("--dry_run", action="store_true", help="print the commands, run nothing")
    args = p.parse_args()

    if args.pack_bundle:
        return pack_bundle(args.pack_bundle)

    all_stages = stages(args)
    if args.list:
        for sid, desc, cmds in all_stages:
            print(f"  {sid:16s} {desc}"
                  + (f"  [{len(cmds)} chunk(s) outstanding]" if sid.startswith("04_") else ""))
        return

    if not preflight(args) and not (args.dry_run or args.check):
        sys.exit("\nPreflight failed; nothing run.  Fix the items above or use --dry_run.")
    if args.check:
        return

    ids = [s[0] for s in all_stages]
    selected = set(args.only) if args.only else set(ids)
    if args.from_stage:
        selected &= set(ids[ids.index(args.from_stage):])
    if args.to_stage:
        selected &= set(ids[:ids.index(args.to_stage) + 1])
    selected -= set(args.skip)
    state = load_state()
    if args.resume:
        selected -= set(state["completed"])

    plan = [s for s in all_stages if s[0] in selected]
    print(f"\nCHS pipeline: {len(plan)} stage(s) — {', '.join(s[0] for s in plan)}")
    if any(s[0].startswith("04_") for s in plan) and not args.dry_run:
        backup_server_refs(args.n_chunks)

    t_start = time.time()
    failed = []
    for sid, desc, cmds in plan:
        print(f"\n\n### {sid} — {desc}")
        if not cmds:
            print("  nothing outstanding (every chunk already has its output)")
            continue
        for n, cmd in enumerate(cmds):
            tag = f"{sid}_{n}" if len(cmds) > 1 else sid
            if sid.startswith("04_"):
                tag = f"{sid}_chunk{cmd[cmd.index('--chunk') + 1]}"
            rc = run(cmd, os.path.join(LOG_DIR, f"{tag}.log"), dry_run=args.dry_run)
            if rc != 0:
                failed.append(tag)
                if not args.keep_going:
                    save_state(state)
                    sys.exit(f"\n{sid} failed; stopping.  Re-run with --from {sid} once fixed "
                             f"(or --keep_going to push past it).")
        if sid not in state["completed"] and not args.dry_run:
            state["completed"].append(sid)
            save_state(state)
        if sid != "export" and not args.dry_run:
            # keep results/processed/chs/ current, so a partial run can already be copied out
            stage = os.path.basename(cmds[0][0]).replace(".py", "")
            run(["export.py", "--dataset", "chs", "--quiet"],
                os.path.join(LOG_DIR, "export.log"))

    print(f"\n{'=' * 72}")
    print(f"  CHS pipeline finished in {(time.time() - t_start) / 60:.0f} min")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    print("  next: copy results/processed/chs/ out; make_figures.py --dataset chs draws it")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
