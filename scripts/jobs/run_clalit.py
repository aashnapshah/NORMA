#!/usr/bin/env python
"""Run the whole CHS (Clalit) validation pipeline inside the Clalit environment.

    python jobs/run_clalit.py --check                 # preflight only, changes nothing
    python jobs/run_clalit.py                         # everything, all chunks
    python jobs/run_clalit.py --from 04_norma         # resume from a stage   (run from scripts/)
    python jobs/run_clalit.py --only 13_cox 14_patient
    python jobs/run_clalit.py --n_chunks 5 --dry_run  # print the commands
    python jobs/run_clalit.py --no_norma              # baselines only, no model
    python jobs/run_clalit.py --reuse_refs --n_chunks 2   # no NORMA / Cohen reruns, 2-chunk test

--reuse_refs runs everything below the reference intervals on the intervals the chunks
already hold: no raw reprocessing or re-split (01_process, 02_index_labs), no NORMA inference (04_norma), no Cohen
fitting (04_baselines gets an empty --cohen_models, so only the Gaussian arms it lacks are
fitted).  Before the first stage it prints which methods each chunk carries.  Every
downstream stage then gets --drop_arms for the arms some chunk has no rows for, and, when
the chunks carry an older NORMA run instead of datasets.NORMA_RUN_ID, --norma_alias so
those rows are scored as "NORMA"; the run actually used is written to
results/processed/<cohort>/NORMA_RUN.txt, so it travels out with the figure data.  The
per-chunk caches an earlier pipeline version left (classification, eval / prevalence
counts, mortality extract) are renamed to <file>.bak-reuse once, so they are rebuilt from
the intervals rather than reused; rename them back to undo.  --norma_run RUN picks the
NORMA run by hand.  Put old NORMA intervals back first with jobs/import_old_norma.py.
With --n_chunks N (fewer than all) results go to results/raw/chs_Nchunks/
and results/processed/chs_Nchunks/, never over a full run's.

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


# What each stage needs finished before it can start.  Everything below 07 reads the
# classification and writes its own result file, so those stages are independent of each
# other and can run at the same time (--jobs).  The per-chunk caches they build are
# per stage, so parallel runs do not collide.
DEPENDS = {
    "02_index_labs": ["01_process"],
    "03_cohort": ["02_index_labs"],
    "04_norma": ["02_index_labs"],
    "04_baselines": ["02_index_labs"],
    "07_classify": ["04_baselines"],
    "06_calibration": ["07_classify"],
    "05_forecasting": ["04_baselines"],
    "08_variability": ["07_classify"],
    "09_age_ri": ["07_classify"],
    "10_mortality": ["07_classify"],
    "11_lead_time": ["07_classify"],
    "12_eval": ["07_classify"],
    "13_cox": ["07_classify"],
    "14_patient": ["07_classify"],
    "16_benchmark": ["07_classify"],
    "11_lead": ["07_classify"],
    "17_outcomes": ["07_classify"],
}
# export copies out whatever is written, so it waits for every other stage in the plan
DEPENDS["export"] = [s for s in DEPENDS if s != "export"]


# ────────────────────────────────────────────────────────────── stages

def stages(args):
    """(id, description, [argv...] or a callable taking (args) -> list of argv)."""
    ds = ["--dataset", "chs"]
    # every stage from 04 down shares add_dataset_args, so it takes --no_norma;
    # 02_index_labs and 03_cohort_summary have their own parsers and never look at a
    # method, so they run unchanged either way.
    # --force reaches every stage below 04 through the shared dataset parser, so one flag
    # rebuilds the classification, the per-chunk count caches and every result file.
    dsn = (ds + (["--no_norma"] if args.no_norma else [])
           + (["--force"] if args.force else []))   # --reuse_refs flags are added at run time
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
                   + (["--no_norma"] if args.no_norma else [])
                   + (["--cohen_models"] if args.reuse_refs else []),      # empty: fit no Cohen model
                   reuse_baselines_done if args.reuse_refs else baselines_done)),
        ("07_classify", "classify every index measurement per method (+ exposure markers), prevalence",
         [["07_classify.py"] + dsn + nc + an]),
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
         [["12_eval.py", "--only", "metrics", "auroc", "deviation"] + dsn + nc + an]),
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
         [["export.py", "--dataset", output_sub(args), "--txt"]]),
    ]
    if args.no_norma:
        stage_list = [s for s in stage_list if s[0] != "04_norma"]
    if args.reuse_refs:
        # the chunks already hold index_labs, and NORMA is what is on disk
        stage_list = [s for s in stage_list if s[0] not in ("01_process", "02_index_labs", "04_norma")]
    return stage_list


def output_sub(args):
    """results/{raw,processed}/<this>/ -- chs, chs_<N>chunks, or sandbox."""
    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    return datasets.CHSDataset(n_chunks=args.n_chunks).output_sub()


def per_chunk(args, script, extra, done_fn):
    """One invocation per chunk, skipping the chunks that already have output."""
    cmds = []
    # --reuse_refs means the intervals on disk are the input, so --force (which is about
    # recomputing everything DOWNSTREAM of them) must not send them through again.
    redo = args.force and not args.reuse_refs
    for i, chunk_dir in chunk_dirs(args.n_chunks):
        if not redo and done_fn(chunk_dir):
            continue
        cmds.append([script, "--dataset", "chs", "--chunk", str(i)] + extra
                    + (["--analytes", args.analytes] if args.analytes else []))
    return cmds


def norma_done(chunk_dir):
    return os.path.exists(os.path.join(chunk_dir, "norma_predictions.parquet"))


def chunk_methods(chunk_dir):
    """{method: patient-analyte pairs} over both ref_intervals files of a chunk."""
    import pandas as pd
    out = {}
    for name in ("ref_intervals.parquet", "ref_intervals_norma.parquet"):
        p = os.path.join(chunk_dir, name)
        if os.path.exists(p):
            r = pd.read_parquet(p, columns=["patient_id", "analyte", "method"]).drop_duplicates()
            for m, n in r["method"].value_counts().items():
                out[m] = out.get(m, 0) + int(n)
    return out


def reuse_baselines_done(chunk_dir):
    # --reuse_refs never fits Cohen, so a chunk is done once core + Gaussian are there
    return {m for m in REFS_METHODS if not m.startswith("cohen_")} <= set(chunk_methods(chunk_dir))


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


# ────────────────────────────────────────────────────────────── --reuse_refs

# per-chunk files an earlier pipeline version cached from the old intervals
STALE_CHUNK_FILES = ["classification.parquet", "classification_norma.parquet", "classification.csv",
                     "eval_counts.parquet", "prevalence_counts.parquet", "mortality_extract.parquet"]
ARM_PREFIXES = ("cohen_", "gaussian_", "norma_")


def inventory(n_chunks=None):
    """Print which methods (patient-analyte pairs) each chunk holds; return {chunk: methods}."""
    import pandas as pd
    per = {os.path.basename(d): chunk_methods(d) for _, d in chunk_dirs(n_chunks)}
    table = pd.DataFrame(per).T.fillna(0).astype(int)
    extra = pd.DataFrame({os.path.basename(d): {
        "norma_predictions": int(norma_done(d)),
        "index_labs": int(os.path.exists(os.path.join(d, "index_labs.parquet"))),
        "diagnosis": int(os.path.exists(os.path.join(d, "diagnosis.parquet")))}
        for _, d in chunk_dirs(n_chunks)}).T
    print("\n  ref_intervals pairs per method (both files), and inputs present (1 = yes):")
    print("    " + table.join(extra).to_string().replace("\n", "\n    "))
    return per


def reuse_flags(args):
    """--drop_arms / --norma_alias for the downstream stages, from what is on disk now."""
    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    per = {os.path.basename(d): set(chunk_methods(d)) for _, d in chunk_dirs(args.n_chunks)}
    everywhere = set.intersection(*per.values()) if per else set()
    base = datasets.BaseDataset
    expected = ([f"cohen_{m}" for m in base.cohen_models]
                + [f"gaussian_{m}" for m in base.gaussian_models])
    drop = [m for m in expected if m not in everywhere]
    flags, main = [], f"norma_{datasets.NORMA_RUN_ID}"
    runs = sorted(m[len("norma_"):] for m in everywhere if m.startswith("norma_"))
    if args.norma_run:
        run = args.norma_run
        if f"norma_{run}" not in everywhere:
            sys.exit(f"--norma_run {run}: not every chunk has norma_{run} rows (found {runs})")
    elif main in everywhere:
        run = datasets.NORMA_RUN_ID
    else:
        run = runs[0] if runs else None
        if len(runs) > 1:
            print(f"  NOTE: several NORMA runs in every chunk {runs}; using {run} (--norma_run to pick)")
    if run is None:
        drop.append(main)
    elif run != datasets.NORMA_RUN_ID:
        flags += ["--norma_alias", run]
    partial = sorted({m for ms in per.values() for m in ms if m.startswith(ARM_PREFIXES)} - everywhere)
    if partial:
        print(f"  NOTE: in some chunks only, left out: {partial}")
    if drop:
        flags += ["--drop_arms", ",".join(drop)]
    print(f"  NORMA run scored as NORMA: {run or 'none (no NORMA rows in every chunk)'}")
    print(f"  arms left out (no rows): {', '.join(drop) or 'none'}")
    return flags, run


def set_aside_stale_caches(n_chunks=None):
    """Rename an earlier version's per-chunk caches to <file>.bak-reuse, once per file."""
    moved = 0
    for _, d in chunk_dirs(n_chunks):
        for name in STALE_CHUNK_FILES:
            p = os.path.join(d, name)
            if os.path.exists(p) and not os.path.exists(p + ".bak-reuse"):
                os.replace(p, p + ".bak-reuse")
                moved += 1
    if moved:
        print(f"  set aside {moved} cached per-chunk file(s) as *.bak-reuse (rebuilt from the intervals)")


def write_norma_note(args, run):
    d = os.path.join(VAL_DIR, "results", "processed", output_sub(args))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "NORMA_RUN.txt"), "w") as f:
        f.write(f"NORMA rows scored as NORMA on this cohort: norma_{run}\n" if run
                else "No NORMA rows on this cohort; NORMA not scored.\n")


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
                + ([] if args.no_norma or args.reuse_refs else ["torch"])):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print(f"  packages      {'all present' if not missing else 'MISSING: ' + ', '.join(missing)}")
    ok &= not missing

    sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))
    import datasets
    if args.no_norma or args.reuse_refs:
        print(f"  NORMA         not run ({'--no_norma' if args.no_norma else '--reuse_refs'}), "
              f"no checkpoint needed")
    else:
        run_id, ckpt = datasets.NORMA_RUN_ID, datasets.NORMA_CHECKPOINT
        weights = os.path.join(datasets.MODEL_LOG_DIR, run_id, f"checkpoint_{ckpt}.pth")
        print(f"  NORMA {run_id}  checkpoint_{ckpt}: "
              f"{weights if os.path.exists(weights) else 'NOT FOUND at ' + weights}")
        ok &= os.path.exists(weights)

    if args.reuse_refs:
        print("  Cohen models  not fitted (--reuse_refs), none needed")
    else:
        cohen = datasets.artifact("cohen_dev_models.pkl")
        print(f"  Cohen models  {cohen if os.path.exists(cohen) else 'NOT FOUND at ' + cohen}")
        ok &= os.path.exists(cohen)

    root = data_root()
    n_out, n_raw = len(chunk_dirs()), count_chunks(raw=True)
    print(f"  chunks        {n_out} under {root}"
          + (f" (source has {n_raw})" if n_raw and not n_out else ""))
    if n_out:
        done_b = sum(baselines_done(d) for _, d in chunk_dirs())
        if args.reuse_refs:
            done_r = sum(reuse_baselines_done(d) for _, d in chunk_dirs(args.n_chunks))
            print(f"  04_refs done  pop/per/Gaussian in {done_r}/{len(chunk_dirs(args.n_chunks))} selected chunks")
        elif args.no_norma:
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

def run_parallel(plan, args, extra_for):
    """Run the plan with up to --jobs stages at once, in dependency order.

    A stage starts when every stage it depends on that is ALSO in this plan has
    finished; a dependency already done in an earlier run is not in the plan and so does
    not hold anything up.  Each stage keeps its own log, and its output is printed when
    it finishes rather than interleaved with the others'.
    """
    todo = {sid: (desc, cmds) for sid, desc, cmds in plan}
    pending = dict(todo)
    running = {}          # sid -> (Popen, log path, [remaining commands], started)
    done, failed = set(), []

    def ready(sid):
        return all(dep not in pending and dep not in running
                   for dep in DEPENDS.get(sid, []) if dep in todo)

    def start(sid):
        desc, cmds = pending.pop(sid)
        cmds = [c + extra_for(sid) for c in cmds]
        if not cmds:
            print(f"### {sid} — nothing outstanding")
            done.add(sid)
            return
        print(f"### {sid} — {desc}  [started]")
        log_path = os.path.join(LOG_DIR, f"{sid}.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        log = open(log_path, "w")
        proc = subprocess.Popen([sys.executable] + cmds[0], cwd=SCRIPTS_DIR,
                                stdout=log, stderr=subprocess.STDOUT)
        running[sid] = (proc, log, cmds[1:], time.time(), log_path)

    while pending or running:
        for sid in [s for s in pending if ready(s)]:
            if len(running) >= args.jobs:
                break
            start(sid)
        if not running:
            if pending:                     # everything left waits on something that failed
                print(f"  not run (a dependency failed): {', '.join(sorted(pending))}")
            break
        time.sleep(2)
        for sid, (proc, log, rest, t0, log_path) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            if proc.returncode == 0 and rest:      # a multi-command stage: next command
                log = open(log_path, "a")
                nxt = subprocess.Popen([sys.executable] + rest[0], cwd=SCRIPTS_DIR,
                                       stdout=log, stderr=subprocess.STDOUT)
                running[sid] = (nxt, log, rest[1:], t0, log_path)
                continue
            running.pop(sid)
            mins = (time.time() - t0) / 60
            if proc.returncode == 0:
                done.add(sid)
                print(f"### {sid} — done in {mins:.1f} min   ({log_path})")
            else:
                failed.append(sid)
                print(f"### {sid} — FAILED ({proc.returncode}) after {mins:.1f} min   "
                      f"see {log_path}")
                if not args.keep_going:
                    for other, (p, l, _, _, _) in running.items():
                        p.terminate()
                        l.close()
                    return failed
    return failed


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
    p.add_argument("--reuse_refs", action="store_true",
                   help="use the intervals the chunks already hold: no 01_process, no NORMA "
                        "inference, no Cohen fitting; missing arms are left out downstream")
    p.add_argument("--norma_run", help="--reuse_refs: the NORMA run to score as NORMA "
                                       "(default: the main run if present, else the one on disk)")
    p.add_argument("--workers", type=int, default=4, help="processes for 05_forecasting")
    p.add_argument("--force", action="store_true",
                   help="redo everything: rebuild the classification, the per-chunk caches "
                        "and every result, ignoring what is already written.  With "
                        "--reuse_refs the reference intervals themselves are still reused")
    p.add_argument("--resume", action="store_true",
                   help="skip whole stages recorded complete in logs/clalit/progress.json")
    p.add_argument("--keep_going", action="store_true", help="carry on after a failing stage")
    p.add_argument("--jobs", type=int, default=1,
                   help="stages to run at once (default 1).  Everything below 07_classify is "
                        "independent, so --jobs 3-4 is the useful range; each still writes its "
                        "own log in logs/clalit/")
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
    if args.reuse_refs:
        inventory(args.n_chunks)
    # stages that build their own parser (no --drop_arms / --norma_alias)
    own_parser = ("01_process", "02_index_labs", "03_cohort", "04_norma", "04_baselines", "export")
    extra = None

    t_start = time.time()
    failed = []
    if args.jobs > 1 and not args.dry_run:
        def extra_for(sid):
            if args.reuse_refs and sid not in own_parser:
                nonlocal_extra = reuse_flags(args)[0]
                return nonlocal_extra
            return []
        failed = run_parallel(plan, args, extra_for) or []
        print(f"\n{'=' * 72}")
        print(f"  CHS pipeline finished in {(time.time() - t_start) / 60:.0f} min "
              f"({args.jobs} stages at a time)")
        if failed:
            print(f"  failed: {', '.join(failed)}")
        print(f"  next: copy results/processed/{output_sub(args)}/ out")
        print(f"{'=' * 72}")
        return
    for sid, desc, cmds in plan:
        print(f"\n\n### {sid} — {desc}")
        if not cmds:
            print("  nothing outstanding (every chunk already has its output)")
            continue
        if args.reuse_refs and sid not in own_parser and extra is None:
            # after 04_baselines, so the Gaussian arms it just fitted count as present
            extra, norma_run = reuse_flags(args)
            if not args.dry_run:
                set_aside_stale_caches(args.n_chunks)
                write_norma_note(args, norma_run)
        if args.reuse_refs and sid not in own_parser:
            cmds = [c + extra for c in cmds]
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
            run(["export.py", "--dataset", output_sub(args), "--quiet"],
                os.path.join(LOG_DIR, "export.log"))

    print(f"\n{'=' * 72}")
    print(f"  CHS pipeline finished in {(time.time() - t_start) / 60:.0f} min")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    print(f"  next: copy results/processed/{output_sub(args)}/ out; make_figures.py --dataset chs draws it")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
