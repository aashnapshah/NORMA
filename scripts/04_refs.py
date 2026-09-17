#!/usr/bin/env python
"""Reference intervals for every method, from each pair's baseline measurements.

Usage:
    python 04_refs.py --dataset eicu                       # norma + baselines
    python 04_refs.py --dataset inspire --only norma --device cuda
    python 04_refs.py --dataset chs --chunk 3 --only baselines
    python 04_refs.py --dataset mimiciv --max_patients 40000   # same subset for both steps
    python 04_refs.py --state_conditional [--runs q_age_set 334f7e21]
"""
import bootstrap

import argparse
import glob
import importlib.util
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from datasets import MODEL_LOG_DIR, NORMA_CHECKPOINT, NORMA_RUN_ID, EXCLUDE_LABS, dev_results_dir, add_dataset_args, get_dataset, result_path, REF_COLUMNS, read_ref_intervals_raw, ref_paths, is_norma_method, upsert_ref_rows, write_ref_intervals
from metrics import STATE_NAMES, build_pairs, pairs_frame, subsample_patients, describe, REFERENCE_INTERVALS, population_reference_range, sex_key
from process.config import TEST_VOCAB

MODEL_DIR = bootstrap.MODEL_DIR
sys.path.insert(0, os.path.join(MODEL_DIR, "baselines"))      # cohen.py, gaussian.py

from gaussian import gmm_setpoint                             # noqa: E402  the Per_RI setpoint

STEPS = ("norma", "baselines")
QUANTILE_COLS = ["q025", "q25", "q50", "q75", "q975"]
MIN_BASELINE_TIMES = 5          # both steps: pairs need >= 5 unique baseline times
STATES = {0: "low", 1: "normal", 2: "high"}


def _import(name, path):
    """Import a module by file path without touching sys.path (lib/datasets.py and
    process/config.py, data/ and model/data.py share names)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model_module(name):
    return _import(f"_norma_{name}", os.path.join(MODEL_DIR, f"{name}.py"))


# norma

def load_norma(run_id, checkpoint="latest", device="cpu"):
    sys.path.insert(0, MODEL_DIR)            # utils.py does `from model import ...`
    utils = _model_module("utils")
    sys.path.remove(MODEL_DIR)
    ckpt, hp = utils.load_checkpoint(MODEL_LOG_DIR, run_id, best=(checkpoint == "best"),
                                     device=device, quiet=True)
    model = utils.create_model(hp, len(TEST_VOCAB), checkpoint=ckpt).to(device).eval()
    is_quantile = (getattr(hp, "output_mode", "gaussian") in ("quantile", "gate", "nig")
                   and getattr(hp, "model", "") == "NORMA2")
    head = "quantile" if is_quantile else "gaussian"
    print(f"  {run_id}: checkpoint_{checkpoint} (epoch {ckpt.get('epoch')}), {head} head")
    return model, hp, is_quantile


def model_covariates(model):
    """Which of the ablation covariates ('age', 'setting', 'co') a loaded model uses."""
    flags = [("use_age_t", "age"), ("use_setting", "setting"), ("use_coanalytes", "co")]
    return {name for flag, name in flags if getattr(model, flag, False)}


def _batch_tensors(recs, idx, cov, panel, normalize):
    """Padded input tensors for the records at `idx` (sorted by history length)."""
    import torch
    B = len(idx)
    T = max(recs[i]["n_hist"] for i in idx)
    K = panel.shape[1] if panel is not None else 0
    x_h = torch.zeros(B, T, 1)
    t_h = torch.zeros(B, T, 1)
    s_h = torch.zeros(B, T, dtype=torch.long)
    pad = torch.ones(B, T, dtype=torch.bool)
    sex = torch.zeros(B, 1, dtype=torch.long)
    age = torch.zeros(B, 1)
    cid = torch.zeros(B, 1, dtype=torch.long)
    t_next = torch.zeros(B, 1)
    ref_lo = np.zeros(B)
    span = np.ones(B)
    extras = {}
    if "age" in cov:
        extras["age_h"] = torch.zeros(B, T)
        extras["age_next"] = torch.zeros(B, 1)
    if "setting" in cov:
        extras["setting_h"] = torch.zeros(B, T, dtype=torch.long)
        extras["setting_next"] = torch.zeros(B, 1, dtype=torch.long)
    co = np.full((B, T, K), np.nan, dtype=np.float32) if "co" in cov else None

    for j, i in enumerate(idx):
        r = recs[i]
        n = r["n_hist"]
        xs = r["x_h"]
        if normalize:
            lo, hi, _ = REFERENCE_INTERVALS[r["analyte"]]["F" if r["sex"] == 1 else "M"]
            ref_lo[j], span[j] = lo, hi - lo
            xs = (xs - lo) / (hi - lo)
        x_h[j, :n, 0] = torch.from_numpy(np.asarray(xs, dtype=np.float32))
        t_h[j, :n, 0] = torch.from_numpy(r["t_h"])
        s_h[j, :n] = torch.from_numpy(r["s_h"])
        pad[j, :n] = False
        sex[j, 0] = r["sex"]
        age[j, 0] = r["age"]
        cid[j, 0] = r["cid"]
        t_next[j, 0] = r["t_next"]
        if "age" in cov:
            extras["age_h"][j, :n] = torch.from_numpy(r["age_h"])
            extras["age_next"][j, 0] = r["age_next"]
        if "setting" in cov:
            # icu -> inpatient, as model/data.py does at training time (collapse_icu)
            extras["setting_h"][j, :n] = torch.from_numpy(np.where(r["setting_h"] == 4, 3, r["setting_h"]))
            extras["setting_next"][j, 0] = 3 if r["setting_next"] == 4 else r["setting_next"]
        if co is not None:
            rows = panel[r["draw_idx"]].astype(np.float32)
            rows[:, r["cid"]] = np.nan    # the target analyte is already the token value
            co[j, :n] = rows
    if co is not None:
        co_t = torch.from_numpy(co)
        extras["co_mask"] = torch.isfinite(co_t).float()
        extras["co_h"] = torch.nan_to_num(co_t, nan=0.0)
    inputs = (x_h, s_h, t_h, sex, age, cid, t_next, pad)
    return inputs, extras, ref_lo, span


def norma_all_states(model, hp, is_quantile, recs, batch_size=1024, device="cpu", panel=None):
    """Run the model with the query token at every state.  Returns (dict of arrays
    keyed 'mu_{q}', 'log_var_{q}' [, 'q*_{q}'] aligned with recs, nstates).
    """
    import torch
    nstates = getattr(hp, "nstates", 3)
    normalize = bool(getattr(hp, "normalize", False))
    cov = model_covariates(model)
    if "co" in cov and panel is None:
        raise ValueError("model uses co-analytes but no draw panel was built")
    for c, field in [("age", "age_h"), ("setting", "setting_h"), ("co", "draw_idx")]:
        if c in cov and recs and field not in recs[0]:
            raise ValueError(f"model needs covariate '{c}' but recs lack {field}")

    N = len(recs)
    keys = ["mu", "log_var"] + (QUANTILE_COLS if is_quantile else [])
    out = {f"{k}_{q}": np.full(N, np.nan) for q in range(nstates) for k in keys}
    order = np.argsort([r["n_hist"] for r in recs], kind="stable")
    t0 = time.time()
    with torch.no_grad():
        for bi, start in enumerate(range(0, N, batch_size)):
            idx = order[start:start + batch_size]
            inputs, extras, ref_lo, span = _batch_tensors(recs, idx, cov, panel, normalize)
            x_h, s_h, t_h, sex, age, cid, t_next, pad = (t.to(device) for t in inputs)
            extras = {k: v.to(device) for k, v in extras.items()}
            for q in range(nstates):
                s_q = torch.full((len(idx), 1), q, dtype=torch.long).to(device)
                o = model(x_h, s_h, t_h, sex, age, cid, s_q, t_next, pad, **extras)
                if is_quantile:
                    arr = o.cpu().numpy().astype(float)
                    if normalize:
                        arr = arr * span[:, None] + ref_lo[:, None]
                    for k, col in enumerate(QUANTILE_COLS):
                        out[f"{col}_{q}"][idx] = arr[:, k]
                    out[f"mu_{q}"][idx] = arr[:, 2]
                    out[f"log_var_{q}"][idx] = 2.0 * np.log((arr[:, 4] - arr[:, 0]) / 3.92 + 1e-8)
                else:
                    mu, lv = o
                    mu = mu.view(-1).cpu().numpy().astype(float)
                    lv = lv.view(-1).cpu().numpy().astype(float)
                    if normalize:
                        mu = mu * span + ref_lo
                        lv = lv + 2.0 * np.log(span + 1e-8)
                    out[f"mu_{q}"][idx] = mu
                    out[f"log_var_{q}"][idx] = lv
            if bi % 100 == 0:
                done = min(start + batch_size, N)
                print(f"    {done:>9,}/{N:,}  {time.time() - t0:5.0f}s", flush=True)
    return out, nstates


def norma_variants(states, nstates, is_quantile, recs):
    """oracle / normal / marginal / marginal_freq point predictions (+ normal interval)."""
    st = _model_module("states")   # state_prior + state_mixture merged
    priors = st.load_state_priors()
    cid = np.array([r["cid"] for r in recs])
    s_last = np.array([r["s_last"] for r in recs])
    s_next = np.array([r["s_next"] for r in recs])
    centre = "q50" if is_quantile else "mu"

    if is_quantile:
        lo, hi = states["q025_1"], states["q975_1"]
    else:
        sd = np.exp(0.5 * states["log_var_1"])
        lo, hi = states["mu_1"] - 1.96 * sd, states["mu_1"] + 1.96 * sd
    out = {
        "oracle": np.select([s_next == q for q in range(nstates)],
                            [states[f"{centre}_{q}"] for q in range(nstates)]),
        "normal": states[f"{centre}_1"],
        "normal_lo": lo,
        "normal_hi": hi,
    }

    mu = np.stack([states[f"mu_{q}"] for q in range(nstates)], 1)
    lv = np.stack([states[f"log_var_{q}"] for q in range(nstates)], 1)
    if is_quantile:
        qarr = np.stack([np.stack([states[f"{c}_{q}"] for c in QUANTILE_COLS], 1)
                         for q in range(nstates)], 1)
    for name, kind in [("marginal", "transition"), ("marginal_freq", "marginal")]:
        w = st.prior_weights(priors, cid, s_last, kind=kind)
        if is_quantile:
            mix = st.mix_quantiles_batched(qarr, w)
        else:
            mix = st.mix_gaussian_batched(mu, lv, w)
        out[name] = mix[centre]
    return out


def norma_ref_rows(out, recs, run_id):
    """norma_<run_id> rows for ref_intervals: the normal-state slice of the FIRST
    index target, for pairs with >= MIN_BASELINE_TIMES history points."""
    first = (out["target_idx"] == 0) & (out["n_hist"] >= MIN_BASELINE_TIMES)
    idx = np.flatnonzero(first.to_numpy())
    lo = out[f"{run_id}_normal_lo"].to_numpy()[idx]
    hi = out[f"{run_id}_normal_hi"].to_numpy()[idx]
    mid = out[f"{run_id}_normal"].to_numpy()[idx]
    rows = pd.DataFrame({
        "patient_id": out["patient_id"].to_numpy()[idx],
        "analyte": out["analyte"].to_numpy()[idx],
        "sex": [recs[i]["sex_raw"] for i in idx],
        "age": [recs[i]["age_raw"] for i in idx],
        "n_bl": out["n_hist"].to_numpy()[idx],
        "t_span": [float(recs[i]["t_h"][-1] - recs[i]["t_h"][0]) for i in idx],
        "method": f"norma_{run_id}",
        "ri_mean": mid,
        "ri_std": (hi - lo) / 3.92,
        "ri_low": lo,
        "ri_high": hi,
    })
    ok = np.isfinite(rows["ri_low"]) & np.isfinite(rows["ri_high"])
    return rows[ok].reset_index(drop=True)


def write_norma_ref_rows(ds, out, recs, runs, analytes=None):
    rows = pd.concat([norma_ref_rows(out, recs, r) for r in runs], ignore_index=True)
    existing = read_ref_intervals_raw(ds, "baselines")
    if existing is not None and (existing["method"] == "pop").any():
        pop = existing.loc[existing["method"] == "pop", ["patient_id", "analyte"]].drop_duplicates()
        mine = rows.loc[rows["method"] == f"norma_{runs[0]}", ["patient_id", "analyte"]]
        m = pop.merge(mine, how="outer", indicator=True)["_merge"].value_counts()
        print(f"  pairs vs the baselines step's pop rows: {m.get('both', 0):,} shared, "
              f"{m.get('left_only', 0):,} pop-only, {m.get('right_only', 0):,} norma-only")
    upsert_ref_rows(ds, rows, [f"norma_{r}" for r in runs], analytes=analytes)
    print(f"  ref_intervals: {len(rows):,} norma rows written")


def _load_models(runs, args):
    """{run_id: (model, hp, is_quantile)} and the union of covariates they need."""
    st = _model_module("states")
    models = {}
    covariates = set()
    for run_id in runs:
        checkpoint = args.checkpoint
        if checkpoint == "auto":
            checkpoint = st.PUBLISHED_CHECKPOINT.get(run_id, "latest")
        models[run_id] = load_norma(run_id, checkpoint=checkpoint, device=args.device)
        covariates |= model_covariates(models[run_id][0])
    if covariates:
        print(f"  covariate inputs needed: {sorted(covariates)}")
    return models, covariates


def _print_mae_by_state(out, runs):
    print("\nMAE (median across analytes) by realized state of the target:")
    lines = []
    for run_id in runs:
        for variant in ("oracle", "normal", "marginal", "marginal_freq"):
            col = f"{run_id}_{variant}"
            d = out.dropna(subset=[col])
            err = (d["x_next"] - d[col]).abs()
            by_state = err.groupby([d["analyte"], d["s_next"]]).mean().unstack()
            line = {"model": col, "n": len(d), "all": err.groupby(d["analyte"]).mean().median()}
            for q in range(3):
                if q in by_state:
                    line[STATE_NAMES[q]] = by_state[q].median()
            lines.append(line)
    print(pd.DataFrame(lines).round(3).to_string(index=False))


def run_norma(ds, args):
    runs = args.runs or list(getattr(ds, "run_ids", [NORMA_RUN_ID]))
    # models first: covariate arms declare the per-measurement inputs build_pairs must carry
    models, covariates = _load_models(runs, args)

    t0 = time.time()
    df = ds.load_index_labs()
    time_unit = df.attrs.get("time_unit", getattr(ds, "time_unit", "days"))
    print(f"{ds.name}: {len(df):,} rows, time_unit={time_unit}, loaded in {time.time() - t0:.0f}s")
    df = subsample_patients(df, args.max_patients)
    built = build_pairs(df, time_unit, target=args.target, max_hist=args.max_hist,
                        exclude=getattr(ds, "exclude_labs", ()), covariates=sorted(covariates))
    recs, panel = built if "co" in covariates else (built, None)
    del df
    if args.max_pairs and len(recs) > args.max_pairs:
        rng = np.random.default_rng(42)
        recs = [recs[i] for i in rng.choice(len(recs), size=args.max_pairs, replace=False)]
    describe(recs, args.target)

    out = pairs_frame(recs)
    out["cohort"] = ds.name
    for run_id, (model, hp, is_quantile) in models.items():
        states, nstates = norma_all_states(model, hp, is_quantile, recs, batch_size=args.batch_size,
                                           device=args.device, panel=panel)
        for k, v in states.items():
            out[f"{run_id}_{k}"] = v
        for k, v in norma_variants(states, nstates, is_quantile, recs).items():
            out[f"{run_id}_{k}"] = v

    out_path = args.out or ds.norma_predictions_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(out):,} targets)")
    if not (args.skip_ref_rows or args.out):
        write_norma_ref_rows(ds, out, recs, runs, analytes=ds._analytes)
    _print_mae_by_state(out, runs)


# baselines

def _eligible_pair_keys(index_labs):
    """(eligible pairs, n_times, enough) -- the pairs a reference interval can
    exist for: >= MIN_BASELINE_TIMES unique baseline times AND an index
    measurement.  Anything else is correctly absent from ref_intervals and must
    not be counted as missing, which is what makes this the right denominator
    for the coverage check as well as the filter for the compute."""
    keys = ["patient_id", "analyte"]
    baseline = index_labs[index_labs["split"] == "baseline"]
    index = index_labs[index_labs["split"] == "index"]
    n_times = baseline.groupby(keys)["timestamp"].nunique()
    enough = set(n_times[n_times >= MIN_BASELINE_TIMES].index)
    has_index = set(map(tuple, index[keys].drop_duplicates().to_numpy()))
    return enough & has_index, n_times, enough


def _eligible_pairs(index_labs):
    """Baseline rows of the eligible pairs, one value per timestamp."""
    keys = ["patient_id", "analyte"]
    keep, n_times, enough = _eligible_pair_keys(index_labs)
    baseline = index_labs[index_labs["split"] == "baseline"]
    baseline = baseline[baseline.set_index(keys).index.isin(keep)]
    print(f"  Skipped: {len(n_times) - len(enough):,} pairs with <{MIN_BASELINE_TIMES} unique "
          f"baseline times, {len(enough) - len(keep):,} pairs with no index")
    before = len(baseline)
    baseline = baseline.drop_duplicates(subset=keys + ["timestamp"], keep="first")
    if before > len(baseline):
        print(f"  Deduplicated baseline: {before:,} -> {len(baseline):,} rows")
    return baseline


def _pair_rows(pair, values, gmm_n_std):
    """base / pop / per rows of one pair (pair = its aggregate record)."""
    if len(values) < 2:
        return []
    pop_low, pop_high = population_reference_range(pair["analyte"], sex_key(pair["sex"]))
    setpoint, sd = gmm_setpoint(values)
    shared = {k: pair[k] for k in ("patient_id", "analyte", "sex", "age", "n_bl", "t_span")}
    return [
        {**shared, "method": "base", "ri_mean": pair["bl_mean"], "ri_std": pair["bl_std"],
         "ri_low": np.nan, "ri_high": np.nan},
        {**shared, "method": "pop", "ri_mean": np.nan, "ri_std": np.nan,
         "ri_low": pop_low, "ri_high": pop_high},
        {**shared, "method": "per", "ri_mean": setpoint, "ri_std": sd,
         "ri_low": setpoint - gmm_n_std * sd, "ri_high": setpoint + gmm_n_std * sd},
    ]


def _rows_for_task(items, gmm_n_std):
    rows = []
    for pair, values in items:
        rows.extend(_pair_rows(pair, values, gmm_n_std))
    return rows


def compute_reference_intervals(index_labs, gmm_n_std=2):
    """base / pop / per rows for every eligible pair."""
    from joblib import Parallel, delayed
    baseline = _eligible_pairs(index_labs)
    keys = ["patient_id", "analyte"]
    agg = baseline.groupby(keys).agg(
        sex=("sex", "first"), age=("age", "first"), n_bl=("value", "count"),
        bl_mean=("value", "mean"), bl_std=("value", "std"),
        t_min=("timestamp", "min"), t_max=("timestamp", "max"),
    ).reset_index()
    agg["t_span"] = agg["t_max"] - agg["t_min"]
    if index_labs.attrs.get("time_unit", "minutes") == "minutes":
        agg["t_span"] = agg["t_span"] / (60 * 24)
    print(f"Computing ref intervals for {len(agg):,} patient-analyte pairs "
          f"({agg['patient_id'].nunique():,} patients, {agg['analyte'].nunique()} analytes)")

    values = baseline.groupby(keys)["value"].apply(lambda s: s.dropna().to_numpy())
    items = [(pair, values.get((pair["patient_id"], pair["analyte"]), np.array([])))
             for pair in agg.to_dict("records")]
    # GMM fits in loky processes, 2,000 pairs per task.
    task = 2_000
    tasks = [items[i:i + task] for i in range(0, len(items), task)]
    n_jobs = min(os.cpu_count() or 1, 16)
    print(f"  Computing base/pop/per: {len(items):,} pairs in {len(tasks):,} tasks on {n_jobs} processes...")
    rows = []
    for chunk in _run_tasks(tasks, gmm_n_std, n_jobs):
        rows.extend(chunk)
    ref_df = pd.DataFrame(rows, columns=REF_COLUMNS if not rows else None)
    print(f"Computed ref intervals: {len(ref_df):,} rows")
    return ref_df


def _run_tasks(tasks, gmm_n_std, n_jobs, verbose=0):
    """Fan the GMM tasks out over loky processes, across joblib versions."""
    from joblib import Parallel, delayed, parallel_backend

    total = sum(len(t) for t in tasks)
    out, done = [], 0
    t0 = time.time()

    def _drain(runner, group):
        nonlocal done
        for chunk in runner(delayed(_rows_for_task)(t, gmm_n_std) for t in group):
            out.append(chunk)
        done += sum(len(t) for t in group)

    def _report(i):
        rate = done / max(time.time() - t0, 1e-9)
        left = (total - done) / rate if rate else float("nan")
        print(f"    {done:,}/{total:,} pairs, task {i}/{len(tasks)}"
              f"  ({rate:,.0f} pairs/s, ~{left / 60:.1f} min left)", flush=True)

    groups = [tasks[i:i + n_jobs] for i in range(0, len(tasks), n_jobs)]
    try:                                   # only the backend build may TypeError
        ctx = parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1)
    except TypeError:                      # joblib too old for inner_max_num_threads
        print("  (joblib without inner_max_num_threads; BLAS threads left to the workers)")
        ctx = parallel_backend("loky", n_jobs=n_jobs)
    with ctx:
        runner = Parallel(n_jobs=n_jobs, verbose=verbose)
        for gi, group in enumerate(groups, 1):
            _drain(runner, group)
            _report(min(gi * n_jobs, len(tasks)))
    return out


def _missing_pairs(index_labs, ref_df, expected=()):
    """Eligible (patient, analyte) pairs without a `pop` row in ref_df."""
    keys = ["patient_id", "analyte"]
    all_pairs = index_labs[keys].drop_duplicates()
    keep, _, _ = _eligible_pair_keys(index_labs)
    split_pairs = all_pairs[pd.MultiIndex.from_frame(all_pairs).isin(keep)]
    ref_pairs = ref_df.loc[ref_df["method"] == "pop", keys].drop_duplicates()
    merged = split_pairs.merge(ref_pairs, on=keys, how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    ref_all = ref_df[keys].drop_duplicates()
    both = ref_all.merge(all_pairs, on=keys, how="inner")
    ref_only = len(ref_all) - len(both)
    split_only = len(all_pairs) - len(both)
    both_elig = len(both.merge(split_pairs, on=keys))
    lost_pat = len(set(ref_df["patient_id"]) - set(index_labs["patient_id"]))
    new_pat = len(set(index_labs["patient_id"]) - set(ref_df["patient_id"]))

    print(f"\n  Coverage check (patient-analyte pairs):")
    print(f"    index_labs only:    {split_only:8,}   no reference interval yet"
          f"   ({new_pat:,} patients not in ref_intervals)")
    print(f"    both:               {len(both):8,}   of which eligible {both_elig:,}")
    print(f"    ref_intervals only: {ref_only:8,}   not in index_labs at all"
          f"   ({lost_pat:,} patients not in the split)")
    print(f"    ---")
    print(f"    index_labs:         {len(all_pairs):8,}  "
          f"(eligible {len(split_pairs):,}: ≥{MIN_BASELINE_TIMES} baseline times + an index)")
    print(f"    ref_intervals:      {len(ref_all):8,}  (with a pop row {len(ref_pairs):,})")
    # Two missing counts, because the default only works on the intersection: the one that
    # matters is inside it, the wing is what --fill_missing would add.
    focus_missing = len(missing.merge(both, on=keys))
    print(f"    Missing on the intersection (no pop row): {focus_missing:,}")
    if len(missing) - focus_missing:
        print(f"    Missing outside it (index_labs only):     "
              f"{len(missing) - focus_missing:,}   --fill_missing to compute")
    if ref_only:
        print(f"    NOTE: {ref_only:,} ref pairs are not in this split — the two files were "
              f"built from different cohorts; delete ref_intervals.parquet to rebuild cleanly")
    # The table is about the intersection: pairs that are in ref_intervals AND in this split AND
    # eligible.
    core = ref_df.loc[ref_df["method"].isin(CORE_METHODS), keys].drop_duplicates()
    focus = core.merge(all_pairs, on=keys).merge(split_pairs, on=keys)
    non_core = len(both) - len(core.merge(all_pairs, on=keys))
    if non_core > 0:
        print(f"    {non_core:,} pairs in both have no {'/'.join(CORE_METHODS)} row "
              f"(norma-only rows from 01_process) — excluded from the table")
    ref_focus = ref_df.merge(focus, on=keys, how="inner")
    _coverage_table(focus, ref_focus, expected)
    return missing


def _coverage_table(split_pairs, ref_df, expected=()):
    """Pairs without a row, per analyte per method."""
    n_split = split_pairs.groupby("analyte").size()
    have = (ref_df.drop_duplicates(["patient_id", "analyte", "method"])
                  .groupby(["analyte", "method"]).size().unstack(fill_value=0))
    extra = [m for m in have.columns if m not in expected]
    cols = list(expected) + extra
    have = have.reindex(index=n_split.index, columns=cols, fill_value=0).fillna(0)
    table = have.rsub(n_split, axis=0).clip(lower=0).astype(int)
    table.insert(0, "pairs", n_split)
    print("\n  Missing methods per analyte, on the intersection (0 = fully covered):")
    print("    " + table.to_string().replace("\n", "\n    "))
    totals = "  ".join(f"{m} {int(table[m].sum()):,}" for m in table.columns[1:])
    print(f"    TOTAL pairs {int(table['pairs'].sum()):,}   {totals}")
    leaked = [m for m in extra if is_norma_method(m)]
    if leaked:
        print(f"    note: {leaked} are NORMA rows inside ref_intervals.parquet "
              f"(a 01_process chunk); 04_refs writes them to ref_intervals_norma.parquet")


CORE_METHODS = ["base", "pop", "per"]


def _expected_methods(args):
    """Every baseline method this run is meant to produce, in report order."""
    return (["base", "pop", "per"]
            + [f"gaussian_{m}" for m in (args.gaussian_methods or [])]
            + [f"cohen_{m}" for m in (args.cohen_models or [])])


def _fill_missing(ref_df, index_labs, missing, gmm_n_std):
    """Append base / pop / per rows for pairs absent from ref_df (never duplicates)."""
    keys = ["patient_id", "analyte"]
    baseline_pairs = index_labs[index_labs["split"] == "baseline"][keys].drop_duplicates()
    todo = missing.merge(baseline_pairs, on=keys, how="inner")
    print(f"\n  {len(todo):,} completely missing pairs have baseline data — computing...")
    if not len(todo):
        return ref_df
    todo_set = set(map(tuple, todo[keys].to_numpy()))
    subset = index_labs[index_labs.set_index(keys).index.isin(todo_set)].copy()
    new = compute_reference_intervals(subset, gmm_n_std=gmm_n_std)
    if new.empty:
        print("  No new ref rows (all candidate pairs filtered out)")
        return ref_df
    new = new[pd.MultiIndex.from_frame(new[keys]).isin(todo_set)]
    existing_keys = pd.MultiIndex.from_frame(ref_df[keys + ["method"]])
    new = new[~pd.MultiIndex.from_frame(new[keys + ["method"]]).isin(existing_keys)]
    print(f"  Appending {len(new):,} new ref rows for missing pairs")
    return pd.concat([ref_df, new], ignore_index=True)


def run_baselines(ds, args):
    os.makedirs(ds.data_dir, exist_ok=True)
    # Both paths up front: with --chunk these are the chunk directory, without it the cohort-
    # level results dir, and reading one while writing the other is the failure that looks like
    # the stage...
    base_p = ref_paths(ds)[0]
    print(f"  index_labs from: {ds.data_dir}")
    print(f"  ref_intervals:   {base_p}"
          f"  [{'exists' if os.path.exists(base_p) else 'not yet written'}]")
    index_labs = subsample_patients(ds.load_index_labs(), args.max_patients)   # same draw as norma
    index_labs["analyte"] = index_labs["analyte"].replace("", "NA").fillna("NA")
    ref_df = read_ref_intervals_raw(ds, "baselines")   # the norma half is not ours to touch

    if ref_df is None:
        print("  no existing baseline ref intervals — computing every pair")
        ref_df = compute_reference_intervals(index_labs, gmm_n_std=args.gmm_n_std)
    else:
        print(f"  read {len(ref_df):,} existing baseline rows")
        ref_df["analyte"] = ref_df["analyte"].replace("", "NA").fillna("NA")
        missing = _missing_pairs(index_labs, ref_df, _expected_methods(args))
        if args.check_only:
            return
        if args.fill_missing and len(missing):
            ref_df = _fill_missing(ref_df, index_labs, missing, args.gmm_n_std)
        else:
            # Default: work only on pairs that already have an interval.
            keys = ["patient_id", "analyte"]
            have = ref_df.loc[ref_df["method"].isin(CORE_METHODS), keys].drop_duplicates()
            before = len(index_labs[keys].drop_duplicates())
            index_labs = index_labs.merge(have, on=keys, how="inner")
            print(f"  Existing pairs only: {len(have):,} already have "
                  f"{'/'.join(CORE_METHODS)}; index_labs restricted {before:,} -> "
                  f"{len(have):,} pairs, {len(missing):,} missing left uncomputed "
                  f"(--fill_missing to compute them)")

    if args.cohen_models:       # Cohen et al. 2021: trained on the dev cohorts, transferred here
        from cohen import augment_cohen
        ref_df = augment_cohen(ref_df, index_labs, models=args.cohen_models, z=args.cohen_z,
                               retrain=args.cohen_retrain, force=args.cohen_force,
                               train_sources=tuple(args.cohen_train_sources),
                               min_pairs=args.cohen_min_pairs, norm_frac=args.cohen_norm_frac)
    if args.gaussian_methods:   # PopRI-normal-history fits: MLE / truncated / empirical Bayes
        from gaussian import augment_gaussian
        ref_df = augment_gaussian(ref_df, index_labs, methods=args.gaussian_methods,
                                  z=args.gaussian_z, prior_source=args.gaussian_prior,
                                  train_sources=tuple(args.gaussian_prior_sources),
                                  rebuild_prior=args.gaussian_rebuild_prior,
                                  force=args.gaussian_force)
    write_ref_intervals(ds, ref_df, "baselines")


# --state_conditional  (Referee 1.4)

SC_NOISE_FRAC = 0.10   # sd of the synthetic history noise, as a fraction of the Pop_RI width
SC_N_DRAWS = 25        # histories averaged per analyte
SC_N_HIST = 10         # measurements per history, SC_SPACING days apart (model/sensitivity_analysis.py)
SC_SPACING = 90
SC_HORIZON = 30        # days from the last measurement to the query


def synthetic_records(model, analyte, sex, age, n_draws, seed):
    """n_draws synthetic histories of one analyte in the record format build_pairs
    produces: SC_N_HIST values drawn around the Pop_RI midpoint, queried SC_HORIZON
    days after the last one.  Covariate arms get the inputs model/sensitivity_analysis.py
    gives them: age counted back from the query, setting 'unknown', no co-analytes."""
    low, high, _ = REFERENCE_INTERVALS[analyte]["F" if sex == 1 else "M"]
    mid, span = (low + high) / 2.0, high - low
    cov = model_covariates(model)
    t = np.arange(SC_N_HIST, dtype=np.float32) * SC_SPACING
    t_next = float(t[-1] + SC_HORIZON)
    recs = []
    for i in range(n_draws):
        rng = np.random.default_rng(seed + i)
        x = rng.normal(mid, SC_NOISE_FRAC * span, SC_N_HIST)
        state = np.where(x < low, 0, np.where(x > high, 2, 1))
        r = {"analyte": analyte, "cid": TEST_VOCAB[analyte], "sex": sex, "age": age,
             "x_h": x.astype(np.float32), "s_h": state.astype(np.int64), "t_h": t,
             "n_hist": SC_N_HIST, "t_next": t_next}
        if "age" in cov:
            r["age_h"] = np.clip(age - (t_next - t) / 365.25, 0.0, None).astype(np.float32)
            r["age_next"] = float(age)
        if "setting" in cov:
            r["setting_h"] = np.zeros(SC_N_HIST, dtype=np.int64)
            r["setting_next"] = 0
        if "co" in cov:
            r["draw_idx"] = np.zeros(SC_N_HIST, dtype=np.int64)
        recs.append(r)
    panel = None
    if "co" in cov:
        panel = np.full((1, getattr(model, "n_panel", len(TEST_VOCAB))), np.nan, dtype=np.float32)
    return recs, panel


def overlap(m1, s1, m2, s2):
    """Overlap coefficient of two Gaussians: integral of the pointwise minimum."""
    s1, s2 = max(s1, 1e-9), max(s2, 1e-9)
    lo = min(m1 - 6 * s1, m2 - 6 * s2)
    hi = max(m1 + 6 * s1, m2 + 6 * s2)
    xs = np.linspace(lo, hi, 20000)

    def density(m, s):
        return np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    return float(np.trapz(np.minimum(density(m1, s1), density(m2, s2)), xs))


def state_conditional_row(model, hp, is_quantile, analyte, args):
    """Mean (mu, sigma) of p(x | H, s) over the synthetic histories, per state, and
    the pairwise overlaps of the three distributions."""
    recs, panel = synthetic_records(model, analyte, args.sex, args.age, args.n_draws, args.seed)
    states, _ = norma_all_states(model, hp, is_quantile, recs, batch_size=len(recs), panel=panel)
    mu, sd = {}, {}
    for q, name in STATES.items():
        mu[name] = float(np.mean(states[f"mu_{q}"]))
        if is_quantile:
            sd[name] = float(np.mean((states[f"q975_{q}"] - states[f"q025_{q}"]) / 3.92))
        else:
            sd[name] = float(np.mean(np.exp(0.5 * states[f"log_var_{q}"])))
    low, high, unit = REFERENCE_INTERVALS[analyte]["F" if args.sex == 1 else "M"]
    return {
        "mu_low": mu["low"], "mu_norm": mu["normal"], "mu_high": mu["high"],
        "sd_low": sd["low"], "sd_norm": sd["normal"], "sd_high": sd["high"],
        "ov_low_norm": overlap(mu["low"], sd["low"], mu["normal"], sd["normal"]),
        "ov_norm_high": overlap(mu["normal"], sd["normal"], mu["high"], sd["high"]),
        "ov_low_high": overlap(mu["low"], sd["low"], mu["high"], sd["high"]),
        "ref_low": low, "ref_high": high, "unit": unit,
    }


def run_state_conditional(args):
    args.device = "cpu"
    runs = args.runs or [NORMA_RUN_ID]
    models, _ = _load_models(runs, args)
    analytes = sorted(a for a in REFERENCE_INTERVALS if a in TEST_VOCAB and a not in EXCLUDE_LABS)
    for run_id, (model, hp, is_quantile) in models.items():
        rows = {}
        for analyte in analytes:
            row = state_conditional_row(model, hp, is_quantile, analyte, args)
            rows[analyte] = row
            print(f"  {analyte:<5s} mu low/normal/high = {row['mu_low']:.3g} / {row['mu_norm']:.3g} "
                  f"/ {row['mu_high']:.3g}   overlap L-N {row['ov_low_norm']:.2f}  "
                  f"N-H {row['ov_norm_high']:.2f}")
        df = pd.DataFrame(rows).T
        df.index.name = "analyte"
        suffix = "" if run_id == NORMA_RUN_ID else f"_{run_id}"
        out = result_path(dev_results_dir("04_refs"), f"state_conditional{suffix}.csv")
        df.to_csv(out)
        num = df.drop(columns="unit").astype(float)
        violations = num[(num.mu_low >= num.mu_norm) | (num.mu_norm >= num.mu_high)]
        print(f"\n  median overlap  L-N {num.ov_low_norm.median():.3f}   "
              f"N-H {num.ov_norm_high.median():.3f}   L-H {num.ov_low_high.median():.3f}")
        print(f"  ordering violations (low < normal < high): "
              f"{', '.join(violations.index) if len(violations) else 'none'}")
        print(f"Wrote {out} ({len(df)} analytes)")


# main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_dataset_args(p, required=False)
    p.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS),
                   help="which steps to run (default: both)")
    p.add_argument("--max_patients", type=int, default=None,
                   help="random patient subsample (seed 42), the same draw for both steps")
    p.add_argument("--runs", nargs="+", default=None,
                   help="NORMA run ids (default: dataset.run_ids; --state_conditional: NORMA_RUN_ID)")
    p.add_argument("--checkpoint", default=NORMA_CHECKPOINT, choices=["auto", "latest", "best"],
                   help="default = config.NORMA_CHECKPOINT; auto = the weights behind each "
                        "run's published predictions")

    g = p.add_argument_group("norma")
    g.add_argument("--target", default="first", choices=["first", "all"],
                   help="first index measurement (the reference-interval target) or every one")
    g.add_argument("--max_hist", type=int, default=128)
    g.add_argument("--max_pairs", type=int, default=None, help="smoke test: random subset of targets")
    g.add_argument("--batch_size", type=int, default=1024)
    g.add_argument("--device", default="cpu")
    g.add_argument("--out", default=None, help="predictions path override (implies --skip_ref_rows)")
    g.add_argument("--skip_ref_rows", action="store_true",
                   help="do not write the norma_<run_id> rows into ref_intervals")

    g = p.add_argument_group("baselines")
    g.add_argument("--gmm_n_std", type=float, default=2)
    g.add_argument("--check_only", action="store_true", help="only report coverage of the existing file")
    g.add_argument("--fill_missing", action="store_true",
                   help="also compute base/pop/per for eligible pairs that have none "
                        "(default: only pairs already in ref_intervals)")
    g.add_argument("--cohen_models", type=str, nargs="*", default=["m2", "m3", "m4"],
                   choices=["m2", "m3", "m4"], help="Cohen variants to append (no values = skip)")
    g.add_argument("--cohen_z", type=float, default=1.96,
                   help="Cohen interval half-width in prediction SDs (1.0 = paper-faithful Fig. 4c band)")
    g.add_argument("--cohen_retrain", action="store_true", help="retrain the Cohen dev models even if cached")
    g.add_argument("--cohen_force", action="store_true", help="recompute cohen_* rows even if present")
    g.add_argument("--cohen_norm_frac", type=float, default=0.8,
                   help="min fraction of a dev pair's history inside PopRI to count as a healthy "
                        "training trajectory (1.0 = Cohen's all-within-norm rule)")
    g.add_argument("--cohen_min_pairs", type=int, default=50,
                   help="minimum healthy training pairs before an analyte gets a Cohen model")
    g.add_argument("--cohen_train_sources", type=str, nargs="*", default=["mimiciv", "ehrshot"],
                   choices=["mimiciv", "ehrshot"], help="dev sources for Cohen training")
    g.add_argument("--gaussian_methods", type=str, nargs="*", default=["mle", "trunc", "eb"],
                   choices=["mle", "trunc", "eb"],
                   help="PopRI-normal-history fits to append (no values = skip)")
    g.add_argument("--gaussian_z", type=float, default=1.96, help="gaussian interval half-width in SDs")
    g.add_argument("--gaussian_prior", type=str, default="popri", choices=["popri", "cohort", "dev"],
                   help="EB prior: popri (default) = the population interval itself is the mean "
                        "prior (mu = midpoint, tau = half-width/z), variance prior from the cached "
                        "dev fit; cohort / dev = type-II ML from other patients' histories")
    g.add_argument("--gaussian_prior_sources", type=str, nargs="*", default=["mimiciv", "ehrshot"],
                   choices=["mimiciv", "ehrshot"], help="dev sources for the EB prior (--gaussian_prior dev)")
    g.add_argument("--gaussian_rebuild_prior", action="store_true", help="re-estimate the cached dev prior")
    g.add_argument("--gaussian_force", action="store_true", help="recompute gaussian_* rows even if present")

    g = p.add_argument_group("state_conditional")
    g.add_argument("--state_conditional", action="store_true",
                   help="run the R1-4 diagnostic on synthetic histories instead of a cohort")
    g.add_argument("--sex", type=int, default=0, help="0 = male, 1 = female")
    g.add_argument("--age", type=float, default=50)
    g.add_argument("--n_draws", type=int, default=SC_N_DRAWS)
    g.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if not args.state_conditional and args.dataset is None:
        p.error("--dataset is required (or pass --state_conditional)")
    return args


def _run_steps(ds, args):
    if "norma" in args.only:
        print("=== norma ===")
        run_norma(ds, args)
    if "baselines" in args.only:
        print("=== baselines ===")
        run_baselines(ds, args)


def _chs_chunk_indices(n_chunks=None):
    """The cohort's chunk indices in numeric order, first `n_chunks` of them."""
    from datasets import DATASETS as _DATASETS   # figlib's star-import rebinds the name
    root = _DATASETS["chs"]().data_root
    idx = sorted(int(m.group(1)) for m in
                 (re.search(r"chunk_(\d+)$", d) for d in glob.glob(os.path.join(root, "chunk_*")))
                 if m)
    if not idx:
        raise SystemExit(f"No chunk_* under {root}; run 02_index_labs.py --dataset chs first")
    return idx[:n_chunks] if n_chunks else idx


def run_chs_chunks(args):
    """CHS without --chunk: run the steps once per chunk, writing into each chunk_i/."""
    chunks = _chs_chunk_indices(args.n_chunks)
    print(f"CHS: {len(chunks)} chunk(s), steps {list(args.only)}")
    for n, i in enumerate(chunks, 1):
        print(f"\n───── chunk_{i}  ({n}/{len(chunks)}) ─────")
        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.chunk = i
        chunk_args.n_chunks = None          # --chunk scopes it; N is the outer loop
        _run_steps(get_dataset(chunk_args), chunk_args)


def main():
    args = parse_args()
    if args.state_conditional:
        run_state_conditional(args)
        return
    if args.dataset == "chs" and args.chunk is None:
        # --n_chunks alone used to mean "one cohort out of the first N chunks", which writes
        # where no CHS reader looks; it now means the same as looping --chunk, which is what
        # every CHS run actually wants.
        if getattr(args, "no_norma", False) and "norma" in args.only:
            raise SystemExit("--no_norma leaves the norma step nothing to do; "
                             "run 04_refs.py --only baselines")
        return run_chs_chunks(args)
    ds = get_dataset(args)
    if ds.no_norma and "norma" in args.only:
        raise SystemExit("--no_norma leaves the norma step nothing to do; "
                         "run 04_refs.py --only baselines")
    _run_steps(ds, args)


# Figures and tables

import os

from figlib import *  # noqa: F401,F403
from datasets import dev_results_dir, result_path

STATE_ORDER = models.STATE_ORDER       # lib/models.py, via figlib
STATE_COLORS = models.STATE_COLORS

# NORMA versions with a results/state_conditional[_<run>].csv; the file suffix is the training
# run id (None = published run, see 04_refs.py --state_conditional).
NORMA_RUN_SUFFIXES = [None] + list(ABLATION_RUN_IDS) + ["q_age_set_co"]


def _style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE, loc="left")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    hide_spines(ax)


def _load_state_conditional(version=None):
    path = find_in(dev_results_dir("04_refs"), f"state_conditional{'_' + version if version else ''}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, keep_default_na=False, na_values=[""])
    num = [c for c in df.columns if c != "unit"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.isin(EXCLUDE_ANALYTES)]
    return df if len(df) else None


def _density_panel(ax, lab, r):
    """The three state-conditional Gaussians for one analyte, Pop_RI bounds dashed."""
    states = [("low", r.mu_low, max(r.sd_low, (r.ref_high - r.ref_low) * 1e-3)),
              ("normal", r.mu_norm, max(r.sd_norm, (r.ref_high - r.ref_low) * 1e-3)),
              ("high", r.mu_high, max(r.sd_high, (r.ref_high - r.ref_low) * 1e-3))]
    for st, mu, sd in states:
        xs = np.linspace(mu - 4.5 * sd, mu + 4.5 * sd, 600)
        ys = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(xs, ys, color=STATE_COLORS[st], lw=1.0)
        ax.fill_between(xs, ys, color=STATE_COLORS[st], alpha=0.15)
    for bnd in (r.ref_low, r.ref_high):
        ax.axvline(bnd, color=DARK, ls=(0, (3, 2)), lw=0.5)
    # x range = what is actually drawn (densities out to 3.5 sd, where they are 0.2% of peak)
    # together with the Pop_RI bounds, so the bounds always stay in frame.
    xlo = min([r.ref_low] + [mu - 3.5 * sd for _, mu, sd in states])
    xhi = max([r.ref_high] + [mu + 3.5 * sd for _, mu, sd in states])
    pad = 0.06 * (xhi - xlo)
    xlo = max(0.0, xlo - pad) if r.ref_low >= 0 else xlo - pad
    ax.set_xlim(xlo, xhi + pad); ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=5)
    ax.locator_params(axis="x", nbins=4)
    _style(ax, None, None, f"{lab} ({r.unit})")


def _overlap_strip(ax, df):
    """Pairwise overlap coefficient per analyte, sorted by the normal-high overlap."""
    d = df.sort_values("ov_norm_high"); x = np.arange(len(d))
    ax.scatter(x, d.ov_low_norm, s=9, color=STATE_COLORS["low"], label="Low vs Normal", zorder=3)
    ax.scatter(x, d.ov_norm_high, s=9, color=STATE_COLORS["high"], label="Normal vs High", zorder=3)
    ax.scatter(x, d.ov_low_high, s=9, color=PALETTE["grey"], label="Low vs High", zorder=3)
    ax.axhline(0.5, color=DARK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=90, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_ylim(-0.03, 1.28); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=FONT_LEGEND)
    _style(ax, None, "Overlap coefficient", None)


def _state_conditional_fig(df):
    """p(x | H, s) for s = low / normal / high, one panel per analyte (top);
    the pairwise overlap of those three densities, every analyte (bottom)."""
    analytes = analyte_panel_order(set(df.index))
    ncol = 5; nrow = int(np.ceil(len(analytes) / ncol))
    # Heights in inches; h_gap is a spacer row so the grid-to-strip gap is exactly that (an
    # hspace here scales with the average row height, i.e. > 1 inch.
    h_top, h_grid, h_gap, h_strip, h_bot = 0.62, 1.15 * nrow, 0.4, 1.9, 0.35
    H = h_top + h_grid + h_gap + h_strip + h_bot
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(3, 1, height_ratios=[h_grid, h_gap, h_strip], hspace=0,
                          top=1 - h_top / H, bottom=h_bot / H, left=0.06, right=0.99)
    grid = gs[0].subgridspec(nrow, ncol, hspace=0.95, wspace=0.18)
    axes = [fig.add_subplot(grid[k // ncol, k % ncol]) for k in range(nrow * ncol)]
    for ax, lab in zip(axes, analytes):
        _density_panel(ax, lab, df.loc[lab])
    for ax in axes[len(analytes):]:
        ax.axis("off")
    handles = [Line2D([], [], color=STATE_COLORS[k], lw=1.5, label=f"Queried {k}") for k in STATE_ORDER]
    handles.append(Line2D([], [], color=DARK, ls=(0, (3, 2)), lw=0.5,
                          label=f"{RI_LABELS['PopRI']} bounds"))
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / H),
               frameon=False, fontsize=FONT_LEGEND, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.3)
    _overlap_strip(fig.add_subplot(gs[2]), df)
    return fig


def fig_state_conditional():
    """One state-conditional figure per NORMA version that has a results CSV."""
    figs = {v: _state_conditional_fig(df) for v in NORMA_RUN_SUFFIXES
            if (df := _load_state_conditional(v)) is not None}
    return figs or None


FIGURES = [
    FigSpec("04_refs", "state_conditional", fig_state_conditional, False, (), None),
]


if __name__ == "__main__":
    main()
