#!/usr/bin/env python
"""Cohen et al. 2021 (Nat Med) personalized lab test models as RI benchmarks.

Implements the lab test regression models of Cohen et al., "Personalized lab
test models to quantify disease potentials in healthy individuals", Nature
Medicine 27, 1582-1591 (2021), doi:10.1038/s41591-021-01468-6, adapted to the
NORMA validation splits:

  m2  single-lab single-time point: XGBoost on (age, sex, baseline mean of
      the target analyte).
  m3  multi-lab single-time point: XGBoost on (age, sex, baseline means of
      all analytes).
  m4  multi-lab multi-time point: XGBoost on (age, sex, per-time-bin means
      of the top-15 analytes), analytes selected per fold by mean absolute
      SHAP values of the corresponding m3 model, as in the paper.

Training follows their protocol exactly: separate model per lab test,
XGBoost 'gbtree' booster with squared-error objective and their published
hyperparameters, fivefold cross-validation with folds controlling for age
and sex distribution, and downsampling of the training population to a
uniform age distribution at 5-year resolution. Each patient's prediction is
out-of-fold. Following their healthy-trajectory training criterion
("showing within normal levels at least until prediction date", Fig. 4a),
training is restricted to pairs whose baseline values all fall within the
population reference interval; predictions are still produced for every
pair, as in their patient report. (Their diagnosis/medication-based healthy
filter is not reproducible in ICU-timescale cohorts; the within-norm
criterion is the component that is defined in all cohorts.) Personalized range = prediction +/- z * s.d. of the predicted values for that
analyte
(their Fig. 4c report uses z=1; default here is z=1.96 for coverage parity
with the 95% intervals of the other methods -- ri_std is stored so either
band can be derived).

Documented adaptations (their exact windows are undefined on ICU-timescale
cohorts): features are drawn from each patient's baseline split rather than
calendar windows 2-3y (m2/m3) or 2-6y (m4) before prediction, m4 uses 8
equal-width bins over the patient's baseline span in place of 6-month
calendar bins, and missing bins are filled with the patient's baseline mean
of that analyte (in place of their iterative 6-month regression imputation),
falling back to XGBoost's native missing-value handling when the patient
never measured the analyte.

Their healthy-cohort criteria (ST2 diagnoses, ST3 drug-lab pairs, pregnancy and
hospitalization) are implemented at the end of this file and are applied to the
development cohorts, where calendar timestamps and raw records exist; build the
event tables once with `python process/cohen_events.py --build_healthy`.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from common import REFERENCE_INTERVALS, detect_cols, sex_key, sex_idx
# Healthy-cohort event tables are built by process/cohen_events.py, which
# reads raw MIMIC-IV / EHRSHOT records; see its --build_healthy CLI.
from process.cohen_events import load_events, healthy_mask

N_FOLDS = 5
N_BINS = 8  # m4: eight time periods, as in the paper
TOP_K_LABS = 15  # m4: top 15 labs by m3 mean absolute SHAP, as in the paper

# Exact hyperparameters from Cohen et al. Methods ("Lab test regression
# models"). 'reg:linear' in xgboost 0.81 is 'reg:squarederror' today.
XGB_PARAMS = {
    "m2": dict(num_boost_round=750, subsample=1.0, max_depth=2,
               colsample_bytree=1.0, eta=0.025, min_child_weight=3),
    "m3": dict(num_boost_round=1250, subsample=0.75, max_depth=5,
               colsample_bytree=0.8, eta=0.01, min_child_weight=2),
    "m4": dict(num_boost_round=1950, subsample=0.5, max_depth=6,
               colsample_bytree=0.6, eta=0.01, min_child_weight=3, gamma=0.7),
}


def _frac_within_popri(baseline, c):
    """Per-pair fraction of baseline values within the population RI."""
    sex_keys = baseline[c["sex"]].map(sex_key)
    lows = np.full(len(baseline), -np.inf)
    highs = np.full(len(baseline), np.inf)
    known = np.zeros(len(baseline), dtype=bool)
    analyte_vals = baseline[c["analyte"]].values
    sex_vals = sex_keys.values
    for analyte in pd.unique(analyte_vals):
        if analyte not in REFERENCE_INTERVALS:
            continue
        for sk in ("M", "F"):
            low, high, _ = REFERENCE_INTERVALS[analyte][sk]
            m = (analyte_vals == analyte) & (sex_vals == sk)
            lows[m], highs[m], known[m] = low, high, True
    vals = baseline[c["value"]].values.astype(float)
    # Unknown RIs count as within-norm so those analytes stay trainable
    in_norm = (~known) | ((vals >= lows) & (vals <= highs))
    return pd.Series(in_norm, index=baseline.index).groupby(
        [baseline[c["pid"]], baseline[c["analyte"]]]).mean()


def build_pair_table(split_df, min_bl=5):
    """One row per (patient, analyte) with baseline features and index target.

    Returns (pairs, blmean_wide, bin_wide):
      pairs       -- pair-level frame with shared cols, bl_mean, target
      blmean_wide -- patient-level baseline mean per analyte (m3 features)
      bin_wide    -- patient-level per-analyte per-bin means (m4 features)
    """
    c = detect_cols(split_df)
    df = split_df.copy()
    df[c["analyte"]] = df[c["analyte"]].replace("", "NA").fillna("NA")

    baseline = df[df["split"] == "baseline"].dropna(subset=[c["value"]])
    baseline = baseline.drop_duplicates(subset=[c["pid"], c["analyte"], c["time"]])

    # Target: first index measurement per pair (their "sampled lab test")
    index = df[df["split"] == "index"].dropna(subset=[c["value"]])
    index = index.sort_values(c["time"]).drop_duplicates(
        subset=[c["pid"], c["analyte"]], keep="first")
    target = index.set_index([c["pid"], c["analyte"]])[c["value"]].rename("target")

    agg = baseline.groupby([c["pid"], c["analyte"]]).agg(
        sex=(c["sex"], "first"),
        age=("age", "first"),
        n_bl=(c["value"], "count"),
        bl_mean=(c["value"], "mean"),
        t_min=(c["time"], "min"),
        t_max=(c["time"], "max"),
    )

    time_unit = split_df.attrs.get("time_unit", "minutes")
    per_day = {"minutes": 60 * 24, "hours": 24, "days": 1}[time_unit]
    agg["t_span"] = (agg["t_max"] - agg["t_min"]) / per_day

    # Fraction of baseline values within the population reference interval
    # (Cohen et al. train on trajectories "showing within normal levels at
    # least until prediction date"; pairs with frac < 1 are excluded from
    # training but still receive predictions)
    agg["frac_bl_norm"] = _frac_within_popri(baseline, c)

    pairs = agg.join(target, how="inner").reset_index()
    pairs = pairs[pairs["n_bl"] >= min_bl].copy()
    pairs = pairs.rename(columns={c["pid"]: "patient_id", c["analyte"]: "analyte"})
    pairs["sex_idx"] = pairs["sex"].map(sex_idx)
    pairs["age_num"] = pd.to_numeric(pairs["age"], errors="coerce")
    pairs = pairs.sort_values(["patient_id", "analyte"]).reset_index(drop=True)

    # m3 features: patient-level baseline mean of every analyte
    blmean_wide = baseline.pivot_table(
        index=c["pid"], columns=c["analyte"], values=c["value"], aggfunc="mean")
    blmean_wide.index.name = "patient_id"

    # m4 features: 8 equal-width bins over each patient's overall baseline span
    bl = baseline[[c["pid"], c["analyte"], c["time"], c["value"]]].copy()
    t0 = bl.groupby(c["pid"])[c["time"]].transform("min")
    span = bl.groupby(c["pid"])[c["time"]].transform("max") - t0
    span = span.replace(0, np.nan)
    bl["bin"] = np.minimum(
        ((bl[c["time"]] - t0) / span * N_BINS).fillna(N_BINS - 1).astype(int),
        N_BINS - 1)
    bin_wide = bl.pivot_table(
        index=c["pid"], columns=[c["analyte"], "bin"], values=c["value"],
        aggfunc="mean")
    bin_wide.columns = [f"{a}__b{b}" for a, b in bin_wide.columns]
    bin_wide.index.name = "patient_id"

    return pairs, blmean_wide, bin_wide


def assign_folds(sub, n_folds=N_FOLDS, seed=0):
    """Fold assignment controlling for age and sex distribution across folds
    (per-analyte, as in the paper: round-robin within (sex, 5y-age) strata)."""
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(sub), dtype=int)
    age_bin = (sub["age_num"].fillna(65) // 5).astype(int)
    strata = sub["sex_idx"].astype(str) + "_" + age_bin.astype(str)
    for _, idx in pd.Series(np.arange(len(sub)), index=strata.values).groupby(level=0):
        order = rng.permutation(idx.values)
        folds[order] = np.arange(len(order)) % n_folds
    return folds


def uniform_age_mask(sub, eligible=None, floor=20, seed=0):
    """Downsample to a uniform age distribution at 5-year resolution
    (training only; every pair still receives an out-of-fold prediction).
    `eligible` restricts the pool (e.g. the within-norm healthy filter)."""
    rng = np.random.RandomState(seed)
    if eligible is None:
        eligible = np.ones(len(sub), dtype=bool)
    age_bin = (sub["age_num"].fillna(65) // 5).astype(int)
    counts = age_bin[eligible].value_counts()
    if len(counts) == 0:
        return np.zeros(len(sub), dtype=bool)
    n_target = max(int(counts.min()), floor)
    mask = np.zeros(len(sub), dtype=bool)
    pos = np.arange(len(sub))[eligible]
    for _, idx in pd.Series(pos, index=age_bin[eligible].values).groupby(level=0):
        take = idx.values if len(idx) <= n_target else rng.choice(
            idx.values, n_target, replace=False)
        mask[take] = True
    return mask


def _train_predict(X, y, folds, train_ok, params, sigma_ok=None, seed=0):
    """Per-fold XGBoost train/predict. Returns (oof_pred, sigma_per_fold, models).

    sigma_ok restricts which pairs' predictions enter the interval-scale estimate
    (Cohen et al.'s sigma is estimated on the healthy CV population)."""
    params = dict(params)
    n_rounds = params.pop("num_boost_round")
    xgb_params = {
        "booster": "gbtree", "objective": "reg:squarederror",
        "eval_metric": "rmse", "tree_method": "hist",
        "seed": seed, "verbosity": 0, **params,
    }
    oof = np.full(len(y), np.nan)
    models = {}
    for k in range(N_FOLDS):
        tr = (folds != k) & train_ok
        if tr.sum() < 10:
            continue
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        bst = xgb.train(xgb_params, dtrain, num_boost_round=n_rounds)
        te = folds == k
        if te.any():
            oof[te] = bst.predict(xgb.DMatrix(X[te]))
        models[k] = bst

    sigma = _oof_sigma(y, oof, folds, sigma_ok)
    return oof, sigma, models


def _sigma_scalar(pred):
    """Cohort-level interval scale for one analyte: the s.d. of the *predicted*
    values, per Cohen et al. Fig. 4c ("range (+/-1 x s.d.) of values predicted
    using our personalized model"). One scalar per analyte, so it sets where the
    z=1 classification threshold falls but leaves per-analyte AUROC unchanged."""
    v = np.asarray(pred, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size >= 2 else float("nan")


def _oof_sigma(y, oof, folds, sigma_ok=None):
    """Per-fold interval scale from the out-of-fold predictions of the other
    folds, restricted to sigma_ok pairs (the healthy population)."""
    ok = np.isfinite(y) & np.isfinite(oof)
    if sigma_ok is not None:
        ok &= sigma_ok
    sigma = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        others = (folds != k) & ok
        src = others if others.sum() >= 2 else (ok if ok.sum() >= 2 else None)
        if src is not None:
            sigma[folds == k] = _sigma_scalar(oof[src])
    return sigma


def _m3_top_analytes(bst, X_tr, feat_names, analytes):
    """Top analytes by mean absolute SHAP of an m3 model (paper's m4 selection)."""
    contribs = bst.predict(xgb.DMatrix(X_tr, feature_names=feat_names),
                           pred_contribs=True)
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)  # drop bias term
    imp = {f: v for f, v in zip(feat_names, mean_abs)}
    ranked = sorted(analytes, key=lambda a: -imp.get(f"bl__{a}", 0.0))
    return ranked[:TOP_K_LABS]


def compute_cohen_refs(split_df, models=("m2", "m3", "m4"), z=1.96,
                       min_pairs=50, age_downsample=True, healthy_train=True,
                       norm_frac=1.0,
                       seed=0):
    """Compute Cohen et al. reference intervals for every valid pair.

    Returns a long-format frame matching ref_intervals.csv, with
    method in {cohen_m2, cohen_m3, cohen_m4}.
    """
    pairs, blmean_wide, bin_wide = build_pair_table(split_df)
    analytes = sorted(pairs["analyte"].unique())
    print(f"  Cohen benchmark: {len(pairs):,} pairs, {len(analytes)} analytes, "
          f"models={list(models)}")

    # Patient-level feature blocks aligned to pairs
    m3_feats = [f"bl__{a}" for a in analytes]
    blmean_aligned = blmean_wide.reindex(pairs["patient_id"])
    blmean_aligned.columns = [f"bl__{a}" for a in blmean_aligned.columns]
    blmean_aligned = blmean_aligned.reindex(columns=m3_feats)

    bin_cols = list(bin_wide.columns)
    bin_aligned = bin_wide.reindex(pairs["patient_id"])
    # m4 imputation adaptation: fill a missing bin with the patient's
    # baseline mean of that analyte (paper uses an iterative 6-month
    # regression model); analytes never measured stay NaN (xgboost native).
    for col in bin_cols:
        a = col.rsplit("__b", 1)[0]
        src = f"bl__{a}"
        if src in blmean_aligned.columns:
            vals = bin_aligned[col].to_numpy(dtype=float, copy=True)
            fill = blmean_aligned[src].to_numpy(dtype=float)
            missing = np.isnan(vals)
            vals[missing] = fill[missing]
            bin_aligned[col] = vals

    rows = []
    for analyte in analytes:
        sub_mask = (pairs["analyte"] == analyte).values
        sub = pairs[sub_mask].reset_index(drop=True)
        if len(sub) < min_pairs:
            print(f"    {analyte}: {len(sub)} pairs < {min_pairs}, skipped")
            continue

        folds = assign_folds(sub, seed=seed)
        healthy_ok = (sub["frac_bl_norm"] >= norm_frac).values if healthy_train \
            else np.ones(len(sub), dtype=bool)
        if age_downsample:
            train_ok = uniform_age_mask(sub, eligible=healthy_ok, seed=seed)
        else:
            train_ok = healthy_ok
        if healthy_train:
            print(f"    {analyte}: {healthy_ok.sum():,}/{len(sub):,} pairs "
                  f"within-norm baseline, {train_ok.sum():,} in training set")
        if train_ok.sum() < min_pairs:
            print(f"    {analyte}: training set < {min_pairs}, skipped")
            continue
        y = sub["target"].values.astype(float)
        base = np.column_stack([sub["age_num"].fillna(np.nan).values,
                                sub["sex_idx"].values.astype(float)])

        preds = {}

        if "m2" in models or "m3" in models or "m4" in models:
            X2 = np.column_stack([base, sub["bl_mean"].values])
            preds["m2"] = _train_predict(X2, y, folds, train_ok,
                                         XGB_PARAMS["m2"],
                                         sigma_ok=healthy_ok, seed=seed)

        m3_models = None
        if "m3" in models or "m4" in models:
            X3 = np.column_stack([base, blmean_aligned.values[sub_mask]])
            oof3, sig3, m3_models = _train_predict(
                X3, y, folds, train_ok, XGB_PARAMS["m3"],
                sigma_ok=healthy_ok, seed=seed)
            preds["m3"] = (oof3, sig3, m3_models)

        if "m4" in models:
            feat_names = ["age", "sex"] + m3_feats
            X3 = np.column_stack([base, blmean_aligned.values[sub_mask]])
            Xbin = bin_aligned.values[sub_mask]
            oof4 = np.full(len(sub), np.nan)
            for k, bst in (m3_models or {}).items():
                tr = (folds != k) & train_ok
                top = _m3_top_analytes(bst, X3[tr], feat_names, analytes)
                keep_cols = [i for i, cname in enumerate(bin_cols)
                             if cname.rsplit("__b", 1)[0] in top]
                X4 = np.column_stack([base, Xbin[:, keep_cols]])
                p = dict(XGB_PARAMS["m4"])
                n_rounds = p.pop("num_boost_round")
                xgb_params = {"booster": "gbtree",
                              "objective": "reg:squarederror",
                              "eval_metric": "rmse", "tree_method": "hist",
                              "seed": seed, "verbosity": 0, **p}
                bst4 = xgb.train(xgb_params, xgb.DMatrix(X4[tr], label=y[tr]),
                                 num_boost_round=n_rounds)
                te = folds == k
                if te.any():
                    oof4[te] = bst4.predict(xgb.DMatrix(X4[te]))
            sig4 = _oof_sigma(y, oof4, folds, sigma_ok=healthy_ok)
            preds["m4"] = (oof4, sig4, None)

        for m in models:
            if m not in preds:
                continue
            oof, sigma, _ = preds[m]
            valid = np.isfinite(oof) & np.isfinite(sigma)
            # R2 on the healthy population, comparable to the paper's figures
            hv = valid & healthy_ok
            r2 = np.nan
            if hv.sum() > 2 and np.var(y[hv]) > 0:
                r2 = 1 - np.var(y[hv] - oof[hv]) / np.var(y[hv])
            print(f"    {analyte} cohen_{m}: {valid.sum():,}/{len(sub):,} "
                  f"predicted, healthy R2={r2:.3f}")
            for i in np.where(valid)[0]:
                rows.append({
                    "patient_id": sub.at[i, "patient_id"],
                    "analyte": analyte,
                    "sex": sub.at[i, "sex"], "age": sub.at[i, "age"],
                    "n_bl": sub.at[i, "n_bl"], "t_span": sub.at[i, "t_span"],
                    "method": f"cohen_{m}",
                    "ri_mean": float(oof[i]), "ri_std": float(sigma[i]),
                    "ri_low": float(oof[i] - z * sigma[i]),
                    "ri_high": float(oof[i] + z * sigma[i]),
                })

    ref_df = pd.DataFrame(rows)
    print(f"  Cohen benchmark: {len(ref_df):,} rows")
    return ref_df


# ── Development-cohort training (MIMIC-IV + EHRSHOT, same split as NORMA) ────
#
# Trains the Cohen models on the same combined development split NORMA was
# trained on (load_and_split_data, random_state=42) and applies them to the
# validation cohorts, mirroring NORMA's train-once / transfer evaluation.
# The within-norm healthy filter is applied to the development sequences:
# a sequence enters training only if its history stays within the population
# reference interval, and interval scales are estimated on the healthy
# development validation split.

import os
import sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # model/baselines
ROOT_DIR = os.path.dirname(os.path.dirname(_BASE_DIR))   # norma root
# The processed dev cohort (EHRSHOT + MIMIC-IV) lives outside the repo because
# both sources are access-controlled; point NORMA_DATA_DIR at your own copy.
DEFAULT_DEV_DIR = os.environ.get(
    "NORMA_DATA_DIR", os.path.join(os.path.dirname(ROOT_DIR), "data", "processed"))
# model/logs/baselines/ here and inside Clalit (jobs/run_clalit.py --pack_bundle carries it in)
DEFAULT_ARTIFACT = os.path.join(ROOT_DIR, "model", "logs", "baselines", "cohen_dev_models.pkl")


def load_dev_sequences(dev_dir, source="combined", nstates=2):
    """Load dev sequences with NORMA's own split function (random_state=42)."""
    from data import load_and_split_data   # local: data.py pulls in torch
    return load_and_split_data(dev_dir, source, print_info=True, nstates=nstates)


def build_dev_frames(seqs, min_hist=5):
    """Per-sequence history features + patient-level wide frames.

    Each sequence contributes one training pair: history = x[:-1] (features),
    target = x[-1]. frac_norm is the fraction of history values within the
    population RI (Cohen et al.'s healthy-trajectory training criterion).
    """
    ri = REFERENCE_INTERVALS

    # Pass 1: per-sequence stats + patient-level history time spans
    recs = []
    pid_tmin, pid_tmax = {}, {}
    for seq in seqs:
        x = np.asarray(seq["x"], dtype=float)
        t = np.asarray(seq["t"], dtype=float)
        ok = np.isfinite(x)
        x, t = x[ok], t[ok]
        if len(x) < min_hist + 1:
            continue
        hist, th = x[:-1], t[:-1]
        pk = f"{seq['source']}|{seq['pid']}"
        sex_idx = 1 if seq["sex"] == 1 else 0  # 0=M, 1=F (process convention)
        analyte = seq["test_name"]
        if analyte in ri:
            low, high, _ = ri[analyte]["F" if sex_idx == 1 else "M"]
            frac_norm = float(np.mean((hist >= low) & (hist <= high)))
        else:
            frac_norm = 1.0
        pid_tmin[pk] = min(pid_tmin.get(pk, np.inf), float(th.min()))
        pid_tmax[pk] = max(pid_tmax.get(pk, -np.inf), float(th.max()))
        recs.append((pk, analyte, sex_idx, float(seq["age"]),
                     float(hist.mean()), len(hist), frac_norm,
                     float(x[-1]), hist, th))

    pairs = pd.DataFrame(recs, columns=[
        "pid_key", "analyte", "sex_idx", "age_num", "hist_mean", "n_hist",
        "frac_bl_norm", "target", "_hist", "_th"])

    # m3 features: patient-level history mean per analyte
    blmean_wide = pairs.pivot_table(index="pid_key", columns="analyte",
                                    values="hist_mean", aggfunc="mean")

    # m4 features: 8 equal-width bins over each patient's overall history span
    bin_recs = []
    for pk, analyte, hist, th in zip(pairs["pid_key"], pairs["analyte"],
                                     pairs["_hist"], pairs["_th"]):
        t0, t1 = pid_tmin[pk], pid_tmax[pk]
        span = t1 - t0
        if span > 0:
            bins = np.minimum(((th - t0) / span * N_BINS).astype(int), N_BINS - 1)
        else:
            bins = np.full(len(th), N_BINS - 1, dtype=int)
        for b in np.unique(bins):
            bin_recs.append((pk, f"{analyte}__b{b}", float(hist[bins == b].mean())))
    bin_df = pd.DataFrame(bin_recs, columns=["pid_key", "col", "val"])
    bin_wide = bin_df.pivot_table(index="pid_key", columns="col",
                                  values="val", aggfunc="mean")

    pairs = pairs.drop(columns=["_hist", "_th"])
    return pairs, blmean_wide, bin_wide


def build_dev_frames_cohen_healthy(dev_dir, min_hist=5, source="combined",
                                   train_sources=("ehrshot",)):
    """Build train/val frames from dev lab data after applying Cohen et al.'s
    healthy-cohort criteria at the point level (diagnoses, medications,
    pregnancy, hospitalization -- see the healthy-cohort criteria section
    at the end of this file).

    Uses combined_lab_data_v2.csv (calendar timestamps) restricted to the
    sequence keys of NORMA's own train/val split, drops unhealthy points,
    and reassembles the surviving points into sequences.
    """
    train_seq, val_seq, _ = load_dev_sequences(dev_dir, source=source)
    key_split = {}
    for s in train_seq:
        key_split[(s["source"], s["pid"], s["test_name"])] = "train"
    for s in val_seq:
        key_split[(s["source"], s["pid"], s["test_name"])] = "val"

    lab_path = os.path.join(dev_dir, "combined_lab_data_v2.csv")
    print(f"  Loading dev lab points from {lab_path}")
    # keep_default_na=False so sodium's test_name "NA" is not read as NaN;
    # real missing values in the numeric columns are empty strings.
    lab = pd.read_csv(lab_path,
                      usecols=["source", "subject_id", "sex", "test_name",
                               "time", "age", "numeric_value"],
                      parse_dates=["time"], keep_default_na=False,
                      na_values={"numeric_value": ["", "nan", "NaN"],
                                 "age": ["", "nan", "NaN"], "time": [""]})
    lab = lab.dropna(subset=["numeric_value", "time"])

    keys_df = pd.DataFrame(
        [(k[0], k[1], k[2], v) for k, v in key_split.items()],
        columns=["source", "subject_id", "test_name", "split"])
    lab = lab.merge(keys_df, on=["source", "subject_id", "test_name"],
                    how="inner")
    if train_sources is not None:
        lab = lab[lab["source"].isin(train_sources)]
        print(f"  Restricted to sources {list(train_sources)}")
    print(f"  {len(lab):,} points in train/val sequences")

    healthy = np.zeros(len(lab), dtype=bool)
    for src in lab["source"].unique():
        m = (lab["source"] == src).values
        ev = load_events(src)
        pts = lab.loc[m, ["subject_id", "test_name", "time"]].rename(
            columns={"test_name": "analyte"})
        healthy[m] = healthy_mask(pts, ev)
        print(f"    {src}: {healthy[m].sum():,}/{m.sum():,} points "
              f"({healthy[m].mean():.1%}) in healthy context")
    lab = lab[healthy]

    # Reassemble surviving points into sequences per split
    seqs = {"train": [], "val": []}
    lab = lab.sort_values(["source", "subject_id", "test_name", "time"])
    for (src, pid, tn, split), g in lab.groupby(
            ["source", "subject_id", "test_name", "split"], sort=False):
        if len(g) < min_hist + 1:
            continue
        tdays = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 86400.0
        seqs[split].append({
            "source": src, "pid": pid, "test_name": tn,
            "sex": g["sex"].iloc[0], "age": g["age"].iloc[0],
            "x": g["numeric_value"].values.astype(float),
            "t": tdays.values,
        })
    print(f"  Healthy-context sequences: {len(seqs['train']):,} train, "
          f"{len(seqs['val']):,} val")
    tr = build_dev_frames(seqs["train"], min_hist=min_hist)
    va = build_dev_frames(seqs["val"], min_hist=min_hist)
    return tr, va


def _fill_bins_from_blmean(bin_aligned, blmean_aligned):
    """Fill missing bins with the patient's history mean of that analyte."""
    for col in bin_aligned.columns:
        a = col.rsplit("__b", 1)[0]
        src = a if a in blmean_aligned.columns else None
        if src is None:
            continue
        vals = bin_aligned[col].to_numpy(dtype=float, copy=True)
        fill = blmean_aligned[src].to_numpy(dtype=float)
        missing = np.isnan(vals)
        vals[missing] = fill[missing]
        bin_aligned[col] = vals
    return bin_aligned


def _fit_single(X, y, params, seed=0):
    params = dict(params)
    n_rounds = params.pop("num_boost_round")
    xgb_params = {"booster": "gbtree", "objective": "reg:squarederror",
                  "eval_metric": "rmse", "tree_method": "hist",
                  "seed": seed, "verbosity": 0, **params}
    return xgb.train(xgb_params, xgb.DMatrix(X, label=y),
                     num_boost_round=n_rounds)


def train_dev_cohen(dev_dir, models=("m2", "m3", "m4"), min_pairs=50,  # noqa: E501

                    age_downsample=True, healthy_train=True, seed=0,
                    source="combined", cohen_healthy=True,
                    train_sources=("ehrshot",), norm_frac=1.0):
    """Train Cohen m2/m3/m4 on the NORMA development train split.

    cohen_healthy=True applies Cohen et al.'s full healthy-cohort criteria
    (ST2 diagnoses, ST3 drug-lab pairs, pregnancy, hospitalization) at the
    point level before sequence assembly; the within-norm-history criterion
    (healthy_train) applies on top, as in their Fig. 4a; norm_frac is the
    minimum fraction of a pair's history inside PopRI to count as healthy
    (1.0 = every value, Cohen's definition; <1 relaxes it for thin cohorts).

    Returns an artifact dict: per-analyte boosters, interval scales (estimated
    on the healthy dev validation split), and feature-column metadata.
    """
    if cohen_healthy:
        (tr_pairs, tr_blmean, tr_bins), (va_pairs, va_blmean, va_bins) = \
            build_dev_frames_cohen_healthy(dev_dir, source=source,
                                           train_sources=train_sources)
    else:
        train_seq, val_seq, _ = load_dev_sequences(dev_dir, source=source)
        print(f"  Building dev frames: {len(train_seq):,} train / "
              f"{len(val_seq):,} val sequences")
        tr_pairs, tr_blmean, tr_bins = build_dev_frames(train_seq)
        va_pairs, va_blmean, va_bins = build_dev_frames(val_seq)
    analytes = sorted(tr_pairs["analyte"].unique())
    m3_cols = list(analytes)
    bin_cols = sorted(set(tr_bins.columns))
    print(f"  {len(tr_pairs):,} train pairs, {len(analytes)} analytes")

    def _features(pairs, blmean, bins, sub_mask):
        sub = pairs[sub_mask]
        base = np.column_stack([sub["age_num"].values,
                                sub["sex_idx"].values.astype(float)])
        bl = blmean.reindex(sub["pid_key"]).reindex(columns=m3_cols)
        bn = bins.reindex(sub["pid_key"]).reindex(columns=bin_cols)
        bn = _fill_bins_from_blmean(bn, bl)
        X2 = np.column_stack([base, sub["hist_mean"].values])
        X3 = np.column_stack([base, bl.values])
        return sub, base, X2, X3, bn.values

    artifact = {"analytes": m3_cols, "bin_cols": bin_cols, "models": {},
                "meta": {"dev_dir": dev_dir, "models": list(models),
                         "healthy_train": healthy_train,
                         "norm_frac": norm_frac,
                         "cohen_healthy": cohen_healthy,
                         "train_sources": list(train_sources) if train_sources else None,
                         "age_downsample": age_downsample, "seed": seed},
                "dev_medians": {}}

    for analyte in analytes:
        tr_mask = (tr_pairs["analyte"] == analyte).values
        va_mask = (va_pairs["analyte"] == analyte).values
        sub, base, X2, X3, Xbin = _features(tr_pairs, tr_blmean, tr_bins, tr_mask)
        vsub, vbase, vX2, vX3, vXbin = _features(va_pairs, va_blmean, va_bins, va_mask)
        y, vy = sub["target"].values, vsub["target"].values

        healthy = (sub["frac_bl_norm"] >= norm_frac).values if healthy_train \
            else np.ones(len(sub), dtype=bool)
        v_healthy = (vsub["frac_bl_norm"] >= norm_frac).values if healthy_train \
            else np.ones(len(vsub), dtype=bool)
        if age_downsample:
            train_ok = uniform_age_mask(sub.reset_index(drop=True),
                                        eligible=healthy, seed=seed)
        else:
            train_ok = healthy
        print(f"    {analyte}: {healthy.sum():,}/{len(sub):,} within-norm (>={norm_frac:g}), "
              f"{train_ok.sum():,} in training set, {v_healthy.sum():,} healthy val")
        if train_ok.sum() < min_pairs or v_healthy.sum() < 2:
            # Each Cohen variant is a per-analyte gradient-boosted model whose
            # interval WIDTH comes from the SD of held-out healthy predictions, so a
            # handful of patients gives both an unreliable fit and an unreliable
            # width. Skipping is our guard, not the paper's; the cost is that the
            # analyte gets no Cohen interval at all and silently drops out of the
            # benchmark. Tune with --cohen_min_pairs.
            print(f"    {analyte}: too few healthy pairs "
                  f"({train_ok.sum():,} train < {min_pairs}, {v_healthy.sum():,} val), skipped")
            continue

        artifact["dev_medians"][analyte] = float(np.median(sub["hist_mean"]))
        entry = {}

        if "m2" in models:
            bst = _fit_single(X2[train_ok], y[train_ok], XGB_PARAMS["m2"], seed)
            _p = bst.predict(xgb.DMatrix(vX2[v_healthy]))
            entry["m2"] = {"bst": bst, "sigma": _sigma_scalar(_p)}

        m3_bst = None
        if "m3" in models or "m4" in models:
            m3_bst = _fit_single(X3[train_ok], y[train_ok], XGB_PARAMS["m3"], seed)
            _p = m3_bst.predict(xgb.DMatrix(vX3[v_healthy]))
            if "m3" in models:
                entry["m3"] = {"bst": m3_bst, "sigma": _sigma_scalar(_p)}

        if "m4" in models:
            feat_names = ["age", "sex"] + [f"bl__{a}" for a in m3_cols]
            top = _m3_top_analytes(m3_bst, X3[train_ok], feat_names, m3_cols)
            keep_idx = [i for i, cname in enumerate(bin_cols)
                        if cname.rsplit("__b", 1)[0] in top]
            X4 = np.column_stack([base, Xbin[:, keep_idx]])
            vX4 = np.column_stack([vbase, vXbin[:, keep_idx]])
            bst4 = _fit_single(X4[train_ok], y[train_ok], XGB_PARAMS["m4"], seed)
            _p = bst4.predict(xgb.DMatrix(vX4[v_healthy]))
            entry["m4"] = {"bst": bst4, "sigma": _sigma_scalar(_p),
                           "bin_idx": keep_idx}

        for m, e in entry.items():
            Xv = {"m2": vX2, "m3": vX3}.get(m)
            if Xv is None:
                Xv = np.column_stack([vbase, vXbin[:, entry["m4"]["bin_idx"]]])
            pred = e["bst"].predict(xgb.DMatrix(Xv[v_healthy]))
            r2 = np.nan
            if v_healthy.sum() > 2 and np.var(vy[v_healthy]) > 0:
                r2 = 1 - np.var(vy[v_healthy] - pred) / np.var(vy[v_healthy])
            print(f"      {m}: sigma={e['sigma']:.4g}, healthy val R2={r2:.3f}")

        artifact["models"][analyte] = entry

    return artifact


def apply_dev_cohen(split_df, artifact, z=1.96):
    """Apply dev-trained Cohen models to a validation cohort's pairs.

    Returns long-format ref interval rows (method = cohen_m2/m3/m4)."""
    pairs, blmean_wide, bin_wide = build_pair_table(split_df)
    m3_cols = artifact["analytes"]
    bin_cols = artifact["bin_cols"]

    # Unit sanity check: cohort baseline medians vs dev training medians
    for analyte, med_dev in sorted(artifact["dev_medians"].items()):
        sub = pairs.loc[pairs["analyte"] == analyte, "bl_mean"]
        if len(sub) == 0 or med_dev == 0:
            continue
        ratio = float(np.median(sub)) / med_dev
        if not (0.67 <= ratio <= 1.5):
            print(f"    WARNING {analyte}: cohort/dev median ratio {ratio:.2f} "
                  f"-- check units before trusting transfer")

    rows = []
    for analyte, entry in artifact["models"].items():
        sub_mask = (pairs["analyte"] == analyte).values
        sub = pairs[sub_mask].reset_index(drop=True)
        if len(sub) == 0:
            continue
        base = np.column_stack([sub["age_num"].values,
                                sub["sex_idx"].values.astype(float)])
        bl = blmean_wide.reindex(sub["patient_id"]).reindex(columns=m3_cols)
        bn = bin_wide.reindex(sub["patient_id"]).reindex(columns=bin_cols)
        bn = _fill_bins_from_blmean(bn, bl)

        X = {"m2": np.column_stack([base, sub["bl_mean"].values])}
        if "m3" in entry or "m4" in entry:
            X["m3"] = np.column_stack([base, bl.values])
        if "m4" in entry:
            X["m4"] = np.column_stack([base, bn.values[:, entry["m4"]["bin_idx"]]])

        for m, e in entry.items():
            pred = e["bst"].predict(xgb.DMatrix(X[m]))
            sigma = e["sigma"]
            valid = np.isfinite(pred)
            print(f"    {analyte} cohen_{m}: {valid.sum():,}/{len(sub):,} predicted")
            for i in np.where(valid)[0]:
                rows.append({
                    "patient_id": sub.at[i, "patient_id"], "analyte": analyte,
                    "sex": sub.at[i, "sex"], "age": sub.at[i, "age"],
                    "n_bl": sub.at[i, "n_bl"], "t_span": sub.at[i, "t_span"],
                    "method": f"cohen_{m}",
                    "ri_mean": float(pred[i]), "ri_std": float(sigma),
                    "ri_low": float(pred[i] - z * sigma),
                    "ri_high": float(pred[i] + z * sigma),
                })

    ref_df = pd.DataFrame(rows)
    print(f"  Cohen benchmark (dev-trained): {len(ref_df):,} rows")
    return ref_df


def augment_cohen(ref_df, split_df, models=("m2", "m3", "m4"), z=1.96,
                  artifact_path=None, dev_dir=None, retrain=False,
                  min_pairs=50, age_downsample=True, healthy_train=True,
                  cohen_healthy=True, train_sources=("ehrshot",), seed=0,
                  force=False, norm_frac=1.0):
    """Append Cohen benchmark rows (cohen_m2/m3/m4) to an existing ref_df.

    Trains on the dev cohorts once (cached at artifact_path) and applies to
    split_df's pairs, restricted to the patient-analyte pairs already covered
    by the existing methods. Called from 04_compute_refs.py.
    """
    import pickle
    artifact_path = artifact_path or DEFAULT_ARTIFACT
    dev_dir = dev_dir or DEFAULT_DEV_DIR

    wanted = {f"cohen_{m}" for m in models}
    existing = set(ref_df["method"].unique()) & wanted
    if retrain and existing:
        # New models must replace rows produced by the old ones.
        force = True
    if existing and not force:
        todo = sorted(wanted - existing)
        if not todo:
            print(f"  All of {sorted(wanted)} already present "
                  f"(--cohen_force to recompute)")
            return ref_df
        models = [m.replace("cohen_", "") for m in todo]
        print(f"  {sorted(existing)} already present; computing {todo}")
    elif existing:
        print(f"  Recomputing {sorted(existing)}")
        ref_df = ref_df[~ref_df["method"].isin(wanted)]

    if os.path.exists(artifact_path) and not retrain:
        print(f"  Loading trained Cohen models from {artifact_path}")
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        missing = [m for m in models
                   if not any(m in e for e in artifact["models"].values())]
        if missing:
            raise RuntimeError(
                f"Cohen artifact lacks models {missing} -- rerun with retrain")
    elif not os.path.isdir(dev_dir):
        # Inside Clalit there are no dev cohorts: training needs
        # combined_sequences_v2.pkl, NORMA's training data, which is not carried
        # in.  Skip the Cohen arms and let every other method proceed -- 04_refs'
        # coverage table reports them as missing, so nothing is silently absent.
        # --cohen_retrain still fails loudly, since that asks for training.
        if retrain:
            raise SystemExit(
                f"\n  --cohen_retrain needs the dev cohorts, and {dev_dir} does not exist.")
        print(f"  No Cohen models at {artifact_path} and no dev cohort at {dev_dir}"
              f" -- skipping the Cohen arms.\n"
              f"    To include them: python jobs/cohen_portable.py --import <export dir>")
        return ref_df
    else:
        print(f"  Training Cohen models on dev cohorts ({dev_dir})")
        artifact = train_dev_cohen(
            dev_dir, models=models, min_pairs=min_pairs,
            age_downsample=age_downsample, healthy_train=healthy_train,
            cohen_healthy=cohen_healthy, train_sources=train_sources,
            seed=seed, norm_frac=norm_frac)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"  Saved trained models to {artifact_path}")

    cohen_df = apply_dev_cohen(split_df, artifact, z=z)
    cohen_df = cohen_df[cohen_df["method"].isin(wanted)]
    if len(cohen_df) == 0:
        print("  No Cohen rows computed")
        return ref_df

    # Same-pair coverage restriction
    base_pairs = set(zip(
        ref_df.loc[ref_df["method"] == "pop", "patient_id"].astype(str),
        ref_df.loc[ref_df["method"] == "pop", "analyte"],
    ))
    if base_pairs:
        in_base = [(str(p), a) in base_pairs
                   for p, a in zip(cohen_df["patient_id"], cohen_df["analyte"])]
        n_drop = len(cohen_df) - sum(in_base)
        if n_drop:
            print(f"  Dropping {n_drop:,} Cohen rows outside existing pair coverage")
        cohen_df = cohen_df[in_base]

    return pd.concat([ref_df, cohen_df], ignore_index=True)
