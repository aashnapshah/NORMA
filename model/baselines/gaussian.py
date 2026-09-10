#!/usr/bin/env python
"""Gaussian reference intervals estimated from a patient's own history.

Every method here fits a Gaussian to the patient's baseline values; they differ
in which values they keep and what they shrink toward.

The Per_RI setpoint stands slightly apart from the other three -- it takes the
history as it comes and lets a mixture decide which mode is the patient's
normal:

  setpoint  gmm_setpoint(): dominant component of a 1-3 component Gaussian
            mixture over the raw baseline values, AIC-selected. The Per_RI of
            the paper; the interval is setpoint +/- n_std * SD (04_refs
            --gmm_n_std, default 2). Method key in ref_intervals: per.

The other three share NORMA's estimand -- the individual's distribution in the
population-defined normal state -- but reach it by *filtering* the baseline
history to values inside the population reference interval (PopRI) rather than
by conditioning on a queried state:

  mle    Gaussian fit to the PopRI-normal baseline values:
         mean +/- z * SD (SD with ddof=1).
  trunc  Truncated-Gaussian fit to the same values: maximum-likelihood
         (mu, sigma) of a normal truncated to [PopRI_low, PopRI_high], which
         corrects the SD deflation caused by the filtering; the interval is
         mu +/- z * sigma of the untruncated latent normal.
  eb     Empirical-Bayes Gaussian on the same values. Per analyte and sex:
             theta_i ~ N(mu, tau^2),  y_ij | theta_i ~ N(theta_i, sigma_i^2),
             sigma_i^2 ~ Scaled-Inv-chi2(nu0, s0^2).
         Hyperparameters are estimated by maximising the marginal likelihood
         (type-II ML) over the patient-analyte histories being shrunk:
           * (nu0, s0^2) from the sample variances, using the exact marginal
             s_i^2 / s0^2 ~ F(n_i - 1, nu0);
           * (mu, tau^2) from the sample means, using
             ybar_i ~ N(mu, tau^2 + sigma_hat_i^2 / n_i)
             with sigma_hat_i^2 the posterior-shrunk within-patient variance.
         prior_source="popri" (the benchmark's definition since 2026-09-03):
         the MEAN prior is the published population reference interval --
         mu = PopRI midpoint, tau = half-width / z -- so the method shrinks
         each patient's history toward the population range and no patient
         data enters the mean prior; the variance prior (nu0, s0^2) comes
         from the dev-cohort type-II ML fit (cached, transferred like NORMA
         and Cohen). "cohort" and "dev" estimate (mu, tau^2) by type-II ML
         from other patients' histories instead (in-cohort / transferred).
         The interval is the plug-in posterior-predictive band
             theta_hat +/- z * sqrt(sigma_hat^2 + V),
         with sigma_hat^2 = (nu0 s0^2 + (n-1) s^2) / (nu0 + n - 1),
         precision = 1/tau^2 + n/sigma_hat^2, theta_hat the precision-
         weighted mean of mu and ybar, V = 1/precision.

Fallback: pairs with fewer PopRI-normal baseline values than a method needs
(mle/trunc: 2; eb: none -- its prior predictive is a complete interval) receive
the PopRI interval itself. Fallback rows are
marked by ri_mean = ri_std = NaN, so they remain distinguishable in
ref_intervals. Pairs whose PopRI is undefined get NaN intervals.

Method keys in ref_intervals: per / gaussian_mle / gaussian_trunc / gaussian_eb.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(_BASE_DIR))   # norma root
# The processed dev cohort (EHRSHOT + MIMIC-IV) lives outside the repo because
# both sources are access-controlled; point NORMA_DATA_DIR at your own copy.
DEFAULT_DEV_DIR = os.environ.get(
    "NORMA_DATA_DIR", os.path.join(os.path.dirname(ROOT_DIR), "data", "processed"))
# model/logs/baselines/ here and inside Clalit (jobs/run_clalit.py --pack_bundle carries it in)
DEFAULT_PRIOR_PATH = os.path.join(ROOT_DIR, "model", "logs", "baselines", "gaussian_eb_prior_dev.pkl")
# Text twin of the pickle, checked into the repo.  The dev fit contributes only
# nu0 and s02 per (analyte, sex) -- 60 keys of six numbers -- and a pickle cannot
# be carried into Clalit, so the prior travels as JSON and the loader prefers it.
DEFAULT_PRIOR_JSON = os.path.splitext(DEFAULT_PRIOR_PATH)[0] + ".json"


def _read_prior_json(path):
    import json
    with open(path) as f:
        payload = json.load(f)
    prior = {}
    for row in payload["prior"]:
        row = dict(row)
        key = (row.pop("analyte"), row.pop("sex"))
        prior[key] = row
    return {"prior": prior, "meta": payload.get("meta", {})}

METHODS = ("mle", "trunc", "eb")
MIN_N = {"mle": 2, "trunc": 2, "eb": 2}   # eb needs s^2 from the patient
NU0_BOUNDS = (2.1, 500.0)


# ---------------------------------------------------------------------------
# Per_RI setpoint (Gaussian mixture over the raw history)
# ---------------------------------------------------------------------------
# The one estimator here that does not filter to the PopRI-normal values. Shared
# by 04_refs (the `per` rows), 06_sensitivity (synthetic histories) and the web
# app; it was in scripts/lib/metrics.py until the 2026-09-08 cleanup, which put
# every Gaussian method in this file.

GMM_WEIGHT_THRESHOLDS = {2: 0.70, 3: 0.45}   # dominant component must carry this much


def gmm_setpoint(values, max_components=3):
    """Fit a 1-3 component Gaussian mixture (AIC-selected) and return the dominant
    component's (mean, std); falls back to the plain mean / SD when no mixture
    is convincingly better or the dominant component is too light."""
    from sklearn.mixture import GaussianMixture
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)

    n = len(vals)
    n_unique = len(np.unique(vals))
    mean_1, var_1 = np.mean(vals), np.var(vals)
    if var_1 == 0:
        return mean_1, 0.0

    log_lik_1 = -0.5 * n * (np.log(2 * np.pi * var_1) + 1)
    best_aic = 2 * 2 - 2 * log_lik_1
    best_model = None

    for nc in range(2, max_components + 1):
        if n < nc or n_unique < nc:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(n_components=nc, covariance_type="full",
                                  max_iter=300, reg_covar=0.001, random_state=0).fit(vals.reshape(-1, 1))
        aic = gmm.aic(vals.reshape(-1, 1))
        if aic < best_aic and gmm.converged_:
            best_aic, best_model = aic, gmm

    if best_model is not None:
        weights = best_model.weights_
        dom = int(np.argmax(weights))
        if weights[dom] > GMM_WEIGHT_THRESHOLDS[best_model.n_components]:
            return best_model.means_[dom, 0], np.sqrt(best_model.covariances_[dom, 0, 0])

    return mean_1, np.sqrt(var_1)


# ---------------------------------------------------------------------------
# Per-pair estimators (PopRI-normal history)
# ---------------------------------------------------------------------------

def fit_mle(v):
    """(mean, SD ddof=1) of v; requires len(v) >= 2."""
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def fit_trunc(v, a, b):
    """MLE (mu, sigma) of a normal truncated to [a, b] given values v in [a, b].

    Falls back to the plain Gaussian fit when the SD is zero, the bounds are
    not finite, or the optimiser fails.
    """
    from scipy.optimize import minimize
    from scipy.special import ndtr

    v = np.asarray(v, dtype=float)
    n = len(v)
    m, s = fit_mle(v)
    w = b - a
    if s == 0 or not np.isfinite(w) or w <= 0:
        return m, s

    def nll(p):
        mu, ls = p
        sig = np.exp(ls)
        z = ndtr((b - mu) / sig) - ndtr((a - mu) / sig)
        return (n * ls + 0.5 * np.sum(((v - mu) / sig) ** 2)
                + n * np.log(max(z, 1e-300)))

    x0 = [m, np.log(max(s, 1e-3 * w))]
    bounds = [(a - 3 * w, b + 3 * w), (np.log(1e-3 * w), np.log(10 * w))]
    try:
        res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    except Exception:
        return m, s
    mu, sig = float(res.x[0]), float(np.exp(res.x[1]))
    if not (np.isfinite(mu) and np.isfinite(sig)):
        return m, s
    return mu, sig


def _shrunk_var(n, s2, nu0, s02):
    return (nu0 * s02 + (n - 1) * s2) / (nu0 + n - 1)


def eb_posterior(v, prior, low=None, high=None):
    """Plug-in posterior predictive (mean, SD), or (nan, nan) when n < 2.

    Normal-normal empirical Bayes: the prior gives (mu, tau^2), the patient's
    own values give their latent (mean, variance) and n, and theta_hat is the
    precision-weighted mean of the two.  The band is the posterior predictive,
    theta_hat +/- z*sqrt(sigma^2 + V) with V = 1/precision.

    The patient's values are the ones inside the PopRI, so their sample mean and
    variance describe a *truncated* normal and both are biased -- the variance
    downward, which is what fit_trunc exists to correct.  Given the bounds, the
    latent (mu_i, sigma_i) therefore come from the truncated-normal MLE, so this
    is trunc with shrinkage toward the population rather than mle with
    shrinkage.  Without bounds it falls back to the plain sample moments.

    A prior carrying (nu0, s0^2) -- prior_source "cohort" or "dev" -- shrinks
    the variance toward s0^2 first and can answer at any n, including 0.  The
    PopRI prior has no such term, so n < 2 has no within-person variance and the
    caller falls back to the PopRI itself, as mle and trunc do.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    mu, tau2 = prior["mu"], prior["tau2"]
    nu0, s02 = prior.get("nu0"), prior.get("s02")
    bounded = (low is not None and high is not None
               and np.isfinite(low) and np.isfinite(high) and high > low)

    if nu0 is None or s02 is None:
        if n < 2:
            return np.nan, np.nan
        ybar, sig = fit_trunc(v, low, high) if bounded else fit_mle(v)
        sig2 = sig ** 2
        # The truncated MLE is unidentified when the values fill the window:
        # a very wide latent normal is almost flat over [low, high] and fits
        # them just as well, so sigma runs away (hundreds, for a PopRI 29 wide).
        # Cap the within-person variance at the population variance -- an
        # individual's own spread cannot exceed the population's, and at the cap
        # the personalised band is simply no narrower than the PopRI.
        if not np.isfinite(sig2) or sig2 > tau2:
            sig2 = min(float(np.var(v, ddof=1)), tau2)
            ybar = float(v.mean())
    else:
        if n == 0:                               # prior predictive
            return float(mu), float(np.sqrt(s02 + tau2))
        ybar = float(v.mean())
        s2 = float(v.var(ddof=1)) if n >= 2 else 0.0
        sig2 = _shrunk_var(n, s2, nu0, s02)
    if not np.isfinite(sig2) or sig2 <= 0:
        return np.nan, np.nan
    prec = 1.0 / tau2 + n / sig2
    theta = (mu / tau2 + n * ybar / sig2) / prec
    return theta, float(np.sqrt(sig2 + 1.0 / prec))


# ---------------------------------------------------------------------------
# Empirical-Bayes hyperparameter estimation (type-II maximum likelihood)
# ---------------------------------------------------------------------------

def _fit_variance_prior(n, s2):
    """Type-II ML of (nu0, s0^2) from sample variances via s2/s0^2 ~ F(n-1, nu0)."""
    from scipy.optimize import minimize
    from scipy.stats import f as fdist

    n = np.asarray(n, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    ok = (n >= 2) & (s2 > 0)
    n, s2 = n[ok], s2[ok]
    if len(n) < 5:
        return None
    dfn = n - 1.0
    m = float(np.median(s2))

    def nll(p):
        lnu, ls02 = p
        nu0, s02 = np.exp(lnu), np.exp(ls02)
        return -np.sum(fdist.logpdf(s2 / s02, dfn, nu0) - ls02)

    best = None
    for nu_init in (4.0, 10.0, 30.0):
        res = minimize(nll, x0=[np.log(nu_init), np.log(m)], method="L-BFGS-B",
                       bounds=[(np.log(NU0_BOUNDS[0]), np.log(NU0_BOUNDS[1])),
                               (np.log(m * 1e-4), np.log(m * 1e4))])
        if best is None or res.fun < best.fun:
            best = res
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def _fit_mean_prior(ybar, var_i):
    """Type-II ML of (mu, tau^2) from ybar_i ~ N(mu, tau^2 + var_i)."""
    from scipy.optimize import minimize_scalar

    ybar = np.asarray(ybar, dtype=float)
    var_i = np.asarray(var_i, dtype=float)
    vy = float(ybar.var(ddof=1))
    if not vy > 0:
        return float(ybar.mean()), 1e-12

    def profile_nll(ltau2):
        tau2 = np.exp(ltau2)
        w = 1.0 / (tau2 + var_i)
        mu = np.sum(w * ybar) / np.sum(w)
        return 0.5 * np.sum(np.log(tau2 + var_i) + w * (ybar - mu) ** 2)

    lo, hi = np.log(vy * 1e-6), np.log(vy * 1e2)
    res = minimize_scalar(profile_nll, bounds=(lo, hi), method="bounded")
    tau2 = float(np.exp(res.x))
    w = 1.0 / (tau2 + var_i)
    mu = float(np.sum(w * ybar) / np.sum(w))
    return mu, tau2


def estimate_eb_prior(stats, min_patients=20):
    """Empirical-Bayes hyperparameters per (analyte, sex_key).

    stats: DataFrame with columns analyte, sex_key, n, ybar, s2 -- one row per
    patient-analyte history restricted to PopRI-normal values (n >= 1; s2 is
    NaN when n == 1). Histories with n >= 2 inform the variance prior; all
    histories inform the mean prior.
    """
    prior = {}
    for (analyte, sk), g in stats.groupby(["analyte", "sex_key"]):
        n = g["n"].values.astype(float)
        ybar = g["ybar"].values.astype(float)
        s2 = g["s2"].values.astype(float)
        if (n >= 2).sum() < min_patients:
            continue
        vp = _fit_variance_prior(n, s2)
        if vp is None:
            continue
        nu0, s02 = vp
        s2_filled = np.where(n >= 2, s2, 0.0)
        var_i = _shrunk_var(n, s2_filled, nu0, s02) / n
        mu, tau2 = _fit_mean_prior(ybar, var_i)
        prior[(analyte, sk)] = {"mu": mu, "tau2": tau2, "nu0": nu0, "s02": s02,
                                "n_patients": int(len(g)),
                                "median_ybar": float(np.median(ybar))}
    return prior


def _popri_lookup():
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from process.config import REFERENCE_INTERVALS
    return REFERENCE_INTERVALS


def _sex_key(sex_val):
    if isinstance(sex_val, str):
        return "F" if sex_val[:1].upper() == "F" else "M"
    return "F" if sex_val == 1 else "M"


def _pair_stats_from_values(records):
    """records: iterable of (analyte, sex_key, values_within_popri) -> stats frame."""
    rows = []
    for analyte, sk, vals in records:
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 1:
            continue
        rows.append((analyte, sk, len(vals), float(vals.mean()),
                     float(vals.var(ddof=1)) if len(vals) >= 2 else np.nan))
    return pd.DataFrame(rows, columns=["analyte", "sex_key", "n", "ybar", "s2"])


def _print_prior(prior):
    for (analyte, sk), p in sorted(prior.items()):
        print(f"    {analyte}/{sk}: n={p['n_patients']:,} mu={p['mu']:.4g} "
              f"tau={np.sqrt(p['tau2']):.3g} s0={np.sqrt(p['s02']):.3g} nu0={p['nu0']:.1f}")


def build_dev_prior(dev_dir=DEFAULT_DEV_DIR, source="combined",
                    train_sources=("mimiciv", "ehrshot"), min_patients=20):
    """Estimate the EB prior from NORMA's dev train split (transfer setting).

    Uses every unique (patient, analyte, time) history point of the train
    sequences (target excluded), keeping only values inside the PopRI.
    """
    from cohen import load_dev_sequences
    ri = _popri_lookup()
    train_seq, _, _ = load_dev_sequences(dev_dir, source=source)
    keep_src = set(train_sources) if train_sources else None
    pts = {}
    for seq in train_seq:
        if keep_src and seq["source"] not in keep_src:
            continue
        analyte = seq["test_name"]
        if analyte not in ri:
            continue
        sk = "F" if seq["sex"] == 1 else "M"
        low, high, _ = ri[analyte][sk]
        x = np.asarray(seq["x"], dtype=float)[:-1]
        t = np.asarray(seq["t"], dtype=float)[:-1]
        ok = np.isfinite(x) & (x >= low) & (x <= high)
        key = (f"{seq['source']}|{seq['pid']}", analyte, sk)
        d = pts.setdefault(key, {})
        for ti, xi in zip(t[ok], x[ok]):
            d.setdefault(float(ti), float(xi))
    records = [(analyte, sk, list(d.values())) for (_, analyte, sk), d in pts.items()]
    stats = _pair_stats_from_values(records)
    prior = estimate_eb_prior(stats, min_patients=min_patients)
    print(f"  Dev EB prior: {len(stats):,} patient-analyte histories -> "
          f"{len(prior)} (analyte, sex) priors")
    _print_prior(prior)
    return {"prior": prior,
            "meta": {"dev_dir": dev_dir, "source": source,
                     "train_sources": list(train_sources) if train_sources else None,
                     "min_patients": min_patients}}


def build_cohort_prior(pair_vals, min_patients=20):
    """Estimate the EB prior from the cohort's own PopRI-normal baseline."""
    stats = _pair_stats_from_values(
        (analyte, sk, vals) for (_, analyte, sk, vals, _, _) in pair_vals)
    prior = estimate_eb_prior(stats, min_patients=min_patients)
    print(f"  Cohort EB prior: {len(stats):,} patient-analyte histories -> "
          f"{len(prior)} (analyte, sex) priors")
    _print_prior(prior)
    return {"prior": prior, "meta": {"source": "cohort"}}


def build_popri_prior(z=1.96):
    """EB prior from the published population reference interval and nothing else.

    Per (analyte, sex), read straight out of process.config.REFERENCE_INTERVALS --
    the same table the `pop` method uses -- so the prior differs by analyte and
    sex exactly as the reference interval does:

        mu    = PopRI midpoint
        tau^2 = (PopRI half-width / z)^2

    That is the whole prior.  The within-person variance comes from the
    patient's own values (s^2), not from a prior on it, so nothing has to split
    the population spread into within- and between-person parts -- the quantity
    a reference interval cannot supply and a fitted artifact used to.
    """
    ri = _popri_lookup()
    prior = {}
    for analyte, by_sex in ri.items():
        for sk, entry in by_sex.items():
            lo, hi = float(entry[0]), float(entry[1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            prior[(analyte, sk)] = {"mu": (lo + hi) / 2.0,
                                    "tau2": ((hi - lo) / 2.0 / z) ** 2}
    print(f"  PopRI EB prior: population reference interval for {len(prior)} "
          f"(analyte, sex) keys; no fitted artifact")
    return {"prior": prior, "meta": {"source": "popri"}}


def load_or_build_prior(prior_source="cohort", prior_path=None, rebuild=False,
                        pair_vals=None, dev_dir=DEFAULT_DEV_DIR,
                        train_sources=("mimiciv", "ehrshot"), min_patients=20):
    import pickle
    if prior_source == "cohort":
        return build_cohort_prior(pair_vals, min_patients=min_patients)
    if prior_source == "popri":
        # The reference interval IS the prior: nothing is loaded or fitted.
        return build_popri_prior()
    prior_path = prior_path or DEFAULT_PRIOR_PATH
    if os.path.exists(prior_path) and not rebuild:
        print(f"  Loading EB prior from {prior_path}")
        with open(prior_path, "rb") as f:
            return pickle.load(f)
    json_path = (os.path.splitext(prior_path)[0] + ".json") if prior_path else DEFAULT_PRIOR_JSON
    if os.path.exists(json_path) and not rebuild:
        print(f"  Loading EB prior from {json_path}")
        return _read_prior_json(json_path)
    if not os.path.isdir(dev_dir):
        # Inside Clalit there are no dev cohorts: estimating the prior needs
        # combined_sequences_v2.pkl, NORMA's training data, which is not carried
        # in.  Note prior_source="popri" reaches here too -- it takes its
        # variance prior from the dev fit -- so this fires on the default path.
        raise SystemExit(
            f"\n  No EB prior at {prior_path} and no dev cohort at {dev_dir}.\n"
            f"  The prior is estimated on MIMIC/EHRSHOT and carried in, not fitted here.\n"
            f"  Copy model/logs/baselines/gaussian_eb_prior_dev.pkl (4.5 KB) to this repo,\n"
            f"  or drop the Gaussian arms:  04_refs.py ... --gaussian_methods")
    print("  Estimating EB prior on dev cohorts")
    artifact = build_dev_prior(dev_dir=dev_dir, train_sources=train_sources,
                               min_patients=min_patients)
    os.makedirs(os.path.dirname(prior_path), exist_ok=True)
    with open(prior_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved prior to {prior_path}")
    return artifact


# ---------------------------------------------------------------------------
# Cohort application
# ---------------------------------------------------------------------------

def _detect_cols(df):
    return {
        "pid": "patient_id" if "patient_id" in df.columns else "uniquepid",
        "analyte": "analyte" if "analyte" in df.columns else "lab_code",
        "value": "value" if "value" in df.columns else "labresult",
        "time": "timestamp" if "timestamp" in df.columns else "labresultoffset",
    }


def build_pair_values(split_df, pop_rows):
    """For each pop row (one per patient-analyte pair), collect PopRI-normal
    baseline values.

    Returns list of (row_dict, analyte, sex_key, normal_values, pop_low, pop_high).
    """
    c = _detect_cols(split_df)
    df = split_df[split_df["split"] == "baseline"].dropna(subset=[c["value"]])
    df = df.drop_duplicates(subset=[c["pid"], c["analyte"], c["time"]])
    wanted = set(zip(pop_rows["patient_id"].astype(str), pop_rows["analyte"]))
    keys = list(zip(df[c["pid"]].astype(str), df[c["analyte"]]))
    mask = np.fromiter((k in wanted for k in keys), dtype=bool, count=len(keys))
    df = df[mask]
    groups = {k: g[c["value"]].values.astype(float) for k, g in
              df.groupby([df[c["pid"]].astype(str), c["analyte"]])}

    out = []
    for row in pop_rows.to_dict("records"):
        key = (str(row["patient_id"]), row["analyte"])
        vals = groups.get(key)
        if vals is None:
            continue
        low = pd.to_numeric(row["ri_low"], errors="coerce")
        high = pd.to_numeric(row["ri_high"], errors="coerce")
        if np.isfinite(low) and np.isfinite(high):
            normal = vals[(vals >= low) & (vals <= high)]
        else:
            normal = np.array([])
        out.append((row, row["analyte"], _sex_key(row["sex"]), normal, low, high))
    return out


def _rows_for_pair(item, methods, z, prior):
    row, analyte, sk, normal, low, high = item
    shared = {k: row[k] for k in ("patient_id", "analyte", "sex", "age", "n_bl", "t_span")
              if k in row}
    n = len(normal)
    out = []
    for m in methods:
        rec = {**shared, "method": f"gaussian_{m}"}
        if not (np.isfinite(low) and np.isfinite(high)):
            rec.update(ri_mean=np.nan, ri_std=np.nan, ri_low=np.nan, ri_high=np.nan)
        elif n < MIN_N[m] or (m == "eb" and prior.get((analyte, sk)) is None):
            # Fallback: population interval, marked by NaN mean/SD
            rec.update(ri_mean=np.nan, ri_std=np.nan, ri_low=low, ri_high=high)
        else:
            if m == "mle":
                mu, sig = fit_mle(normal)
            elif m == "trunc":
                mu, sig = fit_trunc(normal, low, high)
            else:
                mu, sig = eb_posterior(normal, prior[(analyte, sk)], low, high)
            rec.update(ri_mean=mu, ri_std=sig, ri_low=mu - z * sig, ri_high=mu + z * sig)
        out.append(rec)
    return out


def compute_gaussian_refs(pair_vals, methods=METHODS, z=1.96, prior=None, n_jobs=None):
    from joblib import Parallel, delayed
    prior = prior or {}
    n_jobs = n_jobs or min(os.cpu_count() or 1, 16)
    n_chunks = max(1, min(n_jobs * 4, len(pair_vals) // 500 or 1))
    chunks = np.array_split(np.arange(len(pair_vals)), n_chunks)

    def _do(idx):
        rows = []
        for i in idx:
            rows.extend(_rows_for_pair(pair_vals[i], methods, z, prior))
        return rows

    backend = "loky" if n_jobs > 1 else "sequential"
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=5)(
        delayed(_do)(idx) for idx in chunks if len(idx))
    rows = [r for rs in results for r in rs]
    out = pd.DataFrame(rows)
    if len(out):
        n_fb = out["ri_mean"].isna() & out["ri_low"].notna()
        for m in methods:
            mm = out["method"] == f"gaussian_{m}"
            print(f"    gaussian_{m}: {mm.sum():,} rows, "
                  f"{(mm & n_fb).sum():,} PopRI fallbacks")
    return out


def augment_gaussian(ref_df, split_df, methods=METHODS, z=1.96, prior_source="cohort",
                     prior_path=None, rebuild_prior=False, dev_dir=None,
                     train_sources=("mimiciv", "ehrshot"), force=False):
    """Append gaussian_* rows to an existing ref_df (called from 04_compute_refs.py).

    Pairs are those already covered by the 'pop' method; the PopRI bounds are
    taken from those rows so the normal filter matches the pipeline exactly.
    """
    wanted = {f"gaussian_{m}" for m in methods}
    existing = set(ref_df["method"].unique()) & wanted
    if existing and not force:
        todo = sorted(wanted - existing)
        if not todo:
            print(f"  All of {sorted(wanted)} already present (--gaussian_force to recompute)")
            return ref_df
        methods = [m.replace("gaussian_", "") for m in todo]
        print(f"  {sorted(existing)} already present; computing {todo}")
    elif existing:
        print(f"  Recomputing {sorted(existing)}")
        ref_df = ref_df[~ref_df["method"].isin(wanted)]

    pop_rows = ref_df[ref_df["method"] == "pop"].drop_duplicates(
        subset=["patient_id", "analyte"])
    print(f"  Collecting PopRI-normal baseline values for {len(pop_rows):,} pairs")
    pair_vals = build_pair_values(split_df, pop_rows)
    n_norm = np.array([len(p[3]) for p in pair_vals])
    print(f"  {len(pair_vals):,} pairs with baseline data; PopRI-normal points per pair: "
          f"median {np.median(n_norm) if len(n_norm) else 0:.0f}, "
          f"{(n_norm == 0).sum():,} with none, {(n_norm == 1).sum():,} with one")

    prior = {}
    if "eb" in methods:
        artifact = load_or_build_prior(
            prior_source=prior_source, prior_path=prior_path, rebuild=rebuild_prior,
            pair_vals=pair_vals, dev_dir=dev_dir or DEFAULT_DEV_DIR,
            train_sources=train_sources)
        prior = artifact["prior"]
        by_key = {}
        for (_, analyte, sk, vals, _, _) in pair_vals:
            if len(vals):
                by_key.setdefault((analyte, sk), []).append(float(np.mean(vals)))
        for key, meds in sorted(by_key.items()):
            p = prior.get(key)
            if p is None:
                print(f"    WARNING {key[0]}/{key[1]}: no prior -- eb falls back to PopRI")
                continue
            if prior_source == "dev":
                ratio = float(np.median(meds)) / p["median_ybar"] if p["median_ybar"] else np.nan
                if not (0.67 <= ratio <= 1.5):
                    print(f"    WARNING {key[0]}/{key[1]}: cohort/prior median ratio "
                          f"{ratio:.2f} -- check units before trusting transfer")

    new_df = compute_gaussian_refs(pair_vals, methods=methods, z=z, prior=prior)
    if len(new_df) == 0:
        print("  No gaussian rows computed")
        return ref_df
    print(f"  Gaussian baselines: {len(new_df):,} rows")
    return pd.concat([ref_df, new_df], ignore_index=True)
