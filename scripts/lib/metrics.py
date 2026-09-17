"""Shared analysis helpers for the validation stages."""
import os as _os
import sys as _sys

import numpy as np
import pandas as pd
from scipy import stats

# process/config.py holds the published reference intervals; scripts/ on sys.path.
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
if _SCRIPTS_DIR not in _sys.path:
    _sys.path.append(_SCRIPTS_DIR)
from process.config import REFERENCE_INTERVALS, TEST_VOCAB  # noqa: E402


# Pop_RI estimator / sex coding

# The published Pop_RI for a lab and sex, shared by 04_refs (where it builds the `pop` rows),
# 06_sensitivity (synthetic histories) and 16_benchmark (baseline abnormality burden).

def sex_key(sex):
    """Normalise any sex encoding used in this codebase to "M" / "F"."""
    if isinstance(sex, str):
        s = sex.strip()
        if s in ("0", "1"):                      # numeric code stored as text
            return "F" if s == "1" else "M"
        return "M" if s[:1].upper() == "M" else "F"
    try:
        return "F" if int(sex) == 1 else "M"
    except (TypeError, ValueError):
        return "F"


def population_reference_range(lab_code, sex):
    """Population reference interval (low, high) for a lab and sex; (None, None)
    for an analyte without a published interval."""
    entry = REFERENCE_INTERVALS.get(lab_code)
    if entry is None:
        return None, None
    low, high, _ = entry[sex_key(sex)]
    return low, high


# Metrics for comparing reference-interval methods

def method_prefix(method):
    """Display method name -> its ri_low/ri_high column prefix."""
    return method.lower().replace("popri", "pop").replace("perri", "per") + "_ri"


def deviation_z(df, method):
    """|value - centre| / halfwidth for one method; NaN where bounds are missing."""
    stored = f"{method}_z"
    if stored in df.columns:
        return pd.to_numeric(df[stored], errors="coerce")
    pfx = method_prefix(method)
    lo = pd.to_numeric(df[f"{pfx}_low"], errors="coerce")
    hi = pd.to_numeric(df[f"{pfx}_high"], errors="coerce")
    val = pd.to_numeric(df["value"], errors="coerce")
    half = (hi - lo) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return (val - (lo + hi) / 2.0).abs() / half.where(half > 0)


def patient_level(df, method, event_col, cls_col=None):
    """One row per patient: max deviation z, any-abnormal flag, and the outcome."""
    out = pd.DataFrame({
        "patient_id": df["patient_id"].to_numpy(),
        "z": deviation_z(df, method).to_numpy(),
        "event": pd.to_numeric(df[event_col], errors="coerce").to_numpy(),
    })
    cls_col = cls_col or f"{method}_class"
    if cls_col in df.columns:
        cls = df[cls_col]
        out["abn"] = np.where(cls.notna(), (cls != 1).astype(float), np.nan)
    else:
        out["abn"] = np.nan
    out = out.dropna(subset=["event"])
    if out.empty:
        return out
    return (out.groupby("patient_id")
            .agg(z=("z", "max"), abn=("abn", "max"), event=("event", "first"))
            .dropna(subset=["event"]))


def auroc(score, label):
    """Rank-based AUROC (mid-ranks for ties); NaN if either class is empty."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(score) & np.isfinite(label)
    score, label = score[ok], label[ok]
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ppv_at_budget(score, label, budget):
    """PPV when the top `budget` fraction of patients is flagged."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(score) & np.isfinite(label)
    score, label = score[ok], label[ok]
    if len(score) < 20:
        return np.nan, np.nan
    k = max(1, int(round(budget * len(score))))
    thresh = np.sort(score)[::-1][k - 1]
    flagged = score >= thresh
    if flagged.sum() == 0:
        return np.nan, np.nan
    return float(label[flagged].mean()), float(flagged.mean())


def _clean(score, label):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(score) & np.isfinite(label)
    return score[ok], label[ok]


def operating_point(score, label, thresh):
    """Confusion-derived metrics when every patient with score >= thresh is flagged."""
    score, label = _clean(score, label)
    n = len(score)
    if n == 0 or not np.isfinite(thresh):
        return {k: np.nan for k in ("flag_rate", "sensitivity", "specificity",
                                    "ppv", "npv", "lift", "n", "n_events", "n_flagged")}
    flag = score >= thresh
    pos = label == 1
    tp = int((flag & pos).sum()); fp = int((flag & ~pos).sum())
    fn = int((~flag & pos).sum()); tn = int((~flag & ~pos).sum())
    base = pos.mean()
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return {
        "flag_rate": flag.mean(),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": ppv,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "lift": ppv / base if base > 0 and np.isfinite(ppv) else np.nan,
        "n": n, "n_events": int(pos.sum()), "n_flagged": int(flag.sum()),
    }


def threshold_for_rate(score, rate):
    """Score cut that flags the top `rate` fraction (ties at the cut all flagged)."""
    score = np.asarray(score, dtype=float)
    score = score[np.isfinite(score)]
    if len(score) == 0 or not (0 < rate <= 1):
        return np.nan
    k = max(1, int(round(rate * len(score))))
    return float(np.sort(score)[::-1][k - 1])


def threshold_for_sensitivity(score, label, target):
    """Lowest cut whose sensitivity is >= target (the least alerting way to reach it)."""
    score, label = _clean(score, label)
    pos = np.sort(score[label == 1])[::-1]
    if len(pos) == 0:
        return np.nan
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(k, 1) - 1])


def threshold_for_specificity(score, label, target):
    """Highest-sensitivity cut whose specificity is >= target."""
    score, label = _clean(score, label)
    neg = np.sort(score[label == 0])
    if len(neg) == 0:
        return np.nan
    k = int(np.ceil(target * len(neg)))          # number of negatives that must stay unflagged
    if k >= len(neg):
        return float(np.nextafter(neg[-1], np.inf))
    # cut strictly above the k-th smallest negative: flag = score >= cut
    return float(np.nextafter(neg[k - 1], np.inf)) if k > 0 else float(neg[0])


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values; NaN p-values pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def signed_z(df, method):
    """(value - centre) / halfwidth for one method: positive above the centre,
    negative below; NaN where bounds are missing. 07_classify stores it as
    `<method>_zs`; older files fall back to the bounds. |signed_z| == deviation_z."""
    stored = f"{method}_zs"
    if stored in df.columns:
        return pd.to_numeric(df[stored], errors="coerce")
    pfx = method_prefix(method)
    lo = pd.to_numeric(df[f"{pfx}_low"], errors="coerce")
    hi = pd.to_numeric(df[f"{pfx}_high"], errors="coerce")
    val = pd.to_numeric(df["value"], errors="coerce")
    half = (hi - lo) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return (val - (lo + hi) / 2.0) / half.where(half > 0)


def pop_side(df):
    """+1 where the value is nearer the UPPER Pop_RI bound, -1 where nearer the
    lower one (0 if equidistant / unknown). Stored by 07_classify as `pop_side`."""
    if "pop_side" in df.columns:
        return pd.to_numeric(df["pop_side"], errors="coerce").fillna(0).astype(int)
    val = pd.to_numeric(df["value"], errors="coerce")
    plo = pd.to_numeric(df["pop_ri_low"], errors="coerce")
    phi = pd.to_numeric(df["pop_ri_high"], errors="coerce")
    side = np.sign((val - plo) - (phi - val))          # >0: closer to the top
    return pd.Series(np.nan_to_num(side, nan=0.0).astype(int), index=df.index)


def deviates_toward_bound(df, method):
    """True where the value's deviation from `method`'s centre points at the NEARER
    Pop_RI bound (sign(zs) == pop_side) rather than back toward the population
    centre. DIRECTION ONLY -- how far the value is from the centre is the caller's
    business (a z threshold, or z > 1 for the method's own interval).
    """
    zs = signed_z(df, method).to_numpy(float)
    side = pop_side(df).to_numpy()
    return np.isfinite(zs) & (np.sign(zs) == side) & (side != 0)


# DeLong's test for two correlated ROC curves

# DeLong's test for two correlated ROC curves (Sun & Xu 2014 fast version).

def _midrank(x):
    """Midranks of x (ties share the average rank)."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_auc_cov(scores, labels):
    """AUCs and their covariance matrix for k methods scored on the same samples."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    finite = np.isfinite(labels) & np.isfinite(scores).all(axis=0)
    if not finite.all():
        # NaN labels would otherwise be counted as negatives, and NaN scores sort to the end and
        # take the highest midranks, inflating the AUC.
        scores, labels = scores[:, finite], labels[finite]
    pos = labels == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        k = scores.shape[0]
        return np.full(k, np.nan), np.full((k, k), np.nan)

    x = scores[:, pos]          # (k, m) positives
    y = scores[:, neg]          # (k, n) negatives
    k = scores.shape[0]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(x[r])
        ty[r] = _midrank(y[r])
        tz[r] = _midrank(np.concatenate([x[r], y[r]]))

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n                      # (k, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m                # (k, n)
    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, cov


def delong_test(scores_a, scores_b, labels):
    """Two-sided DeLong test of AUC(a) - AUC(b) on paired samples."""
    aucs, cov = delong_auc_cov(np.vstack([scores_a, scores_b]), labels)
    if not np.all(np.isfinite(aucs)):
        return dict(auc_a=np.nan, auc_b=np.nan, delta=np.nan, se=np.nan,
                    z=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not np.isfinite(var) or var <= 0:
        # np.cov returns NaN when a class has a single member (ddof=1), and the variance is
        # genuinely 0 only when the two scores are identical.
        p = 1.0 if delta == 0 else np.nan
        return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=delta,
                    se=np.nan, z=np.nan, p=p, ci_low=np.nan, ci_high=np.nan)
    se = float(np.sqrt(var))
    z = delta / se
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=delta, se=se,
                z=float(z), p=float(2 * stats.norm.sf(abs(z))),
                ci_low=delta - 1.96 * se, ci_high=delta + 1.96 * se)


# Flag rules shared by 11_lead_time and 17_outcomes

# Flag rules shared by 11_lead_time and 17_outcomes: how a method's deviation score becomes a
# yes/no flag on a comparable footing.

MIN_STRATUM = 20       # age band x sex strata below this use the pooled cutoff
TO_HOURS = {"hours": 1.0, "minutes": 1.0 / 60.0, "days": 24.0}


def hours(values, unit):
    return pd.to_numeric(values, errors="coerce") * TO_HOURS[unit]


def jitter(z, seed=0):
    """Break exact ties deterministically.  Lab values are reported at coarse
    resolution, so z (especially Pop_RI's) has many identical values; a quantile
    cutoff with `>=` then flags every tied value and overshoots the target.  The
    jitter is far below the data resolution and only orders ties."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    scale = (np.nanstd(z[ok]) if ok.sum() > 1 else 1.0) or 1.0
    rng = np.random.default_rng(seed)
    out = z.copy()
    out[ok] = z[ok] + rng.uniform(0, 1e-6 * scale, ok.sum())
    return out


def age_band(age):
    """Cohen's 10-year age resolution."""
    return np.floor(pd.to_numeric(age, errors="coerce") / 10.0) * 10.0


def cut_at_sensitivity(score, y, target):
    """Lowest threshold reaching `target` sensitivity.  -inf scores (wrong-direction
    deviations) stay in the positive count but can never be flagged, so sensitivity
    is measured against ALL positives."""
    positives = np.asarray(score, float)[np.asarray(y) == 1]
    positives = positives[~np.isnan(positives)]
    if len(positives) < 5:
        return np.nan
    k = int(round(target * len(positives)))
    finite = np.sort(positives[np.isfinite(positives)])[::-1]
    if k <= 0 or not len(finite):
        return np.inf
    return float(finite[min(k, len(finite)) - 1])


def strata_of(df):
    """{(age band, sex): row positions}: Cohen's 10-year age resolution x sex, or one
    stratum when the cohort carries no demographics."""
    if not {"age", "sex"} <= set(df.columns):
        return {("all", "all"): np.arange(len(df))}
    band = age_band(df["age"]).to_numpy()
    sex = pd.to_numeric(df["sex"], errors="coerce").to_numpy()
    return pd.DataFrame({"band": band, "sex": sex}).groupby(["band", "sex"]).indices


def matched_sensitivity_flags(z, y, strata, target):
    """Flag = z >= the cutoff reaching `target` sensitivity within each stratum; thin
    strata (< MIN_STRATUM) or strata without enough positives use the pooled cutoff.
    Returns (flag, the strata that got a cutoff)."""
    fallback = cut_at_sensitivity(z, y, target)
    flag = np.zeros(len(z), bool)
    thresholded = set()
    for key, idx in strata.items():
        idx = np.asarray(idx)
        threshold = np.nan
        if len(idx) >= MIN_STRATUM:
            threshold = cut_at_sensitivity(z[idx], y[idx], target)
        if not np.isfinite(threshold):
            threshold = fallback
        if not np.isfinite(threshold):
            continue
        flag[idx] = z[idx] >= threshold
        thresholded.add(key)
    return flag, thresholded


def threshold_at_rate(z, rate):
    """Score threshold flagging a `rate` fraction of ALL measurements.  -inf marks
    measurements that may never be flagged (wrong-direction exits); they count in
    the denominator, so past the eligible fraction every eligible one is flagged."""
    z = np.asarray(z, float)
    known = ~np.isnan(z)
    if known.sum() < 50 or not (0 < rate < 1):
        return np.nan
    eligible = np.isfinite(z)
    k = int(round(rate * known.sum()))
    if k <= 0:
        return np.inf
    if k >= eligible.sum():
        return float(z[eligible].min()) if eligible.any() else np.nan
    return float(np.sort(z[eligible])[::-1][k - 1])


# Which index measurement represents a stay (exposure)

WINDOW_HOURS = 48
EXPOSURES = ("first", "window_worst", "window_any")


def stay_col(df):
    return "patientunitstayid" if "patientunitstayid" in df.columns else "patient_id"


def hours_from_admit(df, time_unit=None):
    """Hours since admission for every measurement (eICU: unit admission; INSPIRE: hospital)."""
    if "days_from_admit" in df.columns:
        return pd.to_numeric(df["days_from_admit"], errors="coerce") * 24.0
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    return ts * 24.0 if time_unit == "days" else ts / 60.0


def to_hours(series, unit):
    """Outcome offsets -> hours. unit: 'minutes' | 'hours' | 'days'."""
    v = pd.to_numeric(series, errors="coerce")
    return {"minutes": v / 60.0, "hours": v, "days": v * 24.0}[unit]


def mark_exposures(df, time_unit=None, window_hours=WINDOW_HOURS):
    """Add t_hours, t0_hours, exp_first, exp_window, exp_window_worst columns."""
    df = df.copy()
    df["t_hours"] = hours_from_admit(df, time_unit)
    sc = stay_col(df)
    d = df[df["t_hours"].notna()].sort_values("t_hours", kind="mergesort")
    for c in ("exp_first", "exp_window", "exp_window_worst"):
        df[c] = False
    df["t0_hours"] = np.nan
    if d.empty:
        return df
    grp = d.groupby([sc, "analyte"], sort=False)
    df.loc[grp.head(1).index, "exp_first"] = True
    t0 = grp["t_hours"].transform("min")
    df.loc[d.index, "t0_hours"] = t0
    inwin = d[d["t_hours"] <= t0 + window_hours]
    df.loc[inwin.index, "exp_window"] = True
    lo, hi = pd.to_numeric(inwin["pop_ri_low"], errors="coerce"), pd.to_numeric(inwin["pop_ri_high"], errors="coerce")
    dev = (pd.to_numeric(inwin["value"], errors="coerce") - (lo + hi) / 2).abs() / ((hi - lo) / 2)
    dev = dev.fillna(-1.0)                       # rows without Pop_RI bounds never win
    worst = dev.groupby([inwin[sc], inwin["analyte"]]).idxmax()
    df.loc[worst.dropna().to_numpy(), "exp_window_worst"] = True
    return df


def exposure_rows(df, exposure, window_hours=WINDOW_HOURS):
    """(rows, landmark_hours) for one exposure definition; markers computed if absent."""
    if "exp_first" not in df.columns:
        raise ValueError("classification table has no exposure markers; run 07_classify again")
    if exposure == "first":
        rows = df[df["exp_first"]]
        return rows, rows["t0_hours"]
    if exposure == "window_worst":
        rows = df[df["exp_window_worst"]]
        return rows, rows["t0_hours"] + window_hours
    if exposure == "window_any":
        rows = df[df["exp_window"]]
        return rows, rows["t0_hours"] + window_hours
    raise ValueError(exposure)


# Forecasting targets shared by 04_refs (norma step) and 05_forecasting

# Forecasting targets shared by 04_refs.py (norma step) and 05_forecasting.py.

STATE_NAMES = {0: 'low', 1: 'normal', 2: 'high'}
TARGET_KEYS = ['patient_id', 'analyte', 'target_idx']
BASE_COLS = TARGET_KEYS + ['cid', 'sex', 'age', 'n_hist', 't_next', 'horizon_days',
                           'x_next', 's_next', 's_last']
TIME_SCALE = {'minutes': 1 / 1440.0, 'hours': 1 / 24.0, 'days': 1.0}


def sex_str(v):
    """Any sex encoding used in this codebase -> 'M' / 'F' (0 = male, 1 = female)."""
    if isinstance(v, str):
        return 'M' if v[:1].upper() == 'M' else 'F'
    try:
        return 'F' if int(v) == 1 else 'M'
    except (TypeError, ValueError):
        return 'F'


def subsample_patients(df, max_patients, seed=42, pid_col='patient_id'):
    """Random patient subset, same draw as 04_refs.py --max_patients (sorted ids,
    RandomState(seed)) so NORMA predictions and reference intervals cover the same
    patients."""
    if not max_patients:
        return df
    pids = np.sort(df[pid_col].unique())
    if len(pids) <= max_patients:
        return df
    keep = np.random.RandomState(seed).choice(pids, size=max_patients, replace=False)
    out = df[df[pid_col].isin(set(keep))].copy()
    print(f'  --max_patients: {len(pids):,} -> {max_patients:,} patients, {len(out):,} rows')
    return out


def build_pairs(df, time_unit, target='first', max_hist=128, exclude=(), covariates=()):
    """One record per forecasting target: history arrays + target, times in days
    from the first history measurement.
    """
    want_co = 'co' in covariates
    cols = ['patient_id', 'analyte', 'timestamp', 'value', 'sex', 'age', 'split']
    if 'setting' in covariates:
        if 'setting' not in df.columns:
            raise ValueError("covariates include 'setting' but df has no setting column "
                             "(01_process must write it; see process/covariates.py)")
        cols.append('setting')
    df = df[cols].dropna(subset=['value'])
    df = df[df['analyte'].isin(TEST_VOCAB)]
    df = df.assign(t=df['timestamp'].astype(float) * TIME_SCALE[time_unit or 'days'])

    panel = draw_id = None
    if want_co:
        # Draw table over ALL vocab analytes (pre-exclusion), keyed (patient, time).
        key_df = df[['patient_id', 't']]
        draw_id, _ = pd.factorize(pd.MultiIndex.from_frame(key_df), sort=False)
        sx = df['sex'].map(sex_str)
        lo = np.array([REFERENCE_INTERVALS[a][s][0] for a, s in zip(df['analyte'], sx)], dtype=float)
        hi = np.array([REFERENCE_INTERVALS[a][s][1] for a, s in zip(df['analyte'], sx)], dtype=float)
        vnorm = np.clip((df['value'].to_numpy(dtype=float) - lo) / (hi - lo), -5.0, 5.0)
        panel = np.full((draw_id.max() + 1, len(TEST_VOCAB)), np.nan, dtype=np.float16)
        panel[draw_id, df['analyte'].map(TEST_VOCAB).to_numpy()] = vnorm.astype(np.float16)
        df = df.assign(draw_id=draw_id)

    df = df[~df['analyte'].isin(set(exclude))]
    df = df.sort_values(['patient_id', 'analyte', 't'], kind='stable').reset_index(drop=True)
    # one history value per timestamp (first kept), as the baselines step of 04_refs.py does
    dup = (df['split'] == 'baseline') & df.duplicated(subset=['patient_id', 'analyte', 'split', 't'])
    if dup.any():
        print(f'  history de-duplicated on timestamp: {int(dup.sum()):,} rows dropped')
        df = df[~dup].reset_index(drop=True)

    # population-defined state of every measurement (0 low / 1 normal / 2 high)
    sx = df['sex'].map(sex_str)
    lo = np.array([REFERENCE_INTERVALS[a][s][0] for a, s in zip(df['analyte'], sx)], dtype=float)
    hi = np.array([REFERENCE_INTERVALS[a][s][1] for a, s in zip(df['analyte'], sx)], dtype=float)
    v = df['value'].to_numpy(dtype=float)
    state = np.where(v < lo, 0, np.where(v > hi, 2, 1))

    key = df['patient_id'].astype(str) + '\x00' + df['analyte'].astype(str)
    starts = np.flatnonzero(np.r_[True, key.to_numpy()[1:] != key.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(df)]
    is_bl = (df['split'].to_numpy() == 'baseline')
    t = df['t'].to_numpy(dtype=float)
    pid = df['patient_id'].to_numpy()
    analyte = df['analyte'].to_numpy()
    sex = df['sex'].to_numpy()
    age = df['age'].to_numpy(dtype=float)
    setting = df['setting'].to_numpy(dtype=np.int64) if 'setting' in covariates else None
    drawix = df['draw_id'].to_numpy(dtype=np.int64) if want_co else None

    recs = []
    for a, b in zip(starts, ends):
        bl = np.flatnonzero(is_bl[a:b]) + a
        idx = np.flatnonzero(~is_bl[a:b]) + a
        if len(bl) == 0 or len(idx) == 0:
            continue
        targets = idx[:1] if target == 'first' else idx
        for k, j in enumerate(targets):
            hist = np.arange(a, j) if target == 'all' else bl
            hist = hist[-max_hist:]
            t0 = t[hist[0]]
            r = {
                'patient_id': pid[a], 'analyte': analyte[a], 'target_idx': k,
                'cid': TEST_VOCAB[analyte[a]],
                'sex': 0 if sex_str(sex[a]) == 'M' else 1, 'sex_raw': sex[a],
                'age': 65.0 if np.isnan(age[a]) else float(age[a]), 'age_raw': age[a],
                'x_h': v[hist].astype(np.float32), 's_h': state[hist].astype(np.int64),
                't_h': (t[hist] - t0).astype(np.float32),
                'n_hist': len(hist), 't_next': float(t[j] - t0), 'horizon_days': float(t[j] - t[hist[-1]]),
                'x_next': float(v[j]), 's_next': int(state[j]), 's_last': int(state[hist[-1]]),
            }
            if 'age' in covariates:
                # `age` is the pair's static age column; the per-draw age drifts with the
                # anchored time axis (exact for cohorts whose age is at first draw, off by <= the
                # stay length for admission-anchored ages).
                r['age_h'] = (r['age'] + r['t_h'] / 365.25).astype(np.float32)
                r['age_next'] = float(r['age'] + r['t_next'] / 365.25)
            if setting is not None:
                r['setting_h'] = setting[hist]
                r['setting_next'] = int(setting[j])
            if drawix is not None:
                r['draw_idx'] = drawix[hist]
            recs.append(r)
    return (recs, panel) if want_co else recs


def pairs_frame(recs):
    """The scalar part of the records as a DataFrame (one row per target)."""
    return pd.DataFrame([{k: r[k] for k in BASE_COLS} for r in recs])


def describe(recs, target):
    print(f'  {len(recs):,} forecasting targets ({target}); '
          f'median history {int(np.median([r["n_hist"] for r in recs]))}, '
          f'median horizon {np.median([r["horizon_days"] for r in recs]):.2f} days')
