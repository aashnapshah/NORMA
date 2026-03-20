"""
Shared utility functions for the validation pipeline.
"""
import pandas as pd
import numpy as np
import torch
import sys
import os

from config import (
    EICU_DATA_DIR, CACHE_DIR, DATA_DIR, MODEL_LOG_DIR,
    EICU_LAB_MAP, REVERSE_LAB_MAP, ALL_EICU_NAMES,
    DISEASES, OUTCOMES, DEFAULT_TIME_WINDOWS, EXCLUDE_LAB_CODES,
)

# Add model path for NORMA imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Data loading (raw eICU)
# ============================================================

def load_patient():
    """Load patient table, caching as pickle."""
    cache_path = os.path.join(CACHE_DIR, "patient.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached patient data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading patient.csv (first time, will cache)...")
    patient = pd.read_csv(os.path.join(EICU_DATA_DIR, "patient.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    patient.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return patient


def load_labs(min_tests=3):
    """Load lab table, filter to our 34 labs and patients with >= min_tests per lab."""
    cache_path = os.path.join(CACHE_DIR, "labs_filtered.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached lab data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading lab.csv (first time, will cache)...")
    lab = pd.read_csv(os.path.join(EICU_DATA_DIR, "lab.csv"))

    # Track attrition through processing steps
    processing_attrition = []
    n_raw = len(lab)
    n_raw_patients = lab["patientunitstayid"].nunique()
    processing_attrition.append({
        "Step": "Raw eICU lab table",
        "N Measurements": n_raw,
        "N Stays": n_raw_patients,
    })

    lab = lab[lab["labname"].isin(ALL_EICU_NAMES)].copy()
    processing_attrition.append({
        "Step": "Filter to 34 target lab codes",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    lab["lab_code"] = lab["labname"].map(REVERSE_LAB_MAP)
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab["days_from_admit"] = lab["labresultoffset"] / (60 * 24)
    lab = lab.sort_values(["patientunitstayid", "lab_code", "labresultoffset"])
    lab["days_since_last"] = lab.groupby(["patientunitstayid", "lab_code"])["days_from_admit"].diff()

    # Remove outliers (1.5*IQR per lab)
    outlier_log = []
    cleaned = []
    for lab_code, group in lab.groupby("lab_code"):
        vals = group["labresult"]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (vals >= lower) & (vals <= upper)
        n_removed = (~mask).sum()
        outlier_log.append({
            "lab_code": lab_code, "n_before": len(group),
            "n_removed": n_removed,
            "pct_removed": round(n_removed / len(group) * 100, 2) if len(group) > 0 else 0,
            "iqr_lower": lower, "iqr_upper": upper,
        })
        cleaned.append(group[mask])
    lab = pd.concat(cleaned, ignore_index=True)
    outlier_df = pd.DataFrame(outlier_log)
    print("Outlier removal per lab:")
    print(outlier_df.to_string(index=False))
    processing_attrition.append({
        "Step": "Remove IQR outliers",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    # Keep only patients with enough measurements
    counts = lab.groupby(["patientunitstayid", "lab_code"]).size().reset_index(name="n")
    valid = counts[counts["n"] >= min_tests][["patientunitstayid", "lab_code"]]
    before = len(lab)
    lab = lab.merge(valid, on=["patientunitstayid", "lab_code"], how="inner")
    print(f"\nMin tests filter: {before} -> {len(lab)} rows (min {min_tests} per patient-lab)")
    processing_attrition.append({
        "Step": f"Require ≥{min_tests} tests per stay-lab",
        "N Measurements": len(lab),
        "N Stays": lab["patientunitstayid"].nunique(),
    })

    os.makedirs(CACHE_DIR, exist_ok=True)
    lab.to_pickle(cache_path)
    outlier_df.to_pickle(os.path.join(CACHE_DIR, "outlier_log.pkl"))
    # Save processing attrition for downstream scripts
    pd.DataFrame(processing_attrition).to_csv(
        os.path.join(CACHE_DIR, "processing_attrition.csv"), index=False
    )
    print(f"Cached to {cache_path}")
    return lab


def load_diagnosis():
    """Load diagnosis table, caching as pickle."""
    cache_path = os.path.join(CACHE_DIR, "diagnosis.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached diagnosis data from {cache_path}")
        return pd.read_pickle(cache_path)
    print("Reading diagnosis.csv (first time, will cache)...")
    diagnosis = pd.read_csv(os.path.join(EICU_DATA_DIR, "diagnosis.csv"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    diagnosis.to_pickle(cache_path)
    print(f"Cached to {cache_path}")
    return diagnosis


def merge_labs_patients(lab, patient):
    """Merge lab data with patient demographics."""
    patient_cols = patient[["patientunitstayid", "uniquepid", "gender", "age",
                            "ethnicity", "hospitaldischargeyear"]].copy()
    patient_cols["age"] = pd.to_numeric(patient_cols["age"], errors="coerce")
    patient_cols.rename(columns={"hospitaldischargeyear": "year"}, inplace=True)
    return lab.merge(patient_cols, on="patientunitstayid", how="left")


def load_processed():
    """Load processed eICU data from data/ directory."""
    path = os.path.join(DATA_DIR, "eicu_processed.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 01_process_eicu.py first.")
    print(f"Loading processed data from {path}")
    return pd.read_pickle(path)


def load_diagnosis_processed():
    """Load processed diagnosis data from data/ directory."""
    path = os.path.join(DATA_DIR, "eicu_diagnosis.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 01_process_eicu.py first.")
    print(f"Loading diagnosis data from {path}")
    return pd.read_pickle(path)


# ============================================================
# Summary statistics
# ============================================================

def patient_demographics(merged):
    """Print and return age/sex distributions for unique patients."""
    patients = merged.drop_duplicates(subset="uniquepid")
    age_stats = patients["age"].describe()
    sex_counts = patients["gender"].value_counts()
    sex_pct = patients["gender"].value_counts(normalize=True) * 100
    print(f"=== Unique Patients: {len(patients)} ===\n")
    print("--- Age Distribution ---")
    print(age_stats.to_string())
    print(f"\n--- Sex Distribution ---")
    for sex in sex_counts.index:
        print(f"  {sex}: {sex_counts[sex]} ({sex_pct[sex]:.1f}%)")
    return patients


def lab_summary_stats(merged):
    """Summary statistics for each lab code."""
    results = []
    for lab_code in sorted(c for c in merged["lab_code"].unique() if c not in EXCLUDE_LAB_CODES):
        df = merged[merged["lab_code"] == lab_code].copy()
        vals = df["labresult"].dropna()
        meas_per_person = df.groupby("uniquepid").size()
        patient_spans = df.groupby("uniquepid")["labresultoffset"].agg(["min", "max"])
        patient_spans["span_hours"] = (patient_spans["max"] - patient_spans["min"]) / 60
        spans = patient_spans.loc[meas_per_person[meas_per_person > 1].index, "span_hours"]
        results.append({
            "lab_code": lab_code,
            "eicu_names": ", ".join(EICU_LAB_MAP[lab_code]),
            "n_measurements": len(vals), "mean": vals.mean(), "std": vals.std(),
            "min": vals.min(), "p25": vals.quantile(0.25), "median": vals.median(),
            "p75": vals.quantile(0.75), "max": vals.max(),
            "n_patients": df["uniquepid"].nunique(),
            "meas_per_person_mean": meas_per_person.mean(),
            "meas_per_person_median": meas_per_person.median(),
            "shortest_span_hours": spans.min() if len(spans) > 0 else np.nan,
            "avg_span_hours": spans.mean() if len(spans) > 0 else np.nan,
            "longest_span_hours": spans.max() if len(spans) > 0 else np.nan,
        })
    return pd.DataFrame(results)


# ============================================================
# Baseline/index splitting
# ============================================================

def split_patients(merged, baseline_pct=0.75):
    """Split each patient-lab group into baseline and index measurements.

    Args:
        merged: DataFrame with lab data
        baseline_pct: fraction of measurements for baseline (0-1)

    Returns:
        DataFrame with added column "split": "baseline" or "index"
    """
    result = merged.sort_values(["uniquepid", "lab_code", "labresultoffset"]).copy()
    grp = result.groupby(["uniquepid", "lab_code"])
    rank = grp.cumcount()
    size = grp["labresultoffset"].transform("size")
    cutoff = (size * baseline_pct).astype(int).clip(lower=1)
    result["split"] = np.where(rank < cutoff, "baseline", "index")

    n_baseline = (result["split"] == "baseline").sum()
    n_index = (result["split"] == "index").sum()
    print(f"Split complete ({baseline_pct:.0%} baseline): {n_baseline} baseline, {n_index} index measurements")
    return result


def sample_patients(merged, n_per_code=40, seed=42):
    """Sample n_per_code unique patients per lab code."""
    rng = np.random.RandomState(seed)
    sampled = []
    for lab_code, group in merged.groupby("lab_code"):
        pids = group["uniquepid"].unique()
        chosen = rng.choice(pids, size=min(n_per_code, len(pids)), replace=False)
        sampled.append(group[group["uniquepid"].isin(chosen)])
    result = pd.concat(sampled, ignore_index=True)
    print(f"Sampled {result['uniquepid'].nunique()} unique patients across "
          f"{result['lab_code'].nunique()} lab codes ({len(result)} rows)")
    return result


# ============================================================
# Reference range computation
# ============================================================

def population_reference_range(lab_code, sex):
    """Population reference interval (low, high) for a lab and sex."""
    from process.config import REFERENCE_INTERVALS
    sex_key = "M" if sex == "Male" else "F"
    if lab_code in REFERENCE_INTERVALS:
        low, high, _ = REFERENCE_INTERVALS[lab_code][sex_key]
        return low, high
    return None, None


def gmm_setpoint(values, max_components=3):
    """GMM setpoint from a series of values. Returns (mean, std)."""
    import warnings
    from sklearn.mixture import GaussianMixture
    vals = values.dropna().values
    if len(vals) < 2:
        return (np.mean(vals), np.std(vals)) if len(vals) > 0 else (np.nan, np.nan)

    n = len(vals)
    n_unique = len(np.unique(vals))
    mean_1 = np.mean(vals)
    var_1 = np.var(vals)
    if var_1 == 0:
        return mean_1, 0.0

    log_lik_1 = -0.5 * n * (np.log(2 * np.pi * var_1) + 1)
    best_aic = 2 * 2 - 2 * log_lik_1
    best_model = None
    thresholds = {2: 0.70, 3: 0.45}

    for nc in range(2, max_components + 1):
        if n < nc or n_unique < nc:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=nc, covariance_type="full",
                max_iter=300, reg_covar=0.001, random_state=0,
            ).fit(vals.reshape(-1, 1))
        aic = gmm.aic(vals.reshape(-1, 1))
        if aic < best_aic and gmm.converged_:
            best_aic = aic
            best_model = gmm

    if best_model is not None:
        weights = best_model.weights_
        dom_idx = np.argmax(weights)
        if weights[dom_idx] > thresholds[best_model.n_components]:
            return best_model.means_[dom_idx, 0], np.sqrt(best_model.covariances_[dom_idx, 0, 0])

    return mean_1, np.sqrt(var_1)


def classify_value(value, low, high):
    """Classify a lab value: 0=low, 1=normal, 2=high."""
    if low is None or high is None:
        return 1
    if value < low:
        return 0
    elif value > high:
        return 2
    return 1


def load_norma_model(run_id, device="cpu"):
    """Load a NORMA model from a run_id checkpoint."""
    from model.utils import load_checkpoint, create_model
    from process.config import TEST_VOCAB
    checkpoint, hparams = load_checkpoint(MODEL_LOG_DIR, run_id, best=True, device=device)
    model = create_model(hparams, len(TEST_VOCAB), checkpoint=checkpoint)
    model.to(device).eval()
    return model, hparams


def norma_reference_range(model, hparams, baseline_vals, baseline_offsets,
                          lab_code, sex, age, device="cpu"):
    """NORMA personalized reference range from baseline. Returns (low, high)."""
    from process.config import REFERENCE_INTERVALS, TEST_VOCAB

    values = baseline_vals.dropna().values
    offsets = baseline_offsets.values
    if len(values) == 0:
        return None, None

    cid = TEST_VOCAB.get(lab_code)
    if cid is None:
        return None, None

    pop_low, pop_high = population_reference_range(lab_code, sex)
    nstates = getattr(hparams, 'nstates', getattr(hparams, 'num_states', 2))
    raw_states = np.array([classify_value(v, pop_low, pop_high) for v in values])
    if nstates == 2:
        states = (raw_states == 1).astype(np.int64)
    else:
        states = raw_states

    sex_idx = 0 if sex == "Male" else 1
    age_val = float(age if not np.isnan(age) else 65)

    max_len = getattr(hparams, 'max_seq_len', 128)
    if len(values) > max_len:
        values, offsets, states = values[-max_len:], offsets[-max_len:], states[-max_len:]

    seq_len = len(values)
    batch = {
        "x_h": torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
        "t_h": torch.tensor(offsets, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
        "s_h": torch.tensor(states, dtype=torch.long).unsqueeze(0),
        "sex": torch.tensor([sex_idx], dtype=torch.long),
        "age": torch.tensor([age_val], dtype=torch.float32),
        "cid": torch.tensor([cid], dtype=torch.long),
        "s_next": torch.tensor([1], dtype=torch.long),
        "t_next": torch.tensor([offsets[-1] + 60], dtype=torch.float32).unsqueeze(-1),
        "pad_mask": torch.zeros(1, seq_len, dtype=torch.bool),
    }
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        output = model(
            batch["x_h"], batch["s_h"], batch["t_h"], batch["sex"],
            batch["age"], batch["cid"], batch["s_next"], batch["t_next"], batch["pad_mask"]
        )

    is_quantile = hasattr(model, 'output_mode') and model.output_mode == 'quantile'
    if is_quantile:
        q = output.cpu().numpy()[0]
        return float(q[0]), float(q[4])
    else:
        mu, log_var = output
        sigma = np.exp(0.5 * log_var.item())
        return mu.item() - 1.96 * sigma, mu.item() + 1.96 * sigma


# ============================================================
# Disease & mortality helpers
# ============================================================

def get_disease_offsets(diagnosis):
    """For each disease, get earliest diagnosis offset per patientunitstayid.

    Returns dict: disease_name -> Series(patientunitstayid -> offset)
    """
    offsets = {}
    for name, cfg in DISEASES.items():
        dx_mask = diagnosis["diagnosisstring"].str.contains(cfg["dx_pattern"], case=False, na=False)
        icd_mask = diagnosis["icd9code"].str.contains(cfg["icd_pattern"], na=False)
        earliest = diagnosis.loc[dx_mask | icd_mask].groupby("patientunitstayid")["diagnosisoffset"].min()
        offsets[name] = earliest
        print(f"  {name:20s}: {len(earliest):>6d} stays")
    return offsets


def get_mortality():
    """Get mortality info from patient table."""
    patient = load_patient()
    mort = patient[["patientunitstayid", "unitdischargestatus",
                     "hospitaldischargestatus", "unitdischargeoffset",
                     "hospitaldischargeoffset"]].copy()
    mort["died_in_unit"] = mort["unitdischargestatus"] == "Expired"
    mort["died_in_hospital"] = mort["hospitaldischargestatus"] == "Expired"
    mort["death_offset"] = np.where(
        mort["died_in_unit"],
        mort["unitdischargeoffset"],
        np.where(mort["died_in_hospital"], mort["hospitaldischargeoffset"], np.nan),
    )
    return mort[["patientunitstayid", "died_in_hospital", "died_in_unit",
                 "death_offset", "unitdischargeoffset", "hospitaldischargeoffset"]]


def attach_outcomes(df):
    """Attach disease diagnoses, mortality, and derived outcome columns.

    Adds has_{disease}, {disease}_offset columns for each disease in DISEASES,
    plus mortality columns (died_in_hospital, death_offset, etc.),
    plus prolonged_los, pop_abnormal columns.
    Skips if columns already exist.
    """
    if "died_in_hospital" in df.columns:
        print("Outcome columns already present, skipping.")
        return df

    df = df.copy()

    # Disease diagnoses
    print("Attaching disease diagnoses...")
    diagnosis = load_diagnosis_processed()
    disease_offsets = get_disease_offsets(diagnosis)
    for name, earliest in disease_offsets.items():
        df[f"has_{name}"] = df["patientunitstayid"].isin(earliest.index)
        df[f"{name}_offset"] = df["patientunitstayid"].map(earliest)

    # Mortality
    print("Attaching mortality...")
    mort = get_mortality()
    df = df.merge(mort, on="patientunitstayid", how="left")

    # Prolonged LOS (>7 days in ICU)
    print("Computing prolonged LOS...")
    df["prolonged_los"] = df["unitdischargeoffset"] > (7 * 24 * 60)
    n_prolonged = df.drop_duplicates("patientunitstayid")["prolonged_los"].sum()
    print(f"  {n_prolonged} stays with LOS > 7 days")

    # Pop-abnormal: whether any index measurement leaves population ref interval
    # Computed per (patient, analyte) and broadcast back to all rows
    pid_col = "patient_id" if "patient_id" in df.columns else "uniquepid"
    analyte_col = "analyte" if "analyte" in df.columns else "lab_code"
    time_col = "timestamp" if "timestamp" in df.columns else "labresultoffset"

    print("Computing pop-abnormal outcome...")
    if "PopRI_class" in df.columns:
        df["_is_pop_abn"] = (df["PopRI_class"] != 1).astype(int)
        pop_abn = df.groupby([pid_col, analyte_col]).agg(
            has_pop_abnormal=("_is_pop_abn", "max"),
            pop_abnormal_offset=(time_col, lambda s: (
                s[df.loc[s.index, "_is_pop_abn"] == 1].min()
                if (df.loc[s.index, "_is_pop_abn"] == 1).any()
                else np.nan
            )),
            last_index_offset=(time_col, "max"),
        ).reset_index()
        df = df.drop(columns=["_is_pop_abn"])
        for col in ["has_pop_abnormal", "pop_abnormal_offset", "last_index_offset"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(pop_abn, on=[pid_col, analyte_col], how="left")
        n_pop_abn = pop_abn["has_pop_abnormal"].sum()
        print(f"  {n_pop_abn} / {len(pop_abn)} patient-lab pairs with pop-abnormal index values")
    else:
        print("  PopRI_class column not found, skipping pop_abnormal outcome")

    return df


# ============================================================
# Cox model helpers
# ============================================================

def method_display(method):
    """Map method name to display label and color using scripts/plots.py scheme."""
    if method == "pop":
        return r"Pop$_{RI}$", "Population"
    elif method == "gmm":
        return r"Per$_{RI}$", "Personalized"
    elif method.startswith("norma"):
        return r"NORMA$_{RI}$", "NORMA"
    return method, "Other"


def method_color(method):
    """Get color for a method using scripts/plots.py scheme_colors."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import plots
    _, scheme_key = method_display(method)
    return plots.scheme_colors.get(scheme_key, "#999999")


def detect_methods(df):
    """Detect available ref range classification columns."""
    methods = ["pop", "gmm"]
    for col in df.columns:
        if col.startswith("norma_") and col.endswith("_class"):
            methods.append(col.replace("_class", ""))
    return methods


def filter_by_time_window(df, min_hours, max_hours):
    """Filter measurements to a time window from admission (uses labresultoffset in minutes)."""
    min_min = min_hours * 60
    filtered = df[df["labresultoffset"] >= min_min].copy()
    if max_hours is not None:
        filtered = filtered[filtered["labresultoffset"] < max_hours * 60]
    return filtered


def prepare_cox_data(df, outcome_name, method):
    """Prepare one-row-per-(patient, lab_code) DataFrame for Cox regression.

    Returns DataFrame with columns: duration, event, age, sex, abnormal
    or None if insufficient data.
    """
    cfg = OUTCOMES[outcome_name]
    # Skip tautological method-outcome combinations
    if method in cfg.get("skip_methods", []):
        return None
    cls_col = f"{method}_class"
    if cls_col not in df.columns:
        return None

    grouped = df.groupby(["uniquepid", "lab_code"]).agg(
        age=("age", "first"),
        sex=("gender", "first"),
        abnormal=(cls_col, lambda s: int((s.dropna() != 1).any())),
        event=(cfg["event_col"], "first"),
        event_time=(cfg["time_col"], "first") if cfg["time_col"] in df.columns else ("labresultoffset", "first"),
        censor_time=(cfg["censor_col"], "first"),
        first_index_offset=("labresultoffset", "min"),
    ).reset_index()

    # Exclude patients who already had the condition
    if cfg["exclude_col"] is not None:
        has_prior = grouped["event"] & (grouped["event_time"] <= grouped["first_index_offset"])
        grouped = grouped[~has_prior].copy()

    # Duration = time from first index measurement to event or censoring (in days)
    grouped["duration"] = np.where(
        grouped["event"],
        grouped["event_time"] - grouped["first_index_offset"],
        grouped["censor_time"] - grouped["first_index_offset"],
    ) / (60 * 24)

    grouped = grouped[grouped["duration"] > 0].copy()
    grouped["sex"] = (grouped["sex"] == "Female").astype(int)

    n_events = grouped["event"].sum()
    n_total = len(grouped)
    if n_events < 3 or (n_total - n_events) < 3:
        return None
    if grouped["abnormal"].nunique() < 2:
        return None

    cox_df = grouped[["duration", "event", "age", "sex", "abnormal"]].copy()
    cox_df["event"] = cox_df["event"].astype(int)
    return cox_df.dropna()


def fit_cox_model(cox_df):
    """Fit Cox PH model. Returns (CoxPHFitter, success_bool)."""
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df, duration_col="duration", event_col="event")
        return cph, True
    except Exception:
        return None, False


# ============================================================
# Evaluation metrics helpers
# ============================================================

def compute_metrics_single(df, outcome_name, lab_code, method):
    """Compute PPV, NPV, sensitivity, specificity, F1 for one (outcome, lab, method)."""
    cfg = OUTCOMES[outcome_name]
    if method in cfg.get("skip_methods", []):
        return None
    cls_col = f"{method}_class"
    if cls_col not in df.columns:
        return None

    lab_df = df[df["lab_code"] == lab_code].copy()
    if len(lab_df) == 0:
        return None

    patient_df = lab_df.groupby("uniquepid").agg(
        abnormal=(cls_col, lambda s: int((s.dropna() != 1).any())),
        event=(cfg["event_col"], "first"),
    ).reset_index()
    patient_df = patient_df.dropna()
    patient_df["event"] = patient_df["event"].astype(int)
    patient_df["abnormal"] = patient_df["abnormal"].astype(int)

    # Exclude patients who had the condition before index
    if cfg["exclude_col"] is not None and cfg["time_col"] in lab_df.columns:
        first_offsets = lab_df.groupby("uniquepid")["labresultoffset"].min()
        event_offsets = lab_df.groupby("uniquepid")[cfg["time_col"]].first()
        prior_mask = (event_offsets <= first_offsets) & lab_df.groupby("uniquepid")[cfg["event_col"]].first()
        exclude_pids = prior_mask[prior_mask].index
        patient_df = patient_df[~patient_df["uniquepid"].isin(exclude_pids)]

    if len(patient_df) < 5:
        return None

    tp = ((patient_df["abnormal"] == 1) & (patient_df["event"] == 1)).sum()
    fp = ((patient_df["abnormal"] == 1) & (patient_df["event"] == 0)).sum()
    tn = ((patient_df["abnormal"] == 0) & (patient_df["event"] == 0)).sum()
    fn = ((patient_df["abnormal"] == 0) & (patient_df["event"] == 1)).sum()

    n_flagged = tp + fp
    n_not_flagged = tn + fn
    n_events = tp + fn
    n_no_events = tn + fp

    ppv = tp / n_flagged if n_flagged > 0 else np.nan
    npv = tn / n_not_flagged if n_not_flagged > 0 else np.nan
    sensitivity = tp / n_events if n_events > 0 else np.nan
    specificity = tn / n_no_events if n_no_events > 0 else np.nan
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
          if (ppv and sensitivity and (ppv + sensitivity) > 0) else np.nan)

    # Clean method name: pop -> Pop_RI, gmm -> Per_RI, norma_* -> NORMA_RI
    run_id = ""
    if method == "pop":
        method_label = "Pop_RI"
    elif method == "gmm":
        method_label = "Per_RI"
    elif method.startswith("norma_"):
        run_id = method[len("norma_"):]
        method_label = "NORMA_RI"
    else:
        method_label = method

    return {
        "outcome": outcome_name, "lab_code": lab_code,
        "method": method_label, "run_id": run_id,
        "n": len(patient_df), "n_events": n_events, "n_flagged": n_flagged,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "ppv": ppv, "npv": npv, "sensitivity": sensitivity,
        "specificity": specificity, "f1": f1,
    }


def compute_metrics_pop_restricted(df, outcome_names=None):
    """Compute recall/specificity/precision restricted to PopRI-normal measurements.

    Per the manuscript: restrict to measurements within PopRI (PopRI_class == 1),
    then compute metrics for PerRI (GMM) and NORMA_RI. This isolates the
    added value of personalized/NORMA intervals over population intervals.

    Returns DataFrame with columns: outcome, lab_code, method, sensitivity,
    specificity, ppv, plus confusion matrix counts.
    """
    if outcome_names is None:
        outcome_names = list(OUTCOMES.keys())

    if "PopRI_class" not in df.columns:
        print("PopRI_class column not found — cannot compute PopRI-restricted metrics")
        return pd.DataFrame()

    # Restrict to measurements classified as normal by PopRI
    pop_normal = df[df["PopRI_class"] == 1].copy()
    print(f"PopRI-restricted: {len(pop_normal)} / {len(df)} measurements "
          f"({len(pop_normal)/len(df)*100:.1f}%)")

    methods = detect_methods(df)
    # Exclude pop — tautological (all normal by definition)
    methods = [m for m in methods if m != "pop"]
    lab_codes = sorted(c for c in pop_normal["lab_code"].unique() if c not in EXCLUDE_LAB_CODES)

    results = []
    for outcome_name in outcome_names:
        cfg = OUTCOMES[outcome_name]
        for lab_code in lab_codes:
            for method in methods:
                if method in cfg.get("skip_methods", []):
                    continue
                result = compute_metrics_single(pop_normal, outcome_name, lab_code, method)
                if result is not None:
                    results.append(result)

    return pd.DataFrame(results)
