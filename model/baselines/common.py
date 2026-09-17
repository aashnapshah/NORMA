"""Helpers shared by the reference-interval baselines."""
import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from process.config import REFERENCE_INTERVALS  # noqa: F401  -- re-exported


def detect_cols(df):
    """Map this cohort's column names onto the names the baselines use."""
    return {
        "pid": "patient_id" if "patient_id" in df.columns else "uniquepid",
        "analyte": "analyte" if "analyte" in df.columns else "lab_code",
        "value": "value" if "value" in df.columns else "labresult",
        "time": "timestamp" if "timestamp" in df.columns else "labresultoffset",
        "sex": "sex" if "sex" in df.columns else "gender",
    }


def sex_key(sex_val):
    """Normalise a sex value to the "M" / "F" keys of REFERENCE_INTERVALS."""
    if isinstance(sex_val, str):
        return "F" if sex_val[:1].upper() == "F" else "M"
    return "F" if sex_val == 1 else "M"


def sex_idx(sex_val):
    """0 = male, 1 = female (matches get_sex_str in the processing stage)."""
    return 1 if sex_key(sex_val) == "F" else 0
