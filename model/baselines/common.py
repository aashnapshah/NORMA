"""Helpers shared by the reference-interval baselines.

cohen.py and gaussian.py each carried their own copy of every function here.
The two _detect_cols implementations were identical except that gaussian's was
missing the "sex" entry, and _popri_lookup existed twice, each with its own
sys.path insert to reach process.config.
"""
import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from process.config import REFERENCE_INTERVALS  # noqa: F401  -- re-exported


def detect_cols(df):
    """Map this cohort's column names onto the names the baselines use.

    The cohorts disagree: the processed dev frames use patient_id / analyte /
    value / timestamp / sex, eICU uses uniquepid / lab_code / labresult /
    labresultoffset / gender.
    """
    return {
        "pid": "patient_id" if "patient_id" in df.columns else "uniquepid",
        "analyte": "analyte" if "analyte" in df.columns else "lab_code",
        "value": "value" if "value" in df.columns else "labresult",
        "time": "timestamp" if "timestamp" in df.columns else "labresultoffset",
        "sex": "sex" if "sex" in df.columns else "gender",
    }


def sex_key(sex_val):
    """Normalise a sex value to the "M" / "F" keys of REFERENCE_INTERVALS.

    Deliberately NOT metrics.sex_key, which the stage scripts use. The two
    agree on every encoding the baselines actually see -- detect_cols prefers
    the int 0/1 `sex` column, which every frame under results/raw/ carries --
    but they disagree on a free-text `gender`: metrics.sex_key sends "Other"
    and "Unknown" to F, this sends them to M. Kept as it was so no baseline
    number moves; worth unifying deliberately, not as a side effect.
    """
    if isinstance(sex_val, str):
        return "F" if sex_val[:1].upper() == "F" else "M"
    return "F" if sex_val == 1 else "M"


def sex_idx(sex_val):
    """0 = male, 1 = female (matches get_sex_str in the processing stage)."""
    return 1 if sex_key(sex_val) == "F" else 0
