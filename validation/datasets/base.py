"""
Base dataset interface. Each dataset must implement these methods.

The pipeline expects a standardized DataFrame with these columns:
  - patient_id  : unique patient identifier
  - lab_code    : standardized lab code (e.g., "CRE", "ALB")
  - value       : numeric lab result
  - timestamp   : measurement time (numeric; units defined by dataset)
  - gender      : "Male" / "Female"
  - age         : numeric age
  - split       : "baseline" or "index"

Outcome columns are dataset-specific and defined in OUTCOMES dict.
"""

from abc import ABC, abstractmethod


class BaseDataset(ABC):

    # ── Required attributes (override in subclass) ──────────────────

    name = None                 # e.g., "eicu", "chs"
    time_unit = None            # "minutes" or "days"
    exclude_labs = []           # lab codes to exclude from analysis

    # Outcomes: {name: {"event_col": str, "time_col": str, "censor_col": str, "exclude_col": str|None}}
    outcomes = {}
    primary_outcomes = []       # subset of outcome keys to run by default
    mortality_outcome = None    # key in outcomes used for per-lab NNS

    # ── Required methods ────────────────────────────────────────────

    @abstractmethod
    def load_processed(self):
        """Load processed lab data. Return DataFrame with standard columns:
        patient_id, lab_code, value, timestamp, gender, age.
        """
        pass

    @abstractmethod
    def attach_outcomes(self, df):
        """Attach outcome columns to DataFrame. Return df with outcome columns
        as defined in self.outcomes.
        """
        pass

    # ── Optional overrides ──────────────────────────────────────────

    def split_data(self, df, baseline_frac=0.75, min_baseline=5):
        """Split each patient-lab group into baseline/index by time.
        Default implementation: first baseline_frac by chronological order.
        """
        import numpy as np

        df = df.sort_values(["patient_id", "lab_code", "timestamp"])

        def _split_group(g):
            n = len(g)
            n_baseline = max(int(n * baseline_frac), 1)
            labels = ["baseline"] * n_baseline + ["index"] * (n - n_baseline)
            return labels

        df["split"] = df.groupby(["patient_id", "lab_code"]).transform(
            lambda g: _split_group(g)
        )["timestamp"]  # dummy — replaced below

        # Redo properly with apply
        splits = []
        for _, g in df.groupby(["patient_id", "lab_code"]):
            n = len(g)
            n_bl = max(int(n * baseline_frac), 1)
            splits.extend(["baseline"] * n_bl + ["index"] * (n - n_bl))
        df["split"] = splits

        # Filter: require min_baseline measurements
        bl_counts = df[df["split"] == "baseline"].groupby(["patient_id", "lab_code"]).size()
        valid = bl_counts[bl_counts >= min_baseline].index
        df = df.set_index(["patient_id", "lab_code"])
        df = df.loc[df.index.isin(valid)].reset_index()

        return df

    def get_reference_intervals(self):
        """Return population reference intervals dict.
        Default: load from process.config.REFERENCE_INTERVALS.
        """
        from process.config import REFERENCE_INTERVALS
        return REFERENCE_INTERVALS
