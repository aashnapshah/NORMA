"""
CHS (Clalit Health Services) dataset adapter.

CHS data is organized in 250 chunk folders, each containing:
  - patient.csv   : patient demographics
  - labs.csv      : lab measurements
  - outcomes.csv  : clinical outcomes

Update DATA_ROOT and column names below to match your CHS server paths.
"""

import os
import glob
import pandas as pd

from .base import BaseDataset


class CHSDataset(BaseDataset):

    name = "chs"
    time_unit = "days"
    exclude_labs = []  # update if needed

    # ── Outcomes (update column names to match your outcomes.csv) ────
    outcomes = {
        "mortality": {
            "event_col": "died_10yr",           # TODO: update
            "time_col": "death_days",            # TODO: update
            "censor_col": "followup_days",       # TODO: update
            "exclude_col": None,
        },
        "t2d": {
            "event_col": "has_t2d",              # TODO: update
            "time_col": "t2d_days",              # TODO: update
            "censor_col": "followup_days",       # TODO: update
            "exclude_col": "has_t2d",
        },
        "ckd": {
            "event_col": "has_ckd",              # TODO: update
            "time_col": "ckd_days",              # TODO: update
            "censor_col": "followup_days",       # TODO: update
            "exclude_col": "has_ckd",
        },
    }

    primary_outcomes = ["mortality", "t2d", "ckd"]
    mortality_outcome = "mortality"

    # ── Paths (update these) ────────────────────────────────────────
    DATA_ROOT = "/path/to/chs/chunks"           # TODO: set this
    RESULTS_DIR = "/path/to/chs/results"        # TODO: set this
    FIGURES_DIR = "/path/to/chs/figures"         # TODO: set this

    # ── Column mappings (update to match your CSVs) ─────────────────
    # These map your CSV column names → standard pipeline names
    PATIENT_COLS = {
        "patient_id": "patient_id",             # TODO: update
        "gender": "gender",                     # TODO: update
        "age": "age",                           # TODO: update
    }
    LAB_COLS = {
        "patient_id": "patient_id",             # TODO: update (join key)
        "lab_code": "lab_code",                 # TODO: update
        "value": "value",                       # TODO: update
        "timestamp": "timestamp",               # TODO: update (days from baseline, date, etc.)
    }
    OUTCOME_COLS = {
        "patient_id": "patient_id",             # TODO: update (join key)
        # outcome columns are mapped via self.outcomes dict
    }

    def __init__(self, data_root=None, results_dir=None, figures_dir=None):
        self.data_root = data_root or self.DATA_ROOT
        self.results_dir = results_dir or self.RESULTS_DIR
        self.figures_dir = figures_dir or self.FIGURES_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def _chunk_dirs(self):
        """Find all chunk directories."""
        pattern = os.path.join(self.data_root, "*/")
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            raise FileNotFoundError(f"No chunk directories found in {self.data_root}")
        print(f"Found {len(dirs)} chunk directories")
        return dirs

    def _load_chunks(self, filename, col_map, usecols=None):
        """Load and concatenate a CSV from all chunk directories."""
        frames = []
        for d in self._chunk_dirs():
            path = os.path.join(d, filename)
            if os.path.exists(path):
                chunk = pd.read_csv(path, usecols=usecols)
                frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)

        # Rename columns to standard names
        rename = {v: k for k, v in col_map.items() if v != k}
        if rename:
            df = df.rename(columns=rename)

        return df

    def load_processed(self):
        """Load CHS data from chunks, merge patients + labs."""
        print("Loading CHS patient data ...")
        patients = self._load_chunks("patient.csv", self.PATIENT_COLS)
        patients = patients.drop_duplicates(subset=["patient_id"])

        print("Loading CHS lab data ...")
        labs = self._load_chunks("labs.csv", self.LAB_COLS)

        print("Merging ...")
        df = labs.merge(patients, on="patient_id", how="left")

        print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients")
        return df

    def attach_outcomes(self, df):
        """Load outcomes from chunks and merge onto df."""
        print("Loading CHS outcome data ...")
        outcomes = self._load_chunks("outcomes.csv", self.OUTCOME_COLS)
        outcomes = outcomes.drop_duplicates(subset=["patient_id"])

        df = df.merge(outcomes, on="patient_id", how="left")
        return df
