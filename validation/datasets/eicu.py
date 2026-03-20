"""
eICU dataset adapter.

Wraps existing eICU processing code into the standard dataset interface.
"""

import os
import sys
import pandas as pd

from .base import BaseDataset

# Add parent dir so we can import existing utils/config
_VAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VAL_DIR not in sys.path:
    sys.path.insert(0, _VAL_DIR)


class EICUDataset(BaseDataset):

    name = "eicu"
    time_unit = "minutes"
    exclude_labs = ["LDH", "CRP", "GGT", "PT"]

    outcomes = {
        "mortality": {
            "event_col": "died_in_hospital",
            "time_col": "death_offset",
            "censor_col": "hospitaldischargeoffset",
            "exclude_col": None,
        },
        "aki": {
            "event_col": "has_aki",
            "time_col": "aki_offset",
            "censor_col": "unitdischargeoffset",
            "exclude_col": "has_aki",
        },
        "sepsis": {
            "event_col": "has_sepsis",
            "time_col": "sepsis_offset",
            "censor_col": "unitdischargeoffset",
            "exclude_col": "has_sepsis",
        },
        "liver_injury": {
            "event_col": "has_liver_disease",
            "time_col": "liver_disease_offset",
            "censor_col": "unitdischargeoffset",
            "exclude_col": "has_liver_disease",
        },
        "prolonged_los": {
            "event_col": "prolonged_los",
            "time_col": "unitdischargeoffset",
            "censor_col": "unitdischargeoffset",
            "exclude_col": None,
        },
        "pop_abnormal": {
            "event_col": "has_pop_abnormal",
            "time_col": "pop_abnormal_offset",
            "censor_col": "last_index_offset",
            "exclude_col": None,
        },
    }

    primary_outcomes = ["mortality", "aki", "sepsis", "liver_injury", "prolonged_los", "pop_abnormal"]
    mortality_outcome = "mortality"

    def __init__(self, data_dir=None, results_dir=None):
        import config as eicu_config
        self.data_dir = data_dir or eicu_config.DATA_DIR
        self.results_dir = results_dir or eicu_config.RESULTS_DIR
        self.figures_dir = eicu_config.FIGURES_DIR
        self.cache_dir = eicu_config.CACHE_DIR

    def load_processed(self):
        """Load processed eICU data with standard column names."""
        path = os.path.join(self.data_dir, "split_df.pkl")
        print(f"Loading {path} ...")
        df = pd.read_pickle(path)

        # Rename to standard columns
        df = df.rename(columns={
            "uniquepid": "patient_id",
            "lab_code": "lab_code",
            "labresult": "value",
            "labresultoffset": "timestamp",
            "gender": "gender",
            "age": "age",
        })
        return df

    def load_classification(self):
        """Load pre-built classification detail with standard column names."""
        path = os.path.join(self.results_dir, "classification_detail.pkl")
        print(f"Loading {path} ...")
        df = pd.read_pickle(path)

        df = df.rename(columns={
            "uniquepid": "patient_id",
            "labresult": "value",
            "labresultoffset": "timestamp",
        })
        return df

    def attach_outcomes(self, df):
        """Attach eICU outcomes using existing utils."""
        import utils as eicu_utils
        return eicu_utils.attach_outcomes(df)
