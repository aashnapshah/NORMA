"""
Shared constants for the validation pipeline.
"""
import os

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EICU_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "physionet.org", "files", "eicu-crd", "2.0")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "eicu")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODEL_LOG_DIR = os.path.join(BASE_DIR, "..", "model", "logs")
NORMA_RUN_ID = "334f7e21"

for d in [CACHE_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# eICU lab code mapping
# ============================================================

EICU_LAB_MAP = {
    # CBC
    'HGB': ['Hgb'], 'HCT': ['Hct'], 'RBC': ['RBC'],
    'PLT': ['platelets x 1000'], 'MCH': ['MCH'], 'MCHC': ['MCHC'],
    'MCV': ['MCV'], 'MPV': ['MPV'], 'RDW': ['RDW'], 'WBC': ['WBC x 1000'],
    # BMP
    'NA': ['sodium'], 'K': ['potassium'], 'CL': ['chloride'],
    'CO2': ['bicarbonate', 'Total CO2', 'HCO3'], 'BUN': ['BUN'],
    'CRE': ['creatinine'], 'GLU': ['glucose', 'bedside glucose'],
    'A1C': [], 'CA': ['calcium', 'ionized calcium'],
    # HFP
    'ALT': ['ALT (SGPT)'], 'GGT': [], 'AST': ['AST (SGOT)'],
    'LDH': ['LDH'], 'PT': ['PT'], 'ALP': ['alkaline phos.'],
    'TBIL': ['total bilirubin'], 'DBIL': ['direct bilirubin'],
    'ALB': ['albumin'], 'TP': ['total protein'], 'CRP': ['CRP', 'CRP-hs'],
    # Lipids
    'TC': ['total cholesterol'], 'HDL': ['HDL'], 'LDL': ['LDL'],
    'TGL': ['triglycerides'],
}

REVERSE_LAB_MAP = {}
for code, names in EICU_LAB_MAP.items():
    for name in names:
        REVERSE_LAB_MAP[name] = code
ALL_EICU_NAMES = set(REVERSE_LAB_MAP.keys())

# Labs to exclude from classification (too few samples or not clinically meaningful)
EXCLUDE_LAB_CODES = ['LDH', 'CRP', 'GGT', 'PT']

# ============================================================
# Disease definitions
# ============================================================

DISEASES = {
    "diabetes": {
        "dx_pattern": "diabetes",
        "icd_pattern": r"^250",
    },
    "ckd": {
        "dx_pattern": "chronic kidney|chronic renal",
        "icd_pattern": r"^585",
    },
    "aki": {
        "dx_pattern": "acute renal failure|acute kidney",
        "icd_pattern": r"^584",
    },
    "resp_failure": {
        "dx_pattern": "acute respiratory failure",
        "icd_pattern": r"^518\.8",
    },
    "sepsis": {
        "dx_pattern": "sepsis|septicemia",
        "icd_pattern": r"^995\.9|^038",
    },
    "chf": {
        "dx_pattern": "congestive heart failure|CHF",
        "icd_pattern": r"^428",
    },
    "afib": {
        "dx_pattern": "atrial fibrillation",
        "icd_pattern": r"^427\.31",
    },
    "hypertension": {
        "dx_pattern": "hypertension(?!.*pulmonary)",
        "icd_pattern": r"^401",
    },
    "copd": {
        "dx_pattern": "COPD|chronic obstructive",
        "icd_pattern": r"^491|^492|^496",
    },
    "pneumonia": {
        "dx_pattern": "pneumonia",
        "icd_pattern": r"^486|^481|^482|^483|^484|^485",
    },
    "liver_disease": {
        "dx_pattern": "cirrhosis|hepatic failure|liver failure",
        "icd_pattern": r"^571|^572",
    },
    "stroke": {
        "dx_pattern": "stroke|cerebrovascular accident|CVA",
        "icd_pattern": r"^430|^431|^432|^433|^434|^436",
    },
}

# ============================================================
# Cox model outcome definitions
# ============================================================

OUTCOMES = {
    # --- Primary outcomes (acute, appropriate for ICU) ---
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
        "skip_methods": ["pop"],  # pop predicting pop is tautological
    },
    # --- Secondary outcomes (for Clalit comparability) ---
    "ckd": {
        "event_col": "has_ckd",
        "time_col": "ckd_offset",
        "censor_col": "unitdischargeoffset",
        "exclude_col": "has_ckd",
    },
    "t2d": {
        "event_col": "has_diabetes",
        "time_col": "diabetes_offset",
        "censor_col": "unitdischargeoffset",
        "exclude_col": "has_diabetes",
    },
}

# Primary outcomes to run by default (acute/ICU-appropriate)
PRIMARY_OUTCOMES = ["mortality", "aki", "sepsis", "prolonged_los"]

# ============================================================
# Time windows for stratified analysis
# ============================================================

DEFAULT_TIME_WINDOWS = [
    ("0-24h",   0,   24),
    ("24-72h", 24,   72),
    ("72h+",   72, None),
]
