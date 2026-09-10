"""Dataset adapters, paths and pipeline constants for the validation pipeline.

One module (2026-09-08; was config.py + discover.py + datasets.py + chunks.py + refs_io.py):
  paths and run constants          SCRIPTS_DIR ... NORMA_RUN_ID, EXCLUDE_LABS, PANELS
  the pipeline stages              analysis_dirs(), load_registries()
  where a stage writes             analysis_name(), output_dirs(), data_dir(), results_dir()
  the cohort adapters              DATASETS = {eicu, chs, inspire, ehrshot, mimiciv}
  CHS per-chunk caching            cached_chunk_frames(), read_chunk_classification()
  classification I/O               read_classification(), write_classification() -- two files
  ref_intervals I/O                read_ref_intervals(), upsert_ref_rows(), ... -- two files
Dataset-specific configuration (outcomes, exclude_labs, diseases, primary_outcomes)
lives on the adapter classes.
"""
import glob
import importlib.util
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import constants       # the constants the analysis and the figures share

# ============================================================
# Paths and run constants
# ============================================================

# Layout (2026-09-08).  Code: scripts/<nn>_<stage>.py -- one script per pipeline
# stage, holding that stage's analysis and its figures and tables -- driven by
# scripts/make_figures.py and scripts/make_tables.py.  Results are keyed by kind
# first, then cohort:
#   data/<cohort>/                       inputs + heavy intermediates (<stage>/ subfolders)
#   results/raw/<cohort>/                every CSV the stages write (flat: one
#                                        folder per cohort, filenames are unique)
#   results/processed/<cohort>/           the slice the figures and tables read (scripts/export.py)
#   results/figures/<tag>/<nn>_<name>.pdf        tag = cohort key, or "all"
#   results/tables/<tag>/{tex,csv,pdf}/<nn>_<name>.<ext>
# Fitted things are not results: NORMA checkpoints live in model/logs/<run_id>/
# and the dev-fitted baseline pickles in model/logs/baselines/ (BASELINE_DIR),
# which is also what the Clalit bundle carries in (jobs/run_clalit.py --pack_bundle).
# so the Clalit round trip is: scripts/ in, results/processed/chs/ out.
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # norma/scripts
BASE_DIR = os.path.dirname(SCRIPTS_DIR)                                     # norma
VAL_DIR = BASE_DIR
REPO_DIR = os.path.dirname(BASE_DIR)
# Raw PhysioNet downloads live outside the repo, in REPO_DIR/data/ (see also
# process/covariates.py DATA_ROOT, the processed/ + raw/ source tables).
PHYSIONET_DIR = os.path.join(REPO_DIR, "data", "physionet.org", "files")
EICU_DATA_DIR = os.path.join(PHYSIONET_DIR, "eicu-crd", "2.0")
INSPIRE_DATA_DIR = os.path.join(PHYSIONET_DIR, "inspire", "1.3")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODEL_LOG_DIR = os.path.join(BASE_DIR, "model", "logs")     # NORMA checkpoints, one folder per run
BASELINE_DIR = os.path.join(MODEL_LOG_DIR, "baselines")     # dev-fitted Cohen / Gaussian pickles

# cohort key -> folder under data/, for the keys whose folder is not the key itself.
# INSPIRE's baseline/index split fraction is part of its data path, not its key.
# Everything reaches these through data_dir(key) -- no module hardcodes a folder name.
DATA_SUBDIR = {"inspire": os.path.join("inspire", "25-75"),
               "chs": "clalit", "sandbox": os.path.join("clalit", "sandbox")}
DEV_KEY = "dev"          # the "cohort" of cohort-free outputs (synthetic histories, dev split)
RESULT_KINDS = ("raw", "processed")


def data_dir(sub):
    """data/<cohort>/ for a cohort key such as "eicu" or "inspire"."""
    return os.path.join(DATA_DIR, DATA_SUBDIR.get(sub, sub))


def results_dir(sub, stage=None, kind="raw"):
    """results/<kind>/<sub>/ — kind is "raw" (what the stages write) or "processed"
    (what the figures read; scripts/export.py derives it from raw).  Flat: no
    per-stage subfolder, since every result filename is unique within a cohort
    (that is what let find_result() search by filename alone).  `stage` is
    accepted and ignored so callers can keep passing their stage name."""
    return os.path.join(RESULTS_DIR, kind, sub)


def dev_results_dir(stage=None):
    """results/raw/dev/ for outputs that belong to no cohort."""
    d = results_dir(DEV_KEY)
    os.makedirs(d, exist_ok=True)
    return d


def artifact(name):
    """A dev-fitted baseline (cohen_dev_models.pkl, gaussian_eb_prior_dev.pkl) in
    model/logs/baselines/, here and inside Clalit -- the bundle mirrors these paths."""
    return os.path.join(BASELINE_DIR, name)
# Main model: NORMA2, QuantileLoss, with age at draw + care setting as
# per-measurement covariates (model/logs/q_age_set). Chosen 2026-09-03 over
# q_age_set_co (all three covariates): the co-analyte channel costs ~0.08 RR on
# the future-abnormality analysis and loses 77/100 eICU AUROC cells, age and
# setting are neutral-to-helpful, and setting answers R1.1/R1.2. 334f7e21 (no
# covariates) stays in the pipeline as the "None" arm. CHS still carries
# 334f7e21 until the Clalit bundle is rebuilt with this run; HF weights / app
# likewise.
NORMA_RUN_ID = "q_age_set"
# Covariate-ablation arms (model/run_covariate_ablation.sh) carried through the
# pipeline alongside NORMA_RUN_ID: ref_intervals methods norma_<arm>, downstream
# tables/figures via dataset.run_ids. Every arm is labelled by the covariates it
# contains (lib/models.RUN_COVARIATES). q_age_co, q_set_co and q_co_q are
# appended once trained and pushed through 04_refs.py (norma step).
# Shown as an additive ladder: sex -> +age -> +setting (main) -> +analytes.
# q_set / q_co (setting-only, analytes-only) exist in ref_intervals but are not
# shown: Aashna 2026-09-03, "extra stuff for not that much new info".
NORMA_ABLATION_RUN_IDS = ["334f7e21", "q_age", "q_age_set_co"]
# Which weights of NORMA_RUN_ID every script loads (04_refs.py (norma step),
# 16_benchmark, predict_states). "latest" (epoch 39) is what produced the
# published dev-set predictions_combined.csv and all leak-free variant files;
# "best" (epoch 40) would require regenerating those. Flip here, nowhere else.
NORMA_CHECKPOINT = "latest"
EXCLUDE_LABS = list(constants.EXCLUDE_LABS)  # lib/constants.py owns the exclusions

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

PANELS = {
    "CBC": ["HGB", "HCT", "RBC", "PLT", "MCH", "MCHC", "MCV", "MPV", "RDW", "WBC"],
    "BMP": ["NA", "K", "CL", "CO2", "BUN", "CRE", "GLU", "A1C", "CA"],
    "HFP": ["ALT", "GGT", "AST", "LDH", "PT", "ALP", "TBIL", "DBIL", "ALB", "TP", "CRP"],
    "Lipid": ["TC", "HDL", "LDL", "TGL"],
}
CODE_TO_PANEL = {code: panel for panel, codes in PANELS.items() for code in codes}


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline stages
# ═══════════════════════════════════════════════════════════════════════════

_STAGE = re.compile(r"^\d\d_\w+")


def analysis_dirs():
    """Stage names in pipeline order: "01_process", "02_index_labs", ..."""
    names = set()
    for e in os.listdir(SCRIPTS_DIR):
        if not _STAGE.match(e):
            continue
        if e.endswith(".py"):
            names.add(e[:-3])
        elif os.path.isdir(os.path.join(SCRIPTS_DIR, e)):
            names.add(e)
    return sorted(names)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_registries(attr):
    """Concatenate `attr` ("FIGURES" or "TABLES") from every stage script that defines it.

    A stage script holds its own figure and table definitions, so the registry is
    assembled by importing scripts/<stage>.py -- the same file that computed the
    results it draws."""
    out = []
    for d in analysis_dirs():
        path = os.path.join(SCRIPTS_DIR, f"{d}.py")
        if not os.path.exists(path):
            continue
        module = load_module(path, f"stage_{d}")
        out.extend(getattr(module, attr, []))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Where a stage writes
# ═══════════════════════════════════════════════════════════════════════════

def analysis_name():
    """Stage of the script that is running, e.g. "12_eval" for scripts/12_eval.py
    or "01_process" for scripts/process/eicu.py.

    results/ is flat, so this is what stage_file() turns into the <nn>_ prefix on
    every file the stage writes.  Returns None when there is no owning stage — an
    interactive session, or a script under lib/ jobs/ process/ — and those outputs
    are then written unprefixed.
    """
    import __main__
    path = getattr(__main__, "__file__", None)
    if not path:
        return None
    path = os.path.realpath(path)
    folder = os.path.dirname(path)
    scripts = os.path.realpath(SCRIPTS_DIR)
    if folder == scripts:
        name = os.path.splitext(os.path.basename(path))[0]
    elif os.path.dirname(folder) == scripts:
        name = os.path.basename(folder)
    else:
        return None
    return name if name in ANALYSIS_DIRS() else None


def ANALYSIS_DIRS():
    """The folders that count as analyses."""
    return set(analysis_dirs())


def stage_num(analysis=None):
    """"12" for stage 12_eval -- the numeric prefix every output file carries."""
    a = analysis or analysis_name()
    return a.split("_", 1)[0] if a else None


def stage_file(name, analysis=None):
    """<nn>_<name>: the file a stage writes, tagged with its pipeline stage.

    results/ is flat, so the prefix is what says which stage produced a file --
    12_eval.csv, 13_cox.csv, 17_incidence.csv -- and it keeps a name that repeats
    across stages apart without inventing a per-stage synonym for it.  Names that
    already carry a prefix pass through unchanged; so does anything written
    outside a stage (analysis_name() is None there).
    """
    n = stage_num(analysis)
    base = os.path.basename(name)
    return f"{n}_{base}" if n and not re.match(r"^\d\d_", base) else base


def result_path(results_dir, name, analysis=None):
    """The path a stage writes `name` to, with its stage prefix."""
    return os.path.join(results_dir, stage_file(name, analysis))


def find_in(directory, filename):
    """`filename` inside `directory`, resolving the stage prefix the writer added.

    Callers know the bare name ("state_conditional.csv"); the file is
    "04_state_conditional.csv".  Returns the unprefixed path when nothing matches,
    so an `os.path.exists` check on the result still behaves.
    """
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return path
    hits = sorted(glob.glob(os.path.join(directory, f"[0-9][0-9]_{filename}")))
    return hits[0] if hits else path


def _under_results(directory):
    """results/ files carry the stage prefix; data/ (the CHS chunk dirs) never does."""
    return os.path.realpath(directory).startswith(os.path.realpath(RESULTS_DIR))


def stage_path(directory, name, analysis):
    """Where `name` lives in `directory`, honouring the <nn>_ stage prefix.

    An existing file wins whichever form it is in (find_in resolves both), so a
    CHS chunk dir under data/ keeps its bare names and a fresh results/ file gets
    the prefix.  `analysis` names the OWNING stage, which is not always the
    running one -- 07_classify reads 04_refs' ref_intervals.
    """
    existing = find_in(directory, name)
    if os.path.exists(existing):
        return existing
    return (result_path(directory, name, analysis) if _under_results(directory)
            else os.path.join(directory, name))


REFS_STAGE = "04_refs"
CLASSIFY_STAGE = "07_classify"


def output_dirs(sub):
    """results/raw/<sub>/, created.  There is no per-stage cache under data/ any
    more (2026-09-08): data/ holds inputs and raw-ingest intermediates only, and
    everything a stage computes -- classification, reference intervals, every
    prediction file -- is a raw result, however heavy."""
    res_dir = results_dir(sub)
    os.makedirs(res_dir, exist_ok=True)
    return res_dir
ROOTDIR = REPO_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Cohort adapters
# ═══════════════════════════════════════════════════════════════════════════

def _natural_sort_key(path):
    """Sort paths numerically so chunk_2 comes before chunk_10."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', path)]


def fix_analyte(df):
    """Replace empty analyte values with 'NA' (sodium lab code gets eaten by pandas)."""
    if "analyte" in df.columns:
        df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df



RI_NUM_COLS = ["ri_mean", "ri_std", "ri_low", "ri_high", "age", "n_bl", "t_span"]


def _coerce_ri(df):
    """Coerce RI numeric columns to float."""
    for col in RI_NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



# ── Base Dataset ─────────────────────────────────────────────

class BaseDataset(ABC):

    name = None
    time_unit = None
    outcome_time_unit = None      # unit of the outcome time/censor columns (defaults to time_unit)
    exclude_labs = []
    run_ids = []
    primary_norma_run = None  # first run_id used as canonical "NORMA"
    # --no_norma: the cohort is scored on the baseline methods alone, as if the
    # model had never been run (datasets.get_dataset empties run_ids as well).
    no_norma = False

    # Outcomes: {name: {"event_col", "time_col", "censor_col", "exclude_col"}}
    outcomes = {}
    primary_outcomes = []
    mortality_outcome = None

    # Disease detection patterns (dataset-specific)
    diseases = {}

    # Evaluation time windows for multi-analyte models (in days, since durations are standardized)
    eval_windows = []

    # Analyte filter (set via constructor; None = all)
    _analytes = None

    def _filter_analytes(self, df):
        if self._analytes is not None and "analyte" in df.columns:
            df = df[df["analyte"].isin(self._analytes)]
        return df

    # Cohen et al. 2021 (Nat Med) benchmark variants (see cohen.py)
    cohen_models = ["m2", "m3", "m4"]
    # Per-patient Gaussian fits to PopRI-normal history (see gaussian.py)
    gaussian_models = ["mle", "trunc", "eb"]

    def _filter_methods(self, df):
        """Keep only base/pop/per, Cohen/Gaussian benchmarks, and configured NORMA run IDs."""
        # --no_norma leaves run_ids empty on purpose, and that has to still filter:
        # otherwise a leftover ref_intervals_norma.parquet would come back in.
        if "method" in df.columns and (self.run_ids or self.no_norma):
            keep = ({'base', 'pop', 'per'}
                    | {f'cohen_{m}' for m in self.cohen_models}
                    | {f'gaussian_{m}' for m in self.gaussian_models}
                    | {f'norma_{rid}' for rid in self.run_ids})
            df = df[df["method"].isin(keep)]
        return df

    @property
    def methods(self):
        return (['PopRI', 'PerRI']
                + [f'Cohen_{m}' for m in self.cohen_models]
                + [f'Gaussian_{m}' for m in self.gaussian_models]
                + [f'NORMA_{rid}' for rid in self.run_ids])

    @property
    def norma_methods(self):
        return [f'NORMA_{rid}' for rid in self.run_ids]

    def _filter_norma(self, df):
        """--no_norma: drop the NORMA columns an earlier run left in the classification
        files, so a baselines-only run cannot silently pick them up again."""
        if not self.no_norma:
            return df
        return df.drop(columns=[c for c in df.columns if is_norma_column(c)])

    @abstractmethod
    def load_processed(self):
        pass

    @abstractmethod
    def attach_outcomes(self, df):
        pass

    def output_sub(self):
        """Sub-path under results/ (and key into datasets.DATA_SUBDIR) for this cohort."""
        return self.name

    def setup_output(self):
        """results/raw/<cohort>/ — where every file this stage writes goes."""
        return output_dirs(self.output_sub())

    NORMA_PREDICTIONS_FILE = "norma_predictions.parquet"

    def norma_predictions_path(self):
        """Where 04_refs (norma step) writes this cohort's per-state predictions."""
        return result_path(output_dirs(self.output_sub()), self.NORMA_PREDICTIONS_FILE, "04_refs")

    def load_norma_predictions(self, columns=None):
        """Read them back (chunked cohorts concatenate their chunks)."""
        if self.no_norma:
            return None
        p = self.norma_predictions_path()
        if not os.path.exists(p):
            return None
        return pd.read_parquet(p, columns=columns)

    def load_index_labs(self):
        p = os.path.join(self.data_dir, "index_labs.parquet")
        if os.path.exists(p):
            print(f"  Loading {p}")
            df = self._filter_analytes(self._standardize(pd.read_parquet(p)))
            df.attrs["time_unit"] = self.time_unit
            return df
        raise FileNotFoundError(f"No index_labs.parquet found at {p}")

    def load_ref_intervals(self):
        d = results_dir(self.output_sub())
        print(f"  Loading reference intervals from {d}")
        df = read_ref_intervals(d)
        if df is None:
            raise FileNotFoundError(f"No {REF_FILE} found in {d}")
        return self._filter_methods(self._filter_analytes(_coerce_ri(fix_analyte(df))))

    def classification_dir(self):
        """results/raw/<cohort>/ — 07_classify's two output files live here."""
        return results_dir(self.output_sub())

    def load_classification(self, usecols=None):
        d = self.classification_dir()
        print(f"  Loading classification from {d}")
        df = read_classification(d, usecols)
        if df is None:
            raise FileNotFoundError(f"No {CLASSIFICATION_FILE} found in {d}")
        return self._filter_norma(self._filter_analytes(fix_analyte(df)))

    def classification_columns(self):
        """Column names of both classification halves without reading the data."""
        names = classification_column_names(self.classification_dir())
        if names is None:
            raise FileNotFoundError(
                f"No {CLASSIFICATION_FILE} found in {self.classification_dir()}")
        return names

    def iter_chunks(self):
        raise NotImplementedError

    def get_reference_intervals(self):
        from process.config import REFERENCE_INTERVALS
        return REFERENCE_INTERVALS


# ── eICU Dataset ─────────────────────────────────────────────

class EICUDataset(BaseDataset):

    name = "eicu"
    # index_labs.timestamp is days re-anchored per series (process/eicu.py via
    # _standardize); the raw eICU offset columns (labresultoffset, unitdischargeoffset,
    # ...) and days_from_admit keep the ICU-admission-relative clock.
    time_unit = "days"
    outcome_time_unit = "minutes"
    exclude_labs = EXCLUDE_LABS
    run_ids = [NORMA_RUN_ID, *NORMA_ABLATION_RUN_IDS]
    primary_norma_run = NORMA_RUN_ID

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
            # Not a time-to-event outcome: the event IS "this duration exceeded 7 days",
            # so the label is a deterministic function of the follow-up time and the
            # two are nearly disjoint (eICU CRE: median 21.6 h for non-events, 82.2 h
            # for events). Harrell's C then rests on almost no comparable pairs.
            # Survival code must skip it; it stays a binary outcome for 12_eval.
            "survival": False,
        },
        "pop_abnormal": {
            "event_col": "has_pop_abnormal",
            "time_col": "pop_abnormal_offset",
            "censor_col": "last_index_offset",
            "exclude_col": None,
            "skip_methods": ["pop"],
        },
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

    primary_outcomes = ["mortality", "aki", "sepsis", "prolonged_los"]
    mortality_outcome = "mortality"
    eval_windows = [1, 2, 7, 14, 30]  # days (durations already converted to days)

    diseases = {
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

    def __init__(self, data_dir=None, cache_dir=None, analytes=None, **kwargs):
        self.data_dir = data_dir or globals()["data_dir"]("eicu")
        # Resolved by setup_output(); an explicit cache_dir= still wins.
        self.cache_dir = cache_dir or output_dirs("eicu")
        self._analytes = analytes
        os.makedirs(self.cache_dir, exist_ok=True)

    def _standardize(self, df):
        """Rename eICU columns to standard names and encode sex as int."""
        rename = {}
        if "uniquepid" in df.columns and "patient_id" not in df.columns:
            rename["uniquepid"] = "patient_id"
        if "lab_code" in df.columns and "analyte" not in df.columns:
            rename["lab_code"] = "analyte"
        if "labresult" in df.columns and "value" not in df.columns:
            rename["labresult"] = "value"
        needs_time_convert = False
        if "labresultoffset" in df.columns and "timestamp" not in df.columns:
            rename["labresultoffset"] = "timestamp"
            needs_time_convert = True
        if rename:
            df = df.rename(columns=rename)
        # Encode sex as int (0=M, 1=F)
        if "gender" in df.columns and "sex" not in df.columns:
            df["sex"] = (df["gender"] == "Female").astype(int)
        # Convert minutes → days, then re-anchor to 0 per patient-analyte
        if needs_time_convert:
            df["timestamp"] = df["timestamp"] / (60 * 24)
            if "patient_id" in df.columns and "analyte" in df.columns:
                df["timestamp"] = df["timestamp"] - df.groupby(["patient_id", "analyte"])["timestamp"].transform("min")
        return fix_analyte(df)

    def load_processed(self):
        p = os.path.join(self.data_dir, "processed.parquet")
        if os.path.exists(p):
            print(f"  Loading {p}")
            return self._filter_analytes(self._standardize(pd.read_parquet(p)))
        raise FileNotFoundError(f"No processed.parquet found at {p}; run ../../process/eicu.py")

    def load_diagnosis(self):
        for p in [
            os.path.join(self.data_dir, "eicu_diagnosis.parquet"),
            os.path.join(self.data_dir, "diagnosis.parquet"),
        ]:
            if os.path.exists(p):
                print(f"  Loading {p}")
                return pd.read_parquet(p)
        raise FileNotFoundError(f"No diagnosis parquet found in {self.data_dir}")

    def iter_chunks(self):
        index_labs = self.load_index_labs()
        ref_df = self.load_ref_intervals()
        yield 0, self.cache_dir, index_labs, ref_df

    def get_mortality(self):
        """Load patient table and derive mortality event/time columns."""
        cache_path = os.path.join(data_dir("eicu"), "01_process", "patient.pkl")
        if os.path.exists(cache_path):
            patient = pd.read_pickle(cache_path)
        else:
            patient = pd.read_csv(
                os.path.join(EICU_DATA_DIR, "patient.csv")
            )

        patient["died_in_hospital"] = (
            (patient["unitdischargestatus"] == "Expired")
            | (patient["hospitaldischargestatus"] == "Expired")
        )
        patient["death_offset"] = np.where(
            patient["died_in_hospital"],
            patient["hospitaldischargeoffset"],
            np.nan,
        )
        cols = [
            "patientunitstayid", "died_in_hospital", "death_offset",
            "unitdischargeoffset", "hospitaldischargeoffset",
        ]
        return patient[[c for c in cols if c in patient.columns]].copy()

    def load_diagnosis_processed(self):
        return self.load_diagnosis()

    def attach_outcomes(self, df):
        if "died_in_hospital" not in df.columns and "patientunitstayid" in df.columns:
            mort = self.get_mortality()
            cols = [c for c in mort.columns
                    if c not in df.columns or c == "patientunitstayid"]
            df = df.merge(mort[cols], on="patientunitstayid", how="left")

        if "prolonged_los" not in df.columns and "unitdischargeoffset" in df.columns:
            df["prolonged_los"] = df["unitdischargeoffset"] > (7 * 24 * 60)

        if "patientunitstayid" in df.columns:
            try:
                diagnosis = self.load_diagnosis()
            except FileNotFoundError:
                return df
            for disease_name, cfg in self.diseases.items():
                event_col = f"has_{disease_name}"
                if event_col in df.columns:
                    continue
                dx_mask = diagnosis["diagnosisstring"].str.contains(
                    cfg["dx_pattern"], case=False, na=False
                )
                icd_mask = diagnosis["icd9code"].str.contains(
                    cfg["icd_pattern"], na=False
                )
                earliest = (
                    diagnosis.loc[dx_mask | icd_mask]
                    .groupby("patientunitstayid")["diagnosisoffset"]
                    .min()
                )
                df[event_col] = df["patientunitstayid"].isin(earliest.index)
                df[f"{disease_name}_offset"] = df["patientunitstayid"].map(earliest)

        return df


# ── CHS (Clalit Health Services) Dataset ─────────────────────

class CHSDataset(BaseDataset):

    name = "chs"
    time_unit = "days"
    exclude_labs = []
    # main run only: the covariate-ablation arms would need their four extra
    # checkpoints carried into Clalit and four more CPU passes (decided 2026-09-01)
    run_ids = [NORMA_RUN_ID]
    primary_norma_run = NORMA_RUN_ID

    outcomes = {
        "mortality": {
            "event_col": "died_10yr",
            "time_col": "death_days",
            "censor_col": "followup_days",
            "exclude_col": None,
        },
        "t2d": {
            "event_col": "has_t2d",
            "time_col": "t2d_days",
            "censor_col": "followup_days",
            "exclude_col": "has_t2d",
        },
        "ckd": {
            "event_col": "has_ckd",
            "time_col": "ckd_days",
            "censor_col": "followup_days",
            "exclude_col": "has_ckd",
        },
        "anemia_unspecified": {
            "event_col": "has_anemia_unspecified",
            "time_col": "anemia_unspecified_days",
            "censor_col": "followup_days",
            "exclude_col": "has_anemia_unspecified",
        },
    }

    primary_outcomes = ["mortality", "t2d", "ckd", "anemia_unspecified"]
    mortality_outcome = "mortality"
    eval_windows = [365, 1095, 1825, 3650]  # days: 1yr, 3yr, 5yr, 10yr

    PATIENT_COLS = {
        "patient_id": "patient_id",
        "gender": "gender",
        "age": "age",
    }
    LAB_COLS = {
        "patient_id": "patient_id",
        "lab_code": "analyte",
        "value": "value",
        "timestamp": "timestamp",
    }
    OUTCOME_COLS = {"patient_id": "patient_id"}

    def __init__(self, data_root=None, results_dir=None, figures_dir=None, n_chunks=None,
                 analytes=None, chunk=None, **kwargs):
        chs_root = os.path.join(VAL_DIR, "data", "clalit")
        sandbox = os.path.join(chs_root, "sandbox")
        if data_root:
            self.data_root = data_root
        elif glob.glob(os.path.join(chs_root, "chunk_*")):
            self.data_root = chs_root
        else:
            self.data_root = sandbox
        # --chunk i scopes the whole cohort to one chunk directory, so the heavy
        # per-pair stages (04_refs) hold one chunk in memory and write straight
        # into chunk_i/ where every CHS reader already looks.
        self.chunk = chunk
        self.data_dir = (os.path.join(self.data_root, f"chunk_{chunk}")
                         if chunk is not None else self.data_root)
        if chunk is not None and not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"No such chunk directory: {self.data_dir}")
        self._is_sandbox = (self.data_root == sandbox)
        # Count total available chunks before limiting
        all_dirs = sorted(glob.glob(os.path.join(self.data_root, "chunk_*")))
        self._total_chunks = len(all_dirs) if all_dirs else 0
        self.n_chunks = n_chunks  # None = use all
        self._is_subset = (n_chunks is not None and n_chunks < self._total_chunks)
        self._analytes = analytes

        self.results_dir = results_dir or output_dirs("chs")
        self.figures_dir = figures_dir          # unused: figures are built by make_figures.py
        os.makedirs(self.results_dir, exist_ok=True)

    def _output_suffix(self):
        if self._is_subset:
            return f"{self.name}_{self.n_chunks}chunks"
        return "sandbox" if self._is_sandbox else self.name

    def output_sub(self):
        return self._output_suffix()

    def _chunk_dirs(self):
        if self.chunk is not None:
            return [self.data_dir]
        dirs = sorted(glob.glob(os.path.join(self.data_root, "chunk_*")), key=_natural_sort_key)
        if not dirs:
            dirs = sorted(glob.glob(os.path.join(self.data_root, "*/")), key=_natural_sort_key)
        if not dirs:
            raise FileNotFoundError(
                f"No chunk directories found in {self.data_root}"
            )
        if self.n_chunks is not None:
            dirs = dirs[:self.n_chunks]
        print(f"Found {len(dirs)} chunk directories")
        return dirs

    def _load_chunks_csv(self, filename, col_map=None, usecols=None):
        frames = []
        for d in self._chunk_dirs():
            path = find_in(d, filename)
            if os.path.exists(path):
                frames.append(pd.read_csv(path, usecols=usecols))
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if col_map:
            rename = {v: k for k, v in col_map.items() if v != k}
            if rename:
                df = df.rename(columns=rename)
        return df

    def _standardize(self, df):
        """Encode sex as int and convert datetime timestamps to days."""
        if "gender" in df.columns and "sex" not in df.columns:
            df["sex"] = (df["gender"] == "F").astype(int)
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            min_t = df["timestamp"].min()
            df["timestamp"] = (df["timestamp"] - min_t).dt.days
        return fix_analyte(df)

    def load_processed(self):
        print("Loading CHS patient data ...")
        patients = self._load_chunks_csv("patient.csv", self.PATIENT_COLS)
        patients = patients.drop_duplicates(subset=["patient_id"])

        print("Loading CHS lab data ...")
        labs = self._load_chunks_csv("labs.csv", self.LAB_COLS)

        print("Merging ...")
        df = labs.merge(patients, on="patient_id", how="left")
        print(
            f"  {len(df):,} measurements, "
            f"{df['patient_id'].nunique():,} patients"
        )
        return self._filter_analytes(self._standardize(df))

    def _load_from_chunks(self, filename, label, postprocess=None, usecols=None):
        dirs = self._chunk_dirs()
        frames = []
        cols = usecols if isinstance(usecols, list) else None
        for i, d in enumerate(dirs):
            p = find_in(d, filename)
            if os.path.exists(p):
                frames.append(pd.read_parquet(p, columns=cols))
            if (i + 1) % 50 == 0 or i == len(dirs) - 1:
                print(f"    {i + 1}/{len(dirs)} {label} loaded")
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if postprocess:
            df = postprocess(df)
        return self._filter_analytes(df)

    def load_index_labs(self):
        df = self._load_from_chunks(
            "index_labs.parquet", "index_labs",
            postprocess=self._standardize,
        )
        if df is None:
            raise FileNotFoundError(
                f"No index_labs.parquet in {self.data_root}/chunk_*/"
            )
        df.attrs["time_unit"] = self.time_unit
        return df

    def load_ref_intervals(self):
        frames = []
        dirs = self._chunk_dirs()
        for i, d in enumerate(dirs):
            part = read_ref_intervals(d)
            if part is not None:
                frames.append(part)
            if (i + 1) % 50 == 0 or i == len(dirs) - 1:
                print(f"    {i + 1}/{len(dirs)} ref_intervals loaded")
        if not frames:
            raise FileNotFoundError(
                f"No {REF_FILE} in {self.data_root}/chunk_*/"
            )
        df = self._filter_analytes(_coerce_ri(fix_analyte(pd.concat(frames, ignore_index=True))))
        return self._filter_methods(df)

    def classification_dir(self):
        """CHS keeps its classification per chunk, next to that chunk's index_labs."""
        return self.data_dir

    def load_classification(self, usecols=None):
        frames = []
        dirs = self._chunk_dirs()
        for i, d in enumerate(dirs):
            part = read_classification(d, usecols)
            if part is not None:
                frames.append(part)
            if (i + 1) % 50 == 0 or i == len(dirs) - 1:
                print(f"    {i + 1}/{len(dirs)} classification loaded")
        if not frames:
            raise FileNotFoundError(
                f"No {CLASSIFICATION_FILE} in {self.data_root}/chunk_*/"
            )
        return self._filter_norma(
            self._filter_analytes(fix_analyte(pd.concat(frames, ignore_index=True))))

    def norma_predictions_path(self):
        """Per chunk, next to that chunk's index_labs — 04_refs.py (norma step) runs
        chunk by chunk on CPU inside Clalit, so there is no cohort-wide file."""
        return stage_path(self.data_dir, self.NORMA_PREDICTIONS_FILE, REFS_STAGE)

    def load_norma_predictions(self, columns=None):
        if self.no_norma:
            return None
        return self._load_from_chunks(
            self.NORMA_PREDICTIONS_FILE, "norma_predictions",
            postprocess=fix_analyte, usecols=columns)

    def classification_columns(self):
        """Schema of the first chunk that has one — the chunks share a layout."""
        for d in self._chunk_dirs():
            names = classification_column_names(d)
            if names is not None:
                return names
        raise FileNotFoundError(
            f"No {CLASSIFICATION_FILE} in {self.data_root}/chunk_*/"
        )

    def load_diagnosis(self):
        df = self._load_from_chunks("diagnosis.parquet", "diagnosis")
        if df is None:
            raise FileNotFoundError(
                f"No diagnosis.parquet in {self.data_root}/chunk_*/"
            )
        return df

    def iter_chunks(self):
        dirs = sorted(glob.glob(os.path.join(self.data_root, "chunk_*")), key=_natural_sort_key)
        if not dirs:
            raise FileNotFoundError(f"No chunk_* in {self.data_root}")
        if self.n_chunks is not None:
            dirs = dirs[:self.n_chunks]
        print(f"  Streaming {len(dirs)} chunks from {self.data_root}")
        for i, d in enumerate(dirs):
            sp_path = os.path.join(d, "index_labs.parquet")
            index_labs = self._filter_analytes(self._standardize(pd.read_parquet(sp_path))) if os.path.exists(sp_path) else None
            raw_ref = read_ref_intervals(d)
            ref_df = self._filter_analytes(_coerce_ri(fix_analyte(raw_ref))) if raw_ref is not None else None
            if ref_df is not None and self.no_norma:
                ref_df = ref_df[~ref_df["method"].map(is_norma_method)]
            yield i, d, index_labs, ref_df
            if (i + 1) % 50 == 0 or i == len(dirs) - 1:
                print(f"    {i + 1}/{len(dirs)} chunks processed")

    def attach_outcomes(self, df):
        """Derive event flags and time-to-event from CHS diagnosis data."""
        print("Loading CHS outcome data ...")
        diagnosis = self.load_diagnosis()
        diagnosis = diagnosis.drop_duplicates(subset=["patient_id"])

        # Reference date for followup (baseline cutoff)
        ref_date = pd.Timestamp("2015-01-01")

        # Mortality: died_10yr, death_days, followup_days
        if "death_date" in diagnosis.columns:
            diagnosis["death_date"] = pd.to_datetime(diagnosis["death_date"])
            diagnosis["membership_end"] = pd.to_datetime(diagnosis["membership_end"])
            censor_date = diagnosis["membership_end"].fillna(pd.Timestamp("2025-01-01"))
            diagnosis["death_days"] = (diagnosis["death_date"] - ref_date).dt.days
            diagnosis["followup_days"] = (censor_date - ref_date).dt.days
            diagnosis["died_5yr"] = (
                diagnosis["death_date"].notna()
                & (diagnosis["death_days"] <= 1825)
            )
            diagnosis["died_10yr"] = (
                diagnosis["death_date"].notna()
                & (diagnosis["death_days"] <= 3650)
            )

        # Disease outcomes: convert date columns to has_X / X_days
        for outcome_name in ["t2d", "ckd", "anemia_unspecified"]:
            if outcome_name in diagnosis.columns:
                diagnosis[outcome_name] = pd.to_datetime(diagnosis[outcome_name])
                diagnosis[f"has_{outcome_name}"] = diagnosis[outcome_name].notna()
                diagnosis[f"{outcome_name}_days"] = (
                    diagnosis[outcome_name] - ref_date
                ).dt.days

        return df.merge(diagnosis, on="patient_id", how="left")


# ── INSPIRE Dataset ────────────────────────────────────────────

class INSPIREDataset(BaseDataset):

    name = "inspire"
    time_unit = "days"
    outcome_time_unit = "minutes"   # inhosp_death_time / discharge_time / icuin_time are minutes
    exclude_labs = EXCLUDE_LABS
    run_ids = [NORMA_RUN_ID, *NORMA_ABLATION_RUN_IDS]
    primary_norma_run = NORMA_RUN_ID

    outcomes = {
        "mortality": {
            "event_col": "died_in_hospital",
            "time_col": "inhosp_death_time",
            "censor_col": "discharge_time",
            "exclude_col": None,
        },
        "prolonged_los": {
            "event_col": "prolonged_los",
            "time_col": "discharge_time",
            "censor_col": "discharge_time",
            "exclude_col": None,
            "survival": False,   # see the eICU entry: the event is defined by the duration
        },
        "unplanned_icu": {
            "event_col": "unplanned_icu",
            "time_col": "icuin_time",
            "censor_col": "discharge_time",
            "exclude_col": None,
        },
        "periop_infection": {
            "event_col": "has_periop_infection",
            "time_col": "periop_infection_offset",
            "censor_col": "discharge_time",
            "exclude_col": None,
        },
    }

    primary_outcomes = ["mortality", "prolonged_los", "unplanned_icu", "periop_infection"]
    mortality_outcome = "mortality"
    eval_windows = [1, 2, 7, 14, 30]  # days

    # ICD-10 patterns (INSPIRE uses ICD-10-CM, not ICD-9)
    diseases = {
        "diabetes": {
            "icd_pattern": r"^E1[01]",
        },
        "ckd": {
            "icd_pattern": r"^N18",
        },
        "aki": {
            "icd_pattern": r"^N17",
        },
        "sepsis": {
            "icd_pattern": r"^A4[01]|^R65\.2",
        },
        "chf": {
            "icd_pattern": r"^I50",
        },
        "afib": {
            "icd_pattern": r"^I48",
        },
        "hypertension": {
            "icd_pattern": r"^I10",
        },
        "copd": {
            "icd_pattern": r"^J4[1234]",
        },
        "pneumonia": {
            "icd_pattern": r"^J1[2345678]",
        },
        "liver_disease": {
            "icd_pattern": r"^K7[0-6]",
        },
        "stroke": {
            "icd_pattern": r"^I6[0-4]",
        },
        "periop_infection": {
            "icd_pattern": r"^A4[01]|^R65\.2|^T81\.[34]|^J1[2-8]|^K65|^N39\.0",
        },
    }

    def __init__(self, data_dir=None, cache_dir=None, analytes=None, **kwargs):
        self.data_dir = data_dir or globals()["data_dir"]("inspire")
        self._analytes = analytes
        self.cache_dir = cache_dir or output_dirs("inspire")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _standardize(self, df):
        """Rename INSPIRE columns to standard names and encode sex as int."""
        rename = {}
        if "subject_id" in df.columns and "patient_id" not in df.columns:
            rename["subject_id"] = "patient_id"
        if "lab_code" in df.columns and "analyte" not in df.columns:
            rename["lab_code"] = "analyte"
        needs_time_convert = "chart_time" in df.columns and "timestamp" not in df.columns
        if needs_time_convert:
            rename["chart_time"] = "timestamp"
        if rename:
            df = df.rename(columns=rename)
        # Encode sex as int (0=M, 1=F)
        if "sex" in df.columns and df["sex"].dtype == object:
            df["sex"] = (df["sex"] == "F").astype(int)
        # Convert minutes → days, then re-anchor to 0 per patient-analyte.
        # Guard: only convert if we just renamed from chart_time (raw minutes),
        # not when loading already-standardized parquets.
        if needs_time_convert and "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"] / (60 * 24)
            if "patient_id" in df.columns and "analyte" in df.columns:
                df["timestamp"] = df["timestamp"] - df.groupby(["patient_id", "analyte"])["timestamp"].transform("min")
        return fix_analyte(df)

    def load_processed(self):
        p = os.path.join(self.data_dir, "processed.parquet")
        if os.path.exists(p):
            print(f"  Loading {p}")
            return self._filter_analytes(self._standardize(pd.read_parquet(p)))
        raise FileNotFoundError(f"No processed.parquet found at {p}")

    def load_diagnosis(self):
        p = os.path.join(self.data_dir, "diagnosis.parquet")
        if os.path.exists(p):
            print(f"  Loading {p}")
            return pd.read_parquet(p)
        raise FileNotFoundError(f"No diagnosis.parquet found at {p}")

    def output_sub(self):
        return "inspire"

    def iter_chunks(self):
        index_labs = self.load_index_labs()
        ref_df = self.load_ref_intervals()
        cache_dir = self.setup_output()
        yield 0, cache_dir, index_labs, ref_df

    def get_mortality(self):
        """Derive mortality columns from operations data."""
        ops_path = os.path.join(self.data_dir, "processed.parquet")
        df = pd.read_parquet(ops_path)
        # Deduplicate to one row per patient
        patient = df.drop_duplicates(subset=["subject_id"], keep="first")
        patient["died_in_hospital"] = patient["inhosp_death_time"].notna()
        cols = [
            "subject_id", "died_in_hospital", "inhosp_death_time",
            "allcause_death_time", "discharge_time",
            "icuin_time", "icuout_time",
        ]
        return patient[[c for c in cols if c in patient.columns]].copy()

    def attach_outcomes(self, df):
        if "died_in_hospital" not in df.columns and "patient_id" in df.columns:
            mort = self.get_mortality()
            mort = mort.rename(columns={"subject_id": "patient_id"})
            cols = [c for c in mort.columns
                    if c not in df.columns or c == "patient_id"]
            df = df.merge(mort[cols], on="patient_id", how="left")

        if "prolonged_los" not in df.columns and "discharge_time" in df.columns:
            df["prolonged_los"] = df["discharge_time"] > (7 * 24 * 60)

        # Unplanned ICU: admitted to ICU > 24h after hospital admission
        if "unplanned_icu" not in df.columns and "icuin_time" in df.columns:
            df["unplanned_icu"] = df["icuin_time"].notna() & (df["icuin_time"] > 24 * 60)

        if "patient_id" in df.columns:
            try:
                diagnosis = self.load_diagnosis()
            except FileNotFoundError:
                return df
            # INSPIRE diagnosis has: subject_id, chart_time, icd10_cm
            diagnosis = diagnosis.rename(columns={"subject_id": "patient_id"})
            for disease_name, cfg in self.diseases.items():
                event_col = f"has_{disease_name}"
                if event_col in df.columns:
                    continue
                icd_mask = diagnosis["icd10_cm"].str.contains(
                    cfg["icd_pattern"], na=False
                )
                earliest = (
                    diagnosis.loc[icd_mask]
                    .groupby("patient_id")["chart_time"]
                    .min()
                )
                df[event_col] = df["patient_id"].isin(earliest.index)
                df[f"{disease_name}_offset"] = df["patient_id"].map(earliest)

        return df


# ── Development cohorts (EHRSHOT, MIMIC-IV test split) ───────────
# NORMA's held-out test patients, run through the same baseline/index pipeline
# as the validation cohorts so every cohort is evaluated with the same models
# (process/dev_cohort.py --source {mimiciv,ehrshot} build processed.parquet).  No outcomes: forecasting +
# reference intervals only.

class DevDataset(BaseDataset):

    time_unit = "days"
    exclude_labs = EXCLUDE_LABS
    run_ids = [NORMA_RUN_ID, *NORMA_ABLATION_RUN_IDS]
    primary_norma_run = NORMA_RUN_ID
    outcomes = {}
    primary_outcomes = []
    mortality_outcome = None
    diseases = {}
    eval_windows = []

    def __init__(self, data_dir=None, cache_dir=None, analytes=None, **kwargs):
        self.data_dir = data_dir or os.path.join(VAL_DIR, "data", self.name)
        self._analytes = analytes
        self.cache_dir = cache_dir or output_dirs(self.name)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _standardize(self, df):
        return fix_analyte(df)

    def load_processed(self):
        p = os.path.join(self.data_dir, "processed.parquet")
        if os.path.exists(p):
            print(f"  Loading {p}")
            return self._filter_analytes(self._standardize(pd.read_parquet(p)))
        raise FileNotFoundError(f"No processed.parquet at {p}; run process/dev_cohort.py --source {self.name}")

    def attach_outcomes(self, df):
        return df

    def iter_chunks(self):
        index_labs = self.load_index_labs()
        ref_df = self.load_ref_intervals()
        cache_dir = self.setup_output()
        yield 0, cache_dir, index_labs, ref_df


class EHRSHOTDataset(DevDataset):
    name = "ehrshot"


class MIMICIVDataset(DevDataset):
    name = "mimiciv"


# ── Registry ─────────────────────────────────────────────────

DATASETS = {
    "eicu": EICUDataset,
    "chs": CHSDataset,
    "inspire": INSPIREDataset,
    "ehrshot": EHRSHOTDataset,
    "mimiciv": MIMICIVDataset,
}


def save_csv(df, path, analytes=None, keys=None):
    """Save to CSV, replacing only the rows this run recomputed.

    Two things scope a run: the `analytes` filter, and `keys` -- the factor
    columns of a consolidated file (one file per analysis with a level column,
    rather than one file per level: "outcome" for incidence.csv, "split" for
    cohort.csv).  Rows of the existing file that match the incoming values on
    those columns are dropped and the rest kept, so `--outcomes mortality` or
    `--analytes HGB` updates its slice instead of wiping the file.
    """
    path = os.path.join(os.path.dirname(path), stage_file(path))
    drop_on = (["analyte"] if analytes and "analyte" in df.columns else []) \
              + [k for k in (keys or ()) if k in df.columns]
    if drop_on and os.path.exists(path) and os.path.getsize(path):
        existing = pd.read_csv(path, keep_default_na=False)
        if all(c in existing.columns for c in drop_on):
            incoming = set(map(tuple, df[drop_on].astype(str).drop_duplicates().to_numpy()))
            keep = ~existing[drop_on].astype(str).apply(tuple, axis=1).isin(incoming)
            df = pd.concat([existing[keep], df], ignore_index=True)
            df = df.sort_values(drop_on).reset_index(drop=True)
    df.to_csv(path, index=False)


def add_dataset_args(parser, required=True):
    parser.add_argument("--dataset", required=required, choices=list(DATASETS.keys()))
    parser.add_argument("--force", action="store_true",
                        help="recompute even where the output is already written "
                             "(the default is to reuse it; see datasets.already_done)")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Limit to the first N chunks (CHS only, default: all)")
    parser.add_argument("--analytes", type=str, default=None,
                        help="Comma-separated list of analytes to process (default: all)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="CHS only: scope the run to a single chunk_<i> directory")
    parser.add_argument("--no_norma", action="store_true",
                        help="baselines only: drop every NORMA arm, so the stage runs "
                             "on a cohort the model was never applied to")
    return parser


def already_done(args, results_dir, *names, label=None):
    """True when every one of `names` is already written and --force was not passed.

    Reuse is the DEFAULT: a rerun that only exists to carry the pipeline forward
    should not refit what it already fitted (13_cox re-estimating every landmark
    Cox, 14_patient_level re-training the multi-analyte models) just to write the
    same numbers back.  --force is the one escape hatch, the same bargain
    07_classify has always had.

    Call this at the TOP of a step, before it loads anything -- a guard placed
    after the 1.5 GB classification has been read saves almost nothing.

    The corollary of reuse-by-default: a stage whose upstream input changed keeps
    its stale output until someone passes --force.
    """
    if getattr(args, "force", False):
        return False
    missing = [n for n in names if not os.path.exists(find_in(results_dir, n))]
    what = label or ", ".join(names)
    if missing:
        print(f"  {what}: {', '.join(missing)} not written yet, computing")
        return False
    print(f"  {what}: already written, skipping (use --force to redo)")
    return True


def get_dataset(args):
    cls = DATASETS[args.dataset]
    kwargs = {}
    if args.n_chunks is not None:
        kwargs["n_chunks"] = args.n_chunks
    if getattr(args, "analytes", None) is not None:
        kwargs["analytes"] = [a.strip() for a in args.analytes.split(",")]
    if getattr(args, "chunk", None) is not None:
        kwargs["chunk"] = args.chunk
    ds = cls(**kwargs)
    if getattr(args, "no_norma", False):
        disable_norma(ds)
    return ds


def disable_norma(ds):
    """--no_norma: strip every NORMA arm off a dataset, so `methods`, `norma_methods`
    and `_filter_methods` behave as if the model had never been run on this cohort.
    One switch: every stage derives its method list from `ds.methods`."""
    ds.no_norma = True
    ds.run_ids = []
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# Per-chunk result caching for the chunked cohort (CHS)
# ═══════════════════════════════════════════════════════════════════════════

# A stage that aggregates counts over the classification table cannot hold every
# chunk at once, so it computes one small frame per chunk and caches it next to
# the chunk (`chunk_dir/<cache_file>`).  A rerun reads the caches; `--analytes`
# recomputes only those analytes and patches them into the cache; `--force`
# recomputes everything.

def read_chunk_classification(chunk_dir):
    """The chunk's classification table (both halves, or the older single csv), or None."""
    df = read_classification(chunk_dir)
    if df is not None:
        return df
    csv = os.path.join(chunk_dir, "classification.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv, keep_default_na=False, na_values=[""])
    return None


def cached_chunk_frames(ds, cache_file, compute, force=False):
    """Yield one result frame per chunk.

    compute(chunk_dir) -> DataFrame with an `analyte` column (or None when the chunk
    has no classification).  Frames come from the cache when it exists, except for
    the analytes in ds._analytes, which are recomputed and written back.
    """
    analytes = ds._analytes
    for chunk_dir in ds._chunk_dirs():
        name = os.path.basename(chunk_dir)
        path = os.path.join(chunk_dir, cache_file)
        cached = None
        if os.path.exists(path) and not force:
            cached = pd.read_parquet(path)

        if cached is not None and analytes is None:
            print(f"    {name}: cached")
            yield cached
            continue

        fresh = compute(chunk_dir)
        if fresh is None or fresh.empty:
            if cached is not None:
                yield cached
            continue

        if cached is not None and analytes is not None:
            kept = cached[~cached["analyte"].isin(analytes)]
            fresh = pd.concat([kept, fresh], ignore_index=True)
        fresh.to_parquet(path, index=False)
        print(f"    {name}: {'updated' if analytes else 'computed'}")
        yield fresh


# ═══════════════════════════════════════════════════════════════════════════
# classification.parquet I/O — two files, aligned row for row
# ═══════════════════════════════════════════════════════════════════════════

# 07_classify writes the baseline half and the NORMA half as separate files
# (2026-09-08).  Adding or retraining a NORMA arm then rewrites ~0.4 GB instead of
# the whole 1.5 GB frame -- that growth (766 MB -> 1.5 GB on eICU) is what the
# covariate ablation cost.  The halves are ALIGNED BY POSITION, never joined:
# (patient_id, analyte, timestamp) is not unique (27,833 duplicate triples on
# eICU), so a merge would fan out.  The NORMA half repeats those three columns
# only so a read can assert the halves still line up.
CLASSIFICATION_FILE = "classification.parquet"
CLASSIFICATION_NORMA_FILE = "classification_norma.parquet"
CLASSIFICATION_KEYS = ["patient_id", "analyte", "timestamp"]


def is_norma_column(col):
    """True for a NORMA arm's column: norma_<run>_ri_low, NORMA_<run>_{class,z,zs}."""
    return col.startswith("norma_") or col.startswith("NORMA_")


def split_classification(df):
    """(baseline half, NORMA half) of a full classification frame."""
    norma = [c for c in df.columns if is_norma_column(c)]
    keys = [c for c in CLASSIFICATION_KEYS if c in df.columns]
    return df.drop(columns=norma), df[keys + norma]


def classification_paths(directory):
    """07_classify's two halves, stage-prefixed like every other result -- except
    that a CHS chunk carries the bare names the Clalit bundle shipped, so an
    existing bare file wins.  One rule for both reading and writing."""
    out = []
    for f in (CLASSIFICATION_FILE, CLASSIFICATION_NORMA_FILE):
        prefixed = result_path(directory, f, "07_classify")
        bare = os.path.join(directory, f)
        out.append(bare if not os.path.exists(prefixed) and os.path.exists(bare) else prefixed)
    return tuple(out)


def _align_check(base, norma, where):
    """The two halves must be the same rows in the same order."""
    if len(base) != len(norma):
        raise ValueError(f"{where}: classification halves disagree on length "
                         f"({len(base):,} vs {len(norma):,}) — rerun 07_classify")
    shared = [c for c in CLASSIFICATION_KEYS if c in base.columns and c in norma.columns]
    for c in shared:
        if not base[c].reset_index(drop=True).equals(norma[c].reset_index(drop=True)):
            raise ValueError(f"{where}: classification halves are out of order on "
                             f"'{c}' — rerun 07_classify")


def read_classification(directory, usecols=None):
    """Read both halves out of `directory` and glue them side by side, or None if
    the baseline half is absent.  `usecols` is split across the two files."""
    base_path, norma_path = classification_paths(directory)
    if not os.path.exists(base_path):
        return None
    want = usecols if isinstance(usecols, list) else None
    have_norma = os.path.exists(norma_path)

    if want is None:
        base = pd.read_parquet(base_path)
        if not have_norma:
            return base
        norma = pd.read_parquet(norma_path)
        _align_check(base, norma, directory)
        return pd.concat([base, norma.drop(columns=[c for c in CLASSIFICATION_KEYS
                                                    if c in norma.columns])], axis=1)

    base_cols = [c for c in want if not is_norma_column(c)]
    norma_cols = [c for c in want if is_norma_column(c)]
    if not norma_cols:
        return pd.read_parquet(base_path, columns=base_cols)
    if not have_norma:
        raise FileNotFoundError(f"No {CLASSIFICATION_NORMA_FILE} in {directory}, "
                                f"but {norma_cols[:3]} were requested")
    # read the keys alongside each half so the alignment check has something to compare
    keys = CLASSIFICATION_KEYS
    base = pd.read_parquet(base_path, columns=list(dict.fromkeys(keys + base_cols)))
    norma = pd.read_parquet(norma_path, columns=list(dict.fromkeys(keys + norma_cols)))
    _align_check(base, norma, directory)
    out = pd.concat([base, norma[norma_cols]], axis=1)
    return out[want]


def write_classification(df, directory, atomic=None):
    """Write both halves into `directory`.  `atomic` is 07_classify's writer."""
    os.makedirs(directory, exist_ok=True)
    base, norma = split_classification(df)
    base_path, norma_path = classification_paths(directory)
    write = atomic or (lambda d, p: d.to_parquet(p, index=False))
    write(base, base_path)
    write(norma, norma_path)
    print(f"  Saved {len(base):,} rows x {base.shape[1]} baseline columns -> {base_path}")
    print(f"         {len(norma):,} rows x {norma.shape[1] - len(CLASSIFICATION_KEYS)}"
          f" NORMA columns -> {norma_path}")
    return base_path, norma_path


def classification_column_names(directory):
    """Column names of both halves without reading any data."""
    import pyarrow.parquet as pq
    base_path, norma_path = classification_paths(directory)
    if not os.path.exists(base_path):
        return None
    names = set(pq.ParquetFile(base_path).schema_arrow.names)
    if os.path.exists(norma_path):
        names |= set(pq.ParquetFile(norma_path).schema_arrow.names)
    return names


# ═══════════════════════════════════════════════════════════════════════════
# ref_intervals.{csv,parquet} I/O (04_refs norma + baselines steps)
# ═══════════════════════════════════════════════════════════════════════════

# Two files per cohort (2026-09-08): ref_intervals.parquet holds the baselines
# (base/pop/per/cohen_*/gaussian_*) and ref_intervals_norma.parquet the norma_<run_id>
# arms.  Each 04_refs step owns one file and never reads or rewrites the other, so
# the norma step and the baselines step can now run at the same time on one cohort --
# with a single file the last writer won, and the two could only be run in sequence.

REF_FILE = "ref_intervals.parquet"
REF_NORMA_FILE = "ref_intervals_norma.parquet"
REF_COLUMNS = ["patient_id", "analyte", "sex", "age", "n_bl", "t_span",
               "method", "ri_mean", "ri_std", "ri_low", "ri_high"]
NUMERIC = ["ri_low", "ri_high", "ri_mean", "ri_std", "age"]


def is_norma_method(method):
    """A NORMA arm's rows — method is "norma_<run_id>"."""
    return str(method).startswith("norma")


def ref_half(methods):
    """Which file owns `methods`: "norma", "baselines", or None if they straddle both."""
    kinds = {is_norma_method(m) for m in methods}
    if kinds == {True}:
        return "norma"
    if kinds == {False}:
        return "baselines"
    return None


def _ref_dir(ds):
    """Where this dataset's ref_intervals live.  `--chunk i` scopes CHS to one
    chunk directory, and that is where every CHS reader looks for them
    (CHSDataset.load_ref_intervals, iter_chunks, run_clalit.baselines_done), so
    a chunk-scoped run must not write the cohort-level file instead."""
    return ds.data_dir if getattr(ds, "chunk", None) is not None else results_dir(ds.output_sub())


def ref_paths(ds, directory=None):
    """(baselines path, NORMA path) for this cohort, or inside `directory` (a CHS chunk)."""
    if directory is None and getattr(ds, "chunk", None) is not None:
        directory = ds.data_dir
    d = directory if directory is not None else results_dir(ds.output_sub())
    if directory is not None:          # a CHS chunk keeps the bare names
        return os.path.join(d, REF_FILE), os.path.join(d, REF_NORMA_FILE)
    return (result_path(d, REF_FILE, "04_refs"),
            result_path(d, REF_NORMA_FILE, "04_refs"))


def _sex_as_int(sex):
    """One encoding per file: 1 = female, 0 = male, as every cohort but the legacy
    CHS chunks writes it.  Mixing 'F'/'M' rows with 0/1 rows breaks the parquet write."""
    if sex.dtype == object:
        mapped = sex.astype(str).str.upper().str[0].map({"F": 1, "M": 0})
        sex = mapped.where(mapped.notna(), pd.to_numeric(sex, errors="coerce"))
    return pd.to_numeric(sex, errors="coerce").astype("Int64")


def _read_ref_file(path):
    """One ref_intervals file (parquet, or the pre-2026-08 csv twin), or None."""
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        csv = path.replace(".parquet", ".csv")
        if not os.path.exists(csv):
            return None
        df = pd.read_csv(csv, keep_default_na=False, low_memory=False)
    df["analyte"] = df["analyte"].replace("", "NA").fillna("NA")
    return df


def read_ref_intervals(directory, half=None):
    """Both halves out of `directory`, concatenated, or None.  `half` reads one."""
    base_p = stage_path(directory, REF_FILE, REFS_STAGE)
    norma_p = stage_path(directory, REF_NORMA_FILE, REFS_STAGE)
    want = {"baselines": [base_p], "norma": [norma_p]}.get(half, [base_p, norma_p])
    frames = [d for d in (_read_ref_file(p) for p in want) if d is not None]
    if not frames:
        return None
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def read_ref_intervals_raw(ds, half=None):
    """Every row of the cohort's ref_intervals (no method / analyte filtering), or None.
    Chunk-scoped on CHS when --chunk is given, matching where ref_paths writes."""
    return read_ref_intervals(_ref_dir(ds), half)


def write_ref_intervals(ds, ref_df, half=None, directory=None):
    """Write the cohort's ref_intervals.  `half=None` splits `ref_df` by method and
    writes both files; `half="baselines"` / `"norma"` writes only that file, keeping
    `ref_df`'s rows of that half and leaving the other file untouched."""
    base_p, norma_p = ref_paths(ds, directory)
    os.makedirs(os.path.dirname(base_p), exist_ok=True)
    ref_df = ref_df.copy()
    for col in NUMERIC:
        if col in ref_df.columns:
            ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")
    if "sex" in ref_df.columns:
        ref_df["sex"] = _sex_as_int(ref_df["sex"])
    is_norma = ref_df["method"].map(is_norma_method)
    for name, path, part in (("norma", norma_p, ref_df[is_norma]),
                             ("baselines", base_p, ref_df[~is_norma])):
        if half is not None and half != name:
            continue
        part.to_parquet(path, index=False)
        print(f"Saved {len(part):,} {name} ref intervals to {path}")


def upsert_ref_rows(ds, rows, methods, analytes=None):
    """Replace the rows of `methods` (optionally only for `analytes`) with `rows`
    and write that half back; the other half's file is never opened."""
    rows = rows[REF_COLUMNS]
    half = ref_half(methods)
    existing = read_ref_intervals_raw(ds, half)
    if existing is None:
        out = rows
    else:
        drop = existing["method"].isin(set(methods))
        if analytes:
            drop &= existing["analyte"].isin(set(analytes))
        n_drop = int(drop.sum())
        if n_drop:
            print(f"  replacing {n_drop:,} existing rows of {sorted(set(methods))}")
        out = pd.concat([existing[~drop], rows], ignore_index=True)
    write_ref_intervals(ds, out, half)
    return out
