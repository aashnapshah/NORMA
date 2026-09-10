"""Constants the analysis and its figures have to agree on.

Every stage script (`scripts/<nn>_<stage>.py`) now holds both the analysis and
the figure/table code, so a threshold typed twice is a threshold that can
disagree with itself.  Anything read by both halves, or by more than one stage,
lives here and nowhere else: the cohort sets and the order they are drawn in,
the analyte exclusions, and the cut-offs that decide what is computed and what
is shown.

Three other modules own the rest, and nothing here duplicates them:

    lib/datasets.py     paths, run ids, cohort loaders
    lib/models.py       model / cohort display names, colours, markers
    process/config.py   the analyte dictionary: LOINC codes, units, population
                        reference intervals, the test vocabulary

This module imports nothing, so it can be read from any of them.  It is not
called config.py because the model code (model/data.py, sensitivity_analysis.py,
priors.py, shrinkage.py) imports process/config.py as a bare `import config`,
and lib/ sits ahead of process/ on sys.path.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Cohorts
# ═══════════════════════════════════════════════════════════════════════════

# Development cohorts: NORMA's own train/val/test data, scored on the test split.
DEV_COHORTS = ["ehrshot", "mimiciv"]

# External validation cohorts, in the order composite figures stack their rows.
VAL_COHORTS = ["eicu", "inspire", "chs"]

# The order per-dataset figures and tables are built in (one output per entry;
# `make_figures.py --dataset` restricts it).  Same set as VAL_COHORTS.
DATASET_ORDER = ["eicu", "chs", "inspire"]

# Every cohort with a row in the cohort table and the forecasting composites.
COHORT_ORDER = DEV_COHORTS + VAL_COHORTS

# Splits a cohort's rows are grouped into: baseline/index for the validation
# cohorts, NORMA's sequence split for the development cohorts.
# "all" is the whole-cohort level of a `split` column: a consolidated table
# carries the pooled rows and the per-split rows together, so the pooled ones
# need a name.  Same idea as figlib.OUTPUT_TAG "all".
ALL_SPLIT = "all"
# The `analyte` level of a row aggregated ACROSS analytes.  Not "all": these rows
# are the median over analytes, not a pooled recompute, and calling them "all"
# would read as the latter.  A consolidated table carries the per-analyte rows
# and these together, so readers filter on it.
MEDIAN_ROW = "median"
# The `analyte` level of a row computed on all analytes' rows POOLED (one metric
# over the union), as opposed to MEDIAN_ROW's median of per-analyte metrics.
POOLED_ROW = "pooled"
PSEUDO_ANALYTES = (MEDIAN_ROW, POOLED_ROW)
# The `method` level of rows binned on the deviation from the patient's own
# baseline period rather than on any reference-interval method's z.  Not
# "baseline": that is already a split level (baseline period vs index).
BASELINE_Z = "baseline_z"
SPLIT_ORDER = [ALL_SPLIT, "baseline", "index", "train", "val", "test"]
DEV_SPLITS = ["train", "val", "test"]

# ═══════════════════════════════════════════════════════════════════════════
# Analytes
# ═══════════════════════════════════════════════════════════════════════════

# Dropped before any analysis: too sparse in the validation cohorts to fit an
# interval on.  lib/datasets.py re-exports this as EXCLUDE_LABS.
EXCLUDE_LABS = ["CRP", "GGT", "LDH", "PT"]

# Dropped from the figures and tables only.  GLB (globulin) is computed by every
# stage but has never been drawn; keeping the difference explicit rather than
# hidden in a second literal is the point of this file.
FIGURE_ONLY_EXCLUDE = ["GLB"]
EXCLUDE_ANALYTES = set(EXCLUDE_LABS) | set(FIGURE_ONLY_EXCLUDE)

# ═══════════════════════════════════════════════════════════════════════════
# Thresholds
# ═══════════════════════════════════════════════════════════════════════════

FDR = 0.05             # Benjamini-Hochberg q: a "significant" HR (13_cox, 16_benchmark)

MIN_PATIENTS = 100     # smallest cell that is reported: fewer patients is noise
MIN_EVENTS = 10        # ... and it needs this many events to score an AUROC

# The flag rate every method is compared at, so a difference in performance is
# not a difference in how much each method flags (12_eval, 16_benchmark).
FLAG_RATE = 0.10
ANCHOR = f"{int(FLAG_RATE * 100)}"       # the column suffix: rr_at_10, sens_at_10

# Sensitivity the clinical-endpoint cut-offs are matched at (17_outcomes).
MATCHED_SENSITIVITY = 0.25

# The comparator that flags only what the population interval already flags.
STANDARD_OF_CARE = "StandardOfCare"

# ═══════════════════════════════════════════════════════════════════════════
# Figure conventions
# ═══════════════════════════════════════════════════════════════════════════

PENDING_H = 1.2        # inches given to a cohort row that has no results yet
SMALL_N = 50           # heatmap cells with fewer observations are greyed out
