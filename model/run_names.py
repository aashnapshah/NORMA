"""One place for what each NORMA training run is called.

Training runs are named by their id (model/logs/<run_id>); everything that
shows a run to a reader -- validation figures and tables (validation/lib/models.py),
the dev-set ablation summary (compare_ablation.py), evaluate.py -- takes its label
from here, so an arm is called the same thing everywhere.

The convention: "NORMA | <covariates the run uses>" in legends and
"NORMA (<covariates>)" in tables -- sex listed for every run, then the per-draw
covariates it adds (age at each draw, care setting, same-draw analytes).
Whichever run is the main model (validation/lib/config.NORMA_RUN_ID) is shown as
NORMA_RI in the main figures and as "NORMA (<covariates>) -- main" beside the arms.
"""

# What each run uses, as a reader would list it. Every run also carries age at
# the first draw and the analyte code on its context token (stated once in
# SHORT_KEY rather than repeated in every label).
RUN_COVARIATES = {
    "334f7e21": "sex",
    "q_age": "sex, age",
    "q_set": "sex, setting",
    "q_co": "sex, analytes",
    "q_age_set": "sex, age, setting",
    "q_age_co": "sex, age, analytes",
    "q_set_co": "sex, setting, analytes",
    "q_age_set_co": "sex, age, setting, analytes",
    "q_co_q": "sex, analytes (query)",
    # prior-anchored arms (run_prior_ablation.sh), all on the q_age_set covariates
    "pa_k5": "sex, age, setting; prior anchor k=5",
    "pa_k20": "sex, age, setting; prior anchor k=20",
    "pa_tau": "sex, age, setting; prior anchor k=5, tau=365d",
    "pf_k5": "sex, age, setting; width floor k=5",
    "pg_k5": "sex, age, setting; learned gate k=5",
    "pn_k5": "sex, age, setting; conjugate NIG head",
    "gk_k5": "sex, age, setting; Gaussian KL-aligned k=5",
    # patient-level split arms (run_patient_split.sh, R3 comment 11). Same
    # COMMON as the covariate ablation minus --use_age_t/--use_setting, plus
    # --split_by patient, so their reference point is p_base and not 334f7e21.
    "p_base": "sex; patient split",
    "p_co": "sex, analytes; patient split",
    "p_causal": "sex, causal mask; patient split",
    "p_full": "sex, every past draw, causal mask; patient split",
}

# the patient-split group compares only within itself (different test set)
PATIENT_SPLIT_ORDER = ["p_base", "p_co", "p_causal", "p_full"]

RUN_SHORT = {k: f"NORMA | {v}" for k, v in RUN_COVARIATES.items()}
SHORT_KEY = ("every run also uses age at the first draw and the analyte code; "
             "age = age at each draw, setting = care setting, analytes = the other analytes drawn at the same time")

# additive-ladder order for ablation tables
RUN_ORDER = ["334f7e21", "q_age", "q_set", "q_co", "q_age_set", "q_age_co", "q_set_co",
             "q_age_set_co", "q_co_q"]

# prior-anchored arms (run_prior_ablation.sh): the main model's covariates with a
# different loss or output head
PRIOR_ORDER = ["pa_k5", "pa_k20", "pa_tau", "pf_k5", "pg_k5", "pn_k5", "gk_k5"]

# Every arm trained since the covariate ablation began, in reading order, with the
# group each belongs to. One list, so a figure that wants to show what was tried
# (rather than only what finished) does not grow its own copy -- the mistake
# figlib's ABLATION registry comment already warns about.
ARM_GROUP = ({r: "covariate" for r in RUN_ORDER}
             | {r: "patient split" for r in PATIENT_SPLIT_ORDER}
             | {r: "prior-anchored" for r in PRIOR_ORDER})
ALL_ARMS = RUN_ORDER + PATIENT_SPLIT_ORDER + PRIOR_ORDER


def arm_short(run_id):
    """Legend label for an arm.

    The prior-anchored arms all carry the main model's covariates and differ only
    in the loss, so their shared prefix is dropped -- seven rows repeating
    "sex, age, setting" says nothing, and "NORMA | prior anchor k=5" says what
    the arm is. Every other group keeps its covariates, including the
    patient-split arms, whose semicolon separates the covariates from the split
    marker rather than a shared prefix.
    """
    cov = RUN_COVARIATES.get(run_id)
    if cov is None:
        return f"NORMA_{run_id}"
    if run_id in PRIOR_ORDER and ";" in cov:
        cov = cov.split(";", 1)[1].strip()
    return f"NORMA | {cov}"

# Legend form: "NORMA | sex, age, setting". Same words, no key needed.


def arm_label(run_id, short=False):
    """Legend: 'NORMA | sex, age, setting'; table: 'NORMA (sex, age, setting)'.
    Unknown ids -> NORMA_<id>."""
    if short:
        return RUN_SHORT.get(run_id, f"NORMA_{run_id}")
    cov = RUN_COVARIATES.get(run_id)
    return f"NORMA ({cov})" if cov else f"NORMA_{run_id}"
