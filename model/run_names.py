"""One place for what each NORMA training run is called."""

# What each run uses, as a reader would list it.
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
    # patient-level split arms (run_patient_split.sh, R3 comment 11).
    "p_base": "sex; patient split",
    "p_co": "sex, analytes; patient split",
    "p_causal": "sex, causal mask; patient split",
    "p_full": "sex, every past draw, causal mask; patient split",
    # prior-anchored on the multivariate arm (run_prior_ablation.sh): p_full's inputs and split
    # with each prior-anchored loss, so the only thing varying against p_full is the loss.
    "m_pa_k5":  "sex, age, every past draw, causal mask; patient split; prior anchor k=5",
    "m_pa_k20": "sex, age, every past draw, causal mask; patient split; prior anchor k=20",
    "m_pa_tau": "sex, age, every past draw, causal mask; patient split; prior anchor k=5, tau=365d",
    "m_pf_k5":  "sex, age, every past draw, causal mask; patient split; width floor k=5",
    "m_pg_k5":  "sex, age, every past draw, causal mask; patient split; learned gate k=5",
    "m_pn_k5":  "sex, age, every past draw, causal mask; patient split; conjugate NIG head",
    "m_gk_k5":  "sex, age, every past draw, causal mask; patient split; Gaussian KL-aligned k=5",
}

# the patient-split group compares only within itself (different test set)
PATIENT_SPLIT_ORDER = ["p_base", "p_co", "p_causal", "p_full"]

RUN_SHORT = {k: f"NORMA | {v}" for k, v in RUN_COVARIATES.items()}
SHORT_KEY = ("every run also uses age at the first draw and the analyte code; "
             "age = age at each draw, setting = care setting, analytes = the other analytes drawn at the same time")

# additive-ladder order for ablation tables
RUN_ORDER = ["334f7e21", "q_age", "q_set", "q_co", "q_age_set", "q_age_co", "q_set_co",
             "q_age_set_co", "q_co_q"]

# prior-anchored arms (run_prior_ablation.sh): a different loss or output head on fixed inputs.
MULTI_PRIOR_ORDER = ["m_pa_k5", "m_pa_k20", "m_pa_tau", "m_pf_k5", "m_pg_k5",
                     "m_pn_k5", "m_gk_k5"]
PRIOR_ORDER = ["pa_k5", "pa_k20", "pa_tau", "pf_k5", "pg_k5", "pn_k5", "gk_k5"]

# Every arm trained since the covariate ablation began, in reading order, with the group each
# belongs to.
ARM_GROUP = ({r: "covariate" for r in RUN_ORDER}
             | {r: "patient split" for r in PATIENT_SPLIT_ORDER}
             | {r: "prior-anchored, multivariate" for r in MULTI_PRIOR_ORDER}
             | {r: "prior-anchored" for r in PRIOR_ORDER})
ALL_ARMS = RUN_ORDER + PATIENT_SPLIT_ORDER + MULTI_PRIOR_ORDER + PRIOR_ORDER
PRIOR_GROUPS = set(MULTI_PRIOR_ORDER) | set(PRIOR_ORDER)


def arm_short(run_id):
    """Legend label for an arm."""
    cov = RUN_COVARIATES.get(run_id)
    if cov is None:
        return f"NORMA_{run_id}"
    if run_id in PRIOR_GROUPS and ";" in cov:
        loss = cov.rsplit(";", 1)[1].strip()
        cov = f"{loss}, full panel" if run_id in MULTI_PRIOR_ORDER else loss
    return f"NORMA | {cov}"

# Legend form: "NORMA | sex, age, setting". Same words, no key needed.

def arm_label(run_id, short=False):
    """Legend: 'NORMA | sex, age, setting'; table: 'NORMA (sex, age, setting)'.
    Unknown ids -> NORMA_<id>."""
    if short:
        return RUN_SHORT.get(run_id, f"NORMA_{run_id}")
    cov = RUN_COVARIATES.get(run_id)
    return f"NORMA ({cov})" if cov else f"NORMA_{run_id}"
