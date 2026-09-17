"""The one place that decides what every model and cohort is called and coloured."""

from collections import namedtuple

# Family hues.
import os as _os, sys as _sys
_MODEL_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "model")   # lib is norma/scripts/lib
if _MODEL_DIR not in _sys.path:
    _sys.path.append(_MODEL_DIR)
from run_names import (RUN_COVARIATES, RUN_SHORT, SHORT_KEY, arm_label,  # noqa: E402
                       MULTI_PRIOR_ORDER, PRIOR_ORDER)

HUES = {
    "norma":     "#0097A7",   # teal, the model this paper is about
    "norma_alt": "#00565E",   # darkest teal, for a second NORMA in one figure
    "popri":     "#BDBDBD",   # achromatic: the reference
    "perri":     "#E8734A",
    "gaussian":  "#5C6BC0",
    "cohen":     "#795548",
    "history":   "#5C6B73",   # history-only baselines, one grey ramp
}


def lighten(hex_color, amount):
    """Blend towards white; `amount` 0 = unchanged, 1 = white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


# Model, cohort, state and outcome names are Title Case everywhere (Aashna, 2026-08-31);
# descriptive axis and colourbar text stays sentence case.
Model = namedtuple("Model", "key label short family group marker aliases")


def _m(key, label, short, family, group, marker, aliases=()):
    return Model(key, label, short, family, group, marker, tuple(aliases))


MODELS = {m.key: m for m in [
    # ── reference-interval methods ───────────────────────────────────────────
    _m("PopRI", r"Pop$_{RI}$", None, "PopRI", "ri", "o", ["Pop_RI", "population"]),
    _m("PerRI", r"Per$_{RI}$", None, "PerRI", "ri", "o", ["Per_RI", "personal"]),
    _m("Gaussian_mle", "Gaussian", None, "Gaussian", "ri", "s", ["Gaussian"]),
    _m("Gaussian_trunc", "Trunc. Gaussian", "Trunc. Gauss.", "Gaussian", "ri", "s"),
    _m("Gaussian_eb", "Empirical Bayes", "Emp. Bayes", "Gaussian", "ri", "s", ["Gauss-EB"]),
    _m("Cohen_m2", "Cohen m2", None, "Cohen", "ri", "^"),
    _m("Cohen_m3", "Cohen m3", None, "Cohen", "ri", "^"),
    _m("Cohen_m4", "Cohen", None, "Cohen", "ri", "^", ["Cohen"]),
    _m("NORMA", r"NORMA$_{RI}$", "NORMA", "NORMA", "ri", "D",
       ["NORMA_RI", "norma", "q_age_set", "NORMA_q_age_set", "norma_q_age_set"]),

    # ── forecasters ────────────────────────────────────────────────────────── Two NORMA entries
    # because the query state is what differs: `oracle` is given the realized next state (the
    # published...
    _m("NORMA_oracle", "NORMA (Realized State)", "NORMA-S", "NORMA", "forecast", "D",
       ["NORMA-Quantile", "NORMA_oracle", "oracle"]),
    _m("NORMA_marginal", "NORMA (Leak-Free)", "NORMA-H", "NORMA", "forecast", "D",
       ["NORMA-Quantile (marginal)", "marginal"]),
    _m("Last", "Last Value", "Last", "history", "forecast", "o", ["LVCF", "last"]),
    _m("Mean", "Patient Mean", "Mean", "history", "forecast", "o", ["mean"]),
    _m("ARIMA", "ARIMA", None, "history", "forecast", "o", ["arima"]),

    # ── NORMA covariate-ablation arms ──────────────────────────────────────── Keyed as the
    # pipeline writes them (config.NORMA_ABLATION_RUN_IDS puts norma_<arm> rows in ref_intervals
    # -> method...
    _m("NORMA_334f7e21", arm_label("334f7e21"), RUN_SHORT["334f7e21"], "NORMA_arm", "norma_arm", "D", ['334f7e21', 'norma_334f7e21']),
    _m("NORMA_q_age", arm_label("q_age"), RUN_SHORT["q_age"], "NORMA_arm", "norma_arm", "D", ['q_age']),
    _m("NORMA_q_set", arm_label("q_set"), RUN_SHORT["q_set"], "NORMA_arm", "norma_arm", "D", ['q_set']),
    _m("NORMA_q_co", arm_label("q_co"), RUN_SHORT["q_co"], "NORMA_arm", "norma_arm", "D", ['q_co']),
    _m("NORMA_q_age_co", arm_label("q_age_co"), RUN_SHORT["q_age_co"], "NORMA_arm", "norma_arm", "D", ['q_age_co']),
    _m("NORMA_q_set_co", arm_label("q_set_co"), RUN_SHORT["q_set_co"], "NORMA_arm", "norma_arm", "D", ['q_set_co']),
    _m("NORMA_q_age_set_co", arm_label("q_age_set_co"), RUN_SHORT["q_age_set_co"], "NORMA_arm", "norma_arm", "D", ['q_age_set_co']),
    _m("NORMA_q_co_q", arm_label("q_co_q"), RUN_SHORT["q_co_q"], "NORMA_arm", "norma_arm", "D", ['q_co_q']),
    # patient-split arms: their own group, compared against p_base rather than against the
    # covariate ladder (holding out whole patients changes the test set)
    _m("NORMA_p_base", arm_label("p_base"), RUN_SHORT["p_base"], "NORMA_arm", "norma_arm", "D", ['p_base']),
    _m("NORMA_p_co", arm_label("p_co"), RUN_SHORT["p_co"], "NORMA_arm", "norma_arm", "D", ['p_co']),
    _m("NORMA_p_causal", arm_label("p_causal"), RUN_SHORT["p_causal"], "NORMA_arm", "norma_arm", "D", ['p_causal']),
    _m("NORMA_p_full", arm_label("p_full"), RUN_SHORT["p_full"], "NORMA_arm", "norma_arm", "D", ['p_full']),
    # prior-anchored arms, two input families (run_prior_ablation.sh).
] + [
    _m(f"NORMA_{r}", arm_label(r), RUN_SHORT[r], "NORMA_arm", "norma_arm", "D", [r])
    for r in MULTI_PRIOR_ORDER + PRIOR_ORDER
]}

# Colours, kept next to the registry rather than inside it so a family ramp is visible as a ramp.
COLORS = {
    "PopRI": HUES["popri"],
    "PerRI": HUES["perri"],
    "Gaussian_mle": "#9FA8DA",          # Material indigo 200, the ramp's light end
    "Gaussian_trunc": HUES["gaussian"],
    "Gaussian_eb": "#303F9F",
    "Cohen_m2": lighten(HUES["cohen"], 0.55),
    "Cohen_m3": lighten(HUES["cohen"], 0.30),
    "Cohen_m4": HUES["cohen"],
    "NORMA": HUES["norma"],
    "NORMA_oracle": HUES["norma"],
    "NORMA_marginal": lighten(HUES["norma"], 0.45),
    "Last": HUES["history"],
    "Mean": lighten(HUES["history"], 0.35),
    "ARIMA": lighten(HUES["history"], 0.50),
    # arms: the cross-family hues, never shown beside the RI methods
    "NORMA_q_age_set_co": HUES["popri"],
    "NORMA_q_age_co": lighten(HUES["perri"], 0.45),
    "NORMA_q_set_co": lighten(HUES["cohen"], 0.45),
    "NORMA_q_age": HUES["perri"],
    "NORMA_q_set": HUES["cohen"],
    "NORMA_q_co": HUES["gaussian"],
    "NORMA_334f7e21": HUES["norma_alt"],
    "NORMA_q_co_q": lighten(HUES["gaussian"], 0.45),
    # patient-split group: reuses the same hues, read only against each other
    "NORMA_p_base": HUES["norma_alt"],
    "NORMA_p_co": HUES["gaussian"],
    "NORMA_p_causal": HUES["cohen"],
    "NORMA_p_full": HUES["perri"],
    # prior-anchored families: one ramp of seven, used twice.
    **{f"NORMA_{r}": c for fam in (MULTI_PRIOR_ORDER, PRIOR_ORDER)
       for r, c in zip(fam, [HUES["norma_alt"], HUES["perri"], HUES["cohen"],
                             HUES["gaussian"], HUES["popri"],
                             lighten(HUES["perri"], 0.45),
                             lighten(HUES["cohen"], 0.45)])},
}

# Which query gets an open marker: the realized-state forecast is not a like-for-like number, and
# every figure that shows it flags it the same way.
USES_REALIZED_STATE = {"NORMA_oracle"}

# Linestyle as a second identity channel, for the line figures where two family members sit under
# the dE 15 colour floor (06_calibration/sensitivity.pdf).
LINESTYLES = {
    "Gaussian_mle": (0, (1, 1.2)),
    "Gaussian_trunc": (0, (3.5, 1.4)),
}


_ALIAS = {a: m.key for m in MODELS.values() for a in m.aliases}
_ALIAS.update({k: k for k in MODELS})


def canonical(name):
    """Registry key for whatever a result file called this model ('' if unknown)."""
    return _ALIAS.get(str(name), "")


def label(key, short=False):
    m = MODELS.get(canonical(key))
    if m is None:
        return str(key)
    return (m.short or m.label) if short else m.label


def color(key, default="#999999"):
    return COLORS.get(canonical(key), default)


def marker(key, default="o"):
    m = MODELS.get(canonical(key))
    return m.marker if m else default


def collapse_run_id(name):
    """Fold a bare training-run id onto NORMA, leaving registry names alone."""
    s = str(name)
    known = canonical(s)
    if known:
        return known
    return "NORMA" if s.startswith("NORMA") else s


def group(name):
    """Registry keys of one group, in registry order ('ri' / 'forecast' / 'norma_arm')."""
    return [k for k, m in MODELS.items() if m.group == name]


def labels(keys, short=False):
    return {k: label(k, short) for k in keys}


def colors(keys):
    return {k: color(k) for k in keys}


# Queried state The state token NORMA conditions on.
STATE_ORDER = ["low", "normal", "high"]
STATE_DISPLAY = {"low": "Low", "normal": "Normal", "high": "High"}
STATE_COLORS = {"low": "#5C6BC0", "normal": "#2E7D32", "high": "#FF8F00"}


# Cohorts Display name, colour and marker for every cohort, development ones included (they were
# missing from the marker and colour maps, so any figure that put a development cohort on a
# shared axis...
Cohort = namedtuple("Cohort", "key display color marker")

COHORTS = {c.key: c for c in [
    Cohort("ehrshot", "EHRSHOT", "#E7298A", "^"),
    Cohort("mimiciv", "MIMIC-IV", "#A6761D", "v"),
    Cohort("eicu", "eICU-CRD", "#1B9E77", "o"),
    Cohort("chs", "CHS", "#E6702A", "s"),
    Cohort("inspire", "INSPIRE", "#5C6BC0", "D"),
]}

COHORT_DISPLAY = {k: c.display for k, c in COHORTS.items()}
COHORT_COLORS = {k: c.color for k, c in COHORTS.items()}
COHORT_MARKERS = {k: c.marker for k, c in COHORTS.items()}
