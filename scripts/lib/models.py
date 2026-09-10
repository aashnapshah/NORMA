"""The one place that decides what every model and cohort is called and coloured.

Change a label, a colour or a marker HERE and every figure and table in the
pipeline follows. Nothing else in the repo should define a model label or a
hex colour: `lib/figlib.py` and `lib/figlib.py` re-export views of this
registry under the names the per-stage modules already use
(METHOD_COLORS, RI_LABELS, _BM_COLORS, ABLATION_*, MODEL_COLORS, DATASET_*),
so those modules keep working unchanged.

Three kinds of entry, all in MODELS:

  ri          reference-interval methods, the ones a cohort figure compares:
              Pop_RI, Per_RI, the Gaussian fits, the Cohen models, NORMA_RI
  forecast    next-value forecasters: NORMA under either query, and the
              history-only baselines
  norma_arm   the covariate-ablation arms of NORMA (model/run_covariate_ablation.sh)

`key` is the string that appears in the result CSVs. `aliases` lists the other
spellings older result files use, so `canonical()` can fold them in.

Colours are the colour-vision-deficient-safe set validated 2026-08-31: every
cross-family pair clears Lab dE 15 under deuteranopia and protanopia (Machado
matrices). Within a family (Gaussian, Cohen) the colours are one hue ramp on
purpose, so they read as variants; a figure showing every member of a family
must therefore carry a second identity channel (row or x position, or
linestyle). Pop_RI is achromatic because it is the reference the others are
measured against.

The NORMA arms reuse the cross-family hues rather than a teal ramp: five steps
of one hue are not separable, and an ablation figure never shows the RI methods
at the same time (`figlib.ablation_mode`), so there is no clash.
"""

from collections import namedtuple

# Family hues. Anything needing a raw colour takes it from here, never inline.
# Per-measurement covariates each training run was fitted with, by run id --
# defined once in model/run_names.py so the dev-set scripts use the same names.
import os as _os, sys as _sys
_MODEL_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "model")   # lib is norma/scripts/lib
if _MODEL_DIR not in _sys.path:
    _sys.path.append(_MODEL_DIR)
from run_names import RUN_COVARIATES, RUN_SHORT, SHORT_KEY, arm_label  # noqa: E402

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


# Model, cohort, state and outcome names are Title Case everywhere (Aashna,
# 2026-08-31); descriptive axis and colourbar text stays sentence case.
#
# label  full name, may contain TeX; used where there is room (legends, tables)
# short  compact axis label; falls back to `label` when None
# family variants of one approach share a family, which drives the row/column
#        brackets in the grouped figures and the family marker
# marker per-family marker shape, for figures that encode method by shape
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

    # ── forecasters ──────────────────────────────────────────────────────────
    # Two NORMA entries because the query state is what differs: `oracle` is
    # given the realized next state (the published setting, which the baselines
    # cannot match), `marginal` uses the transition prior and is the
    # like-for-like comparison. Both are the same weights, hence the same hue.
    _m("NORMA_oracle", "NORMA (Realized State)", "NORMA-S", "NORMA", "forecast", "D",
       ["NORMA-Quantile", "NORMA_oracle", "oracle"]),
    _m("NORMA_marginal", "NORMA (Leak-Free)", "NORMA-H", "NORMA", "forecast", "D",
       ["NORMA-Quantile (marginal)", "marginal"]),
    _m("Last", "Last Value", "Last", "history", "forecast", "o", ["LVCF", "last"]),
    _m("Mean", "Patient Mean", "Mean", "history", "forecast", "o", ["mean"]),
    _m("ARIMA", "ARIMA", None, "history", "forecast", "o", ["arima"]),

    # ── NORMA covariate-ablation arms ────────────────────────────────────────
    # Keyed as the pipeline writes them (config.NORMA_ABLATION_RUN_IDS puts
    # norma_<arm> rows in ref_intervals -> method NORMA_<arm> downstream).
    # Every arm is labelled by exactly the per-measurement covariates it was
    # trained with -- one convention, readable whichever run is the main. The
    # main run is folded onto "NORMA" above; in ablation figures figlib labels it
    # by its own covariate set (RUN_COVARIATES) so it is comparable to the arms.
    _m("NORMA_334f7e21", arm_label("334f7e21"), RUN_SHORT["334f7e21"], "NORMA_arm", "norma_arm", "D", ['334f7e21', 'norma_334f7e21']),
    _m("NORMA_q_age", arm_label("q_age"), RUN_SHORT["q_age"], "NORMA_arm", "norma_arm", "D", ['q_age']),
    _m("NORMA_q_set", arm_label("q_set"), RUN_SHORT["q_set"], "NORMA_arm", "norma_arm", "D", ['q_set']),
    _m("NORMA_q_co", arm_label("q_co"), RUN_SHORT["q_co"], "NORMA_arm", "norma_arm", "D", ['q_co']),
    _m("NORMA_q_age_co", arm_label("q_age_co"), RUN_SHORT["q_age_co"], "NORMA_arm", "norma_arm", "D", ['q_age_co']),
    _m("NORMA_q_set_co", arm_label("q_set_co"), RUN_SHORT["q_set_co"], "NORMA_arm", "norma_arm", "D", ['q_set_co']),
    _m("NORMA_q_age_set_co", arm_label("q_age_set_co"), RUN_SHORT["q_age_set_co"], "NORMA_arm", "norma_arm", "D", ['q_age_set_co']),
    _m("NORMA_q_co_q", arm_label("q_co_q"), RUN_SHORT["q_co_q"], "NORMA_arm", "norma_arm", "D", ['q_co_q']),
]}

# Colours, kept next to the registry rather than inside it so a family ramp is
# visible as a ramp. Every value comes from HUES.
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
}

# Which query gets an open marker: the realized-state forecast is not a
# like-for-like number, and every figure that shows it flags it the same way.
USES_REALIZED_STATE = {"NORMA_oracle"}

# Linestyle as a second identity channel, for the line figures where two family
# members sit under the dE 15 colour floor (06_calibration/sensitivity.pdf).
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
    """Fold a bare training-run id onto NORMA, leaving registry names alone.

    Result files name the primary run's method NORMA_<hex run id>. Anything the
    registry knows — including the ablation arms — comes back untouched, so
    NORMA_q_age stays a separate series instead of being averaged into NORMA.
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# Queried state
# ═══════════════════════════════════════════════════════════════════════════
# The state token NORMA conditions on. The integer coding (0/1/2) is the data
# side and lives with the code that reads the model; these are the display
# identities, so a figure that shows the three states colours them the same way
# wherever it sits in the pipeline.
STATE_ORDER = ["low", "normal", "high"]
STATE_DISPLAY = {"low": "Low", "normal": "Normal", "high": "High"}
STATE_COLORS = {"low": "#5C6BC0", "normal": "#2E7D32", "high": "#FF8F00"}


# ═══════════════════════════════════════════════════════════════════════════
# Cohorts
# ═══════════════════════════════════════════════════════════════════════════
# Display name, colour and marker for every cohort, development ones included
# (they were missing from the marker and colour maps, so any figure that put a
# development cohort on a shared axis fell back to a default). Hues are Dark2,
# far enough apart to survive both common CVD types.
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
