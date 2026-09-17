"""Everything the stage scripts' figure and table halves share (2026-09-08; was
plotting.py + figlib.py + tablelib.py).  The figure half of every scripts/<stage>.py does
`from figlib import *`; make_figures.py / make_tables.py import it as P.
"""

import functools
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from datasets import (BASE_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, NORMA_RUN_ID,
                      find_in, stage_file,
                      NORMA_ABLATION_RUN_IDS, DEV_KEY, SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:          # for `process.config` below
    sys.path.append(SCRIPTS_DIR)
import models        # the single registry of model / cohort names and colours
import models as M
# lib/constants.py is the single definition of everything the analysis and the figures must agree
# on; re-exported here so `from figlib import *` still sees it.
from constants import (ALL_SPLIT, ANCHOR, COHORT_ORDER, DATASET_ORDER, DEV_COHORTS, DEV_SPLITS,
                    EXCLUDE_ANALYTES, FDR, FLAG_RATE, MATCHED_SENSITIVITY, MEDIAN_ROW, MIN_EVENTS,
                    BASELINE_Z, POOLED_ROW, PSEUDO_ANALYTES,
                    MIN_PATIENTS, PENDING_H, SMALL_N, SPLIT_ORDER, STANDARD_OF_CARE)
from constants import VAL_COHORTS as _ALL_VAL_COHORTS
from process.config import REFERENCE_INTERVALS   # the intervals the pipeline classifies against

PREDICTION_DIR = os.path.join(RESULTS_DIR, "raw", DEV_KEY)   # model/evaluate.py output
# Where find_result() looks, in order: results/raw/<cohort>/ (what a stage writes) then
# results/processed/<cohort>/ (the figure data, all that comes back from Clalit).
RESULT_KINDS = ["raw", "processed"]

# Datasets & outcomes

# External validation cohorts.
DATASETS = list(DATASET_ORDER)
_ALL_DATASETS = tuple(DATASETS)
# results/figures/<OUTPUT_TAG>/: the SCOPE of the build, not a per-cohort split of the outputs --
# "all" is the full cohort set (the manuscript build, whose figures already carry one row or
# panel per...
OUTPUT_TAG = "all"
_LINKED = []      # module-level cohort lists that --dataset must restrict as well


def link_dataset_list(lst):
    """Register a cohort list (figlib.VAL_COHORTS) so set_datasets() restricts it in place."""
    _LINKED.append(lst)
    return lst
# Development cohorts (cohort table only)
COHORT_DATASETS = ["ehrshot", "mimiciv", "eicu", "chs", "inspire"]

# Views of models.COHORTS — edit the cohort's display name, colour or marker there.
DATASET_DISPLAY = dict(M.COHORT_DISPLAY)
DATASET_MARKERS = dict(M.COHORT_MARKERS)
DATASET_COLORS = dict(M.COHORT_COLORS)

OUTCOMES = {
    "eicu": ["mortality", "aki", "sepsis", "prolonged_los"],
    "chs": ["ckd", "t2d", "mortality"],
    "inspire": ["mortality", "prolonged_los", "unplanned_icu", "periop_infection"],
}
OUTCOME_DISPLAY = {
    "mortality": "Mortality", "aki": "Acute Kidney Injury",
    "sepsis": "Sepsis", "prolonged_los": "Prolonged LOS (>7d)",
    "ckd": "Chronic Kidney Disease", "t2d": "Type 2 Diabetes",
    "anemia_unspecified": "Anemia",
    "unplanned_icu": "Unplanned ICU Admission",
    "periop_infection": "Perioperative Infection",
}
OUTCOME_SHORT = {"mortality": "Mortality", "aki": "AKI", "sepsis": "Sepsis",
                 "prolonged_los": "Prolonged Stay", "unplanned_icu": "Unplanned ICU",
                 "periop_infection": "Periop. Infection", "ckd": "CKD", "t2d": "T2D",
                 "anemia_unspecified": "Anemia"}

def set_datasets(datasets):
    """Restrict the datasets built (used by --dataset); mutates in place so
    every module that imported DATASETS sees the change."""
    global OUTPUT_TAG
    keep = [d for d in DATASETS if d in datasets]
    DATASETS.clear()
    DATASETS.extend(keep)
    for lst in _LINKED:
        lst[:] = [d for d in lst if d in keep]
    OUTPUT_TAG = "all" if set(keep) == set(_ALL_DATASETS) else "_".join(keep)


# Reference-interval methods, palette

PALETTE = {
    "teal": "#0097A7", "coral": "#E85D4A", "grey": "#78909C",
    "green": "#2E7D32", "gold": "#FF8F00", "slate": "#5C6BC0",
    "terracotta": "#C27A56",
}
DARK = "#333333"

METHODS = ["PopRI", "PerRI", "NORMA"]
COX_METHODS = METHODS
# Views of models.MODELS — edit a method's label or colour there, not here.
METHOD_DISPLAY = M.labels(METHODS)
METHOD_COLORS = M.colors(METHODS)
METHOD_MARKERS = {k: M.marker(k) for k in METHODS}

# Balanced accuracy is deliberately NOT a panel: it weights a missed case and a false alarm
# equally, the objection Referee 3 raised (R3.M3).
EVAL_METRICS = ["ppv", "sensitivity", "specificity", "auroc"]
EVAL_METRIC_LABELS = {"ppv": "Precision", "sensitivity": "Sensitivity",
                      "specificity": "Specificity", "auroc": "AUROC"}
EVAL_METRIC_COLORS = {"ppv": PALETTE["teal"], "sensitivity": PALETTE["coral"],
                      "specificity": PALETTE["green"], "auroc": PALETTE["gold"]}

# Forecasting models as the dev-set prediction CSVs name them.
NORMA_MODEL = "NORMA-Quantile"
BASELINE_MODELS = ["ARIMA", "Mean", "Last"]
MODEL_ORDER = [NORMA_MODEL] + BASELINE_MODELS
MODEL_COLORS = M.colors(MODEL_ORDER)


def model_label(m):
    """Display label for a forecasting model named as the prediction CSVs name it."""
    if M.canonical(m):
        return M.label(m, short=True)
    return "NORMA" if str(m).startswith("NORMA") else m


SENSITIVITY_FEATURES = ["history_length", "horizon", "history_std"]
FEATURE_LABELS = {"history_length": "History Length", "horizon": "Prediction Horizon",
                  "history_std": "Within-Person Variability"}

# Analytes

CORE_ANALYTES = sorted([
    "A1C", "ALB", "ALP", "ALT", "AST", "BUN", "CA", "CL", "CO2", "CRE",
    "DBIL", "GLU", "HCT", "HDL", "HGB", "K", "LDL", "MCH", "MCHC", "MCV",
    "MPV", "NA", "PLT", "RBC", "RDW", "TBIL", "TC", "TGL", "TP", "WBC",
])

ANALYTE_PANELS = {
    "CBC": ["HCT", "HGB", "MCH", "MCHC", "MCV", "MPV", "PLT", "RBC", "RDW", "WBC"],
    "BMP": ["NA", "K", "CL", "CO2", "BUN", "CRE", "GLU", "CA", "A1C"],
    "HFP": ["ALT", "AST", "ALP", "TBIL", "DBIL", "ALB", "TP"],
    "Lipid": ["TC", "HDL", "LDL", "TGL"],
}
PANEL_COLORS = {"CBC": PALETTE["coral"], "BMP": PALETTE["teal"], "HFP": PALETTE["gold"], "Lipid": PALETTE["slate"]}

ANALYTE_NAMES = {
    "A1C": "Hemoglobin A1c", "ALB": "Albumin", "ALP": "Alkaline Phosphatase",
    "ALT": "Alanine Aminotransferase", "AST": "Aspartate Aminotransferase",
    "BUN": "Blood Urea Nitrogen", "CA": "Calcium", "CL": "Chloride",
    "CO2": "Bicarbonate", "CRE": "Creatinine", "DBIL": "Direct Bilirubin",
    "GLU": "Glucose", "HCT": "Hematocrit", "HDL": "HDL Cholesterol",
    "HGB": "Hemoglobin", "K": "Potassium", "LDL": "LDL Cholesterol",
    "MCH": "Mean Corpuscular Hemoglobin", "MCHC": "MCH Concentration",
    "MCV": "Mean Corpuscular Volume", "MPV": "Mean Platelet Volume",
    "NA": "Sodium", "PLT": "Platelet Count", "RBC": "Red Blood Cell Count",
    "RDW": "Red Cell Distribution Width", "TBIL": "Total Bilirubin",
    "TC": "Total Cholesterol", "TGL": "Triglycerides", "TP": "Total Protein",
    "WBC": "White Blood Cell Count",
}

_n = len(CORE_ANALYTES)
ANALYTE_COLORS = {a: (plt.cm.tab20(i / 20) if i < 20 else plt.cm.tab20b((i - 20) / 20))
                  for i, a in enumerate(CORE_ANALYTES)}


def all_analytes():
    return sorted(set(ANALYTE_NAMES) - EXCLUDE_ANALYTES)


def analyte_panel(analyte):
    for panel, members in ANALYTE_PANELS.items():
        if analyte in members:
            return panel
    return None


def analyte_panel_order(analytes):
    """Order analytes by clinical panel (CBC, BMP, HFP, Lipid), then the rest alphabetically."""
    order = [a for p in ANALYTE_PANELS.values() for a in p]
    rest = sorted(a for a in analytes if a not in order)
    return [a for a in order if a in analytes] + rest


# Font sizes / styling

FONT_TITLE = 8
FONT_AXIS = 7
FONT_TICK = 6
FONT_LEGEND = 6
MARKER_SIZE = 3


def setup_style(font="Work Sans"):
    """Nature-style matplotlib defaults (Type 42 fonts so text stays editable)."""
    family = font
    try:
        import matplotlib.font_manager as fm
        from pyfonts import load_google_font
        for weight in ("regular", "bold", "medium"):
            fp = load_google_font(font, weight=weight)
            fm.fontManager.addfont(fp.get_file())
        family = fp.get_name()
    except Exception:
        pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_AXIS, "axes.titlesize": FONT_TITLE, "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK, "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": 0.5, "axes.edgecolor": DARK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelpad": 2, "axes.grid": False,
        "xtick.major.size": 3, "xtick.major.width": 0.5, "xtick.direction": "in", "xtick.major.pad": 2,
        "ytick.major.size": 3, "ytick.major.width": 0.5, "ytick.direction": "in", "ytick.major.pad": 2,
        "lines.linewidth": 1.0, "lines.markersize": 3,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.3,
        "figure.constrained_layout.use": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.06, "savefig.format": "pdf",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    """Axis labels, a left-aligned title and tick sizes at the figure's font scale."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE, loc="left")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    hide_spines(ax)


def hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for which in ("major", "minor"):
        ax.tick_params(which=which, top=False, right=False)


def lighten(color, amt=0.35):
    r, g, b = mcolors.to_rgb(color)
    return tuple(c + (1 - c) * amt for c in (r, g, b))


# Result loading

def results_dir(dataset, kind="raw"):
    """results/<kind>/<cohort>/ — read through find_result(), which knows the stage."""
    return os.path.join(RESULTS_DIR, kind, dataset)


def find_result(dataset, filename):
    """Locate a result file in results/<kind>/<cohort>/."""
    for kind in RESULT_KINDS:
        path = find_in(os.path.join(RESULTS_DIR, kind, dataset), filename)
        if os.path.exists(path):
            return path
    return None


def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def has_result(dataset, *filenames):
    """True if ANY of the given result files exists for this dataset."""
    return any(find_result(dataset, f) is not None for f in filenames)


def missing_results(dataset, *filenames):
    return [f for f in filenames if find_result(dataset, f) is None]


def available_datasets(*filenames):
    return [ds for ds in DATASETS if has_result(ds, *filenames)]


def normalize_norma_cols(df, dataset=None):
    """Fold the MAIN run onto plain NORMA: NORMA_<NORMA_RUN_ID>_* columns become
    NORMA_*, method value NORMA_<NORMA_RUN_ID> becomes NORMA. Anything the model
    registry knows (the ablation arms, which since 2026-09-03 include the old
    main 334f7e21 as the "no covariates" arm) is left untouched; an unregistered
    NORMA_<hex> run is a stale leftover and is dropped.
    """
    if df is None:
        return None
    from models import canonical  # lazy: models is a leaf, but keep import order flexible
    main_prefix = f"NORMA_{NORMA_RUN_ID}_"
    main_method = f"NORMA_{NORMA_RUN_ID}"
    hex_col = re.compile(r"^NORMA_([a-f0-9]{6,})_(.*)$")
    hex_val = re.compile(r"^NORMA_[a-f0-9]{6,}$")

    renames, drop = {}, []
    for col in df.columns:
        if col.startswith(main_prefix):
            renames[col] = "NORMA_" + col[len(main_prefix):]
            continue
        m = hex_col.match(col)
        if m and not canonical(f"NORMA_{m.group(1)}"):
            drop.append(col)                       # unregistered stale run
    if renames or drop:
        df = df.rename(columns=renames).drop(columns=drop, errors="ignore")

    if "method" in df.columns:
        meth = df["method"].astype(str)
        stale = meth.str.match(hex_val) & (meth != main_method) & ~meth.map(lambda v: bool(canonical(v)))
        if stale.any():
            df = df[~stale].copy()
        df["method"] = df["method"].astype(str).where(df["method"].astype(str) != main_method, "NORMA")
    if "balanced_accuracy" in df.columns and "accuracy" not in df.columns:
        df = df.rename(columns={"balanced_accuracy": "accuracy"})
    return df


def load_result(dataset, filename, normalize=True):
    path = find_result(dataset, filename)
    df = load_csv(path) if path else None
    return normalize_norma_cols(df, dataset) if normalize else df


def load_prediction(filename):
    return load_csv(find_in(PREDICTION_DIR, filename))


def to_numeric(df, skip=("analyte", "method", "outcome", "model", "setting", "metric", "split", "anchor",
                         "realized_state", "target_state", "dataset", "test_name", "feature",
                         "subset", "state", "code", "reference", "comparator", "favours", "stratum",
                         "panel", "model_type", "time_window", "new_method", "ref_method", "design",
                         # 13_cox landmark columns: coercing these to numeric turned every value
                         # into NaN, so the exposure/encoding filters matched nothing and every
                         # 13_cox figure silently fell back to a placeholder.
                         "exposure", "encoding", "level", "norma_run",
                         # 05_forecasting/norma_versions.csv label columns
                         "version", "source")):
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_metric(sub, analyte, method, metric):
    match = sub[(sub["analyte"] == analyte) & (sub["method"] == method)]
    if len(match) == 1 and pd.notna(match.iloc[0][metric]):
        return match.iloc[0][metric]
    return np.nan


# Figure output

def figure_path(analysis, name):
    """results/figures/<OUTPUT_TAG>/<nn>_<name>.pdf — flat, like the results
    folders, with the drawing stage's number as the prefix."""
    return os.path.join(FIGURES_DIR, OUTPUT_TAG, stage_file(f"{name}.pdf", analysis))


def save_fig(fig, analysis, name):
    path = figure_path(analysis, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 0.06 rather than a hairline: "tight" crops to the ink, so a rotated cohort label on the
    # right edge came out flush against the crop box and read as cut off.
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, BASE_DIR)}")
    return path


def placeholder_fig(message, size=(3.2, 2.4)):
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_TITLE, color="#999999", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#CCCCCC"); s.set_linestyle("--")
    return fig


def save_placeholder(analysis, name, message):
    path = save_fig(placeholder_fig(message), analysis, name)
    print(f"     (placeholder: {message.splitlines()[0]})")
    return path


def pending_message(dataset, missing):
    ds = DATASET_DISPLAY.get(dataset, dataset)
    files = ", ".join(missing[:3]) + (" …" if len(missing) > 3 else "")
    return f"{ds}: pending\nmissing {dataset}/{files}"


# Table output (LaTeX helpers + save)

def tex_escape(s):
    if not isinstance(s, str):
        s = str(s)
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, f"\\{ch}")
    return (s.replace("\u00b1", r"$\pm$").replace("\u2014", "---").replace("\u2013", "--"))


def fmt_num(x, d=2):
    if isinstance(x, str):
        return x
    return "---" if pd.isna(x) else f"{x:.{d}f}"


def fmt_ci(val, lo, hi, d=2):
    if any(pd.isna(v) for v in (val, lo, hi)):
        return "---"
    return f"{val:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"


def fmt_pval(p):
    if pd.isna(p):
        return "---"
    return "$<$0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def fmt_pm(mean, std, d=2):
    m, s = fmt_num(mean, d), fmt_num(std, d)
    return "---" if "---" in (m, s) else f"{m} \u00b1 {s}"


PDF_TABLES = True   # make_tables.py --no-pdf turns LaTeX compilation off


def _tectonic():
    t = shutil.which("tectonic")
    if t:
        return t
    for c in (os.path.expanduser("~/miniconda3/envs/normal/bin/tectonic"),
              os.path.expanduser("~/miniconda3/bin/tectonic")):
        if os.path.exists(c):
            return c
    return None


def compile_latex(tex_path, pdf_dir, name, landscape=False):
    """Compile a table snippet to a standalone PDF via tectonic. Returns path or None."""
    tectonic = _tectonic()
    if tectonic is None:
        print("     [skip pdf] tectonic not found")
        return None
    os.makedirs(pdf_dir, exist_ok=True)
    margin = "margin=0.3in, landscape" if landscape else "margin=0.5in"
    standalone = os.path.join(tempfile.gettempdir(), f"{name}_standalone.tex")
    with open(tex_path) as src, open(standalone, "w") as f:
        f.write("\\documentclass[11pt]{article}\n"
                "\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{amsmath}\n"
                "\\usepackage{lmodern}\n\\renewcommand{\\familydefault}{\\sfdefault}\n"
                f"\\usepackage[{margin}]{{geometry}}\n\\pagestyle{{empty}}\n"
                "\\begin{document}\n" + src.read() + "\n\\end{document}\n")
    r = subprocess.run([tectonic, standalone], capture_output=True, text=True, cwd=tempfile.gettempdir())
    if r.returncode != 0:
        print(f"     [warn] LaTeX failed for {name}: {r.stderr.strip()[-300:]}")
        return None
    dst = os.path.join(pdf_dir, f"{name}.pdf")
    shutil.move(standalone.replace(".tex", ".pdf"), dst)
    return dst


TABLE_FORMATS = ("tex", "csv", "pdf")   # one folder each under results/tables/<tag>/


def save_table(analysis, name, latex_lines, csv_df=None, landscape=False, font_size="footnotesize"):
    """Write results/tables/<OUTPUT_TAG>/{tex,csv,pdf}/<nn>_<name>.<ext> — one
    folder per format (Aashna 2026-09-08), and the <nn> prefix is the stage that
    built it.  Unlike the results CSVs, a table is the SAME table in three
    renderings, so the format is the folder rather than a suffix on one name."""
    base = os.path.join(TABLES_DIR, OUTPUT_TAG)
    tex_dir, csv_dir, pdf_dir = (os.path.join(base, d) for d in TABLE_FORMATS)
    name = stage_file(name, analysis)
    os.makedirs(tex_dir, exist_ok=True)

    latex = "\n".join(latex_lines)
    for size_cmd in (r"\tiny", r"\scriptsize", r"\footnotesize", r"\small", r"\normalsize"):
        latex = latex.replace(size_cmd + "\n", "").replace(size_cmd, "")
    latex = latex.replace(r"\centering", r"\centering" + "\n\\" + font_size)
    if r"\label" not in latex:
        latex = latex.replace(r"\end{table}", r"\label{tab:" + name + "}\n" + r"\end{table}")

    tex_path = os.path.join(tex_dir, f"{name}.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  -> {os.path.relpath(tex_path, BASE_DIR)}")
    if csv_df is not None:
        os.makedirs(csv_dir, exist_ok=True)
        csv_df.to_csv(os.path.join(csv_dir, f"{name}.csv"), index=False)
    if PDF_TABLES:
        compile_latex(tex_path, pdf_dir, name, landscape=landscape)
    return tex_path


def save_placeholder_table(analysis, name, message):
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l}", r"\toprule",
             tex_escape(message.replace("\n", " — ")) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    csv_df = pd.DataFrame([{"status": message.replace("\n", " — ")}])
    print(f"     (placeholder: {message.splitlines()[0]})")
    return save_table(analysis, name, lines, csv_df)


# ── Loaders shared by the figure and table modules ────────────────────────── Defined here
# rather than in figlib/tablelib: both need them, and the two copies had already drifted apart in
# formatting.

EVAL_CSV = "eval.csv"        # one file; the `subset` column says which restriction
# The primary eval subset: per-analyte normal history, scored inside Pop_RI-normal tests.
EVAL_SUBSET = "per_normal_pop_normal"

# Coefficients of variation above this are fitting artefacts, not biology.
_VAR_MAX = 200


def _ri(method):
    return METHOD_DISPLAY.get(method, method)


def _load_variability(ds):
    df = load_result(ds, "variability.csv")
    if df is None:
        return None
    df = df[~df["analyte"].isin(EXCLUDE_ANALYTES)].copy()
    metric_cols = ["cv_intra", "cv_intra_ci_lower", "cv_intra_ci_upper",
                   "cv_inter", "cv_inter_ci_lower", "cv_inter_ci_upper",
                   "individuality_index", "ii_ci_lower", "ii_ci_upper"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].abs() > _VAR_MAX, col] = np.nan
    return df


def attach_auroc(ev, ds, subset="pop_normal"):
    """Add an `auroc` column (threshold-free discrimination of the deviation score,
    12_eval.py (auroc step)) to an eval frame, matched on analyte x method x outcome."""
    if ev is None or len(ev) == 0:
        return ev
    au = load_result(ds, "auroc.csv")
    if au is None or "auc" not in au.columns:
        ev["auroc"] = np.nan
        return ev
    if "subset" in au.columns:
        au = au[au["subset"].astype(str) == subset]
    cols = ["auc"] + [c for c in ("auc_lo", "auc_hi") if c in au.columns]
    au = au[["analyte", "method", "outcome", *cols]].drop_duplicates(["analyte", "method", "outcome"])
    renamed = {"auc": "auroc", "auc_lo": "auroc_lo", "auc_hi": "auroc_hi"}
    return ev.merge(au.rename(columns=renamed), on=["analyte", "method", "outcome"], how="left")


def pooled_rows(df, *dims):
    """The rows of a consolidated table that are NOT broken out along `dims`."""
    if df is None:
        return None
    for d in dims:
        if d in df.columns:
            df = df[df[d].astype(str) == ALL_SPLIT]
    return df


def load_eval(ds, subset=EVAL_SUBSET):
    """One subset of eval.csv (see EVAL_SUBSET); None when the cohort has no rows."""
    df = load_result(ds, EVAL_CSV)
    if df is None or "subset" not in df.columns:
        return df
    df = df[df["subset"].astype(str) == subset]
    return df if len(df) else None


def _load_eval_restricted(ds):
    return attach_auroc(load_eval(ds), ds)


# Figure helpers, FigSpec and the per-stage constants

def _load_mortality_deviation(ds):
    df = load_result(ds, "mortality_deviation.csv")
    if df is None:
        return None
    if "method" in df.columns:
        df = df[df["method"].astype(str) == BASELINE_Z]   # the baseline-period z
    df["analyte"] = df["analyte"].replace("", "NA").replace("TG", "TGL")
    if "n" in df.columns:
        df = df[df["n"] >= 100]
    if ds == "chs":   # z_median computed with a unit mismatch (M/uL vs K/uL)
        df = df[df["analyte"] != "RBC"]
    return df
def _valid_analytes(sub, min_n=100):
    out = []
    for a in all_analytes():
        rows = sub[sub["analyte"] == a]
        if len(rows) and rows["n"].max() >= min_n:
            out.append(a)
    return out
def _method_legend(ax, methods, markers=None, **kw):
    handles = []
    for m in methods:
        mk = (markers or {}).get(m)
        if mk:
            handles.append(Line2D([0], [0], marker=mk, color=METHOD_COLORS[m], markersize=4,
                                  linestyle="none", label=_ri(m)))
        else:
            handles.append(Patch(facecolor=METHOD_COLORS[m], alpha=0.85, label=_ri(m)))
    ax.legend(handles=handles, frameon=False, fontsize=FONT_LEGEND, **kw)


def plan_legend(labels, width_in, fontsize=FONT_LEGEND, handlelength=1.8,
                handletextpad=0.4, columnspacing=1.0, pad_in=0.06):
    """(ncol, height_in) for a centred legend of `labels` across `width_in`."""
    if not len(labels):
        return 1, 0.0
    char_in = 0.52 * fontsize / 72.0          # mean glyph advance at this size
    line_in = 1.45 * fontsize / 72.0          # row pitch
    widest = max(len(l) for l in labels)
    entry_in = (handlelength + handletextpad + columnspacing) * fontsize / 72.0 + widest * char_in
    ncol = max(1, min(len(labels), int(width_in // entry_in)))
    nrow = int(np.ceil(len(labels) / ncol))
    return ncol, nrow * line_in + pad_in


def top_legend(fig, handles, labels, fontsize=FONT_LEGEND, handlelength=1.8,
               handletextpad=0.4, columnspacing=1.0, pad_in=0.06):
    """Legend above the panels, centred, wrapped to as many rows as it needs."""
    if not handles:
        return 0.0
    ncol, height_in = plan_legend(labels, fig.get_figwidth(), fontsize, handlelength,
                                  handletextpad, columnspacing, pad_in)
    fig.legend(handles, labels, frameon=False, fontsize=fontsize, ncol=ncol,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=handlelength,
               handletextpad=handletextpad, columnspacing=columnspacing)
    return height_in


_CIRCOS_NEG = "#C0C0C0"
_CIRCOS_CI = 0.20     # whisker clip: rings are 0.5 apart, so whiskers never meet


def _se_table(sub, metric):
    """Standard error of `metric` per (analyte, method): binomial from the counts for the
    proportions (denominator = flagged / events / non-events), the DeLong interval for
    AUROC.  NaN where the counts are not there (the CHS legacy tables)."""
    d = sub.drop_duplicates(["analyte", "method"]).set_index(["analyte", "method"])
    if metric == "auroc":
        if "auroc_lo" not in d.columns or "auroc_hi" not in d.columns:
            return pd.Series(np.nan, index=d.index)
        return (d["auroc_hi"] - d["auroc_lo"]) / 3.92
    nothing = pd.Series(np.nan, index=d.index)
    if metric not in d.columns:
        return nothing
    p = pd.to_numeric(d[metric], errors="coerce")
    if metric == "ppv":
        m = d["tp"] + d["fp"] if "tp" in d.columns else d.get("n_flagged")
    elif metric == "sensitivity":
        m = d.get("n_events")
    elif metric == "specificity":
        m = d["n"] - d["n_events"] if "n" in d.columns and "n_events" in d.columns else None
    else:
        return nothing
    if m is None:
        return nothing
    m = pd.to_numeric(m, errors="coerce").where(lambda v: v > 0)
    return np.sqrt(p * (1 - p) / m)


def _delta_ci_table(sub, metric, comparators):
    """{(comparator, analyte): 95% half-width of delta(NORMA - comparator)}, the two errors
    added in quadrature (independent-samples: conservative for the paired design)."""
    se = _se_table(sub, metric)
    out = {}
    for c in comparators:
        for a in sub["analyte"].unique():
            if (a, "NORMA") in se.index and (a, c) in se.index:
                out[(c, a)] = 1.96 * float(np.hypot(se[(a, "NORMA")], se[(a, c)]))
    return out


def _ci_whisker(ax, theta, base_r, d, half, scale):
    """Radial 95% whisker around a circos bar, clipped so it stays inside the ring."""
    if not np.isfinite(half):
        return
    lo = np.clip((d - half) * scale, -_CIRCOS_CI, _CIRCOS_CI)
    hi = np.clip((d + half) * scale, -_CIRCOS_CI, _CIRCOS_CI)
    ax.plot([theta, theta], [base_r + lo, base_r + hi], color=DARK, lw=0.45, alpha=0.7, zorder=5,
            solid_capstyle="butt")


def _draw_circos(ax, sub, metric, color, comparator="PerRI"):
    """Radial bars of delta(NORMA - comparator) per analyte. Returns False if nothing to draw."""
    deltas, labels = [], []
    for a in _valid_analytes(sub):
        p, n = get_metric(sub, a, comparator, metric), get_metric(sub, a, "NORMA", metric)
        if not (np.isnan(p) or np.isnan(n)):
            deltas.append(n - p); labels.append(a)
    if not deltas:
        return False
    order = np.argsort(deltas)[::-1]
    deltas = [deltas[i] for i in order]; labels = [labels[i] for i in order]
    n = len(deltas); theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bw = 2 * np.pi / n * 0.85; base_r = 0.6          # room inside for the centre label and whiskers
    scale = 0.35 / max(max(abs(d) for d in deltas), 0.01)
    halves = _delta_ci_table(sub, metric, [comparator])
    for j, (d, a) in enumerate(zip(deltas, labels)):
        if d >= 0:
            ax.bar(theta[j], d * scale, width=bw, bottom=base_r, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        else:
            ax.bar(theta[j], abs(d) * scale, width=bw, bottom=base_r - abs(d) * scale, color=_CIRCOS_NEG,
                   alpha=0.8, edgecolor="white", linewidth=0.3)
        _ci_whisker(ax, theta[j], base_r, d, halves.get((comparator, a), np.nan), scale)
    for t, lab in zip(theta, labels):
        deg = np.degrees(t)
        ax.text(t, base_r + 0.7, lab, ha="center", va="center", fontsize=3.5, rotation=deg - 180 if 90 < deg < 270 else deg)
    if deltas[0] > 0:
        ax.text(0, 0, f"{labels[0]}\n{deltas[0] * 100:+.1f}%", ha="center", va="center", fontsize=FONT_TITLE, color="#333")
    ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False); ax.grid(False)
    return True
_RING_STEP, _RING_HALF, _RING_INNER = 0.5, 0.22, 0.75   # ring spacing, max bar length, innermost baseline


def _draw_circos_rings(ax, sub, metric, color, comparators):
    """One ring per comparator, inner to outer: radial bars of delta(NORMA - comparator)
    per analyte on ONE scale shared by the rings, positive in the metric colour, negative
    grey inward.  The thin circle at each ring's base carries the comparator's colour.
    Analytes are ordered by the innermost ring.  The centre prints, per ring, the analyte
    with the largest gain and that gain in percentage points.  Returns False if nothing
    to draw."""
    analytes = _valid_analytes(sub)
    deltas = {c: {a: get_metric(sub, a, "NORMA", metric) - get_metric(sub, a, c, metric) for a in analytes}
              for c in comparators}
    labels = [a for a in analytes if any(np.isfinite(deltas[c][a]) for c in comparators)]
    if not labels:
        return False
    inner = deltas[comparators[0]]
    labels.sort(key=lambda a: -(inner[a] if np.isfinite(inner[a]) else -np.inf))
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bw = 2 * np.pi / n * 0.85
    vmax = max([abs(d) for c in comparators for d in deltas[c].values() if np.isfinite(d)] + [0.01])
    scale = _RING_HALF / vmax
    halves = _delta_ci_table(sub, metric, comparators)
    circle = np.linspace(0, 2 * np.pi, 200)
    for k, c in enumerate(comparators):
        base_r = _RING_INNER + k * _RING_STEP
        ax.plot(circle, np.full_like(circle, base_r), color=_BM_COLORS.get(c, DARK), lw=0.7, zorder=2)
        for t, a in zip(theta, labels):
            d = deltas[c][a]
            if not np.isfinite(d):
                continue
            if d >= 0:
                ax.bar(t, d * scale, width=bw, bottom=base_r, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.3, zorder=3)
            else:
                ax.bar(t, -d * scale, width=bw, bottom=base_r + d * scale, color=_CIRCOS_NEG, alpha=0.8,
                       edgecolor="white", linewidth=0.3, zorder=3)
            _ci_whisker(ax, t, base_r, d, halves.get((c, a), np.nan), scale)
    r_label = _RING_INNER + (len(comparators) - 1) * _RING_STEP + _RING_HALF + 0.12
    for t, lab in zip(theta, labels):
        deg = np.degrees(t)
        ax.text(t, r_label, lab, ha="center", va="center", fontsize=3.5,
                rotation=deg - 180 if 90 < deg < 270 else deg)
    # centre: best analyte per ring, stacked top to bottom in ring order
    slots = [(np.pi / 2, 0.24), (0.0, 0.0), (-np.pi / 2, 0.24)][:len(comparators)]
    for (t, r), c in zip(slots, comparators):
        finite = {a: d for a, d in deltas[c].items() if np.isfinite(d)}
        if not finite:
            continue
        best = max(finite, key=finite.get)
        ax.text(t, r, f"{best} {finite[best] * 100:+.1f} pp", ha="center", va="center",
                fontsize=4, color=_BM_COLORS.get(c, DARK))
    ax.set_ylim(0, r_label + 0.15)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    return True


def _cox_row(sub, analyte, method):
    match = sub[(sub["analyte"] == analyte) & (sub["method"] == method)]
    if len(match) != 1:
        return None
    r = match.iloc[0]
    if pd.isna(r["HR"]) or not np.isfinite(r["HR"]) or r["HR"] > 50:
        return None
    return r
def _concordance_bars(sub, top, title):
    y = np.arange(len(top)); h = 0.8 / len(COX_METHODS)
    fig, ax = plt.subplots(figsize=(3.0, 0.32 * len(top) + 1.2))
    for mi, method in enumerate(COX_METHODS):
        msub = sub[sub["method"] == method].drop_duplicates("analyte").set_index("analyte")
        vals = [float(msub.loc[a, "concordance_test"]) if a in msub.index and pd.notna(msub.loc[a, "concordance_test"])
                else np.nan for a in top]
        ax.barh(y + (mi - 1) * h, vals, h * 0.9, color=METHOD_COLORS[method], alpha=0.8, label=_ri(method))
    ax.set_yticks(y); ax.set_yticklabels(top, fontsize=FONT_TICK)
    c = pd.to_numeric(sub["concordance_test"], errors="coerce").dropna()
    ax.set_xlim(max(0.45, c.min() - 0.05) if len(c) else 0.45, min(0.95, c.max() + 0.07) if len(c) else 0.95)
    ax.set_xlabel("C-index", fontsize=FONT_AXIS); ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.set_title(title, fontsize=FONT_TITLE, loc="left")
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower right")
    hide_spines(ax); fig.tight_layout()
    return fig
def _lead_time_unit(plot_df):
    med = plot_df["median_lead_hours"].median()
    if med > 24 * 365:
        return 1 / (24 * 365.25), "Lead Time (years)"
    if med > 24 * 30:
        return 1 / (24 * 30.44), "Lead Time (months)"
    return 1.0, "Lead Time (hours)"
# Benchmark method sets.
_BM_MAIN = ["PopRI", "PerRI", "Cohen_m4", "NORMA"]
# Cohen m2 / m3 are computed (04_refs) but not shown: Aashna 2026-08-28, "the only Cohen I want is m4".
_BM_SUPP = ["PopRI", "PerRI", "Gaussian_mle", "Gaussian_trunc", "Gaussian_eb", "Cohen_m4", "NORMA"]
_BM_METHODS = _BM_MAIN

# One naming convention for every reference-interval method, used by calibration, prevalence and
# benchmark figures alike (RI_SHORT / _BM_LABELS are aliases).
_RI_AND_ARMS = models.group("ri") + models.group("norma_arm")
RI_LABELS = models.labels(_RI_AND_ARMS)
_BM_LABELS = RI_LABELS
_BM_LABELS_MAIN = dict(RI_LABELS, Cohen_m4="Cohen")

# Colour-vision-deficient-safe method palette (2026-08-31), now defined once in lib/models.py;
# the rationale lives in that module's docstring.
_BM_COLORS = models.colors(_RI_AND_ARMS)
# Methods that are variants of one underlying approach.
METHOD_FAMILY = {k: models.MODELS[k].family for k in models.group("ri")
                 if models.MODELS[k].family in ("Gaussian", "Cohen")}
# Linestyle is the second identity channel where colour alone is too close (Gaussian_trunc sits
# under the dE 15 floor against Gaussian_eb and NORMA).
METHOD_LINESTYLE = dict(models.LINESTYLES)
# Marker per method family, for the figures that encode the method by shape.
FAMILY_MARKERS = {models.MODELS[k].family: models.marker(k) for k in _RI_AND_ARMS}
FAMILY_MARKERS["NORMA"] = models.marker("NORMA")

# ── NORMA covariate-ablation arms as first-class methods ────────────────────────
# config.NORMA_ABLATION_RUN_IDS puts norma_<arm> rows in ref_intervals, so every downstream
# result carries the arms...
ABLATION_RUN_IDS = list(NORMA_ABLATION_RUN_IDS)
ABLATION_METHODS = ["NORMA"] + [f"NORMA_{r}" for r in ABLATION_RUN_IDS]
# Plots use the shorthand codes (NORMA-B, -A, -S, -C, -AS, ...; key in models.SHORT_KEY for the
# caption); tables use models.label(m) -- the full text.
_MAIN_RUN = NORMA_RUN_ID
ABLATION_LABELS = {m: models.label(m, short=True) for m in ABLATION_METHODS}
ABLATION_LABELS["NORMA"] = models.RUN_SHORT.get(_MAIN_RUN, "NORMA")             # the main run, unmarked
ABLATION_KEY = models.SHORT_KEY
# Any figure that looks an arm up in RI_LABELS (the per-stage legends) gets the same code, so an
# arm is named identically in every plot of the pipeline.
for _m in ABLATION_METHODS:
    if _m != "NORMA":
        RI_LABELS[_m] = ABLATION_LABELS[_m]
ABLATION_COLORS = models.colors(ABLATION_METHODS)
# There used to be a second arm registry here (NORMA_ARM_LABELS / _COLORS) with different labels
# and different hues, so one arm was "+age at draw" in #E8734A in 07_classify and "+ age at draw"
# in...
NORMA_ARM_LABELS = ABLATION_LABELS

# ── Every arm, not only the ones carried through the pipeline ──────────────── ABLATION_METHODS
# ABLATION_METHODS above is what NORMA_ABLATION_RUN_IDS forwards, i.e. what a figure can plot.
from run_names import ALL_ARMS, ARM_GROUP, arm_short   # noqa: E402

ALL_ARM_METHODS = [f"NORMA_{r}" for r in ALL_ARMS]
ALL_ARM_LABELS = {f"NORMA_{r}": arm_short(r) for r in ALL_ARMS}
ALL_ARM_COLORS = {m: models.color(m) for m in ALL_ARM_METHODS}


def arm_trained():
    """Arms that finished and produced predictions, read from model/logs/."""
    import os as _os
    from datasets import MODEL_LOG_DIR
    return {r for r in ALL_ARMS
            if _os.path.exists(_os.path.join(MODEL_LOG_DIR, r, "predictions_combined.csv"))}


def arm_notes(present, methods=None, split_text=None):
    """{method: why it is blank} for arms with nothing to plot."""
    methods = methods or ALL_ARM_METHODS
    trained = arm_trained()
    out = {}
    for m in methods:
        if m in present:
            continue
        run = m.replace("NORMA_", "")
        if split_text and ARM_GROUP.get(run) == "patient split":
            out[m] = split_text
        elif run not in trained:
            out[m] = "not trained"
        else:
            out[m] = "not run here"
    return out
NORMA_ARM_COLORS = ABLATION_COLORS

_METHOD_SET = None      # None = normal figures; a list = ablation mode


def bm_methods(default=None):
    """The method list a fig_* function should draw."""
    if _METHOD_SET is not None:
        return list(_METHOD_SET)
    return list(default if default is not None else _BM_SUPP)


@contextmanager
def ablation_mode(methods=None):
    """Draw the NORMA arms instead of the RI methods. While active, the main run
    is labelled by its shorthand code and marked (NORMA-AS (main)), not NORMA_RI,
    so it reads as one arm among the others."""
    global _METHOD_SET
    prev = _METHOD_SET
    prev_main = RI_LABELS.get("NORMA")
    _METHOD_SET = list(methods or ABLATION_METHODS)
    RI_LABELS["NORMA"] = ABLATION_LABELS["NORMA"]
    try:
        yield
    finally:
        _METHOD_SET = prev
        RI_LABELS["NORMA"] = prev_main


def in_ablation_mode():
    return _METHOD_SET is not None


def ablation_variant(fn):
    """Wrap a fig_* function so it draws the NORMA arms instead of the RI methods."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with ablation_mode():
            return fn(*args, **kwargs)
    return wrapped
_BM_MARKERS = ["o", "s", "^", "D"]   # positional, one per outcome — not per method
# Reclassification rate is no longer a benchmark panel: 07_classify/reclassification.pdf and
# prevalence_overall.pdf show it per lab and per method for every cohort.
BENCHMARK_PANELS = ["ppv", "auroc", "sensitivity", "specificity", "hr_fraction", "concordance"]
_bm_method = models.collapse_run_id   # keeps the ablation arms separable
def _bm_bars(ax, df, value_col, outcomes, ylabel, ref=None, agg="median",
             methods=None, labels=None):
    """x = method; all outcomes pooled as a bar, per-outcome aggregates as markers."""
    methods = methods or _BM_METHODS
    labels = labels or _BM_LABELS
    x = np.arange(len(methods))
    overall = df.groupby("method")[value_col].agg(agg).reindex(methods).to_numpy()
    ax.bar(x, overall, color=[_BM_COLORS[m] for m in methods], width=0.7, alpha=0.9, zorder=1)
    for k, oc in enumerate(outcomes):
        v = df[df.outcome == oc].groupby("method")[value_col].agg(agg).reindex(methods).to_numpy()
        ax.scatter(x, v, s=9, marker=_BM_MARKERS[k % 4], facecolor="white", edgecolor=DARK, linewidth=0.5, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=DARK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in methods], rotation=60, ha="right", fontsize=5.5)
    ax.tick_params(axis="y", labelsize=FONT_TICK); ax.set_ylabel(ylabel, fontsize=FONT_AXIS); hide_spines(ax)
def _bm_legend(ax, outcomes):
    h = [Line2D([], [], marker=_BM_MARKERS[k % 4], ls="", ms=3.5, markerfacecolor="white", markeredgecolor=DARK,
                markeredgewidth=0.5, label=OUTCOME_SHORT.get(o, OUTCOME_DISPLAY.get(o, o))) for k, o in enumerate(outcomes)]
    h.append(Patch(facecolor="#CCCCCC", label="All outcomes pooled"))
    ax.legend(handles=h, fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
              handlelength=1.2, handletextpad=0.4)

# ── Short labels / colours for the reference-interval methods (benchmark + classify) ──
RI_ORDER = _BM_SUPP
RI_SHORT = RI_LABELS
VAL_COHORTS = link_dataset_list(list(_ALL_VAL_COHORTS))   # cohorts with outcomes / classification
CORAL_CMAP = mcolors.LinearSegmentedColormap.from_list("coral_seq", ["#FFFFFF", "#F5C6BE", "#E85D4A", "#8B2D1A"])
TEAL_CMAP = mcolors.LinearSegmentedColormap.from_list("teal_seq", ["#FFFFFF", "#B2DFDB", "#0097A7", "#00565E"])
GREY_CELL = "#E6E8EA"


def ax_inches(fig, W, H, x, y_top, w, h):
    """Axes placed in inches from the top-left corner of a W x H inch figure."""
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])


def pending_handles(present):
    """Legend entries naming the expected cohorts (figlib.DATASETS) with no results
    yet, so an overlay figure still says what is missing instead of silently
    dropping the cohort."""
    missing = [DATASET_DISPLAY.get(ds, ds) for ds in DATASETS if ds not in present]
    if not missing:
        return []
    return [Line2D([], [], ls="none", marker="", label=f"{', '.join(missing)}: pending")]


def pending_axis(ax, cohort):
    ax.text(0.5, 0.5, f"{DATASET_DISPLAY.get(cohort, cohort)}: pending\n(results/processed/{cohort}/\nnot yet copied in)",
            transform=ax.transAxes, ha="center", va="center", fontsize=FONT_TICK, color="#999999", linespacing=1.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#CCCCCC"); sp.set_linestyle("--")


def heatmap_blocks(blocks, columns, cmap, vmin, vmax, fmt, cbar_label, extend="neither",
                   neg_color=None, col_groups=None, legend_extra=(), cell_w=0.255, cell_h=0.175,
                   row_label_fontsize=5, small_n=SMALL_N):
    """Stacked heatmap blocks, one per cohort, sharing the column axis."""
    n_c = len(columns)
    # Left margin holds only the row labels; the cohort name is on the right (Aashna 2026-08-31),
    # between the cells and the colourbar, so `right` carries the name (_coh_w), the bar and the
    # bar's own label.
    left, right, _coh_w = 0.72, 1.20, 0.22
    W = left + cell_w * n_c + right
    y_legend, y_bracket, y = 0.08, 0.50, 0.70
    blk_gap, pending_h, xlab_h = 0.22, 0.30, 0.40
    heights = [cell_h * len(b["rows"]) if b else pending_h for _, b in blocks]
    H = y + sum(h + blk_gap for h in heights) - blk_gap + xlab_h
    fig = plt.figure(figsize=(W, H))
    last_ax = None
    for (cohort, b), h in zip(blocks, heights):
        fig.text((left + cell_w * n_c + _coh_w / 2) / W, 1 - (y + h / 2) / H,
                 DATASET_DISPLAY.get(cohort, cohort), ha="center", va="center",
                 rotation=90, fontsize=FONT_AXIS, color=DARK)
        ax = ax_inches(fig, W, H, left, y, cell_w * n_c, h)
        if b is None:
            pending_axis(ax, cohort)
            ax.set_xlim(-0.5, n_c - 0.5); ax.set_xticks(range(n_c)); ax.tick_params(length=0)
        else:
            mat = np.asarray(b["mat"], float); shown = np.asarray(b.get("shown", mat), float)
            nmat = b.get("nmat"); n_r = len(b["rows"])
            ax.imshow(np.where(np.isnan(mat), np.nan, shown), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            for i in range(n_r):
                for j in range(n_c):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", hatch="////",
                                                   edgecolor="#DDDDDD", lw=0))
                        continue
                    if neg_color is not None and v < 0:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=neg_color, lw=0))
                    sv = shown[i, j]
                    thin = nmat is not None and np.isfinite(nmat[i, j]) and nmat[i, j] < small_n
                    if thin:   # too few observations to colour: the number stays, the
                        # colour goes, so the eye is not drawn to the noisiest cell
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", lw=0))
                    dark = (not thin) and np.isfinite(sv) and (sv - vmin) / (vmax - vmin) > 0.6
                    ax.text(j, i, fmt(v), ha="center", va="center", fontsize=4.0,
                            color=("#DDDDDD" if thin else "white") if dark else ("#A0A0A0" if thin else DARK))
            ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True); ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
            ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
            ax.set_yticks(range(n_r)); ax.set_yticklabels(b["rows"], fontsize=row_label_fontsize)
            ax.set_xticks(range(n_c)); ax.set_xticklabels([]); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            groups = b.get("row_groups")
            if groups:
                for i in range(1, n_r):
                    if groups[i] != groups[i - 1]:
                        ax.axhline(i - 0.5, color=DARK, lw=0.6)
            if col_groups:
                pos = 0
                for _, members in col_groups.items():
                    k = sum(1 for c in columns if c in members)
                    if k and pos:
                        ax.axvline(pos - 0.5, color=DARK, lw=0.6)
                    pos += k
        last_ax = ax
        y += h + blk_gap
    if last_ax is not None:
        last_ax.set_xticklabels(columns, rotation=90, fontsize=5)
    if col_groups:
        pos = 0
        for name, members in col_groups.items():
            k = sum(1 for c in columns if c in members)
            if k == 0:
                continue
            x0, x1 = (left + (pos + 0.1) * cell_w) / W, (left + (pos + k - 0.1) * cell_w) / W
            fig.add_artist(Line2D([x0, x1], [1 - y_bracket / H] * 2, color=DARK, lw=0.6, transform=fig.transFigure))
            fig.text((x0 + x1) / 2, 1 - (y_bracket - 0.03) / H, name, ha="center", va="bottom", fontsize=FONT_TICK, color=DARK)
            pos += k
    # vertical colourbar on the right, cell key centred at the top
    cb_h = min(2.2, 0.5 * (H - y_legend - xlab_h))
    cax = ax_inches(fig, W, H, W - right + _coh_w + 0.20,
                    y_legend + (H - y_legend - xlab_h) / 2 - cb_h / 2, 0.09, cb_h)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap), cax=cax,
                      orientation="vertical", extend=extend)
    cb.set_label(cbar_label, fontsize=FONT_TICK, labelpad=3); cb.ax.tick_params(labelsize=5, length=2)
    cb.outline.set_visible(False)
    handles = list(legend_extra) + [Patch(facecolor="white", hatch="////", edgecolor="#BBBBBB", label="Not available")]
    if any(b and b.get("nmat") is not None for _, b in blocks):
        handles.append(Line2D([], [], ls="", marker="$n$", ms=5, color="#A0A0A0", label=f"Uncoloured: fewer than {small_n:,} observations"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - y_legend / H), ncol=len(handles),
               fontsize=FONT_LEGEND, frameon=False, handlelength=1.2, columnspacing=1.2)
    return fig


def dot_blocks(frames, metrics, methods, colors, labels, open_marker=(), block_key="cohort",
               W=6.4, row_h=0.135, label_rotation=0, row_labels=False, share_x=True,
               zero_line=False, notes=None):
    """One axis per metric; cohorts stacked down the y axis, methods as colours.
    frames: list of (cohort, {method: {metric: (centre, lo, hi)}} or None).
    label_rotation=90 writes the cohort names vertically (narrower left margin).
    row_labels=True names every method on the left y axis instead of relying on the
    legend, so identity is read off the axis rather than matched by colour; the
    legend is then dropped as pure duplication (Aashna 2026-08-31).
    share_x=False gives every cohort its own x axis and ticks, autoscaled to that
    cohort's own values. Use it when the spread between cohorts is much larger than
    the difference between methods the figure is about: the NORMA covariate arms sit
    within ~0.3% of one another inside a cohort but ~20% apart across cohorts, so one
    shared axis stacks every method of a row on a single point.
    zero_line=True marks x=0 on every axis and keeps it inside the autoscaled
    limits, for the figures whose x is a change from a reference method.
    notes={method: text} annotates a method that has no value with why, in grey,
    on the first metric axis. A method with nothing to plot keeps its row either
    way; the note is what distinguishes "we have not run this" from "this
    happened to be missing", which an empty row alone cannot say."""
    import matplotlib.colors as _mc
    n_m = len(methods); blocks = []; y0 = 0.0
    for c, d in frames:
        blocks.append((c, d, y0)); y0 += n_m + 1
    y_max = y0 - 1
    # Cohort names sit on the right of the last column, not the left of the first (Aashna
    # 2026-08-31), so the reading order is metric axes then cohort.
    _lab_w = 0.30 if label_rotation else 0.72
    left, right, gap_c, top, bottom = 0.10, _lab_w, 0.30, 0.42, 0.34
    if row_labels:   # left margin has to hold the longest method name
        left = 0.10 + 0.052 * max(len(str(labels[m])) for m in methods)
        top = 0.16    # no legend strip to reserve
    col_w = (W - left - right - gap_c * (len(metrics) - 1)) / len(metrics)

    def draw(ax, d, metric, yb=0.0, annotate=False):
        """The dots and SD bars of one block on one metric axis."""
        for i, m in enumerate(methods):
            have = m in d and metric in d[m] and np.isfinite(d[m][metric][0])
            if not have:
                if annotate and notes and m in notes:
                    ax.text(0.5, yb + i, notes[m], transform=ax.get_yaxis_transform(),
                            ha="center", va="center", fontsize=FONT_TICK, color="#AAAAAA")
                continue
            v, lo, hi = d[m][metric]
            yy = yb + i; col = colors[m]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot(v, yy, marker="o", ms=4.0, mfc="white" if m in open_marker else col, mec=col,
                    mew=1.0, ls="", zorder=3)

    def pending(ax, y_centre, transform):
        ax.text(0.5, y_centre, "pending", transform=transform, ha="center", va="center",
                fontsize=FONT_TICK, color="#AAAAAA")

    def block_span(d, metric):
        """Data range of one block on one metric, error bars included."""
        vals = [x for m in methods if d and m in d and metric in d[m]
                for x in d[m][metric] if np.isfinite(x)]
        if zero_line:
            vals.append(0.0)      # the reference has to stay on the axis
        return (min(vals), max(vals)) if vals else None

    def apply_xlim(ax, xlim):
        if xlim is not None:
            ax.set_xlim(*xlim)

    def finish(ax, mlabel, show_xlabel=True):
        if zero_line:
            ax.axvline(0, color="#BBBBBB", lw=0.6, ls=(0, (2.5, 1.6)), zorder=1)
        ax.tick_params(axis="x", labelsize=FONT_TICK)
        if show_xlabel:
            ax.set_xlabel(mlabel, fontsize=FONT_AXIS)
        hide_spines(ax); ax.spines["left"].set_visible(False)

    def cohort_ticks(ax, positions, names):
        ax.set_yticks(positions)
        ax.yaxis.set_ticks_position("right")
        ax.set_yticklabels([DATASET_DISPLAY.get(cc, cc) for cc in names],
                           fontsize=FONT_AXIS, rotation=label_rotation, va="center")

    if not share_x:
        # One axis per (metric, cohort): each cohort keeps its own scale, so the ticks have to be
        # repeated under every block rather than once at the foot.
        tick_h = 0.20                       # room for a block's own x tick labels
        block_h = row_h * n_m; block_gap = row_h + tick_h
        h_total = len(blocks) * block_h + (len(blocks) - 1) * block_gap
        H = top + h_total + bottom
        fig = plt.figure(figsize=(W, H))
        for c, (metric, mlabel, xlim) in enumerate(metrics):
            for b, (cohort, d, _) in enumerate(blocks):
                ax = ax_inches(fig, W, H, left + c * (col_w + gap_c),
                               top + b * (block_h + block_gap), col_w, block_h)
                last = b == len(blocks) - 1
                if d is None:
                    pending(ax, 0.5, ax.transAxes)
                    ax.set_xticks([])
                else:
                    draw(ax, d, metric, annotate=(c == 0))
                    span = block_span(d, metric)
                    if span is not None:
                        lo, hi = span
                        pad = 0.12 * (hi - lo) if hi > lo else max(abs(hi) * 0.02, 0.5)
                        ax.set_xlim(lo - pad, hi + pad)
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    apply_xlim(ax, xlim)
                ax.set_ylim(n_m - 0.5, -0.5)
                if row_labels and c == 0:
                    ax.set_yticks(range(n_m))
                    ax.set_yticklabels([str(labels[m]) for m in methods], fontsize=FONT_TICK)
                elif c == len(metrics) - 1:
                    cohort_ticks(ax, [(n_m - 1) / 2], [cohort])
                else:
                    ax.set_yticks([(n_m - 1) / 2]); ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0, pad=4 if label_rotation else 3.5)
                finish(ax, mlabel, show_xlabel=last)
    else:
        h_ax = row_h * y_max; H = top + h_ax + bottom
        fig = plt.figure(figsize=(W, H))
        for c, (metric, mlabel, xlim) in enumerate(metrics):
            ax = ax_inches(fig, W, H, left + c * (col_w + gap_c), top, col_w, h_ax)
            for cohort, d, yb in blocks:
                if d is None:
                    pending(ax, yb + (n_m - 1) / 2, ax.get_yaxis_transform())
                    continue
                draw(ax, d, metric, yb, annotate=(c == 0))
            for k, (_, _, yb) in enumerate(blocks):
                if k:
                    ax.axhline(yb - 1, color="#DDDDDD", lw=0.5, zorder=0)
            ax.set_ylim(y_max - 0.5, -0.5)
            if row_labels and c == 0:
                ax.set_yticks([yb + i for _, _, yb in blocks for i in range(n_m)])
                ax.set_yticklabels([str(labels[m]) for _ in blocks for m in methods], fontsize=FONT_TICK)
            else:
                ax.set_yticks([yb + (n_m - 1) / 2 for _, _, yb in blocks])
                if c == len(metrics) - 1:
                    cohort_ticks(ax, [yb + (n_m - 1) / 2 for _, _, yb in blocks], [cc for cc, _, _ in blocks])
                else:
                    ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=4 if label_rotation else 3.5)
            apply_xlim(ax, xlim)
            finish(ax, mlabel)
    if not row_labels:   # with every row named on the axis the legend says nothing new
        handles = [Line2D([], [], marker="o", ms=4.2, mfc="white" if m in open_marker else colors[m], mec=colors[m],
                          mew=1.0, ls="", label=labels[m]) for m in methods]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.06 / H), ncol=len(handles),
                   fontsize=FONT_LEGEND, frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.3)
    return fig


FigSpec = namedtuple("FigSpec", "analysis base fn per_dataset requires expected")
def _one(base):
    return lambda ds: [f"{base}_{ds}"]
def _per_outcome(base):
    return lambda ds: [f"{base}_{ds}_{o}" for o in OUTCOMES.get(ds, [])]
def _per_outcome_metric(base):
    return lambda ds: [f"{base}_{ds}_{o}_{m}" for o in OUTCOMES.get(ds, []) for m in EVAL_METRICS]
def _suffixes(base, sfx):
    return lambda ds: [f"{base}_{ds}_{s}" for s in sfx]
PREDICTION_INPUTS = ["forecasting_overall.csv", "forecasting_by_analyte.csv", "sensitivity.csv",
                     "forecasting_by_analyte_state_variants.csv"]
def output_name(base, ds, suffix):
    parts = [base] + ([ds] if ds else []) + ([suffix] if suffix else [])
    return "_".join(parts)


# Export everything, including the imports re-exported for the per-analysis modules (load_result,
# DATASETS, METHOD_COLORS, …) and the underscore-prefixed helpers the fig_/table_ functions rely
# on.

# Table helpers and TableSpec

def _fmt_hr(hr, lo, hi):
    if any(pd.isna(v) or not np.isfinite(v) for v in (hr, lo, hi)) or hr > 100:
        return "---"
    return f"{hr:.2f} [{lo:.2f}, {hi:.2f}]"
def _bold_best(values, lower_is_better=True):
    nums = []
    for v in values:
        try:
            nums.append(float(v.split("[")[0].strip()))
        except Exception:
            nums.append(None)
    valid = [x for x in nums if x is not None]
    if not valid:
        return [None] * len(values)
    best = (min if lower_is_better else max)(valid)
    return [v if (n is not None and abs(n - best) < 1e-6) else None for v, n in zip(values, nums)]
def _table(col_spec, header_lines, body_lines):
    return ([r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
            + header_lines + [r"\midrule"] + body_lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])
def format_population_ri(female, male):
    """The population interval for one analyte as a table cell, one interval when
    the sexes share it.  process/config.py stores some bounds as ints and some as
    floats (HGB is 14 for men, 12.0 for women), so every bound of an analyte is
    shown to the same number of decimals rather than printed as stored."""
    bounds = [b for ri in (female, male) for b in ri[:2] if b is not None]
    decimals = max((len(str(float(b)).split(".")[1].rstrip("0")) for b in bounds), default=0)

    def one(ri):
        lo, hi = ri[0], ri[1]
        text = [None if b is None else f"{float(b):.{decimals}f}" for b in (lo, hi)]
        if lo is None:
            return f"<{text[1]}"
        if hi is None:
            return f">{text[0]}"
        return f"{text[0]}–{text[1]}"

    if female[:2] == male[:2]:
        return one(female)
    return f"F: {one(female)}, M: {one(male)}"
METRIC_ORDER = ["MAE", "MAPE", "R2"]
SPLIT_LABELS = {"train": "Train", "val": "Val", "test": "Test"}
LOWER_BETTER = {"MAE": True, "MAPE": True, "R2": False}
def _fmt_val(mean, lo, hi, metric):
    d = 2 if metric == "R2" else 1
    return f"{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
TableSpec = namedtuple("TableSpec", "analysis base fn per_dataset requires expected")


# Export everything, including the imports re-exported for the per-analysis modules (load_result,
# DATASETS, METHOD_COLORS, …) and the underscore-prefixed helpers the fig_/table_ functions rely
# on.

# Export everything, including the underscore-prefixed helpers the fig_/table_ functions rely on.
__all__ = [_n for _n in dir() if not _n.startswith("__")]
