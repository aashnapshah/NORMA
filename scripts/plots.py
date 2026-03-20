import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pyfonts import load_google_font

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'process'))
from config import REFERENCE_INTERVALS

fp = load_google_font("Work Sans")
fm.fontManager.addfont(fp.get_file())
plt.rcParams["font.family"] = fp.get_name()

sys.path.append('../')

rc = {
    "figure.figsize": [3.3, 2.5],
    "figure.dpi": 300,
    "figure.constrained_layout.use": True,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "font.size": 7,
    "axes.titlesize": 8,
    "font.family": fp.get_name(),
    "text.usetex": False,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "k",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.4,
    "xtick.direction": "in",
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.format": "pdf",
    "savefig.dpi": 300,
    "axes.prop_cycle": plt.cycler(color=sns.color_palette("Set2")),
}

plt.rcParams.update(rc)
sns.set_theme(style="white", font=fp.get_name(), rc=rc)
sns.set_context("notebook", rc={"legend.frameon": False})

np.random.seed(42)

# ---------------------------------------------------------------------------
# Method color scheme — Nature-quality, colorblind-safe
# ---------------------------------------------------------------------------
# Master palette — vibrant, professional, Google Health / AMIE inspired
PALETTE = {
    'teal':      '#0097A7',   # NORMA / hero / eICU — vibrant cyan-teal
    'coral':     '#E85D4A',   # PerRI / personalized — warm red-coral
    'grey':      '#78909C',   # PopRI / population — blue-grey
    'terracotta':'#C27A56',   # dataset: CHS — earthy terracotta
    'green':     '#2E7D32',   # metric: specificity — forest green
    'gold':      '#FF8F00',   # metric: accuracy — vivid amber
    'slate':     '#5C6BC0',   # hybrid — indigo
}

METHOD_COLORS = {
    'Population': PALETTE['grey'],
    'Personalized': PALETTE['coral'],
    'NORMA': PALETTE['teal'],
    'Hybrid': PALETTE['slate'],
}

scheme_colors = {
    'Population': METHOD_COLORS['Population'],
    'population': METHOD_COLORS['Population'],
    'Inter-Patient': METHOD_COLORS['Population'],
    'prior': METHOD_COLORS['Population'],
    'reference': METHOD_COLORS['Population'],
    'Personalized': METHOD_COLORS['Personalized'],
    'personalized': METHOD_COLORS['Personalized'],
    'obs': METHOD_COLORS['Personalized'],
    'Intra-Patient': METHOD_COLORS['Personalized'],
    'setpoint': METHOD_COLORS['Personalized'],
    'Hybrid': METHOD_COLORS['Hybrid'],
    'post': METHOD_COLORS['Hybrid'],
    'bayes': METHOD_COLORS['Hybrid'],
    'Other': METHOD_COLORS['Hybrid'],
    'NORMA': METHOD_COLORS['NORMA'],
    'norma': METHOD_COLORS['NORMA'],
}

scheme_map = {
    'Population': 'Population',
    'population': 'Population',
    'Inter-Patient': 'Population',
    'prior': 'Population',
    'reference': 'Population',
    'personalized': 'Personalized',
    'Personalized': 'Personalized',
    'obs': 'Personalized',
    'Intra-Patient': 'Personalized',
    'setpoint': 'Personalized',
    'Hybrid': 'Hybrid',
    'post': 'Hybrid',
    'bayes': 'Hybrid',
    'Other': 'Hybrid',
    'NORMA': 'NORMA',
    'norma': 'NORMA',
}

# LaTeX-style label aliases
for _label, _key in [
    (r"Pop$_{RI}$", "Population"),
    (r"Per$_{RI}$", "Personalized"),
    (r"NORMA$_{RI}$", "NORMA"),
]:
    scheme_colors[_label] = scheme_colors[_key]
    scheme_map[_label] = _key

# ---------------------------------------------------------------------------
# Per-analyte color scheme — single gradient, alphabetical assignment
# ---------------------------------------------------------------------------
# Nature style: one continuous colormap so all analytes are visually cohesive.
# Using a perceptually uniform colormap (viridis family) sampled evenly.

_all_analytes_sorted = sorted([
    'A1C', 'ALB', 'ALP', 'ALT', 'AST', 'BUN', 'CA', 'CL', 'CO2', 'CRE',
    'DBIL', 'GLU', 'HCT', 'HDL', 'HGB', 'K', 'LDL', 'MCH', 'MCHC', 'MCV',
    'MPV', 'NA', 'PLT', 'RBC', 'RDW', 'TBIL', 'TC', 'TGL', 'TP', 'WBC',
])

# Use a muted qualitative palette for 30 analytes
_n_analytes = len(_all_analytes_sorted)
_analyte_pal = [plt.cm.tab20(i / 20) if i < 20 else plt.cm.tab20b((i - 20) / 20)
                for i in range(_n_analytes)]

analyte_colors = {a: _analyte_pal[i] for i, a in enumerate(_all_analytes_sorted)}

analyte_panels = {
    'CBC': ['HCT', 'HGB', 'MCH', 'MCHC', 'MCV', 'MPV', 'PLT', 'RBC', 'RDW', 'WBC'],
    'BMP': ['NA', 'K', 'CL', 'CO2', 'BUN', 'CRE', 'GLU', 'CA', 'A1C'],
    'HFP': ['ALT', 'AST', 'ALP', 'GGT', 'TBIL', 'DBIL', 'ALB', 'TP', 'LDH', 'CRP', 'PT'],
    'Lipid': ['TC', 'HDL', 'LDL', 'TGL'],
}


# Save plot as PDF with dpi=300
def save_pdf(filename):
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, format='pdf', dpi=300)
    plt.show()
    plt.close()

# don't show top and right spines
def hide_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False)
    ax.tick_params(which='major', bottom=False, right =False, top=False, left=False)
    ax.tick_params(which='minor', bottom=False, right =False, top=False, left=False)


# ---------------------------------------------------------------------------
# Shared constants and formatting utilities
# ---------------------------------------------------------------------------

EXCLUDE_ANALYTES = {'CRP', 'GGT', 'LDH', 'PT'}

ANALYTE_NAMES = {
    'A1C': 'Hemoglobin A1c', 'ALB': 'Albumin', 'ALP': 'Alkaline Phosphatase',
    'ALT': 'Alanine Aminotransferase', 'AST': 'Aspartate Aminotransferase',
    'BUN': 'Blood Urea Nitrogen', 'CA': 'Calcium', 'CL': 'Chloride',
    'CO2': 'Bicarbonate', 'CRE': 'Creatinine', 'DBIL': 'Direct Bilirubin',
    'GLU': 'Glucose', 'HCT': 'Hematocrit', 'HDL': 'HDL Cholesterol',
    'HGB': 'Hemoglobin', 'K': 'Potassium', 'LDL': 'LDL Cholesterol',
    'MCH': 'Mean Corpuscular Hemoglobin', 'MCHC': 'MCH Concentration',
    'MCV': 'Mean Corpuscular Volume', 'MPV': 'Mean Platelet Volume',
    'NA': 'Sodium', 'PLT': 'Platelet Count', 'RBC': 'Red Blood Cell Count',
    'RDW': 'Red Cell Distribution Width', 'TBIL': 'Total Bilirubin',
    'TC': 'Total Cholesterol', 'TGL': 'Triglycerides', 'TP': 'Total Protein',
    'WBC': 'White Blood Cell Count',
}


def get_unit(analyte):
    """Get the unit string for an analyte from REFERENCE_INTERVALS."""
    ref = REFERENCE_INTERVALS.get(analyte, {})
    v = ref.get('F', ref.get('M', (None, None, '')))
    return v[2] if len(v) > 2 else ''


def get_pop_ri(analyte):
    """Get population reference interval string for an analyte."""
    ref = REFERENCE_INTERVALS.get(analyte, {})
    f = ref.get('F', (None, None))
    m = ref.get('M', (None, None))
    if f[:2] == m[:2]:
        return f'{f[0]}\u2013{f[1]}'
    return f'F: {f[0]}\u2013{f[1]}, M: {m[0]}\u2013{m[1]}'


def fmt(val, precision=2):
    """Format a number with given precision, returning '—' for missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '\u2014'
    return f'{val:.{precision}f}'


def fmt_pm(mean, std, precision=2):
    """Format as 'mean ± std', returning '—' if either is missing."""
    m = fmt(mean, precision)
    s = fmt(std, precision)
    if m == '\u2014' or s == '\u2014':
        return '\u2014'
    return f'{m} \u00b1 {s}'


def tex_escape(s):
    """Escape a string for use in LaTeX."""
    s = str(s)
    s = s.replace('%', r'\%')
    s = s.replace('±', r'$\pm$')
    s = s.replace('\u00b1', r'$\pm$')
    s = s.replace('—', r'---')
    s = s.replace('\u2014', r'---')
    s = s.replace('–', r'--')
    s = s.replace('\u2013', r'--')
    s = s.replace('³', r'$^3$')
    s = s.replace('⁶', r'$^6$')
    s = s.replace('µ', r'$\mu$')
    s = s.replace('&', r'\&')
    return s


def save_fig(path_no_ext):
    """Save current figure as PDF + PNG and close."""
    plt.savefig(path_no_ext + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(path_no_ext + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"  {path_no_ext}.pdf")
    print(f"  {path_no_ext}.png")


def compile_latex(tex_path, name, out_dir, landscape=False):
    """Compile a .tex file to PDF and PNG via tectonic."""
    standalone = os.path.join(tempfile.gettempdir(), f'{name}_standalone.tex')
    margin_opts = 'margin=0.3in, landscape' if landscape else 'margin=0.5in'
    with open(tex_path, 'r') as src:
        tex_content = src.read()
    with open(standalone, 'w') as f:
        f.write('\\documentclass[11pt]{article}\n')
        f.write('\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{amsmath}\n')
        f.write('\\usepackage{helvet}\n\\renewcommand{\\familydefault}{\\sfdefault}\n')
        f.write(f'\\usepackage[{margin_opts}]{{geometry}}\n\\pagestyle{{empty}}\n')
        f.write('\\begin{document}\n')
        f.write(tex_content + '\n')
        f.write('\\end{document}\n')

    result = subprocess.run(['tectonic', standalone],
                            capture_output=True, text=True, cwd=tempfile.gettempdir())
    if result.returncode != 0:
        print(f"  LaTeX compilation failed: {result.stderr[:200]}")
        return

    pdf_src = standalone.replace('.tex', '.pdf')
    pdf_dst = os.path.join(out_dir, f'{name}.pdf')
    png_dst = os.path.join(out_dir, f'{name}.png')

    shutil.move(pdf_src, pdf_dst)
    print(f"  {pdf_dst}")
    subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                     pdf_dst, png_dst.replace('.png', '')], capture_output=True)
    if os.path.exists(png_dst):
        print(f"  {png_dst}")


def save_table(name, latex_str, csv_df, table_dir, landscape=False):
    """Save a table as LaTeX (.tex), CSV, and compiled PDF/PNG.

    Standardizes font size to \\footnotesize for consistency across all tables.
    """
    import re

    # Standardize: replace any font size command with \footnotesize
    for size_cmd in [r'\tiny', r'\scriptsize', r'\footnotesize', r'\small', r'\normalsize']:
        latex_str = latex_str.replace(size_cmd + '\n', '')
        latex_str = latex_str.replace(size_cmd, '')

    # Insert \footnotesize after \centering
    latex_str = latex_str.replace(r'\centering', r'\centering' + '\n' + r'\footnotesize')

    # Add caption if not present
    pretty_name = name.replace('_', ' ').title()
    if r'\caption' not in latex_str:
        latex_str = latex_str.replace(r'\end{table}',
                                       r'\caption{' + pretty_name + '}\n'
                                       r'\label{tab:' + name + '}\n'
                                       r'\end{table}')

    latex_dir = os.path.join(table_dir, 'latex')
    csv_dir = os.path.join(table_dir, 'csv')
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, f'{name}.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    print(f"  {tex_path}")

    csv_path = os.path.join(csv_dir, f'{name}.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"  {csv_path}")

    compile_latex(tex_path, name, table_dir, landscape=landscape)

