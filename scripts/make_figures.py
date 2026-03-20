#!/usr/bin/env python
"""
Compose multi-panel manuscript figures using LaTeX subfigures.

Each panel is an individual PDF from format_validation.py / format_prediction.py.
This script generates .tex files that arrange them with subfigure labels,
then compiles to PDF.

Usage:
    python scripts/make_figures.py              # all figures
    python scripts/make_figures.py fig2          # just Figure 2
"""
import os
import sys
import subprocess
import tempfile
import shutil

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOTDIR, 'results', 'manuscript_figures')
os.makedirs(FIGDIR, exist_ok=True)


def _path(relpath):
    """Get absolute path for a results file."""
    return os.path.join(ROOTDIR, relpath)


def _exists(relpath):
    return os.path.exists(_path(relpath))


def compile_figure(name, tex_content):
    """Compile a LaTeX figure to PDF.

    Copies referenced images to a temp directory to avoid absolute path issues.
    """
    import re

    work_dir = os.path.join(tempfile.gettempdir(), f'norma_{name}')
    os.makedirs(work_dir, exist_ok=True)

    # Find all image paths in the tex and copy them to work_dir
    img_paths = re.findall(r'\\includegraphics\[.*?\]\{(.*?)\}', tex_content)
    for img_path in img_paths:
        if os.path.exists(img_path):
            dst = os.path.join(work_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst)
            tex_content = tex_content.replace(img_path, os.path.basename(img_path))

    doc = r"""\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage[margin=0.3in]{geometry}
\usepackage{amsmath}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[font=sf,labelfont=bf]{caption}
\pagestyle{empty}
\begin{document}
""" + tex_content + r"""
\end{document}
"""
    tex_path = os.path.join(work_dir, f'{name}.tex')
    with open(tex_path, 'w') as f:
        f.write(doc)

    result = subprocess.run(['tectonic', tex_path],
                            capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        print(f'  LaTeX failed: {result.stderr[:500]}')
        return

    pdf_src = os.path.join(work_dir, f'{name}.pdf')
    pdf_dst = os.path.join(FIGDIR, f'{name}.pdf')
    shutil.move(pdf_src, pdf_dst)
    print(f'  {pdf_dst}')

    subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                     pdf_dst, pdf_dst.replace('.pdf', '')], capture_output=True)
    png = pdf_dst.replace('.pdf', '.png')
    if os.path.exists(png):
        print(f'  {png}')

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Within-person variability and mortality
# ═══════════════════════════════════════════════════════════════════════

def figure2():
    cohort_path = _path('results/manuscript_figures/figure2a_cohort.pdf')
    var_path = _path('results/validation/figures/variability.pdf')
    mort_path = _path('results/validation/figures/mortality_by_quintile.pdf')

    tex = r"""
\begin{figure}[p]

% Two-column layout: left = A + B stacked, right = C full height
\noindent
\begin{minipage}[t]{0.48\textwidth}
  % A — Cohort Selection (top left)
  \raggedright\textbf{\textsf{A}} \textsf{Cohort Selection}\\[4pt]
  \includegraphics[width=\textwidth]{""" + cohort_path + r"""}

  \vspace{0.4cm}

  % B — Variability (bottom left)
  \raggedright\textbf{\textsf{B}} \textsf{Intra-Patient vs Inter-Patient Variation}\\[4pt]
  \includegraphics[width=\textwidth]{""" + var_path + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
  % C — Mortality (full right column)
  \raggedright\textbf{\textsf{C}} \textsf{Mortality Association}\\[4pt]
  \includegraphics[width=\textwidth]{""" + mort_path + r"""}
\end{minipage}

\end{figure}
"""
    compile_figure('figure2', tex)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: NORMA architecture and predictive performance
# ═══════════════════════════════════════════════════════════════════════

def figure3():
    arch_path = _path('results/manuscript_figures/figure3a_architecture.pdf')
    perf_path = _path('results/prediction/figures/norma_quantile_forecasting_accuracy.pdf')
    r2_path = _path('results/prediction/figures/norma_quantile_analyte_r2.pdf')
    sens_path = _path('results/prediction/figures/norma_quantile_sensitivity.pdf')

    tex = r"""
\begin{figure}[p]

% Row 1: A (left, height-matched) | B + C stacked (right)
\noindent
\begin{minipage}[t]{0.25\textwidth}
  \raggedright\textbf{\textsf{A}} \textsf{NORMA Architecture}\\[4pt]
  \includegraphics[width=\textwidth,height=0.55\textheight,keepaspectratio]{""" + arch_path + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.72\textwidth}
  \raggedright\textbf{\textsf{B}} \textsf{NORMA Forecasting Performance}\\[4pt]
  \includegraphics[width=\textwidth]{""" + perf_path + r"""}

  \vspace{0.2cm}

  \raggedright\textbf{\textsf{C}} \textsf{Analyte-Specific Performance}\\[4pt]
  \includegraphics[width=\textwidth]{""" + r2_path + r"""}
\end{minipage}

\vspace{0.2cm}

% D — Sensitivity analysis (full width)
\noindent
\begin{minipage}[t]{\textwidth}
  \raggedright\textbf{\textsf{D}} \textsf{Sensitivity Analysis}\\[4pt]
  \includegraphics[width=\textwidth]{""" + sens_path + r"""}
\end{minipage}

\end{figure}
"""
    compile_figure('figure3', tex)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4/5: Clinical outcome prediction (per dataset)
# ═══════════════════════════════════════════════════════════════════════

def figure4():
    """CHS: A prevalence | B (placeholder reclassification)
            C circos 3×3 (rows=metrics, cols=outcomes) | D AUC bars
    """
    ds = 'chs'
    prev = _path(f'results/{ds}/figures/prevalence.pdf')
    circos = _path(f'results/{ds}/figures/eval_circos.pdf')
    concordance = _path(f'results/{ds}/figures/cox_concordance.pdf')

    tex = r"""
\begin{figure}[p]

% Row 1
\noindent
\begin{minipage}[t]{0.55\textwidth}
  \raggedright\textbf{\textsf{A}} \textsf{Abnormality Prevalence}\\[4pt]
  \includegraphics[width=\textwidth]{""" + prev + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.40\textwidth}
  \raggedright\textbf{\textsf{B}} \textsf{Albumin Reclassification}\\[4pt]
  \vspace{2cm}
  \centering\textit{(Schematic placeholder)}
  \vspace{2cm}
\end{minipage}

\vspace{0.5cm}

% Row 2
\noindent
\begin{minipage}[t]{0.60\textwidth}
  \raggedright\textbf{\textsf{C}} \textsf{Recall, Specificity, Precision Trade-Offs}\\[4pt]
  \includegraphics[width=\textwidth]{""" + circos + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.35\textwidth}
  \raggedright\textbf{\textsf{D}} \textsf{Time-to-Event Discrimination}\\[4pt]
  \includegraphics[width=\textwidth]{""" + concordance + r"""}
\end{minipage}

\end{figure}
"""
    compile_figure('figure4', tex)


def figure5():
    """eICU: Row-based layout.
    Row 1: A reclassification | B hazard ratios
    Row 2: C ROC
    Row 3: E circos | D lead time
    """
    ds = 'eicu'
    prev = _path(f'results/{ds}/figures/reclassification.pdf')
    forest = _path(f'results/{ds}/figures/cox_forest.pdf')
    roc = _path(f'results/{ds}/figures/eval_roc.pdf')
    circos = _path(f'results/{ds}/figures/eval_circos.pdf')
    lead = _path(f'results/{ds}/figures/lead_time.pdf')

    tex = r"""
\begin{figure}[p]

% Row 1: A (prevalence) | B (hazard ratios) — matched height
\noindent
\begin{minipage}[t]{0.63\textwidth}
  \raggedright\textbf{\textsf{A}} \textsf{Reclassification Rate}\\[4pt]
  \includegraphics[width=\textwidth]{""" + prev + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.33\textwidth}
  \raggedright\textbf{\textsf{B}} \textsf{Hazard Ratios}\\[4pt]
  \includegraphics[width=\textwidth]{""" + forest + r"""}
\end{minipage}

\vspace{0.2cm}

% Row 2: C (ROC scatter) — full width
\noindent
\begin{minipage}[t]{\textwidth}
  \raggedright\textbf{\textsf{C}} \textsf{Sensitivity vs False Positive Rate}\\[4pt]
  \includegraphics[width=\textwidth]{""" + roc + r"""}
\end{minipage}

\vspace{0.2cm}

% Row 3: E (circos) | D (lead time)
\noindent
\begin{minipage}[t]{0.63\textwidth}
  \raggedright\textbf{\textsf{E}} \textsf{Recall, Specificity, Precision Trade-Offs}\\[4pt]
  \includegraphics[width=\textwidth]{""" + circos + r"""}
\end{minipage}
\hfill
\begin{minipage}[t]{0.33\textwidth}
  \raggedright\textbf{\textsf{D}} \textsf{Time-to-Event Discrimination}\\[4pt]
  \includegraphics[width=\textwidth]{""" + lead + r"""}
\end{minipage}

\end{figure}
"""
    compile_figure('figure5', tex)


# ═══════════════════════════════════════════════════════════════════════
# Supplementary: Gaussian model performance (combined)
# ═══════════════════════════════════════════════════════════════════════

def supp_gaussian():
    """Gaussian model: forecasting accuracy + analyte R² + sensitivity in one figure."""
    perf_path = _path('results/prediction/figures/norma_gaussian_forecasting_accuracy.pdf')
    r2_path = _path('results/prediction/figures/norma_gaussian_analyte_r2.pdf')
    sens_path = _path('results/prediction/figures/norma_gaussian_sensitivity.pdf')

    tex = r"""
\begin{figure}[p]

% A — Forecasting accuracy
\noindent
\begin{minipage}[t]{\textwidth}
  \raggedright\textbf{\textsf{A}} \textsf{NORMA (Gaussian) Forecasting Performance}\\[4pt]
  \includegraphics[width=\textwidth]{""" + perf_path + r"""}
\end{minipage}

\vspace{0.3cm}

% B — Per-analyte R²
\noindent
\begin{minipage}[t]{\textwidth}
  \raggedright\textbf{\textsf{B}} \textsf{Analyte-Specific Performance}\\[4pt]
  \includegraphics[width=\textwidth]{""" + r2_path + r"""}
\end{minipage}

\vspace{0.3cm}

% C — Sensitivity analysis
\noindent
\begin{minipage}[t]{\textwidth}
  \raggedright\textbf{\textsf{C}} \textsf{Sensitivity Analysis}\\[4pt]
  \includegraphics[width=\textwidth]{""" + sens_path + r"""}
\end{minipage}

\end{figure}
"""
    compile_figure('supp_gaussian', tex)


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

FIGURES = {
    'fig2': figure2,
    'fig3': figure3,
    'fig4': figure4,
    'fig5': figure5,
    'supp_gaussian': supp_gaussian,
}


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURES.keys())
    for name in targets:
        if name not in FIGURES:
            print(f'Unknown: {name}. Available: {", ".join(FIGURES.keys())}')
            sys.exit(1)
    for name in targets:
        print(f'\n  Building {name}...')
        FIGURES[name]()
        print(f'  Done.')


if __name__ == '__main__':
    main()
