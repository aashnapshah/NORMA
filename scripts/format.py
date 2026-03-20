#!/usr/bin/env python
"""
Generate all formatted tables and figures, sync to manuscript/, and compile.

Usage:
    python scripts/format.py              # everything
    python scripts/format.py cohort       # just cohort
    python scripts/format.py prediction   # just prediction
    python scripts/format.py validation   # just validation
"""
import subprocess
import sys
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(SCRIPT_DIR)
MANUSCRIPT_DIR = os.path.join(ROOTDIR, 'manuscript')

HELPERS_DIR = os.path.join(SCRIPT_DIR, 'helpers')

MODULES = {
    'cohort': os.path.join(HELPERS_DIR, 'format_cohort.py'),
    'prediction': os.path.join(HELPERS_DIR, 'format_prediction.py'),
    'validation': os.path.join(HELPERS_DIR, 'format_validation.py'),
}

# Figures to sync: (source relative to ROOTDIR, dest filename in manuscript/figures/)
FIGURE_MAP = {
    # Main figures
    'results/manuscript_figures/figure1.pdf': 'figure1.pdf',
    'results/manuscript_figures/figure2.pdf': 'figure2.pdf',
    'results/manuscript_figures/figure3.pdf': 'figure3.pdf',
    'results/manuscript_figures/figure4.pdf': 'figure4.pdf',
    'results/manuscript_figures/figure5.pdf': 'figure5.pdf',
    # Supplemental figures
    'results/prediction/figures/token_example.pdf': 'token_example.pdf',
    'results/manuscript_figures/supp_gaussian.pdf': 'supp_gaussian.pdf',
    'results/eicu/figures/prevalence.pdf': 'eicu_prevalence.pdf',
    'results/chs/figures/prevalence.pdf': 'chs_prevalence.pdf',
    'results/eicu/figures/cox_concordance.pdf': 'cox_concordance.pdf',
    'results/eicu/figures/mortality_by_deviation.pdf': 'mortality_by_deviation.pdf',
    'results/eicu/figures/age_ri.pdf': 'age_ri.pdf',
}

# Tables to sync: (source relative to ROOTDIR, dest filename in manuscript/tables/)
TABLE_MAP = {
    # Cohort
    'results/tables/latex/analyte_reference.tex': 'analyte_reference.tex',
    'results/tables/latex/cohort_summary.tex': 'cohort_summary.tex',
    'results/tables/latex/design_choices.tex': 'design_choices.tex',
    # Prediction
    'results/prediction/tables/latex/prediction_performance.tex': 'prediction_performance.tex',
    'results/prediction/tables/latex/analyte_performance.tex': 'analyte_performance.tex',
    'results/prediction/tables/latex/sensitivity_summary.tex': 'sensitivity_summary.tex',
    'results/prediction/tables/latex/sensitivity_norma_quantile.tex': 'sensitivity_norma_quantile.tex',
    'results/prediction/tables/latex/sensitivity_norma_gaussian.tex': 'sensitivity_norma_gaussian.tex',
    # Validation — combined
    'results/validation/tables/latex/variability.tex': 'variability.tex',
    'results/validation/tables/latex/prevalence.tex': 'prevalence.tex',
    'results/validation/tables/latex/eval_summary.tex': 'eval_summary.tex',
    # Validation — eICU per-outcome
    'results/eicu/tables/latex/eval_aki.tex': 'eval_aki.tex',
    'results/eicu/tables/latex/eval_mortality.tex': 'eicu_eval_mortality.tex',
    'results/eicu/tables/latex/eval_prolonged_los.tex': 'eval_prolonged_los.tex',
    'results/eicu/tables/latex/eval_sepsis.tex': 'eval_sepsis.tex',
    'results/eicu/tables/latex/lead_time.tex': 'lead_time.tex',
    # Validation — CHS per-outcome
    'results/chs/tables/latex/eval_ckd.tex': 'eval_ckd.tex',
    'results/chs/tables/latex/eval_mortality.tex': 'chs_eval_mortality.tex',
    'results/chs/tables/latex/eval_t2d.tex': 'eval_t2d.tex',
}


def run_formatter(name, path):
    print(f'\n{"=" * 60}')
    print(f'  {name.upper()}')
    print(f'{"=" * 60}')
    return subprocess.run([sys.executable, path], cwd=ROOTDIR).returncode


def sync_manuscript():
    """Copy latest figures and tables into manuscript/ for Overleaf."""
    print(f'\n{"=" * 60}')
    print(f'  SYNCING TO MANUSCRIPT/')
    print(f'{"=" * 60}')

    fig_dir = os.path.join(MANUSCRIPT_DIR, 'figures')
    tab_dir = os.path.join(MANUSCRIPT_DIR, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    synced, skipped = 0, 0
    for src_rel, dst_name in FIGURE_MAP.items():
        src = os.path.join(ROOTDIR, src_rel)
        dst = os.path.join(fig_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            synced += 1
        else:
            skipped += 1
            print(f'  [skip] {src_rel} not found')

    for src_rel, dst_name in TABLE_MAP.items():
        src = os.path.join(ROOTDIR, src_rel)
        dst = os.path.join(tab_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            synced += 1
        else:
            skipped += 1
            print(f'  [skip] {src_rel} not found')

    print(f'  Synced {synced} files, skipped {skipped}')


def compile_supplement():
    """Compile the manuscript supplement PDF."""
    print(f'\n{"=" * 60}')
    print(f'  COMPILING SUPPLEMENT')
    print(f'{"=" * 60}')
    supp_tex = os.path.join(MANUSCRIPT_DIR, 'supplement.tex')
    if os.path.exists(supp_tex):
        result = subprocess.run(
            ['tectonic', supp_tex],
            capture_output=True, text=True,
            cwd=MANUSCRIPT_DIR,
        )
        if result.returncode == 0:
            print(f'  Compiled → {supp_tex.replace(".tex", ".pdf")}')
        else:
            print(f'  Compilation failed: {result.stderr[:500]}')
    else:
        print('  supplement.tex not found, skipping')


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODULES.keys())

    for name in targets:
        if name not in MODULES:
            print(f'Unknown: {name}. Available: {", ".join(MODULES.keys())}')
            sys.exit(1)

    # Run formatters
    failures = []
    for name in targets:
        if run_formatter(name, MODULES[name]) != 0:
            failures.append(name)

    # Build composite manuscript figures
    make_figures = os.path.join(SCRIPT_DIR, 'make_figures.py')
    if os.path.exists(make_figures):
        print(f'\n{"=" * 60}')
        print(f'  MAKE FIGURES')
        print(f'{"=" * 60}')
        if subprocess.run([sys.executable, make_figures], cwd=ROOTDIR).returncode != 0:
            failures.append('make_figures')

    # Sync to manuscript/
    sync_manuscript()

    # Compile supplement
    compile_supplement()

    if failures:
        print(f'\nWarning: {", ".join(failures)} had errors')


if __name__ == '__main__':
    main()
