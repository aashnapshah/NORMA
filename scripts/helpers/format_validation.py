"""
Format validation results: combined tables and figures across datasets.

Auto-detects which datasets (eICU, CHS) have variability.csv available.
Produces combined tables and figures in results/validation/.

Usage:
    python scripts/format_validation.py
    python scripts/format_validation.py --tables variability
    python scripts/format_validation.py --figures variability
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots import (plt, hide_spines, scheme_colors, analyte_colors, PALETTE,
                    EXCLUDE_ANALYTES, ANALYTE_NAMES, save_fig, save_table, tex_escape)

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOTDIR, 'results', 'validation')

ALL_DATASETS = ['eicu', 'chs']
DATASET_LABELS = {'eicu': 'eICU', 'chs': 'CHS'}
DATASET_MARKERS = {'eicu': 'o', 'chs': 's'}


def load_variability(dataset):
    """Load variability.csv for a dataset, or None if missing."""
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'variability.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    df = df[~df['analyte'].isin(EXCLUDE_ANALYTES)]
    return df.set_index('analyte')


def load_mortality(dataset):
    """Load mortality_by_quintile.csv for a dataset, or None if missing."""
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'mortality_by_quintile.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def available_datasets(filename='variability.csv'):
    """Return list of datasets that have the given raw file."""
    return [ds for ds in ALL_DATASETS
            if os.path.exists(os.path.join(ROOTDIR, 'results', ds, 'raw', filename))]


# ── Variability ──────────────────────────────────────────────────────────────

def _single_variability_table(ds, present, all_analytes):
    """Build table for one dataset."""
    label = DATASET_LABELS[ds]
    csv_rows = []
    for analyte in all_analytes:
        if analyte in present.index:
            r = present.loc[analyte]
            csv_rows.append({
                'Analyte': analyte,
                'N': int(r['n_patients']),
                'CV_intra': f'{r["cv_intra"]:.1f} [{r["cv_intra_ci_lower"]:.1f}, {r["cv_intra_ci_upper"]:.1f}]',
                'CV_inter': f'{r["cv_inter"]:.1f} [{r["cv_inter_ci_lower"]:.1f}, {r["cv_inter_ci_upper"]:.1f}]',
                'II': f'{r["individuality_index"]:.1f} [{r["ii_ci_lower"]:.1f}, {r["ii_ci_upper"]:.1f}]',
            })
        else:
            csv_rows.append({
                'Analyte': analyte, 'N': '---', 'CV_intra': '---', 'CV_inter': '---', 'II': '---',
            })

    csv_df = pd.DataFrame(csv_rows)

    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Analyte & N & CV$_{\text{intra}}$ [95\% CI] & CV$_{\text{inter}}$ [95\% CI] & II [95\% CI] \\',
             r'\midrule']
    for row in csv_rows:
        n = f'{row["N"]:,}' if isinstance(row['N'], int) else '---'
        cells = [row['Analyte'], n, row['CV_intra'], row['CV_inter'], row['II']]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
    save_table('variability', '\n'.join(lines), csv_df, tdir)


def _combined_variability_table(datasets, data, all_analytes):
    """Build combined table across datasets."""
    n_ds = len(datasets)

    csv_rows = []
    for analyte in all_analytes:
        row = {'Analyte': analyte}
        for ds in datasets:
            label = DATASET_LABELS[ds]
            present = data[ds]
            if analyte in present.index:
                r = present.loc[analyte]
                row[f'{label} N'] = int(r['n_patients'])
                row[f'{label} CV_intra'] = f'{r["cv_intra"]:.1f} [{r["cv_intra_ci_lower"]:.1f}, {r["cv_intra_ci_upper"]:.1f}]'
                row[f'{label} CV_inter'] = f'{r["cv_inter"]:.1f} [{r["cv_inter_ci_lower"]:.1f}, {r["cv_inter_ci_upper"]:.1f}]'
                row[f'{label} II'] = f'{r["individuality_index"]:.1f} [{r["ii_ci_lower"]:.1f}, {r["ii_ci_upper"]:.1f}]'
            else:
                row[f'{label} N'] = '---'
                row[f'{label} CV_intra'] = '---'
                row[f'{label} CV_inter'] = '---'
                row[f'{label} II'] = '---'
        csv_rows.append(row)

    csv_df = pd.DataFrame(csv_rows)

    n_cols = 4
    col_spec = 'l' + 'rrrr' * n_ds
    lines = [r'\begin{table}[ht]', r'\centering', r'\tiny',
             r'\begin{tabular}{' + col_spec + '}', r'\toprule']

    h1 = ' '
    cmidrules = []
    for i, ds in enumerate(datasets):
        start = 2 + i * n_cols
        h1 += r' & \multicolumn{' + str(n_cols) + r'}{c}{' + DATASET_LABELS[ds] + '}'
        cmidrules.append(f'\\cmidrule(lr){{{start}-{start + n_cols - 1}}}')
    lines.append(h1 + r' \\')
    lines.append(' '.join(cmidrules))

    h2 = 'Analyte'
    for _ in datasets:
        h2 += r' & N & CV$_{\text{intra}}$ & CV$_{\text{inter}}$ & II'
    lines.append(h2 + r' \\')
    lines.append(r'\midrule')

    for row in csv_rows:
        cells = [row['Analyte']]
        for ds in datasets:
            label = DATASET_LABELS[ds]
            n = row[f'{label} N']
            cells.append(f'{n:,}' if isinstance(n, int) else '---')
            cells.append(row[f'{label} CV_intra'])
            cells.append(row[f'{label} CV_inter'])
            cells.append(row[f'{label} II'])
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    tdir = os.path.join(OUT_DIR, 'tables')
    save_table('variability', '\n'.join(lines), csv_df, tdir, landscape=True)


def table_variability():
    datasets = available_datasets()
    data = {ds: load_variability(ds) for ds in datasets}
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    # Per-dataset tables
    for ds in datasets:
        print(f'  {DATASET_LABELS[ds]}:')
        _single_variability_table(ds, data[ds], all_analytes)

    # Combined table
    if len(datasets) > 1:
        print(f'  Combined:')
        _combined_variability_table(datasets, data, all_analytes)


DATASET_COLORS = {
    'eicu': {'inter': PALETTE['teal'], 'intra': '#66C7D2'},
    'chs':  {'inter': PALETTE['terracotta'], 'intra': '#DEBB9C'},
}


def fig_variability():
    datasets = available_datasets()
    data = {ds: load_variability(ds) for ds in datasets}
    n_ds = len(datasets)

    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)
    x = np.arange(len(all_analytes))

    # 2 bars per dataset, small gap between datasets
    bar_width = 0.7 / (n_ds * 2)
    gap = bar_width * 0.3

    fig, (ax_cv, ax_ii) = plt.subplots(
        2, 1, figsize=(5.5, 5.5),
        gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.15},
        sharex=True,
    )

    # CV bars (as %) with error bars
    for di, ds in enumerate(datasets):
        present = data[ds]
        ds_label = DATASET_LABELS[ds]
        colors = DATASET_COLORS.get(ds, {'inter': '#999999', 'intra': '#CCCCCC'})

        # Center each dataset's pair, with gap between datasets
        group_offset = (di - (n_ds - 1) / 2) * (bar_width * 2 + gap)

        for kind, color_key, shift, cv_label in [
            ('cv_inter', 'inter', -bar_width / 2, 'Inter-Patient'),
            ('cv_intra', 'intra', bar_width / 2, 'Intra-Patient'),
        ]:
            vals = []
            err_lo = []
            err_hi = []
            for a in all_analytes:
                if a in present.index:
                    v = present.loc[a, kind] * 100
                    vals.append(v)
                    if f'{kind}_ci_lower' in present.columns:
                        lo = v - present.loc[a, f'{kind}_ci_lower'] * 100
                        hi = present.loc[a, f'{kind}_ci_upper'] * 100 - v
                        err_lo.append(max(lo, 0))
                        err_hi.append(max(hi, 0))
                    else:
                        err_lo.append(0)
                        err_hi.append(0)
                else:
                    vals.append(0)
                    err_lo.append(0)
                    err_hi.append(0)

            color = colors[color_key]
            label = f'{cv_label} ({ds_label})' if n_ds > 1 else cv_label
            ax_cv.bar(x + group_offset + shift, vals, bar_width,
                      color=color, alpha=0.85, label=label,
                      edgecolor='white', linewidth=0.3)
            ax_cv.errorbar(x + group_offset + shift, vals,
                           yerr=[err_lo, err_hi],
                           fmt='none', ecolor='#333333', elinewidth=0.5,
                           capsize=1, alpha=0.6)

    ax_cv.set_ylabel('Coefficient of Variation (%)', labelpad=10, fontsize=9)
    ax_cv.yaxis.set_label_coords(-0.06, 0.5)
    ax_cv.tick_params(axis='both', labelsize=7)
    ax_cv.legend(loc='upper right', frameon=False, fontsize=7)
    hide_spines(ax_cv)

    # Index of Individuality
    ds_ii_colors = {'eicu': PALETTE['teal'], 'chs': PALETTE['terracotta']}
    for di, ds in enumerate(datasets):
        present = data[ds]
        marker = DATASET_MARKERS.get(ds, 'o')
        ds_label = DATASET_LABELS[ds]
        color = ds_ii_colors.get(ds, 'black')

        ii_vals = []
        ii_lo = []
        ii_hi = []
        x_valid = []
        for i, a in enumerate(all_analytes):
            if a in present.index:
                ii = present.loc[a, 'individuality_index']
                ii_vals.append(ii)
                x_valid.append(i)
                if 'ii_ci_lower' in present.columns:
                    ii_lo.append(ii - present.loc[a, 'ii_ci_lower'])
                    ii_hi.append(present.loc[a, 'ii_ci_upper'] - ii)
                else:
                    ii_lo.append(0)
                    ii_hi.append(0)

        offset = (di - (n_ds - 1) / 2) * 0.15
        ax_ii.errorbar(np.array(x_valid) + offset, ii_vals,
                       yerr=[ii_lo, ii_hi],
                       fmt=marker, color=color, markersize=4,
                       elinewidth=0.5, capsize=1, alpha=0.8,
                       label=ds_label)

    ax_ii.axhline(0.6, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax_ii.set_ylabel('Index of Individuality', labelpad=10, fontsize=9)
    ax_ii.yaxis.set_label_coords(-0.06, 0.5)
    ax_ii.tick_params(axis='both', labelsize=7)
    ax_ii.set_xticks(x)
    ax_ii.set_xticklabels(all_analytes, rotation=90, ha='center')
    if n_ds > 1:
        ax_ii.legend(frameon=False)

    # Grey out missing labels (missing from all datasets)
    for i, label in enumerate(ax_ii.get_xticklabels()):
        a = all_analytes[i]
        if not any(a in data[ds].index for ds in data):
            label.set_color('lightgray')

    hide_spines(ax_ii)

    fdir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(fdir, exist_ok=True)
    save_fig(os.path.join(fdir, 'variability'))


# ── Mortality Association ────────────────────────────────────────────────────

def table_mortality():
    datasets = available_datasets('mortality_by_quintile.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_mortality(ds)
        label = DATASET_LABELS[ds]
        present_analytes = set(df['analyte'].unique())

        csv_rows = []
        for analyte in all_analytes:
            row = {'Analyte': analyte}
            if analyte in present_analytes:
                sub = df[df['analyte'] == analyte].sort_values('q_label')
                for _, r in sub.iterrows():
                    q = int(r['q_label'])
                    row[f'Q{q} Mortality (%)'] = f'{r["mortality_pct"]:.1f}'
                    row[f'Q{q} N'] = int(r['n'])
            else:
                for q in range(1, 6):
                    row[f'Q{q} Mortality (%)'] = '---'
                    row[f'Q{q} N'] = '---'
            csv_rows.append(row)

        csv_df = pd.DataFrame(csv_rows)
        tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
        # CSV only (wide table, skip LaTeX)
        csv_dir = os.path.join(tdir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'mortality_by_quintile.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f'  {label}: {csv_path}')


def fig_mortality():
    datasets = available_datasets('mortality_by_quintile.csv')
    data = {ds: load_mortality(ds) for ds in datasets}
    n_ds = len(datasets)

    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)
    n_analytes = len(all_analytes)

    ncols = 4
    nrows = int(np.ceil(n_analytes / ncols))

    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.3),
                             sharex=True, sharey=False)
    axes = axes.flatten()

    ds_colors = {'eicu': PALETTE['teal'], 'chs': PALETTE['terracotta']}

    for i, analyte in enumerate(all_analytes):
        ax = axes[i]
        has_any = False

        for ds in datasets:
            df = data[ds]
            sub = df[df['analyte'] == analyte].sort_values('q_label')
            if len(sub) == 0 or sub['n'].min() < 100:
                continue
            has_any = True
            color = ds_colors.get(ds, 'black')
            ds_label = DATASET_LABELS[ds]

            ax.plot(sub['q_label'], sub['mortality_pct'],
                    '-o', color=color, markersize=3, linewidth=1,
                    label=ds_label if i == 0 else None)
            ax.fill_between(sub['q_label'], sub['ci_lo'], sub['ci_hi'],
                            color=color, alpha=0.15)

        ax.set_title(analyte, fontsize=9, fontweight='bold', pad=4)
        if not has_any:
            ax.set_title(analyte, fontsize=9, pad=4, color='lightgray')
        ax.set_xticks([1, 2, 3, 4, 5])
        if i >= (nrows - 1) * ncols:
            ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='y', labelsize=7)
        hide_spines(ax)

    # Hide unused axes
    for j in range(n_analytes, len(axes)):
        axes[j].set_visible(False)

    # Shared labels
    fig.text(0.5, -0.01, 'Quintile', ha='center', fontsize=9)
    fig.text(-0.02, 0.5, '5-Year Mortality Rate (%)', va='center', rotation='vertical', fontsize=9)

    # Legend
    if n_ds > 1:
        handles = [plt.Line2D([0], [0], color=ds_colors.get(ds, 'black'), lw=2,
                              label=DATASET_LABELS[ds]) for ds in datasets]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
                   ncol=n_ds, frameon=False, fontsize=9)

    plt.tight_layout()

    fdir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(fdir, exist_ok=True)
    save_fig(os.path.join(fdir, 'mortality_by_quintile'))


def fig_mortality_deviation():
    """Mortality by standardized deviation from personal baseline (Supp Fig 1)."""
    datasets = available_datasets('mortality_by_deviation.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        path = os.path.join(ROOTDIR, 'results', ds, 'raw', 'mortality_by_deviation.csv')
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
        ds_label = DATASET_LABELS[ds]

        analytes_present = sorted(df['analyte'].unique())
        n = len(analytes_present)
        ncols = 6
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8),
                                 sharex=True)
        axes = axes.flatten()

        for i, analyte in enumerate(analytes_present):
            ax = axes[i]
            sub = df[df['analyte'] == analyte].sort_values('decile')

            ax.plot(sub['z_median'], sub['mortality_pct'],
                    '-o', color=PALETTE['teal'], markersize=3, linewidth=1)
            ax.fill_between(sub['z_median'], sub['ci_lo'], sub['ci_hi'],
                            color=PALETTE['teal'], alpha=0.15)

            ax.set_title(analyte, fontsize=9, fontweight='bold', pad=4)
            ax.tick_params(axis='both', labelsize=6)
            hide_spines(ax)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.text(0.5, -0.02, 'Standardized Distance from Baseline (|z|)', ha='center', fontsize=9)
        fig.text(-0.02, 0.5, 'Mortality Rate (%)', va='center', rotation='vertical', fontsize=9)
        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'mortality_by_deviation'))


# ── Abnormality Prevalence ───────────────────────────────────────────────────

PREV_METHODS = ['PopRI', 'PerRI', 'NORMA']
PREV_METHOD_COLORS = {
    'PopRI': scheme_colors['Population'],
    'PerRI': scheme_colors['Personalized'],
    'NORMA': scheme_colors['NORMA'],
}


def load_prevalence(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'prevalence.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def table_prevalence():
    datasets = available_datasets('prevalence.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_prevalence(ds)
        if df is None:
            continue
        label = DATASET_LABELS[ds]
        tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
        present = df.set_index('analyte')

        csv_rows = []
        for analyte in all_analytes:
            if analyte in present.index:
                r = present.loc[analyte]
                n_total = int(r['n']) if pd.notna(r['n']) else None
                row = {
                    'Analyte': analyte,
                    'N': f'{n_total:,}' if n_total else '---',
                    'PopRI (%)': f'{r["PopRI_pct"]:.1f}' if pd.notna(r.get('PopRI_pct')) else '---',
                    'PerRI (%)': f'{r["PerRI_pct"]:.1f}' if pd.notna(r.get('PerRI_pct')) else '---',
                    'PerRI RR (%)': f'{r["PerRI_reclass_pct"]:.1f}' if pd.notna(r.get('PerRI_reclass_pct')) else '---',
                    'NORMA (%)': f'{r["NORMA_pct"]:.1f}' if pd.notna(r.get('NORMA_pct')) else '---',
                    'NORMA RR (%)': f'{r["NORMA_reclass_pct"]:.1f}' if pd.notna(r.get('NORMA_reclass_pct')) else '---',
                }
            else:
                row = {'Analyte': analyte, 'N': '---',
                       'PopRI (%)': '---',
                       'PerRI (%)': '---', 'PerRI RR (%)': '---',
                       'NORMA (%)': '---', 'NORMA RR (%)': '---'}
            csv_rows.append(row)

        csv_df = pd.DataFrame(csv_rows)

        # LaTeX — group prevalence + RR under each method
        lines = [r'\begin{table}[ht]', r'\centering', r'\small',
                 r'\begin{tabular}{lrrrrrr}', r'\toprule',
                 r' & & \multicolumn{1}{c}{Pop$_{RI}$} & \multicolumn{2}{c}{Per$_{RI}$} & \multicolumn{2}{c}{NORMA$_{RI}$} \\',
                 r'\cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}',
                 r'Analyte & N & Abn (\%) & Abn (\%) & RR (\%) & Abn (\%) & RR (\%) \\',
                 r'\midrule']
        for row in csv_rows:
            cells = [
                row['Analyte'],
                str(row['N']),
                str(row['PopRI (%)']),
                str(row['PerRI (%)']), str(row['PerRI RR (%)']),
                str(row['NORMA (%)']), str(row['NORMA RR (%)']),
            ]
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        save_table('prevalence', '\n'.join(lines), csv_df, tdir)
        print(f'  {label}: done')

    # Combined prevalence summary (mean across analytes per dataset)
    if len(datasets) > 1:
        summary_rows = []
        for ds in datasets:
            df = load_prevalence(ds)
            if df is None:
                continue
            ds_label = DATASET_LABELS[ds]
            present = df[df['analyte'] != 'Overall'].set_index('analyte')
            summary_rows.append({
                'Dataset': ds_label,
                'N analytes': len(present),
                'PopRI (%)': f'{present["PopRI_pct"].mean():.1f}',
                'PerRI (%)': f'{present["PerRI_pct"].mean():.1f}',
                'NORMA (%)': f'{present["NORMA_pct"].mean():.1f}',
                'PerRI RR (%)': f'{present["PerRI_reclass_pct"].mean():.1f}' if 'PerRI_reclass_pct' in present.columns else '---',
                'NORMA RR (%)': f'{present["NORMA_reclass_pct"].mean():.1f}' if 'NORMA_reclass_pct' in present.columns else '---',
            })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            lines = [r'\begin{table}[ht]', r'\centering', r'\small',
                     r'\begin{tabular}{lrrrrrr}', r'\toprule',
                     r'Dataset & N & Pop$_{RI}$ (\%) & Per$_{RI}$ (\%) & NORMA$_{RI}$ (\%) & Per$_{RI}$ RR (\%) & NORMA$_{RI}$ RR (\%) \\',
                     r'\midrule']
            for row in summary_rows:
                cells = [row['Dataset'], str(row['N analytes']),
                         row['PopRI (%)'], row['PerRI (%)'], row['NORMA (%)'],
                         row['PerRI RR (%)'], row['NORMA RR (%)']]
                lines.append(' & '.join(cells) + r' \\')
            lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

            tdir = os.path.join(OUT_DIR, 'tables')
            save_table('prevalence', '\n'.join(lines), summary_df, tdir)


def fig_prevalence():
    datasets = available_datasets('prevalence.csv')
    data = {ds: load_prevalence(ds) for ds in datasets}
    n_ds = len(datasets)

    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)
    x = np.arange(len(all_analytes))

    method_cols = [
        ('PopRI_pct', 'Pop$_{RI}$', 'PopRI'),
        ('PerRI_pct', 'Per$_{RI}$', 'PerRI'),
        ('NORMA_pct', 'NORMA$_{RI}$', 'NORMA'),
    ]

    for ds in datasets:
        df = data[ds].set_index('analyte')
        ds_label = DATASET_LABELS[ds]

        fig, ax = plt.subplots(figsize=(7, 3))

        for col, display_label, color_key in method_cols:
            color = PREV_METHOD_COLORS[color_key]
            vals = [df.loc[a, col] if a in df.index and pd.notna(df.loc[a, col]) else np.nan
                    for a in all_analytes]
            ax.scatter(x, vals, color=color, s=18, label=display_label, zorder=3, alpha=0.8)

        ax.set_ylabel('Abnormal Measurements (%)')
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        hide_spines(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(all_analytes, rotation=90)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'prevalence'))


def fig_reclassification():
    """Bar chart: reclassification rate (among PopRI-normal) per analyte."""
    datasets = available_datasets('prevalence.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        prev = load_prevalence(ds)
        if prev is None:
            continue
        prev = prev[prev['analyte'] != 'Overall'].set_index('analyte')

        fig, ax = plt.subplots(figsize=(max(8, len(all_analytes) * 0.4), 4.5))
        x = np.arange(len(all_analytes))
        bw = 0.35

        perri_rr = [prev.loc[a, 'PerRI_reclass_pct'] if a in prev.index and pd.notna(prev.loc[a, 'PerRI_reclass_pct']) else 0
                    for a in all_analytes]
        norma_rr = [prev.loc[a, 'NORMA_reclass_pct'] if a in prev.index and pd.notna(prev.loc[a, 'NORMA_reclass_pct']) else 0
                    for a in all_analytes]

        ax.bar(x - bw/2, perri_rr, bw, color=scheme_colors['Personalized'],
               alpha=0.8, label=r'Per$_{RI}$')
        ax.bar(x + bw/2, norma_rr, bw, color=scheme_colors['NORMA'],
               alpha=0.8, label=r'NORMA$_{RI}$')
        ax.set_xticks(x)
        ax.set_xticklabels(all_analytes, rotation=90, fontsize=7)
        ax.set_ylabel('Reclassification Rate (%)', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=8)
        hide_spines(ax)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'reclassification'))


# ── Eval Metrics ─────────────────────────────────────────────────────────────

EVAL_METRICS_COLS = ['ppv', 'sensitivity', 'specificity', 'accuracy']
EVAL_METRIC_LABELS = {'ppv': 'Prec', 'sensitivity': 'Sens', 'specificity': 'Spec', 'accuracy': 'Acc'}
EVAL_METHODS_RECLASS = ['PerRI', 'NORMA']  # PopRI is tautologically normal in this subset

OUTCOME_LABELS = {
    'mortality': 'Mortality',
    'aki': 'Acute Kidney Injury',
    'sepsis': 'Sepsis',
    'prolonged_los': 'Prolonged ICU Stay (>7d)',
    't2d': 'Type 2 Diabetes',
    'ckd': 'Chronic Kidney Disease',
}


def load_eval_restricted(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'eval_metrics_restricted.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    return df[df['subset'] == 'normal_setpoint_popri_normal']


def _ri_label(method):
    return method.replace('PopRI', r'Pop$_{RI}$').replace('PerRI', r'Per$_{RI}$').replace('NORMA', r'NORMA$_{RI}$')


def _get_metric(sub, analyte, method, metric):
    """Get a metric value, return float or NaN."""
    match = sub[(sub['analyte'] == analyte) & (sub['method'] == method)]
    if len(match) == 1 and pd.notna(match.iloc[0][metric]):
        return match.iloc[0][metric]
    return np.nan


def table_eval_metrics():
    datasets = available_datasets('eval_metrics_restricted.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        label = DATASET_LABELS[ds]
        tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
        df = load_eval_restricted(ds)
        if df is None or len(df) == 0:
            continue

        outcomes = sorted(df['outcome'].unique())

        # ── Summary table: mean + win count per outcome ──
        summary_rows = []
        for outcome in outcomes:
            sub = df[df['outcome'] == outcome]
            row = {'Outcome': OUTCOME_LABELS.get(outcome, outcome)}
            for metric in EVAL_METRICS_COLS:
                ml = EVAL_METRIC_LABELS[metric]
                per_vals = sub[sub['method'] == 'PerRI'][metric].dropna()
                norma_vals = sub[sub['method'] == 'NORMA'][metric].dropna()

                row[f'PerRI Mean {ml}'] = f'{per_vals.mean():.2f}' if len(per_vals) > 0 else '---'
                row[f'NORMA Mean {ml}'] = f'{norma_vals.mean():.2f}' if len(norma_vals) > 0 else '---'

                # Delta: mean of per-analyte deltas
                deltas = []
                for analyte in all_analytes:
                    p = _get_metric(sub, analyte, 'PerRI', metric)
                    n = _get_metric(sub, analyte, 'NORMA', metric)
                    if not (np.isnan(p) or np.isnan(n)):
                        deltas.append(n - p)
                if deltas:
                    row[f'Mean Δ {ml}'] = f'{np.mean(deltas):+.2f}'
                    # Higher is better for all three metrics
                    row[f'NORMA wins {ml}'] = f'{sum(d > 0 for d in deltas)}/{len(deltas)}'
                else:
                    row[f'Mean Δ {ml}'] = '---'
                    row[f'NORMA wins {ml}'] = '---'

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Summary LaTeX
        lines = [r'\begin{table}[ht]', r'\centering', r'\small',
                 r'\begin{tabular}{l' + 'rrrr' * len(EVAL_METRICS_COLS) + '}', r'\toprule']
        h1 = ' '
        cmidrules = []
        for i, metric in enumerate(EVAL_METRICS_COLS):
            start = 2 + i * 4
            h1 += r' & \multicolumn{4}{c}{' + EVAL_METRIC_LABELS[metric] + '}'
            cmidrules.append(f'\\cmidrule(lr){{{start}-{start + 3}}}')
        lines.append(h1 + r' \\')
        lines.append(' '.join(cmidrules))

        h2 = 'Outcome'
        for _ in EVAL_METRICS_COLS:
            h2 += r' & Per$_{RI}$ & NORMA$_{RI}$ & $\Delta$ & Wins'
        lines.append(h2 + r' \\')
        lines.append(r'\midrule')

        for row in summary_rows:
            cells = [row['Outcome']]
            for metric in EVAL_METRICS_COLS:
                ml = EVAL_METRIC_LABELS[metric]
                cells.extend([
                    str(row[f'PerRI Mean {ml}']),
                    str(row[f'NORMA Mean {ml}']),
                    str(row[f'Mean Δ {ml}']),
                    str(row[f'NORMA wins {ml}']),
                ])
            lines.append(' & '.join(cells) + r' \\')

        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        save_table('eval_summary', '\n'.join(lines), summary_df, tdir)

        # ── Detail table: N, Flagged, TP, PPV, Sens, Spec per method ──
        detail_rows = []
        for outcome in outcomes:
            sub = df[df['outcome'] == outcome]
            outcome_display = OUTCOME_LABELS.get(outcome, outcome)
            for analyte in all_analytes:
                # N and n_events are the same across methods
                any_match = sub[sub['analyte'] == analyte]
                if len(any_match) > 0:
                    n = int(any_match.iloc[0]['n'])
                    n_events = int(any_match.iloc[0]['n_events'])
                    row = {'Outcome': outcome_display, 'Analyte': analyte,
                           'N': f'{n:,}', 'Events': f'{n_events:,}'}
                else:
                    row = {'Outcome': outcome_display, 'Analyte': analyte,
                           'N': '---', 'Events': '---'}

                for method in EVAL_METHODS_RECLASS:
                    match = sub[(sub['analyte'] == analyte) & (sub['method'] == method)]
                    if len(match) == 1:
                        r = match.iloc[0]
                        row[f'{method} Precision'] = f'{r["ppv"]:.2f}' if pd.notna(r['ppv']) else '---'
                        row[f'{method} Sensitivity'] = f'{r["sensitivity"]:.2f}' if pd.notna(r['sensitivity']) else '---'
                        row[f'{method} Specificity'] = f'{r["specificity"]:.2f}' if pd.notna(r['specificity']) else '---'
                        row[f'{method} Accuracy'] = f'{r["accuracy"]:.2f}' if pd.notna(r.get('accuracy')) else '---'
                    else:
                        for k in ['Precision', 'Sensitivity', 'Specificity', 'Accuracy']:
                            row[f'{method} {k}'] = '---'
                detail_rows.append(row)

        detail_df = pd.DataFrame(detail_rows)

        # One LaTeX table per outcome
        n_per_method = 4  # PPV, Sens, Spec, Acc
        for outcome in outcomes:
            outcome_display = OUTCOME_LABELS.get(outcome, outcome)
            outcome_rows = [r for r in detail_rows if r['Outcome'] == outcome_display]

            lines = [r'\begin{table}[ht]', r'\centering', r'\footnotesize',
                     r'\begin{tabular}{lrr' + 'rrrr' * len(EVAL_METHODS_RECLASS) + '}', r'\toprule']
            h1 = r' & & '
            cmidrules = []
            for i, method in enumerate(EVAL_METHODS_RECLASS):
                start = 4 + i * n_per_method
                h1 += r' & \multicolumn{' + str(n_per_method) + r'}{c}{' + _ri_label(method) + '}'
                cmidrules.append(f'\\cmidrule(lr){{{start}-{start + n_per_method - 1}}}')
            lines.append(h1 + r' \\')
            lines.append(' '.join(cmidrules))

            h2 = 'Analyte & N & Events'
            for _ in EVAL_METHODS_RECLASS:
                h2 += r' & Prec & Sens & Spec & Acc'
            lines.append(h2 + r' \\')
            lines.append(r'\midrule')

            for row in outcome_rows:
                cells = [row['Analyte'], str(row['N']), str(row['Events'])]
                for method in EVAL_METHODS_RECLASS:
                    for k in ['Precision', 'Sensitivity', 'Specificity', 'Accuracy']:
                        cells.append(str(row[f'{method} {k}']))
                lines.append(' & '.join(cells) + r' \\')
            lines += [r'\bottomrule', r'\end{tabular}',
                      r'\caption{' + outcome_display + '}',
                      r'\end{table}']

            outcome_slug = outcome.lower().replace(' ', '_').replace('>', 'gt')
            save_table(f'eval_{outcome_slug}', '\n'.join(lines),
                       pd.DataFrame(outcome_rows), tdir)

        print(f'  {label}: done')

    # ── Combined summary across datasets ──
    if len(datasets) > 1:
        combined_rows = []
        for ds in datasets:
            df = load_eval_restricted(ds)
            if df is None or len(df) == 0:
                continue
            ds_label = DATASET_LABELS[ds]
            for outcome in sorted(df['outcome'].unique()):
                sub = df[df['outcome'] == outcome]
                row = {'Dataset': ds_label, 'Outcome': OUTCOME_LABELS.get(outcome, outcome)}
                for metric in EVAL_METRICS_COLS:
                    ml = EVAL_METRIC_LABELS[metric]
                    per_vals = sub[sub['method'] == 'PerRI'][metric].dropna()
                    norma_vals = sub[sub['method'] == 'NORMA'][metric].dropna()
                    row[f'PerRI {ml}'] = f'{per_vals.mean():.2f}' if len(per_vals) > 0 else '---'
                    row[f'NORMA {ml}'] = f'{norma_vals.mean():.2f}' if len(norma_vals) > 0 else '---'
                    if len(per_vals) > 0 and len(norma_vals) > 0:
                        row[f'Δ {ml}'] = f'{norma_vals.mean() - per_vals.mean():+.2f}'
                    else:
                        row[f'Δ {ml}'] = '---'
                combined_rows.append(row)

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            tdir = os.path.join(OUT_DIR, 'tables')

            lines = [r'\begin{table}[ht]', r'\centering', r'\small',
                     r'\begin{tabular}{ll' + 'rrr' * len(EVAL_METRICS_COLS) + '}', r'\toprule']
            h1 = ' & '
            cmidrules = []
            for i, metric in enumerate(EVAL_METRICS_COLS):
                start = 3 + i * 3
                h1 += r' & \multicolumn{3}{c}{' + EVAL_METRIC_LABELS[metric] + '}'
                cmidrules.append(f'\\cmidrule(lr){{{start}-{start + 2}}}')
            lines.append(h1 + r' \\')
            lines.append(' '.join(cmidrules))

            h2 = 'Dataset & Outcome'
            for _ in EVAL_METRICS_COLS:
                h2 += r' & Per$_{RI}$ & NORMA$_{RI}$ & $\Delta$'
            lines.append(h2 + r' \\')
            lines.append(r'\midrule')

            prev_ds = None
            for row in combined_rows:
                ds_cell = row['Dataset'] if row['Dataset'] != prev_ds else ''
                prev_ds = row['Dataset']
                cells = [ds_cell, row['Outcome']]
                for metric in EVAL_METRICS_COLS:
                    ml = EVAL_METRIC_LABELS[metric]
                    cells.extend([str(row[f'PerRI {ml}']), str(row[f'NORMA {ml}']), str(row[f'Δ {ml}'])])
                lines.append(' & '.join(cells) + r' \\')

            lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

            save_table('eval_summary', '\n'.join(lines), combined_df, tdir)
            print(f'  Combined: done')


def fig_eval_metrics():
    """Radial bar chart (circos-style): Δ(NORMA - PerRI) per analyte.

    Grid: rows = metrics (Sensitivity, Specificity, PPV, F1), columns = outcomes.
    Each radial chart has one bar per analyte arranged around a circle.
    Bars extend outward (NORMA better, colored) or inward (PerRI better, grey).
    Center text shows mean Δ%.
    """
    datasets = available_datasets('eval_metrics_restricted.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    metric_row_labels = {
        'sensitivity': r'$\Delta$ Sensitivity' + '\n' + r'(NORMA$_{RI}$ $-$ Per$_{RI}$)',
        'specificity': r'$\Delta$ Specificity' + '\n' + r'(NORMA$_{RI}$ $-$ Per$_{RI}$)',
        'ppv': r'$\Delta$ Precision' + '\n' + r'(NORMA$_{RI}$ $-$ Per$_{RI}$)',
        'accuracy': r'$\Delta$ Accuracy' + '\n' + r'(NORMA$_{RI}$ $-$ Per$_{RI}$)',
    }

    METRIC_COLORS = {
        'ppv': PALETTE['teal'],
        'sensitivity': PALETTE['coral'],
        'specificity': PALETTE['green'],
        'accuracy': PALETTE['gold'],
    }
    neg_color = '#C0C0C0'

    for ds in datasets:
        df = load_eval_restricted(ds)
        if df is None or len(df) == 0:
            continue

        outcomes = sorted(df['outcome'].unique())
        n_outcomes = len(outcomes)
        n_metrics = len(EVAL_METRICS_COLS)

        cell_size = 2.2
        fig, axes = plt.subplots(
            n_outcomes, n_metrics,
            figsize=(n_metrics * cell_size, n_outcomes * cell_size),
            subplot_kw={'projection': 'polar'},
            squeeze=False,
            constrained_layout=False,
            gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
        )

        for row_idx, outcome in enumerate(outcomes):
            sub = df[df['outcome'] == outcome]

            for col_idx, metric in enumerate(EVAL_METRICS_COLS):
                ax = axes[row_idx, col_idx]

                # Compute deltas per analyte (skip analytes with < 100 patients)
                deltas = []
                labels = []
                for a in all_analytes:
                    match = sub[(sub['analyte'] == a) & (sub['method'] == 'NORMA')]
                    if len(match) == 1 and match.iloc[0]['n'] < 100:
                        continue
                    p = _get_metric(sub, a, 'PerRI', metric)
                    n = _get_metric(sub, a, 'NORMA', metric)
                    if not (np.isnan(p) or np.isnan(n)):
                        deltas.append(n - p)
                        labels.append(a)

                if not deltas:
                    ax.set_visible(False)
                    continue

                # Sort by delta descending
                order = np.argsort(deltas)[::-1]
                deltas = [deltas[i] for i in order]
                labels = [labels[i] for i in order]

                n_bars = len(deltas)
                theta = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
                bar_width = 2 * np.pi / n_bars * 0.85

                base_r = 0.5
                max_abs = max(abs(d) for d in deltas)
                max_abs = max(max_abs, 0.01)
                scale = 0.4 / max_abs

                metric_color = METRIC_COLORS.get(metric, PALETTE['teal'])

                heights = []
                bottoms = []
                colors = []
                for d in deltas:
                    if d >= 0:
                        heights.append(d * scale)
                        bottoms.append(base_r)
                        colors.append(metric_color)
                    else:
                        heights.append(abs(d) * scale)
                        bottoms.append(base_r - abs(d) * scale)
                        colors.append(neg_color)

                ax.bar(theta, heights, width=bar_width, bottom=bottoms,
                       color=colors, alpha=0.8, edgecolor='white', linewidth=0.3)

                # Analyte labels around the outside
                for t, lab in zip(theta, labels):
                    angle_deg = np.degrees(t)
                    if 90 < angle_deg < 270:
                        ax.text(t, base_r + 0.7, lab, ha='center', va='center',
                                fontsize=3.5, rotation=angle_deg - 180)
                    else:
                        ax.text(t, base_r + 0.7, lab, ha='center', va='center',
                                fontsize=3.5, rotation=angle_deg)

                # Center: best gain analyte
                best_idx = np.argmax(deltas)
                best_delta = deltas[best_idx] * 100
                best_label = labels[best_idx]
                ax.text(0, 0, f'{best_label}\n+{best_delta:.1f}%',
                        ha='center', va='center', fontsize=6, fontweight='bold',
                        transform=ax.transData, color=metric_color)

                # Base circle
                circle_theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(circle_theta, [base_r] * 100, color='#DDDDDD',
                        linewidth=0.3, zorder=0)

                ax.set_ylim(0, 1.3)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.spines['polar'].set_visible(False)
                ax.grid(False)

                if row_idx == 0:
                    ax.set_title(EVAL_METRIC_LABELS.get(metric, metric),
                                 fontsize=8, fontweight='bold', pad=12)
                if col_idx == 0:
                    ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome),
                                  fontsize=7, labelpad=15)

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'eval_circos'))


def fig_eval_roc():
    """Sensitivity vs FPR scatter per outcome, showing PerRI → NORMA shift.

    Each analyte shown as an arrow from PerRI (pink dot) to NORMA (blue dot).
    Arrows show the direction of improvement.
    """
    datasets = available_datasets('eval_metrics_restricted.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_eval_restricted(ds)
        if df is None or len(df) == 0:
            continue

        outcomes = sorted(df['outcome'].unique())
        n_outcomes = len(outcomes)

        ncols = n_outcomes
        nrows = 1

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.8, 3),
                                 squeeze=False)

        for idx, outcome in enumerate(outcomes):
            row_idx, col_idx = divmod(idx, ncols)
            ax = axes[row_idx, col_idx]
            sub = df[df['outcome'] == outcome]

            for a in all_analytes:
                match_n = sub[(sub['analyte'] == a) & (sub['method'] == 'NORMA')]
                if len(match_n) == 1 and match_n.iloc[0]['n'] < 100:
                    continue

                per_sens = _get_metric(sub, a, 'PerRI', 'sensitivity')
                per_spec = _get_metric(sub, a, 'PerRI', 'specificity')
                norma_sens = _get_metric(sub, a, 'NORMA', 'sensitivity')
                norma_spec = _get_metric(sub, a, 'NORMA', 'specificity')

                if any(np.isnan(v) for v in [per_sens, per_spec, norma_sens, norma_spec]):
                    continue

                per_fpr = 1 - per_spec
                norma_fpr = 1 - norma_spec

                # Grey line connecting
                ax.plot([per_fpr, norma_fpr], [per_sens, norma_sens],
                        '-', color='#CCCCCC', lw=0.6, alpha=0.5, zorder=1)

                # PerRI = pink, NORMA = blue
                ax.scatter(per_fpr, per_sens, marker='o', color=scheme_colors['Personalized'],
                           s=18, alpha=0.6, edgecolors='none', zorder=3)
                ax.scatter(norma_fpr, norma_sens, marker='D', color=scheme_colors['NORMA'],
                           s=18, alpha=0.8, edgecolors='none', zorder=4)

                # Label at NORMA point
                ax.text(norma_fpr + 0.01, norma_sens, a, fontsize=3.5,
                        color='#333333', alpha=0.6, va='center')

            # Diagonal
            ax.plot([0, 1], [0, 1], '--', color='#DDDDDD', lw=0.5)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('False Positive Rate', fontsize=8)
            ax.set_ylabel('True Positive Rate', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.set_title(OUTCOME_LABELS.get(outcome, outcome), fontsize=8, fontweight='bold')
            ax.set_aspect('equal')
            hide_spines(ax)

        # Hide unused
        for idx in range(n_outcomes, nrows * ncols):
            row_idx, col_idx = divmod(idx, ncols)
            axes[row_idx, col_idx].set_visible(False)

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color=scheme_colors['Personalized'], markersize=5,
                   linestyle='none', alpha=0.6, label=r'Per$_{RI}$'),
            Line2D([0], [0], marker='D', color=scheme_colors['NORMA'], markersize=5,
                   linestyle='none', alpha=0.8, label=r'NORMA$_{RI}$'),
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
                   ncol=3, frameon=False, fontsize=8)

        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'eval_roc'))


# ── Age-dependent RI ─────────────────────────────────────────────────────────

def fig_age_ri():
    """Age-dependent RI trends: mean PerRI and NORMA bounds vs age, faceted by analyte."""
    datasets = available_datasets('age_ri.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    method_colors = {
        'PopRI': scheme_colors['Population'],
        'PerRI': scheme_colors['Personalized'],
        'NORMA': scheme_colors['NORMA'],
    }

    for ds in datasets:
        path = os.path.join(ROOTDIR, 'results', ds, 'raw', 'age_ri.csv')
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])

        analytes_present = sorted(set(df['analyte'].unique()) & set(all_analytes))
        n = len(analytes_present)
        ncols = 6
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 1.8), sharex=True)
        axes = axes.flatten()

        for i, analyte in enumerate(analytes_present):
            ax = axes[i]
            sub = df[df['analyte'] == analyte]

            for method in ['PopRI', 'PerRI', 'NORMA']:
                msub = sub[sub['method'] == method].sort_values('age')
                if len(msub) < 3:
                    continue
                color = method_colors[method]

                # Polynomial fit (3rd order)
                ages = msub['age'].values
                if method == 'PopRI':
                    # PopRI is constant — just shade
                    ax.axhspan(msub['ri_low'].iloc[0], msub['ri_high'].iloc[0],
                               color=color, alpha=0.1)
                else:
                    for bound in ['ri_low', 'ri_high']:
                        vals = msub[bound].values
                        valid = ~np.isnan(vals)
                        if valid.sum() < 4:
                            continue
                        coeffs = np.polyfit(ages[valid], vals[valid], 3)
                        x_smooth = np.linspace(ages[valid].min(), ages[valid].max(), 100)
                        y_smooth = np.polyval(coeffs, x_smooth)
                        ax.plot(x_smooth, y_smooth, color=color, lw=1, alpha=0.8)

            ax.set_title(analyte, fontsize=8, fontweight='bold', pad=3)
            ax.tick_params(axis='both', labelsize=5)
            hide_spines(ax)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.text(0.5, -0.02, 'Age (years)', ha='center', fontsize=9)
        fig.text(-0.02, 0.5, 'Reference Interval Bounds', va='center', rotation='vertical', fontsize=9)

        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=method_colors[m], lw=1.5, label=_ri_label(m))
                   for m in ['PopRI', 'PerRI', 'NORMA']]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
                   ncol=3, frameon=False, fontsize=7)

        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'age_ri'))


# ── Cox Models ───────────────────────────────────────────────────────────────

COX_METHODS = ['PopRI', 'PerRI', 'NORMA']
COX_METHOD_COLORS = {
    'PopRI': scheme_colors['Population'],
    'PerRI': scheme_colors['Personalized'],
    'NORMA': scheme_colors['NORMA'],
}


def load_cox(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'cox_results.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def table_cox():
    datasets = available_datasets('cox_results.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_cox(ds)
        if df is None or len(df) == 0:
            continue
        label = DATASET_LABELS[ds]
        tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
        outcomes = sorted(df['outcome'].unique())

        # Per-analyte table with HR [CI] per method, grouped by outcome
        def _fmt_hr(hr, lo, hi):
            """Format HR [CI], replacing inf/extreme values with ---."""
            if any(not np.isfinite(v) for v in [hr, lo, hi]) or hr > 100 or lo > 100 or hi > 100:
                return '---'
            return f'{hr:.2f} [{lo:.2f}, {hi:.2f}]'

        # One table per outcome
        for outcome in outcomes:
            sub = df[df['outcome'] == outcome]
            outcome_display = OUTCOME_LABELS.get(outcome, outcome)

            csv_rows = []
            for analyte in all_analytes:
                any_match = sub[sub['analyte'] == analyte]
                if len(any_match) > 0:
                    r0 = any_match.iloc[0]
                    n_train = int(r0['n_train']) if pd.notna(r0.get('n_train')) else '---'
                    n_test = int(r0['n_test']) if pd.notna(r0.get('n_test')) else '---'
                    ev_train = int(r0['n_events_train']) if pd.notna(r0.get('n_events_train')) else '---'
                    ev_test = int(r0['n_events_test']) if pd.notna(r0.get('n_events_test')) else '---'
                    row = {'Analyte': analyte,
                           'N (train)': f'{n_train} ({ev_train})',
                           'N (test)': f'{n_test} ({ev_test})'}
                else:
                    row = {'Analyte': analyte, 'N (train)': '---', 'N (test)': '---'}

                for method in COX_METHODS:
                    match = sub[(sub['analyte'] == analyte) & (sub['method'] == method)]
                    if len(match) == 1:
                        r = match.iloc[0]
                        row[f'{method} HR'] = _fmt_hr(r['HR'], r['HR_lower'], r['HR_upper'])
                        row[f'{method} C'] = f'{r["concordance_test"]:.3f}' if pd.notna(r.get('concordance_test')) else '---'
                        row[f'{method} Sig'] = '*' if pd.notna(r.get('p_value')) and r['p_value'] < 0.05 else ''
                    else:
                        row[f'{method} HR'] = '---'
                        row[f'{method} C'] = '---'
                        row[f'{method} Sig'] = ''
                csv_rows.append(row)

            csv_df = pd.DataFrame(csv_rows)

            # LaTeX
            n_per_method = 3  # HR, C, Sig
            col_spec = 'lrr' + 'rlc' * len(COX_METHODS)
            lines = [r'\begin{table}[ht]', r'\centering', r'\tiny',
                     r'\caption{Cox PH: ' + outcome_display + '}',
                     r'\begin{tabular}{' + col_spec + '}', r'\toprule']
            h1 = r' & & '
            cmidrules = []
            for i, method in enumerate(COX_METHODS):
                start = 4 + i * n_per_method
                h1 += r' & \multicolumn{' + str(n_per_method) + r'}{c}{' + _ri_label(method) + '}'
                cmidrules.append(f'\\cmidrule(lr){{{start}-{start + n_per_method - 1}}}')
            lines.append(h1 + r' \\')
            lines.append(' '.join(cmidrules))
            h2 = r'Analyte & N train (events) & N test (events)'
            for _ in COX_METHODS:
                h2 += r' & HR [95\% CI] & C & '
            lines.append(h2 + r' \\')
            lines.append(r'\midrule')

            for row in csv_rows:
                cells = [row['Analyte'], str(row['N (train)']), str(row['N (test)'])]
                for method in COX_METHODS:
                    cells.extend([str(row[f'{method} HR']), str(row[f'{method} C']), str(row[f'{method} Sig'])])
                lines.append(' & '.join(cells) + r' \\')
            lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

            save_table(f'cox_{outcome}', '\n'.join(lines), csv_df, tdir, landscape=True)
        print(f'  {label}: done')


def fig_cox_forest():
    """Forest plot: HR with 95% CI per analyte, mortality only."""
    datasets = available_datasets('cox_results.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_cox(ds)
        if df is None or len(df) == 0:
            continue

        sub = df[df['outcome'] == 'mortality']
        if len(sub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))

        y_pos = 0
        y_ticks, y_labels = [], []
        for analyte in reversed(all_analytes):
            has_any = False
            for mi, method in enumerate(COX_METHODS):
                match = sub[(sub['analyte'] == analyte) & (sub['method'] == method)]
                if len(match) != 1:
                    continue
                r = match.iloc[0]
                if pd.isna(r['HR']) or not np.isfinite(r['HR']) or r['HR'] > 50:
                    continue
                # Only show significant results (p < 0.05)
                if 'p_value' in r and pd.notna(r['p_value']) and r['p_value'] >= 0.05:
                    continue
                has_any = True
                color = COX_METHOD_COLORS[method]
                offset = (mi - 1) * 0.25
                ax.plot([r['HR_lower'], r['HR_upper']], [y_pos + offset, y_pos + offset],
                        '-', color=color, lw=1.5, alpha=0.6)
                ax.plot(r['HR'], y_pos + offset, 'o', color=color, markersize=4, alpha=0.8)
            if has_any:
                y_ticks.append(y_pos)
                y_labels.append(analyte)
                y_pos += 1

        ax.axvline(1, color='#CCCCCC', lw=0.5, ls='--')
        ax.set_xlabel('Hazard Ratio', fontsize=8)
        ax.set_title('Mortality', fontsize=8, fontweight='bold', loc='left')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xscale('log')
        ax.tick_params(axis='x', labelsize=7)

        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], marker='o', color=COX_METHOD_COLORS[m],
                                  markersize=4, linestyle='-', lw=1, alpha=0.7,
                                  label=_ri_label(m)) for m in COX_METHODS],
                  frameon=False, fontsize=7)
        hide_spines(ax)
        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'cox_forest'))


def fig_cox_concordance():
    """Horizontal bar chart: concordance index per analyte × method, faceted by outcome."""
    datasets = available_datasets('cox_results.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_cox(ds)
        if df is None or len(df) == 0:
            continue

        outcomes = sorted(df['outcome'].unique())
        n_outcomes = len(outcomes)

        # Top N analytes per outcome by NORMA concordance
        top_n = 8

        fig, axes = plt.subplots(n_outcomes, 1,
                                 figsize=(3.5, n_outcomes * 2),
                                 squeeze=False)

        for idx, outcome in enumerate(outcomes):
            ax = axes[idx, 0]
            sub = df[df['outcome'] == outcome]

            # Rank by NORMA concordance
            norma = sub[sub['method'] == 'NORMA'].sort_values('concordance_test', ascending=False)
            top_labs = norma.head(top_n)['analyte'].tolist()

            y = np.arange(len(top_labs))
            bar_height = 0.8 / len(COX_METHODS)

            for mi, method in enumerate(COX_METHODS):
                msub = sub[sub['method'] == method].set_index('analyte')
                vals = [msub.loc[a, 'concordance_test'] if a in msub.index and pd.notna(msub.loc[a, 'concordance_test']) else np.nan
                        for a in top_labs]
                positions = y + (mi - 1) * bar_height
                ax.barh(positions, vals, bar_height * 0.9,
                        color=COX_METHOD_COLORS[method], alpha=0.8,
                        label=_ri_label(method) if idx == 0 else None)

            ax.set_yticks(y)
            ax.set_yticklabels(top_labs, fontsize=7)
            ax.set_xlim(0.5, max(0.9, sub['concordance_test'].max() + 0.05))
            ax.set_xlabel('C-index' if idx == n_outcomes - 1 else '')
            ax.set_title(OUTCOME_LABELS.get(outcome, outcome), fontsize=8, fontweight='bold')
            hide_spines(ax)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),
                   ncol=len(COX_METHODS), frameon=False, fontsize=7)
        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'cox_concordance'))


# ── Lead Time ────────────────────────────────────────────────────────────────

def load_lead_time(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'lead_time.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def table_lead_time():
    datasets = available_datasets('lead_time.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_lead_time(ds)
        if df is None or len(df) == 0:
            continue
        label = DATASET_LABELS[ds]
        tdir = os.path.join(ROOTDIR, 'results', ds, 'tables')
        present = df.set_index('analyte')

        csv_rows = []
        for analyte in list(all_analytes) + (['Overall'] if 'Overall' in present.index else []):
            if analyte not in present.index:
                csv_rows.append({'Analyte': analyte, 'NORMA-only flags': '---',
                                 'With later PopRI flag': '---',
                                 'Median lead (h)': '---', 'IQR (h)': '---'})
                continue
            r = present.loc[analyte]
            iqr = f'{r["iqr_lead_hours_25"]:.1f}\u2013{r["iqr_lead_hours_75"]:.1f}' if pd.notna(r['iqr_lead_hours_25']) else '---'
            csv_rows.append({
                'Analyte': analyte,
                'NORMA-only flags': f'{int(r["n_norma_only"]):,}',
                'With later PopRI flag': f'{int(r["n_with_later_pop_flag"]):,}',
                'Median lead (h)': f'{r["median_lead_hours"]:.1f}' if pd.notna(r['median_lead_hours']) else '---',
                'IQR (h)': iqr,
            })

        csv_df = pd.DataFrame(csv_rows)

        lines = [r'\begin{table}[ht]', r'\centering', r'\small',
                 r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Analyte & NORMA-only flags & Later Pop$_{RI}$ flag & Median lead (h) & IQR (h) \\',
                 r'\midrule']
        for row in csv_rows:
            cells = [row['Analyte'], str(row['NORMA-only flags']),
                     str(row['With later PopRI flag']),
                     str(row['Median lead (h)']), tex_escape(str(row['IQR (h)']))]
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        save_table('lead_time', '\n'.join(lines), csv_df, tdir)
        print(f'  {label}: done')


def fig_lead_time():
    datasets = available_datasets('lead_time.csv')
    all_analytes = sorted(set(ANALYTE_NAMES.keys()) - EXCLUDE_ANALYTES)

    for ds in datasets:
        df = load_lead_time(ds)
        if df is None or len(df) == 0:
            continue

        # Exclude Overall row and analytes with no lead time
        plot_df = df[(df['analyte'] != 'Overall') & (df['analyte'].isin(all_analytes))]
        plot_df = plot_df.dropna(subset=['median_lead_hours'])
        plot_df = plot_df.sort_values('median_lead_hours', ascending=True)

        if len(plot_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4.5, 8))

        y = np.arange(len(plot_df))
        colors = [analyte_colors.get(a, '#999999') for a in plot_df['analyte']]

        ax.barh(y, plot_df['median_lead_hours'], color=colors, alpha=0.8, height=0.7)

        # IQR whiskers
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if pd.notna(row['iqr_lead_hours_25']):
                ax.plot([row['iqr_lead_hours_25'], row['iqr_lead_hours_75']], [i, i],
                        '-', color='#333333', lw=0.8, alpha=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(plot_df['analyte'], fontsize=7)
        ax.set_xlabel('Lead Time (hours)', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        hide_spines(ax)
        plt.tight_layout()

        fdir = os.path.join(ROOTDIR, 'results', ds, 'figures')
        os.makedirs(fdir, exist_ok=True)
        save_fig(os.path.join(fdir, 'lead_time'))


# ── Registry ─────────────────────────────────────────────────────────────────

TABLES = {
    'variability': table_variability,
    'mortality': table_mortality,
    'prevalence': table_prevalence,
    'eval_metrics': table_eval_metrics,
    'cox': table_cox,
    'lead_time': table_lead_time,
}

# Ordered to match manuscript multi-panel figures:
#   Fig 2: variability, mortality
#   Fig 4/5: prevalence, circos, roc, lead_time
#   Supp: mortality_deviation, cox, age_ri
FIGURES = {
    # Figure 2 panels
    'variability': fig_variability,
    'mortality': fig_mortality,
    # Figure 4/5 panels
    'prevalence': fig_prevalence,
    'reclassification': fig_reclassification,
    'eval_metrics': fig_eval_metrics,
    'eval_roc': fig_eval_roc,
    'lead_time': fig_lead_time,
    # Supplementary
    'mortality_deviation': fig_mortality_deviation,
    'cox_forest': fig_cox_forest,
    'cox_concordance': fig_cox_concordance,
    'age_ri': fig_age_ri,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', nargs='*', default=None)
    parser.add_argument('--figures', nargs='*', default=None)
    args = parser.parse_args()

    run_tables = args.tables is not None
    run_figures = args.figures is not None
    if not run_tables and not run_figures:
        run_tables = True
        run_figures = True

    all_available = set()
    for f in ['variability.csv', 'mortality_by_quintile.csv', 'prevalence.csv',
               'eval_metrics_restricted.csv', 'cox_results.csv', 'lead_time.csv']:
        all_available.update(available_datasets(f))
    if not all_available:
        print("No datasets found.")
        return
    print(f"Datasets: {', '.join(DATASET_LABELS[ds] for ds in sorted(all_available))}")

    if run_tables:
        names = args.tables if args.tables else list(TABLES.keys())
        print('\nTables:')
        for name in names:
            if name in TABLES:
                TABLES[name]()
            else:
                print(f"  Unknown table: {name}")

    if run_figures:
        names = args.figures if args.figures else list(FIGURES.keys())
        print('\nFigures:')
        for name in names:
            if name in FIGURES:
                FIGURES[name]()
            else:
                print(f"  Unknown figure: {name}")


if __name__ == '__main__':
    main()
