"""
Format prediction results: tables (LaTeX + CSV + PDF/PNG) and figures (PDF + PNG).

Reads from results/prediction/raw/ (raw CSVs).
Writes tables to results/prediction/tables/ and figures to results/prediction/figures/.

Usage:
    python scripts/format_prediction.py
    python scripts/format_prediction.py --tables prediction_performance analyte_performance
    python scripts/format_prediction.py --figures --models NORMA-Gaussian
    python scripts/format_prediction.py --all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots import (plt, hide_spines, analyte_colors, scheme_colors,
                    EXCLUDE_ANALYTES, save_fig, compile_latex, save_table)

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOTDIR, 'results', 'prediction')
TABLE_DIR = os.path.join(ROOTDIR, 'results', 'prediction', 'tables')
FIG_DIR = os.path.join(ROOTDIR, 'results', 'prediction', 'figures')
MODEL_ORDER = ['NORMA-Gaussian', 'NORMA-Quantile', 'ARIMA', 'Mean', 'Last']
METRIC_ORDER = ['MAE', 'MAPE', 'R2']
SPLIT_ORDER = ['train', 'val', 'test']
SPLIT_LABELS = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
ANALYTE_MODELS = ['NORMA-Gaussian', 'NORMA-Quantile', 'ARIMA']
BASELINE_MODELS = ['ARIMA', 'Mean', 'Last']

MODEL_COLORS = {
    'NORMA-Gaussian': scheme_colors.get('NORMA', '#0097A7'),
    'NORMA-Quantile': scheme_colors.get('NORMA', '#0097A7'),
    'ARIMA': '#607D8B',            # blue-grey
    'Mean': '#90A4AE',             # light blue-grey
    'Last': '#B0BEC5',             # pale blue-grey
}

SENSITIVITY_FEATURES = ['history_length', 'horizon', 'history_std']
FEATURE_LABELS = {
    'history_length': 'History Length',
    'horizon': 'Prediction Horizon',
    'history_std': 'Within-Person Variability',
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(name):
    return pd.read_csv(os.path.join(RAW_DIR, name), keep_default_na=False, na_values=[''])


def bold_best(values, lower_is_better=True):
    nums = []
    for v in values:
        try:
            nums.append(float(v.split('[')[0].strip()))
        except:
            nums.append(None)
    best = (min if lower_is_better else max)(x for x in nums if x is not None)
    return [v if (n is not None and abs(n - best) < 1e-6) else None
            for v, n in zip(values, nums)]


def format_val(mean, ci_lower, ci_upper, metric):
    if metric == 'R2':
        return f'{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'
    return f'{mean:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]'


# ── Tables ───────────────────────────────────────────────────────────────────

def table_prediction_performance():
    df = load_csv('forecasting_overall.csv')
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]
    lower_better = {'MAE': True, 'MAPE': True, 'R2': False}

    rows = []
    for metric in METRIC_ORDER:
        for split in SPLIT_ORDER:
            row = {'Metric': metric, 'Split': SPLIT_LABELS[split]}
            for model in models:
                match = df[(df['model'] == model) & (df['split'] == split) & (df['metric'] == metric)]
                if len(match) == 1:
                    r = match.iloc[0]
                    row[model] = format_val(r['mean'], r['ci_lower'], r['ci_upper'], metric)
                else:
                    row[model] = '—'
            rows.append(row)

    # CSV
    csv_df = pd.DataFrame(rows)
    for i in range(len(csv_df)):
        if i % 3 != 0:
            csv_df.loc[i, 'Metric'] = ''

    # LaTeX
    col_spec = 'll' + 'r' * len(models)
    header_labels = [m.replace('NORMA-', 'NORMA ') for m in models]
    lines = [r'\begin{table}[ht]', r'\centering',
             r'\begin{tabular}{' + col_spec + '}', r'\toprule',
             'Metric & & ' + ' & '.join(header_labels) + r' \\', r'\midrule']

    for i, row in enumerate(rows):
        metric = row['Metric']
        vals = [row[m] for m in models]
        best_idx = bold_best(vals, lower_is_better=lower_better[metric])
        cells = [r'\textbf{' + v + '}' if b is not None else v for v, b in zip(vals, best_idx)]
        metric_cell = r'\multirow{3}{*}{' + metric + '}' if i % 3 == 0 else ''
        lines.append(metric_cell + ' & ' + row['Split'] + ' & ' + ' & '.join(cells) + r' \\')
        if i % 3 == 2 and i < len(rows) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    save_table('prediction_performance', '\n'.join(lines), csv_df, TABLE_DIR)


def table_analyte_performance():
    df = load_csv('forecasting_by_analyte.csv')
    df = df[(df['split'] == 'test') & (df['metric'].isin(METRIC_ORDER)) & (~df['analyte'].isin(EXCLUDE_ANALYTES))].dropna(subset=['analyte'])
    models = [m for m in ANALYTE_MODELS if m in df['model'].unique()]
    analytes = sorted(df['analyte'].unique())
    lower_better = {'MAE': True, 'MAPE': True, 'R2': False}

    rows = []
    for analyte in analytes:
        row = {'Analyte': analyte}
        for model in models:
            for metric in METRIC_ORDER:
                match = df[(df['model'] == model) & (df['analyte'] == analyte) & (df['metric'] == metric)]
                col = f'{model}_{metric}'
                if len(match) == 1:
                    v = match.iloc[0]['point_estimate']
                    row[col] = f'{v:.2f}' if metric == 'R2' else f'{v:.1f}'
                else:
                    row[col] = '—'
        rows.append(row)

    # CSV
    csv_df = pd.DataFrame(rows)
    rename = {'Analyte': 'Analyte'}
    for model in models:
        for metric in METRIC_ORDER:
            rename[f'{model}_{metric}'] = f'{model.replace("NORMA-", "NORMA ")} {metric}'
    csv_df = csv_df.rename(columns=rename)

    # LaTeX
    n_metrics = len(METRIC_ORDER)
    header_labels = [m.replace('NORMA-', 'NORMA ') for m in models]
    col_spec = 'l' + 'r' * (len(models) * n_metrics)
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\begin{tabular}{' + col_spec + '}', r'\toprule']

    model_header = ' '
    cmidrules = []
    for i, label in enumerate(header_labels):
        start = 2 + i * n_metrics
        model_header += r' & \multicolumn{' + str(n_metrics) + r'}{c}{' + label + '}'
        cmidrules.append(f'\\cmidrule(lr){{{start}-{start + n_metrics - 1}}}')
    lines += [model_header + r' \\', ' '.join(cmidrules)]

    subheader = 'Analyte'
    for _ in models:
        for metric in METRIC_ORDER:
            subheader += f' & {metric}'
    lines += [subheader + r' \\', r'\midrule']

    for row in rows:
        cells = [row['Analyte']]
        best_per_metric = {}
        for metric in METRIC_ORDER:
            vals = [row[f'{m}_{metric}'] for m in models]
            best_per_metric[metric] = bold_best(vals, lower_is_better=lower_better[metric])
        for mi, model in enumerate(models):
            for metric in METRIC_ORDER:
                v = row[f'{model}_{metric}']
                b = best_per_metric[metric][mi]
                cells.append(r'\textbf{' + v + '}' if b is not None else v)
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    save_table('analyte_performance', '\n'.join(lines), csv_df, TABLE_DIR)


def table_sensitivity(model_label):
    df = load_csv('sensitivity.csv')
    df = df[(df['model'] == model_label) & (~df['test_name'].isin(EXCLUDE_ANALYTES))]
    analytes = sorted(df['test_name'].unique())
    mu_label = 'q50' if 'Quantile' in model_label else 'μ'
    mu_tex = r'$\mu$' if mu_label == 'μ' else mu_label

    rows = []
    for analyte in analytes:
        row = {'Analyte': analyte}
        for feat in SENSITIVITY_FEATURES:
            sub = df[(df['test_name'] == analyte) & (df['feature'] == feat)]
            if len(sub) == 0:
                row[f'{feat}_ci'] = '—'
                row[f'{feat}_mu'] = '—'
            else:
                row[f'{feat}_ci'] = f'{sub["ci_norm_mean"].min():.1f} to {sub["ci_norm_mean"].max():.1f}'
                row[f'{feat}_mu'] = f'{sub["mu_pct_dev"].min():.1f} to {sub["mu_pct_dev"].max():.1f}'
        rows.append(row)

    # CSV
    csv_df = pd.DataFrame(rows)
    rename = {'Analyte': 'Analyte'}
    for feat in SENSITIVITY_FEATURES:
        rename[f'{feat}_ci'] = f'{FEATURE_LABELS[feat]} CI (% ref)'
        rename[f'{feat}_mu'] = f'{FEATURE_LABELS[feat]} {mu_label} (%)'
    csv_df = csv_df.rename(columns=rename)

    # LaTeX
    n_feat = len(SENSITIVITY_FEATURES)
    col_spec = 'l' + 'rr' * n_feat
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    header = ' '
    cmidrules = []
    for i, feat in enumerate(SENSITIVITY_FEATURES):
        start = 2 + i * 2
        header += r' & \multicolumn{2}{c}{' + FEATURE_LABELS[feat] + '}'
        cmidrules.append(f'\\cmidrule(lr){{{start}-{start + 1}}}')
    lines += [header + r' \\', ' '.join(cmidrules)]
    subheader = 'Analyte'
    for _ in SENSITIVITY_FEATURES:
        subheader += f' & CI (\\% ref) & {mu_tex} (\\%)'
    lines += [subheader + r' \\', r'\midrule']
    for row in rows:
        cells = [row['Analyte']]
        for feat in SENSITIVITY_FEATURES:
            cells += [row[f'{feat}_ci'], row[f'{feat}_mu']]
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    suffix = model_label.lower().replace('-', '_').replace(' ', '_')
    save_table(f'sensitivity_{suffix}', '\n'.join(lines), csv_df, TABLE_DIR)


def table_sensitivity_summary(model_label):
    df = load_csv('sensitivity.csv')
    df = df[(df['model'] == model_label) & (~df['test_name'].isin(EXCLUDE_ANALYTES))]
    mu_label = 'q50' if 'Quantile' in model_label else 'μ'
    mu_tex = r'$\mu$' if mu_label == 'μ' else mu_label

    rows = []
    for feat in SENSITIVITY_FEATURES:
        sub = df[df['feature'] == feat]
        if len(sub) == 0:
            continue
        ci_vals = sub.groupby('test_name')['ci_norm_mean'].agg(['min', 'max'])
        ci_ranges = ci_vals['max'] - ci_vals['min']
        mu_vals = sub.groupby('test_name')['mu_pct_dev'].agg(['min', 'max'])
        mu_ranges = mu_vals['max'] - mu_vals['min']
        rows.append({
            'Feature': FEATURE_LABELS[feat],
            'CI range median': f'{ci_ranges.median():.1f}',
            'CI range IQR': f'[{ci_ranges.quantile(0.25):.1f}, {ci_ranges.quantile(0.75):.1f}]',
            f'{mu_label} range median': f'{mu_ranges.median():.1f}',
            f'{mu_label} range IQR': f'[{mu_ranges.quantile(0.25):.1f}, {mu_ranges.quantile(0.75):.1f}]',
        })

    csv_df = pd.DataFrame(rows)

    lines = [r'\begin{table}[ht]', r'\centering', r'\begin{tabular}{lrrrr}', r'\toprule',
             r' & \multicolumn{2}{c}{CI Width Change (\%)} & \multicolumn{2}{c}{' + mu_tex + r' Shift (\%)} \\',
             r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
             r'Feature & Median & IQR & Median & IQR \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(row.values()) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    suffix = model_label.lower().replace('-', '_').replace(' ', '_')
    save_table(f'sensitivity_summary_{suffix}', '\n'.join(lines), csv_df, TABLE_DIR)


def table_sensitivity_summary_combined():
    """Combined sensitivity summary for both Quantile and Gaussian models."""
    df = load_csv('sensitivity.csv')
    df = df[~df['test_name'].isin(EXCLUDE_ANALYTES)]

    all_rows = []
    for model_label in ['NORMA-Quantile', 'NORMA-Gaussian']:
        sub = df[df['model'] == model_label]
        mu_label = 'q50' if 'Quantile' in model_label else 'μ'
        model_short = 'Quantile' if 'Quantile' in model_label else 'Gaussian'
        for feat in SENSITIVITY_FEATURES:
            fsub = sub[sub['feature'] == feat]
            if len(fsub) == 0:
                continue
            ci_vals = fsub.groupby('test_name')['ci_norm_mean'].agg(['min', 'max'])
            ci_ranges = ci_vals['max'] - ci_vals['min']
            mu_vals = fsub.groupby('test_name')['mu_pct_dev'].agg(['min', 'max'])
            mu_ranges = mu_vals['max'] - mu_vals['min']
            all_rows.append({
                'Model': model_short,
                'Feature': FEATURE_LABELS[feat],
                'CI Median': f'{ci_ranges.median():.1f}',
                'CI IQR': f'[{ci_ranges.quantile(0.25):.1f}, {ci_ranges.quantile(0.75):.1f}]',
                'Shift Median': f'{mu_ranges.median():.1f}',
                'Shift IQR': f'[{mu_ranges.quantile(0.25):.1f}, {mu_ranges.quantile(0.75):.1f}]',
            })

    csv_df = pd.DataFrame(all_rows)

    lines = [r'\begin{table}[ht]', r'\centering',
             r'\begin{tabular}{llrrrr}', r'\toprule',
             r' & & \multicolumn{2}{c}{CI Width Change (\%)} & \multicolumn{2}{c}{Midpoint Shift (\%)} \\',
             r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}',
             r'Model & Feature & Median & IQR & Median & IQR \\', r'\midrule']
    prev_model = None
    for row in all_rows:
        if row['Model'] != prev_model:
            if prev_model is not None:
                lines.append(r'\midrule')
            prev_model = row['Model']
        lines.append(' & '.join(row.values()) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    save_table('sensitivity_summary', '\n'.join(lines), csv_df, TABLE_DIR)


TABLES = {
    'prediction_performance': table_prediction_performance,
    'analyte_performance': table_analyte_performance,
    'sensitivity_quantile': lambda: table_sensitivity('NORMA-Quantile'),
    'sensitivity_gaussian': lambda: table_sensitivity('NORMA-Gaussian'),
    'sensitivity_summary_quantile': lambda: table_sensitivity_summary('NORMA-Quantile'),
    'sensitivity_summary_gaussian': lambda: table_sensitivity_summary('NORMA-Gaussian'),
    'sensitivity_summary': table_sensitivity_summary_combined,
}


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_forecasting_accuracy(norma_model):
    df = load_csv('forecasting_overall.csv')
    df = df[df['split'] == 'test']
    models = [norma_model] + BASELINE_MODELS
    df = df[df['model'].isin(models)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df['metric'] == metric].set_index('model').reindex(models)
        colors_list = [MODEL_COLORS.get(m, '#999999') for m in models]
        labels = ['NORMA' if m.startswith('NORMA') else m for m in models]
        ax.barh(range(len(models)), sub['mean'], color=colors_list, height=0.6)
        if 'ci_lower' in sub.columns:
            for i, (m, lo, hi) in enumerate(zip(sub['mean'],
                    sub['mean'] - sub['ci_lower'], sub['ci_upper'] - sub['mean'])):
                ax.plot([m - lo, m + hi], [i, i], color=colors_list[i], lw=1.5, alpha=0.6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric)
        ax.invert_yaxis()
        hide_spines(ax)
    plt.tight_layout()
    prefix = norma_model.lower().replace('-', '_').replace(' ', '_')
    save_fig(os.path.join(FIG_DIR, f'{prefix}_forecasting_accuracy'))


def fig_analyte_r2(norma_model):
    df = load_csv('forecasting_by_analyte.csv')
    df = df[(df['split'] == 'test') & (df['metric'] == 'R2') & (~df['analyte'].isin(EXCLUDE_ANALYTES))]

    # All models to show
    models_to_show = [norma_model] + [m for m in BASELINE_MODELS if m in df['model'].unique()]
    analytes = sorted(df['analyte'].unique())

    fig, ax = plt.subplots(figsize=(8, 2.5))
    x = np.arange(len(analytes))
    n_models = len(models_to_show)
    bar_width = 0.8 / n_models

    model_colors = {
        norma_model: scheme_colors.get('NORMA', '#2A6B6E'),
    }
    for m in BASELINE_MODELS:
        model_colors[m] = MODEL_COLORS.get(m, '#999999')

    for mi, model in enumerate(models_to_show):
        msub = df[df['model'] == model].set_index('analyte')
        vals = [msub.loc[a, 'point_estimate'] if a in msub.index else 0 for a in analytes]
        offset = (mi - (n_models - 1) / 2) * bar_width
        label = 'NORMA' if model.startswith('NORMA') else model
        ax.bar(x + offset, vals, bar_width, color=model_colors.get(model, '#999'),
               alpha=0.85, label=label, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(analytes, rotation=90)
    ax.set_ylabel('R²')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    hide_spines(ax)
    plt.tight_layout()
    prefix = norma_model.lower().replace('-', '_').replace(' ', '_')
    save_fig(os.path.join(FIG_DIR, f'{prefix}_analyte_r2'))


def fig_sensitivity(norma_model):
    df = load_csv('sensitivity.csv')
    df = df[(df['model'] == norma_model) & (~df['test_name'].isin(EXCLUDE_ANALYTES))]
    if len(df) == 0:
        print(f"  No sensitivity data for {norma_model}, skipping")
        return

    ci_col = 'ci_norm_mean' if 'ci_norm_mean' in df.columns else 'pct_change'
    ci_ylabel = 'CI Width (% of ref range)' if ci_col == 'ci_norm_mean' else 'CI Width Change (%)'

    panels = [
        ('history_length', 'Number of Measurements',              ci_col,       ci_ylabel, True),
        ('horizon',        'Prediction Horizon (days)',            ci_col,       ci_ylabel, True),
        ('history_std',    'Within-Person SD (× ref range / 10)', ci_col,       ci_ylabel, False),
        ('history_std',    'Within-Person SD (× ref range / 10)', 'mu_pct_dev', 'Deviation from Midpoint (%)', False),
    ]

    analytes = sorted(df['test_name'].unique())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    for ax, (feat, xlabel, ycol, ylabel, use_log) in zip(axes, panels):
        sub = df[df['feature'] == feat]
        for lab in analytes:
            d = sub[sub['test_name'] == lab].sort_values('value')
            if d.empty:
                continue
            c = analyte_colors.get(lab, '#999999')
            ax.plot(d['value'], d[ycol], '-', color=c, lw=0.8, alpha=0.7)
            if ycol == ci_col and feat == 'history_std' and 'ci_norm_lo' in d.columns:
                ax.fill_between(d['value'], d['ci_norm_lo'], d['ci_norm_hi'],
                                color=c, alpha=0.08)
        if ycol == 'mu_pct_dev':
            ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_xscale('log')
        hide_spines(ax)

    handles = [plt.Line2D([0], [0], color=analyte_colors.get(l, '#999999'), lw=2, label=l)
               for l in analytes]
    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.06),
               loc='lower center', frameon=False, fontsize=9, ncol=10,
               handlelength=1.5, columnspacing=1.2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    prefix = norma_model.lower().replace('-', '_').replace(' ', '_')
    save_fig(os.path.join(FIG_DIR, f'{prefix}_sensitivity'))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', nargs='*', default=None)
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--models', nargs='+', default=['NORMA-Gaussian', 'NORMA-Quantile'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    run_tables = args.all or args.tables is not None
    run_figures = args.all or args.figures

    if not run_tables and not run_figures:
        run_tables = run_figures = True

    if run_tables:
        table_names = args.tables if args.tables else list(TABLES.keys())
        print('Tables:')
        for name in table_names:
            if name in TABLES:
                TABLES[name]()
            else:
                print(f"  Unknown table: {name}")

    if run_figures:
        os.makedirs(FIG_DIR, exist_ok=True)
        print('\nFigures:')
        for model in args.models:
            print(f'\n  {model}:')
            fig_forecasting_accuracy(model)
            fig_analyte_r2(model)
            fig_sensitivity(model)


if __name__ == '__main__':
    main()
