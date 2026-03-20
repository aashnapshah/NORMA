"""
Format cross-dataset manuscript tables and figures.

Reads cohort_summary.csv from results/{dataset}/raw/ and produces master tables.

Usage:
    python scripts/format_cohort.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots import (EXCLUDE_ANALYTES, ANALYTE_NAMES, get_unit, get_pop_ri,
                    fmt, fmt_pm, tex_escape, compile_latex, save_table)

TABLE_DIR = os.path.join(ROOTDIR, 'results', 'tables')

DATASETS = ['ehrshot', 'mimiciv', 'eicu', 'chs']
DATASET_LABELS = {
    'ehrshot': 'EHRSHOT', 'mimiciv': 'MIMIC-IV', 'eicu': 'eICU', 'chs': 'CHS',
}


def load_cohort_summary(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'cohort_summary.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def load_cohort_demographics(dataset):
    path = os.path.join(ROOTDIR, 'results', dataset, 'raw', 'cohort_demographics.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def table_analyte_reference():
    """Analyte reference table: abbreviation, full name, unit, population RI."""
    summaries = {}
    for ds in DATASETS:
        s = load_cohort_summary(ds)
        if s is not None:
            summaries[ds] = s.set_index('analyte')

    all_analytes = sorted(set().union(*(s.index for s in summaries.values())) - EXCLUDE_ANALYTES)

    csv_rows = []
    for analyte in all_analytes:
        csv_rows.append({
            'Analyte': analyte,
            'Full Name': ANALYTE_NAMES.get(analyte, ''),
            'Unit': get_unit(analyte),
            'Population RI': get_pop_ri(analyte),
        })

    csv_df = pd.DataFrame(csv_rows)

    # LaTeX
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\begin{tabular}{llll}', r'\toprule',
             r'Analyte & Full Name & Unit & Population RI \\', r'\midrule']
    for row in csv_rows:
        cells = [tex_escape(c) for c in [row['Analyte'], row['Full Name'], row['Unit'], row['Population RI']]]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    save_table('analyte_reference', '\n'.join(lines), csv_df, TABLE_DIR)


def table_cohort_summary():
    """Per-dataset cohort statistics with demographics header."""
    summaries = {}
    demographics = {}
    for ds in DATASETS:
        s = load_cohort_summary(ds)
        if s is not None:
            summaries[ds] = s.set_index('analyte')
        d = load_cohort_demographics(ds)
        if d is not None:
            demographics[ds] = d.iloc[0]

    all_analytes = sorted(set().union(*(s.index for s in summaries.values())) - EXCLUDE_ANALYTES)

    csv_rows = []
    # Collect tests/person means per dataset for summary
    tpp_values = {ds: [] for ds in DATASETS}
    for analyte in all_analytes:
        row = {'Analyte': analyte}
        for ds in DATASETS:
            label = DATASET_LABELS[ds]
            if ds in summaries and analyte in summaries[ds].index:
                s = summaries[ds].loc[analyte]
                row[f'{label} N'] = int(s['n_patients']) if pd.notna(s['n_patients']) else '—'
                row[f'{label} Mean ± Std'] = fmt_pm(s['value_mean'], s['value_std'], 1)
                tpp_mean = s['tests_per_person_mean']
                if pd.notna(tpp_mean):
                    tpp_values[ds].append(tpp_mean)
            else:
                row[f'{label} N'] = '—'
                row[f'{label} Mean ± Std'] = '—'
        csv_rows.append(row)

    csv_df = pd.DataFrame(csv_rows)

    # LaTeX
    n_ds = len(DATASETS)
    n_cols = 2  # N, Value
    col_spec = 'l' + 'rr' * n_ds
    ds_labels = [DATASET_LABELS[ds] for ds in DATASETS]

    lines = [r'\begin{table}[ht]', r'\centering', r'\tiny',
             r'\begin{tabular}{' + col_spec + '}', r'\toprule']

    # Dataset header
    h1 = ' '
    cmidrules = []
    for i, label in enumerate(ds_labels):
        start = 2 + i * n_cols
        h1 += r' & \multicolumn{' + str(n_cols) + r'}{c}{' + label + '}'
        cmidrules.append(f'\\cmidrule(lr){{{start}-{start + n_cols - 1}}}')
    lines.append(h1 + r' \\')
    lines.append(' '.join(cmidrules))

    # Demographics rows
    def demo_row(label, values):
        cells = [r'\textit{' + label + '}']
        for ds in DATASETS:
            cells.append(r'\multicolumn{' + str(n_cols) + r'}{c}{' + tex_escape(values.get(ds, '—')) + '}')
        return ' & '.join(cells) + r' \\'

    # Time span: years for most datasets, days for eICU
    SPAN_UNITS = {'ehrshot': 'yr', 'mimiciv': 'yr', 'eicu': 'days', 'chs': 'yr'}

    patients = {}
    sequences = {}
    ages = {}
    sex = {}
    spans = {}
    for ds in DATASETS:
        if ds in demographics:
            d = demographics[ds]
            patients[ds] = f'{int(d["n_patients"]):,}' if pd.notna(d['n_patients']) else '—'
            sequences[ds] = f'{int(d["n_sequences"]):,}' if pd.notna(d.get('n_sequences')) else '—'
            ages[ds] = fmt_pm(d.get('age_mean'), d.get('age_std'), 1) if pd.notna(d.get('age_mean')) else '—'
            pf = f'{d["pct_female"]:.0f}' if pd.notna(d.get('pct_female')) else '—'
            pm = f'{d["pct_male"]:.0f}' if pd.notna(d.get('pct_male')) else '—'
            sex[ds] = f'{pf}% F / {pm}% M' if pf != '—' else '—'
            # Time span median [IQR]
            med = d.get('span_days_median')
            q25 = d.get('span_days_q25')
            q75 = d.get('span_days_q75')
            if pd.notna(med):
                unit = SPAN_UNITS.get(ds, 'yr')
                if unit == 'yr':
                    spans[ds] = f'{med / 365.25:.1f} [{q25 / 365.25:.1f}, {q75 / 365.25:.1f}] yr'
                else:
                    spans[ds] = f'{med:.0f} [{q25:.0f}, {q75:.0f}] days'

    lines.append(demo_row('Patients', patients))
    lines.append(demo_row('Sequences', sequences))
    lines.append(demo_row('Age', ages))
    lines.append(demo_row('Sex', sex))
    lines.append(demo_row('Time', spans))

    # Freq: median [IQR] of tests/person across analytes
    freq = {}
    for ds in DATASETS:
        vals = tpp_values[ds]
        if vals:
            import numpy as _np
            arr = _np.array(vals)
            freq[ds] = f'{_np.median(arr):.0f} [{_np.percentile(arr, 25):.0f}, {_np.percentile(arr, 75):.0f}]'
        else:
            freq[ds] = '—'
    lines.append(demo_row('Tests/Person', freq))
    lines.append(r'\midrule')

    # Column sub-headers for analyte rows
    h2 = 'Analyte'
    for _ in DATASETS:
        h2 += r' & N & Value'
    lines.append(h2 + r' \\')
    lines.append(r'\midrule')

    for row in csv_rows:
        cells = [row['Analyte']]
        for ds in DATASETS:
            label = DATASET_LABELS[ds]
            n = row[f'{label} N']
            cells.append(f'{n:,}' if isinstance(n, int) else str(n))
            cells.append(tex_escape(row[f'{label} Mean ± Std']))
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    save_table('cohort_summary', '\n'.join(lines), csv_df, TABLE_DIR, landscape=True)


def table_demographics():
    """Cohort demographics comparison table."""
    rows = []
    for ds in DATASETS:
        demo = load_cohort_demographics(ds)
        if demo is None:
            rows.append({
                'Dataset': DATASET_LABELS[ds],
                'Patients': '—', 'Sequences': '—',
                'Age (mean ± std)': '—', '% Female': '—', '% Male': '—',
            })
            continue
        d = demo.iloc[0]
        rows.append({
            'Dataset': DATASET_LABELS[ds],
            'Patients': f'{int(d["n_patients"]):,}' if pd.notna(d['n_patients']) else '—',
            'Sequences': f'{int(d["n_sequences"]):,}' if pd.notna(d.get('n_sequences')) else '—',
            'Age (mean ± std)': fmt_pm(d.get('age_mean'), d.get('age_std'), 1),
            '% Female': f'{d["pct_female"]:.1f}%' if pd.notna(d.get('pct_female')) else '—',
            '% Male': f'{d["pct_male"]:.1f}%' if pd.notna(d.get('pct_male')) else '—',
        })

    csv_df = pd.DataFrame(rows)

    csv_dir = os.path.join(TABLE_DIR, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'cohort_demographics.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"  {csv_path}")


TABLES = {
    'analyte_reference': table_analyte_reference,
    'cohort_summary': table_cohort_summary,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', nargs='*', default=None)
    args = parser.parse_args()

    table_names = args.tables if args.tables else list(TABLES.keys())

    for name in table_names:
        if name in TABLES:
            print(f"\n{name}:")
            TABLES[name]()
        else:
            print(f"Unknown table: {name}")


if __name__ == '__main__':
    main()
