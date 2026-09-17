"""Compute cohort demographics and per-analyte summary statistics."""
import bootstrap

import os
import argparse
import pickle
import pandas as pd
import numpy as np

from constants import ALL_SPLIT, DEV_COHORTS, SPLIT_ORDER
from datasets import NORMA_RUN_ID, DATASETS as LOADERS, save_csv

ROOTDIR = bootstrap.BASE_DIR      # norma/: model/logs/, data/processed/


def days_per_unit(ds):
    """Timestamps are stored in different units per cohort; convert spans to days."""
    unit = getattr(ds, "time_unit", None) or "days"
    return {"minutes": 1.0 / 1440.0, "hours": 1.0 / 24.0, "days": 1.0}.get(unit, 1.0)


def compute_summary(df, exclude_labs=None, to_days=1.0):
    """Per-analyte summary statistics."""
    if exclude_labs:
        df = df[~df['analyte'].isin(exclude_labs)].copy()
    analytes = sorted(df['analyte'].unique())

    rows = []
    for analyte in analytes:
        sub = df[df['analyte'] == analyte]
        per_patient = sub.groupby('patient_id')
        counts = per_patient.size()
        spans = per_patient['timestamp'].agg(['min', 'max'])
        spans_days = (spans['max'] - spans['min']) * to_days

        rows.append({
            'analyte': analyte,
            'n_patients': sub['patient_id'].nunique(),
            'value_mean': sub['value'].mean(),
            'value_std': sub['value'].std(),
            'time_span_mean': spans_days.mean(),
            'time_span_std': spans_days.std(),
            'tests_per_person_mean': counts.mean(),
            'tests_per_person_std': counts.std(),
        })

    return pd.DataFrame(rows)


def compute_demographics(df, to_days=1.0):
    """Cohort-level demographics."""
    pat = df.drop_duplicates('patient_id')
    n = pat['patient_id'].nunique()
    n_female = (pat['sex'] == 1).sum()
    n_male = (pat['sex'] == 0).sum()
    # Overall time span per patient (days)
    per_patient = df.groupby('patient_id')['timestamp'].agg(['min', 'max'])
    spans_days = (per_patient['max'] - per_patient['min']) * to_days
    return pd.DataFrame([{
        'n_patients': int(n),
        'n_sequences': df.groupby(['patient_id', 'analyte']).ngroups,
        'age_mean': pat['age'].mean(),
        'age_std': pat['age'].std(),
        'pct_female': n_female / (n_female + n_male) * 100 if (n_female + n_male) > 0 else None,
        'pct_male': n_male / (n_female + n_male) * 100 if (n_female + n_male) > 0 else None,
        'span_days_median': spans_days.median(),
        'span_days_q25': spans_days.quantile(0.25),
        'span_days_q75': spans_days.quantile(0.75),
    }])


# ── Loaders ──────────────────────────────────────────────────────────────────

def norma_sequence_splits(run_id=NORMA_RUN_ID):
    """(pid, analyte) -> train/val/test from the run's predictions (unique per sequence)."""
    path = os.path.join(ROOTDIR, 'model', 'logs', run_id, 'predictions_combined.csv')
    p = (pd.read_csv(path, usecols=['pid', 'code', 'split'])
           .drop_duplicates(['pid', 'code'])
           .rename(columns={'pid': 'patient_id', 'code': 'analyte'}))
    p['analyte'] = p['analyte'].replace('', 'NA').fillna('NA')
    return p


def load_sequences(source):
    """Load sequences from combined_sequences_v2.pkl, filtered by source, with NORMA's split."""

    path = os.path.join(os.path.dirname(ROOTDIR), 'data', 'processed', 'combined_sequences_v2.pkl')
    with open(path, 'rb') as f:
        seqs = pickle.load(f)
    seqs = [s for s in seqs if s['source'] == source]
    rows = []
    for s in seqs:
        pid = s['pid']
        analyte = s['test_name']
        sex = s['sex']
        age = s['age']
        for x_val, t_val in zip(s['x'], s['t']):
            rows.append({
                'patient_id': pid, 'analyte': analyte,
                'value': float(x_val), 'timestamp': float(t_val),
                'sex': sex, 'age': age,
            })
    df = pd.DataFrame(rows)
    df['analyte'] = df['analyte'].replace('', 'NA').fillna('NA')
    df = df.merge(norma_sequence_splits(), on=['patient_id', 'analyte'], how='left')
    n_missing = int(df['split'].isna().sum())
    if n_missing:
        # predictions_combined.csv is complete for val (10 %) and test (20 %) — every sequence
        # exactly once — but its train rows come from the training DataLoader, whose
        # WeightedRandomSampler draws WITH...
        print(f"  {n_missing:,} measurements ({n_missing / len(df):.1%}) belong to training sequences "
              f"the resampling train loader never drew -> labelled 'train'")
        df['split'] = df['split'].fillna('train')
    return df


def _chunk_analyte_stats(df, exclude_labs=None, to_days=1.0):
    """Per-analyte sufficient statistics from one chunk."""
    if exclude_labs:
        df = df[~df['analyte'].isin(exclude_labs)]
    rows = []
    for analyte, sub in df.groupby('analyte'):
        per_pat = sub.groupby('patient_id')
        counts = per_pat.size()
        spans = per_pat['timestamp'].agg(['min', 'max'])
        span_days = (spans['max'] - spans['min']) * to_days
        vals = sub['value'].dropna()
        rows.append({
            'analyte': analyte,
            'n_patients': sub['patient_id'].nunique(),
            'n_meas': len(vals),
            'value_sum': vals.sum(),
            'value_ss': (vals ** 2).sum(),
            'span_sum': span_days.sum(),
            'span_ss': (span_days ** 2).sum(),
            'n_pat_spans': len(span_days),
            'tpp_sum': counts.sum(),
            'tpp_ss': (counts ** 2).sum(),
            'n_pat_tpp': len(counts),
        })
    return pd.DataFrame(rows)


def _chunk_demo_stats(df, to_days=1.0):
    """Demographic sufficient statistics from one chunk."""
    pat = df.drop_duplicates('patient_id')
    per_patient = df.groupby('patient_id')
    n_sequences = df.groupby(['patient_id', 'analyte']).ngroups
    spans = per_patient['timestamp'].agg(['min', 'max'])
    span_days = (spans['max'] - spans['min']) * to_days
    return {
        'n_patients': len(pat),
        'n_sequences': n_sequences,
        'n_female': int((pat['sex'] == 1).sum()),
        'n_male': int((pat['sex'] == 0).sum()),
        'age_sum': pat['age'].sum(),
        'age_ss': (pat['age'] ** 2).sum(),
        'span_values': span_days.values,  # collect for quantiles
    }


def _sd(total, sum_sq, n):
    """Sample SD (ddof=1) from power sums, so the streamed CHS numbers match the
    pandas .std() the other cohorts get from compute_summary/compute_demographics.
    Clipped at 0: the one-pass form can go slightly negative on rounding."""
    n = np.asarray(n, float)
    var = (np.asarray(sum_sq, float) - np.asarray(total, float) ** 2 / n) / (n - 1)
    return np.sqrt(np.clip(var, 0, None), where=n > 1, out=np.full_like(var, np.nan))


def _combine_analyte_stats(stats_list):
    """Combine per-chunk analyte statistics into final summary."""
    combined = pd.concat(stats_list, ignore_index=True)
    agg = combined.groupby('analyte').agg({
        'n_patients': 'sum', 'n_meas': 'sum',
        'value_sum': 'sum', 'value_ss': 'sum',
        'span_sum': 'sum', 'span_ss': 'sum', 'n_pat_spans': 'sum',
        'tpp_sum': 'sum', 'tpp_ss': 'sum', 'n_pat_tpp': 'sum',
    }).reset_index()

    agg['value_mean'] = agg['value_sum'] / agg['n_meas']
    agg['value_std'] = _sd(agg['value_sum'], agg['value_ss'], agg['n_meas'])
    agg['time_span_mean'] = agg['span_sum'] / agg['n_pat_spans']
    agg['time_span_std'] = _sd(agg['span_sum'], agg['span_ss'], agg['n_pat_spans'])
    agg['tests_per_person_mean'] = agg['tpp_sum'] / agg['n_pat_tpp']
    agg['tests_per_person_std'] = _sd(agg['tpp_sum'], agg['tpp_ss'], agg['n_pat_tpp'])

    return agg[['analyte', 'n_patients', 'value_mean', 'value_std',
                'time_span_mean', 'time_span_std',
                'tests_per_person_mean', 'tests_per_person_std']]


def _combine_demo_stats(demo_list):
    """Combine per-chunk demographic statistics into final summary."""
    n_patients = sum(d['n_patients'] for d in demo_list)
    n_sequences = sum(d['n_sequences'] for d in demo_list)
    n_female = sum(d['n_female'] for d in demo_list)
    n_male = sum(d['n_male'] for d in demo_list)
    age_sum = sum(d['age_sum'] for d in demo_list)
    age_ss = sum(d['age_ss'] for d in demo_list)
    all_spans = np.concatenate([d['span_values'] for d in demo_list])

    age_mean = age_sum / n_patients
    age_std = float(_sd(age_sum, age_ss, n_patients))
    n_sex = n_female + n_male

    return pd.DataFrame([{
        'n_patients': int(n_patients),
        'n_sequences': int(n_sequences),
        'age_mean': age_mean,
        'age_std': age_std,
        'pct_female': n_female / n_sex * 100 if n_sex > 0 else None,
        'pct_male': n_male / n_sex * 100 if n_sex > 0 else None,
        'span_days_median': float(np.median(all_spans)),
        'span_days_q25': float(np.percentile(all_spans, 25)),
        'span_days_q75': float(np.percentile(all_spans, 75)),
    }])


def run_chs_streaming(out_dir, n_chunks=None, analytes=None, analytes_list=None):
    """Stream CHS chunks to compute cohort summary."""
    os.makedirs(out_dir, exist_ok=True)
    from collections import defaultdict
    analyte_stats = defaultdict(list)     # key: None (all) or split name
    demo_stats = defaultdict(list)

    ds = LOADERS['chs'](n_chunks=n_chunks, analytes=analytes)
    exclude = ds.exclude_labs
    to_days = days_per_unit(ds)
    for i, chunk_dir, index_labs, ref_df in ds.iter_chunks():
        if index_labs is None:
            continue
        cols = ['patient_id', 'analyte', 'value', 'timestamp', 'sex', 'age', 'split']
        df = index_labs[[c for c in cols if c in index_labs.columns]]
        subsets = [(None, df)]
        if 'split' in df.columns:
            subsets += [(name, sub) for name, sub in df.groupby('split')]
        for key, sub in subsets:
            analyte_stats[key].append(_chunk_analyte_stats(sub, exclude_labs=exclude, to_days=to_days))
            demo_stats[key].append(_chunk_demo_stats(sub, to_days=to_days))
        del df

    print("  Combining stats...")
    splits = [k for k in analyte_stats if k is not None]
    splits.sort(key=lambda s: (SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99, s))
    _write_cohort_tables(
        out_dir, analytes_list,
        [(ALL_SPLIT, _combine_analyte_stats(analyte_stats[None]))]
        + [(s, _combine_analyte_stats(analyte_stats[s])) for s in splits],
        [(ALL_SPLIT, _combine_demo_stats(demo_stats[None]))]
        + [(s, _combine_demo_stats(demo_stats[s])) for s in splits])


def _write_cohort_tables(out_dir, analytes_list, summaries, demographics):
    """cohort.csv / demographics.csv from [(split, frame), ...]; the whole-cohort
    frame comes in as split "all"."""
    def _stack(pairs):
        parts = []
        for name, frame in pairs:
            frame = frame.copy(); frame.insert(0, 'split', name); parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    summary = _round_summary(_stack(summaries))
    path = os.path.join(out_dir, 'cohort.csv')
    save_csv(summary, path, analytes=analytes_list, keys=('split',))
    print(f"  {path}")

    demo = _round_demographics(_stack(demographics))
    path = os.path.join(out_dir, 'demographics.csv')
    save_csv(demo, path, keys=('split',))
    print(f"  {path}")


def _round_summary(df):
    return df.round({'value_mean': 2, 'value_std': 2,
                     'time_span_mean': 1, 'time_span_std': 1,
                     'tests_per_person_mean': 1, 'tests_per_person_std': 1})


def _round_demographics(df):
    return df.round({'age_mean': 1, 'age_std': 1,
                     'pct_female': 1, 'pct_male': 1,
                     'span_days_median': 0, 'span_days_q25': 0, 'span_days_q75': 0})


def run_dataset(dataset, out_dir, n_chunks=None, analytes=None, analytes_list=None):
    os.makedirs(out_dir, exist_ok=True)

    if dataset == 'chs':
        return run_chs_streaming(out_dir, n_chunks=n_chunks, analytes=analytes, analytes_list=analytes_list)

    extra_splits = []      # [(split, frame, exclude, to_days)] beyond df's own split column
    if dataset in ('ehrshot', 'mimiciv'):
        source = 'ehrshot' if dataset == 'ehrshot' else 'mimiciv'
        print(f"  Loading {source} sequences...")
        df = load_sequences(source)
        exclude = set()
        to_days = 1.0          # dev sequences carry days since first measurement
        # baseline/index on NORMA's test patients (validation pipeline index_labs)
        try:
            dev = LOADERS[dataset](n_chunks=n_chunks, analytes=analytes).load_index_labs()
            dev_days = days_per_unit(LOADERS[dataset])
            extra_splits = [(s, dev[dev['split'] == s], set(), dev_days) for s in ('baseline', 'index')
                            if (dev['split'] == s).any()]
            print(f"  baseline/index from index_labs: {len(dev):,} measurements, {dev['patient_id'].nunique():,} patients")
        except FileNotFoundError as e:
            print(f"  no baseline/index index_labs for {dataset} ({e}); train/val/test only")
    elif dataset in LOADERS:
        ds = LOADERS[dataset](n_chunks=n_chunks, analytes=analytes)
        df = ds.load_index_labs()
        exclude = ds.exclude_labs
        to_days = days_per_unit(ds)
        print(f"  timestamps in {getattr(ds, 'time_unit', 'days')} "
              f"-> spans scaled by {to_days:g} to report days")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"  {len(df):,} measurements, {df['patient_id'].nunique():,} patients")
    summary = _round_summary(compute_summary(df, exclude_labs=exclude, to_days=to_days))
    demographics = _round_demographics(compute_demographics(df, to_days=to_days))

    subsets = list(extra_splits)
    if 'split' in df.columns and df['split'].notna().any():
        subsets += [(s, df[df['split'] == s], exclude, to_days) for s in df['split'].dropna().unique()]
    subsets.sort(key=lambda t: (SPLIT_ORDER.index(t[0]) if t[0] in SPLIT_ORDER else 99, t[0]))
    _write_cohort_tables(
        out_dir, analytes_list,
        [(ALL_SPLIT, summary)] + [(s, compute_summary(sub, exclude_labs=ex, to_days=td))
                                  for s, sub, ex, td in subsets],
        [(ALL_SPLIT, demographics)] + [(s, compute_demographics(sub, to_days=td))
                                       for s, sub, ex, td in subsets])
    if not subsets:
        print("  no split column -> only the split=all rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ehrshot', 'mimiciv', 'eicu', 'chs', 'inspire'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n_chunks', type=int, default=None,
                        help='Limit to the first N chunks (CHS only, default: all)')
    parser.add_argument('--analytes', type=str, default=None,
                        help='Comma-separated list of analytes to process (default: all)')
    args = parser.parse_args()

    analytes_list = [a.strip() for a in args.analytes.split(",")] if args.analytes else None

    if args.all:
        datasets = ['ehrshot', 'mimiciv', 'eicu', 'chs', 'inspire']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['ehrshot', 'mimiciv', 'eicu', 'chs', 'inspire']

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        if dataset in LOADERS:
            ds = LOADERS[dataset](n_chunks=args.n_chunks, analytes=analytes_list)
            results_dir = ds.setup_output()
        else:
            from datasets import output_dirs
            results_dir = output_dirs(dataset)
        run_dataset(dataset, results_dir, n_chunks=args.n_chunks, analytes=analytes_list, analytes_list=analytes_list)


# Figures and tables

from figlib import *  # noqa: F401,F403


# save_table()'s first argument is the folder the table is written into, so it must match this
# directory name.

def _load_cohort_stats(ds, split):
    """(per-analyte frame indexed by analyte, demographics row) for one dataset and split (None = all)."""
    split = split or ALL_SPLIT          # the whole cohort is a split level now
    s = load_result(ds, "cohort.csv", normalize=False)
    d = load_result(ds, "demographics.csv", normalize=False)
    s = s[s["split"] == split] if s is not None and "split" in s.columns else None
    d = d[d["split"] == split] if d is not None and "split" in d.columns else None
    s_idx = None
    if s is not None and len(s) and "analyte" in s.columns:
        s = s.copy(); s["analyte"] = s["analyte"].replace("", "NA")
        s_idx = s.set_index("analyte")
        if "TG" in s_idx.index:
            if "TGL" in s_idx.index:
                s_idx.loc["TGL"] = s_idx.loc["TGL"].combine_first(s_idx.loc["TG"]); s_idx = s_idx.drop(index="TG")
            else:
                s_idx = s_idx.rename(index={"TG": "TGL"})
    demo = d.iloc[0] if d is not None and len(d) else None
    return s_idx, demo


def _cohort_table(name, split=None, datasets=COHORT_DATASETS):
    """Per-dataset cohort statistics with a demographics header; split=None is all
    measurements, otherwise one stratum of cohort.csv (baseline / index /
    train / val / test)."""
    summaries, demographics = {}, {}
    for ds in datasets:
        s_idx, demo = _load_cohort_stats(ds, split)
        if s_idx is not None:
            summaries[ds] = s_idx
        if demo is not None:
            demographics[ds] = demo
    if not summaries:
        return []

    all_a = sorted(set().union(*(s.index for s in summaries.values())) - EXCLUDE_ANALYTES)
    labels = {ds: DATASET_DISPLAY.get(ds, ds) for ds in datasets}
    csv_rows = []
    for analyte in all_a:
        row = {"Analyte": analyte}
        for ds in datasets:
            if ds in summaries and analyte in summaries[ds].index:
                s = summaries[ds].loc[analyte]
                row[f"{labels[ds]} N"] = int(s["n_patients"]) if pd.notna(s["n_patients"]) else "---"
                row[f"{labels[ds]} Mean"] = fmt_pm(s.get("value_mean"), s.get("value_std"), 1)
            else:
                row[f"{labels[ds]} N"] = "---"; row[f"{labels[ds]} Mean"] = "---"
        csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)

    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{l" + "rr" * len(datasets) + "}", r"\toprule"]
    h1, cmid = " ", []
    for i, ds in enumerate(datasets):
        h1 += r" & \multicolumn{2}{c}{" + labels[ds] + "}"; cmid.append(f"\\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}")
    lines += [h1 + r" \\", " ".join(cmid)]

    def demo_row(label, vals):
        return " & ".join([r"\textit{" + label + "}"] + [r"\multicolumn{2}{c}{" + tex_escape(vals.get(ds, "---")) + "}"
                                                         for ds in datasets]) + r" \\"
    patients, ages, sex, spans = {}, {}, {}, {}
    for ds, d in demographics.items():
        patients[ds] = f'{int(d["n_patients"]):,}' if pd.notna(d["n_patients"]) else "---"
        ages[ds] = fmt_pm(d.get("age_mean"), d.get("age_std"), 1)
        pf = f'{d["pct_female"]:.0f}' if pd.notna(d.get("pct_female")) else "---"
        pm = f'{d["pct_male"]:.0f}' if pd.notna(d.get("pct_male")) else "---"
        sex[ds] = f"{pf}% F / {pm}% M" if pf != "---" else "---"
        med = d.get("span_days_median")
        if pd.notna(med):
            # cohort_summary converts every cohort's timestamps to days.
            q25, q75 = d.get("span_days_q25"), d.get("span_days_q75")
            if med < 90:
                spans[ds] = f"{med:.0f} [{q25:.0f}, {q75:.0f}] d"
            else:
                y = 1.0 / 365.25
                spans[ds] = f"{med * y:.2f} [{q25 * y:.2f}, {q75 * y:.2f}] yr"
    lines += [demo_row("Patients", patients), demo_row("Age", ages), demo_row("Sex", sex), demo_row("Time", spans), r"\midrule"]
    lines += ["Analyte" + r" & N & Value" * len(datasets) + r" \\", r"\midrule"]
    for row in csv_rows:
        cells = [row["Analyte"]]
        for ds in datasets:
            n = row[f"{labels[ds]} N"]
            cells += [f"{n:,}" if isinstance(n, int) else str(n), tex_escape(row[f"{labels[ds]} Mean"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_table("03_cohort_summary", name, lines, csv_df, landscape=True, font_size="scriptsize")
    return [name]


# All measurements; the 02_index_labs strata (every cohort — EHRSHOT / MIMIC-IV = NORMA's test
# patients); NORMA's sequence split (development cohorts only).
def table_cohort():          return _cohort_table("cohort")
def table_cohort_baseline(): return _cohort_table("cohort_baseline", "baseline")
def table_cohort_index():    return _cohort_table("cohort_index", "index")
def table_cohort_train():    return _cohort_table("cohort_train", "train", DEV_COHORTS)
def table_cohort_val():      return _cohort_table("cohort_val", "val", DEV_COHORTS)
def table_cohort_test():     return _cohort_table("cohort_test", "test", DEV_COHORTS)

def table_analyte_reference():
    rows = []
    for a in all_analytes():
        ref = REFERENCE_INTERVALS.get(a, {})
        f, m = ref.get("F", (None, None, "")), ref.get("M", (None, None, ""))
        rows.append({"Analyte": a, "Full Name": ANALYTE_NAMES.get(a, ""), "Unit": f[2] if len(f) > 2 else "",
                     "Population RI": format_population_ri(f, m)})
    csv_df = pd.DataFrame(rows)
    body = [" & ".join(tex_escape(r[c]) for c in ("Analyte", "Full Name", "Unit", "Population RI")) + r" \\" for r in rows]
    save_table("03_cohort_summary", "analyte_reference", _table("llll", [r"Analyte & Full Name & Unit & Pop$_{RI}$ \\"], body), csv_df)
    return ["analyte_reference"]

def table_hyperparams():
    """Design choices of the NORMA model used throughout (datasets.NORMA_RUN_ID)."""
    rows = [("Temporal encoding", "Log-delta-t + periodic"),
            ("Health-state encoding", "Ternary (low / normal / high)"),
            ("Laboratory values", "Within-sequence normalization"),
            ("Age representation", "Binned (decade-wide)"),
            ("Context token", "Yes"),
            ("Output head", r"Quantile ($\tau \in \{.025,.25,.50,.75,.975\}$)"),
            ("Architecture", r"8 layers, d\_model 64, 4 heads"),
            ("Training", "Adam, lr $10^{-4}$, weight decay $10^{-3}$, batch 32, early stopping (patience 10)")]
    csv_df = pd.DataFrame([{"Component": r[0], "NORMA": r[1]} for r in rows])
    body = [f"{r[0]} & {r[1]} \\\\" for r in rows]
    save_table("03_cohort_summary", "hyperparams", _table("lp{8cm}", [r"Component & NORMA \\"], body), csv_df)
    return ["hyperparams"]

TABLES = [
    TableSpec("03_cohort_summary",        "cohort",                  table_cohort,                  False, (), None),
    TableSpec("03_cohort_summary",        "cohort_baseline",         table_cohort_baseline,         False, (), None),
    TableSpec("03_cohort_summary",        "cohort_index",            table_cohort_index,            False, (), None),
    TableSpec("03_cohort_summary",        "cohort_train",            table_cohort_train,            False, (), None),
    TableSpec("03_cohort_summary",        "cohort_val",              table_cohort_val,              False, (), None),
    TableSpec("03_cohort_summary",        "cohort_test",             table_cohort_test,             False, (), None),
    TableSpec("03_cohort_summary",        "analyte_reference",       table_analyte_reference,       False, (), None),
    TableSpec("03_cohort_summary",        "hyperparams",             table_hyperparams,             False, (), None),
]


if __name__ == "__main__":
    main()
