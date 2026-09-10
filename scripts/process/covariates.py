#!/usr/bin/env python3
"""Per-measurement covariates for NORMA v3 sequences.

Adds three things to every measurement of the already-processed dev dataframes
(MIMIC-IV_processed_df.csv / EHRSHOT_processed_df.csv, i.e. the exact rows that
produced the v2 sequences):

  * age_t     – age at the time of the draw (the processed dfs already carry a
                per-row age; v2 sequences kept only the first one)
  * setting   – care setting of the draw (see SETTING_VOCAB)
  * draw_idx  – index into a per-source "draw table": one row per
                (patient, timestamp) holding every analyte drawn at that moment,
                normalised to its own sex-specific population reference interval
                ((x - low) / (high - low), clipped to [-5, 5]); NaN = not drawn.

The v3 sequence list is built in the same group order as v2, so the
train/val/test split in model/data.py is unchanged. `--check` verifies that
against the v2 pickle.

Outputs (in --save-dir):
  {name}_sequences_v3.pkl   list of sequence dicts (v2 keys + age_t, setting, draw_idx)
  {name}_panel_v3.npy       float16 (n_draws, K) draw table, K = len(TEST_VOCAB)
  setting_summary_v3.csv    per-source setting distribution

Setting labellers for the validation cohorts (eICU, INSPIRE, CHS) live here too
so the vocabulary is defined once.
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts/ -> `process.config` (lib/ has its own constants.py)
try:
    from process.config import REFERENCE_INTERVALS  # noqa: E402
except ImportError:  # run from inside process/
    sys.path.append(HERE)
    from config import REFERENCE_INTERVALS  # noqa: E402

TEST_VOCAB = {t: i for i, t in enumerate(REFERENCE_INTERVALS.keys())}
K = len(TEST_VOCAB)
EXCLUDE = ['CRP', 'LDH', 'GGT', 'PT']  # never a *target* sequence (same as process_data.create_sequences)

# 0 is also the padding value, so "unknown" and "padded" share an embedding row.
SETTING_VOCAB = {'unknown': 0, 'outpatient': 1, 'ed': 2, 'inpatient': 3, 'icu': 4}
SETTING_NAMES = {v: k for k, v in SETTING_VOCAB.items()}
N_SETTINGS = len(SETTING_VOCAB)


def report_settings(setting, label):
    """Print (and return) the share of measurements in each care setting (used by
    process/{eicu,inspire,clalit,dev_cohort}.py)."""
    dist = pd.Series(setting).map(SETTING_NAMES).value_counts(normalize=True)
    print(f"  {label} care setting: "
          + ", ".join(f"{name} {frac:.1%}" for name, frac in dist.items()))
    return dist

DATA_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data'))   # outside the repo
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed')
MIMIC_RAW_DIR = os.path.join(DATA_ROOT, 'raw', 'mimiciv')
EHRSHOT_LAB_CSV = os.path.join(PROCESSED_DIR, 'ehrshot', 'lab_measurements.csv')
EHRSHOT_MEDS_DIR = os.path.join(DATA_ROOT, 'raw', 'ehrshot', 'meds-omop-ehrshot')

CLIP = 5.0


def _log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Loading the processed dev dataframes
# ---------------------------------------------------------------------------
def load_processed_df(path):
    """Read a *_processed_df.csv exactly as process_data.py wrote it (sodium is 'NA')."""
    import pyarrow as pa
    import pyarrow.csv as pcsv
    conv = pcsv.ConvertOptions(
        column_types={'test_name': pa.string(), 'source': pa.string(),
                      'subject_id': pa.int64(), 'sex': pa.int8(),
                      'age': pa.float32(), 'numeric_value': pa.float32(),
                      'time_delta': pa.float32(), 'time': pa.timestamp('s')},
        null_values=[''], strings_can_be_null=True,
    )
    tbl = pcsv.read_csv(path, convert_options=conv,
                        read_options=pcsv.ReadOptions(block_size=64 << 20))
    df = tbl.to_pandas()
    df['time'] = df['time'].astype('datetime64[ns]')
    df['test_name'] = df['test_name'].fillna('NA')
    assert df['test_name'].isin(TEST_VOCAB).all(), 'unknown test_name in processed df'
    return df


# ---------------------------------------------------------------------------
# Setting labellers — dev cohorts
# ---------------------------------------------------------------------------
def mimic_setting(df, raw_dir=MIMIC_RAW_DIR):
    """Label each lab by the patient's admission windows (hosp/admissions.csv):
    ED if charted inside an ED stay (edregtime..edouttime), inpatient if inside a
    hospital admission (admittime..dischtime), otherwise outpatient.
    Windows are used instead of labevents.hadm_id because most ED labs carry a
    null hadm_id (they precede the admission) and would otherwise look outpatient.
    ICU is only labelled if icu/icustays.csv(.gz) has been downloaded."""
    adm = pd.read_csv(os.path.join(raw_dir, '3.1', 'hosp', 'admissions.csv'),
                      usecols=['subject_id', 'admittime', 'dischtime', 'edregtime', 'edouttime'],
                      parse_dates=['admittime', 'dischtime', 'edregtime', 'edouttime'])
    labs = df[['subject_id', 'time']].reset_index().sort_values('time', kind='stable')

    def _inside(windows, start, end):
        w = windows.dropna(subset=[start]).sort_values(start)[['subject_id', start, end]]
        j = pd.merge_asof(labs, w, left_on='time', right_on=start, by='subject_id', direction='backward')
        assert (j['index'].values == labs['index'].values).all()
        return (j['time'] <= j[end]).fillna(False).values

    in_ed = _inside(adm, 'edregtime', 'edouttime')
    in_hosp = _inside(adm, 'admittime', 'dischtime')
    lvl = np.where(in_ed, SETTING_VOCAB['ed'],
                   np.where(in_hosp, SETTING_VOCAB['inpatient'], SETTING_VOCAB['outpatient'])).astype(np.int8)

    for icu_name in ('icustays.csv.gz', 'icustays.csv'):
        icustays = os.path.join(raw_dir, '3.1', 'icu', icu_name)
        if os.path.exists(icustays):
            _log(f'MIMIC: {icu_name} found - labelling ICU')
            icu = pd.read_csv(icustays, usecols=['subject_id', 'intime', 'outtime'], parse_dates=['intime', 'outtime'])
            in_icu = _inside(icu, 'intime', 'outtime')
            lvl[in_icu] = SETTING_VOCAB['icu']
            break
    else:
        _log('MIMIC: no icustays file - ICU labs are labelled inpatient')

    setting = np.empty(len(df), dtype=np.int8)
    setting[labs['index'].values] = lvl
    return setting


def ehrshot_visit_code_to_setting(code):
    """Map a MEDS 'visit' row code to SETTING_VOCAB.
    Codes look like 'STANFORD_VISIT/<class>|<urgency>|<type>'; class may be blank
    (Orders Only, History, Telephone, Office Visit ...) which are all ambulatory."""
    if not isinstance(code, str) or not code.startswith('STANFORD_VISIT/'):
        return SETTING_VOCAB['unknown']          # e.g. 'Domain/OMOP generated'
    cls = code.split('/', 1)[1].split('|')[0].strip()
    up = cls.upper()
    if 'ICU' in up or 'INTENSIVE' in up:
        return SETTING_VOCAB['icu']
    if up.startswith('INPATIENT') or '(I)' in cls:
        return SETTING_VOCAB['inpatient']
    if up in ('EMERGENCY', 'EMERGENCY SERVICES', 'OBSERVATION'):
        return SETTING_VOCAB['ed']
    return SETTING_VOCAB['outpatient']


def ehrshot_setting(df, lab_csv=EHRSHOT_LAB_CSV, meds_dir=EHRSHOT_MEDS_DIR):
    """Resolve each lab's visit_id to its MEDS visit row and classify it."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    _log('EHRSHOT: reading lab visit_ids')
    labs = pd.read_csv(lab_csv, usecols=['subject_id', 'time', 'test_name', 'visit_id'],
                       keep_default_na=False, na_values=[''])
    labs['test_name'] = labs['test_name'].replace({'': 'NA', 'TG': 'TGL'})
    labs['time'] = pd.to_datetime(labs['time'])
    labs = labs.dropna(subset=['visit_id']).drop_duplicates(['subject_id', 'time', 'test_name'])

    _log('EHRSHOT: reading MEDS visit rows')
    vs = []
    for f in sorted(glob.glob(os.path.join(meds_dir, 'data', '*.parquet'))):
        t = pq.read_table(f, columns=['code', 'visit_id', 'table'])
        vs.append(t.filter(pc.equal(t['table'], 'visit')).select(['code', 'visit_id']).to_pandas())
    v = pd.concat(vs, ignore_index=True).dropna(subset=['visit_id']).drop_duplicates('visit_id')
    v['visit_id'] = v['visit_id'].astype(float)
    v['setting'] = v['code'].map(ehrshot_visit_code_to_setting).astype(np.int8)
    _log(f'EHRSHOT: {len(v):,} visits; class counts {v["setting"].map(SETTING_NAMES).value_counts().to_dict()}')

    labs = labs.merge(v[['visit_id', 'setting']], on='visit_id', how='left')
    out = df[['subject_id', 'time', 'test_name']].merge(
        labs[['subject_id', 'time', 'test_name', 'setting']],
        on=['subject_id', 'time', 'test_name'], how='left')
    assert len(out) == len(df)
    setting = out['setting'].fillna(SETTING_VOCAB['unknown']).astype(np.int8).values
    return setting


# ---------------------------------------------------------------------------
# Setting labellers — validation cohorts (used by scripts/04_refs.py)
# ---------------------------------------------------------------------------
def eicu_setting(split_df, patient_df):
    """eICU labs are ICU-stay relative: offset >= 0 -> icu; offset < 0 -> the
    unit's admit source (ED -> ed, else inpatient). Uses days_from_admit
    (= labresultoffset / 1440) so it does not depend on the timestamp unit."""
    src = patient_df.set_index('patientunitstayid')['unitadmitsource']
    admit_src = split_df['patientunitstayid'].map(src)
    setting = np.where(split_df['days_from_admit'].values >= 0, SETTING_VOCAB['icu'],
                       np.where(admit_src.values == 'Emergency Department',
                                SETTING_VOCAB['ed'], SETTING_VOCAB['inpatient']))
    return setting.astype(np.int8)


def inspire_setting(split_df, ops_df):
    """INSPIRE: a lab is icu if inside any of the patient's ICU windows, inpatient
    if inside any admission window, else outpatient. Times in the ops table are
    minutes on the same patient clock as labs; split_df.days_from_admit is days."""
    ops = ops_df[['subject_id', 'admission_time', 'discharge_time', 'icuin_time', 'icuout_time']].copy()
    for c in ['admission_time', 'discharge_time', 'icuin_time', 'icuout_time']:
        ops[c] = pd.to_numeric(ops[c], errors='coerce') / (60 * 24)
    labs = split_df[['patient_id', 'days_from_admit']].reset_index()
    m = labs.merge(ops, left_on='patient_id', right_on='subject_id', how='left')
    t = m['days_from_admit']
    in_icu = (t >= m['icuin_time']) & (t <= m['icuout_time'])
    in_hosp = (t >= m['admission_time']) & (t <= m['discharge_time'])
    lvl = np.where(in_icu, SETTING_VOCAB['icu'], np.where(in_hosp, SETTING_VOCAB['inpatient'], SETTING_VOCAB['outpatient']))
    best = pd.Series(lvl).groupby(m['index'].values).max()
    setting = np.full(len(split_df), SETTING_VOCAB['outpatient'], dtype=np.int8)
    setting[best.index.values] = best.values
    return setting


def chs_setting(split_df):
    """CHS carries a per-lab `inpatient` flag."""
    flag = split_df['inpatient'].fillna(False).astype(bool).values
    return np.where(flag, SETTING_VOCAB['inpatient'], SETTING_VOCAB['outpatient']).astype(np.int8)


def collapse_icu(setting):
    """3-level variant: ICU -> inpatient (what the dev cohorts can support)."""
    s = np.asarray(setting).copy()
    s[s == SETTING_VOCAB['icu']] = SETTING_VOCAB['inpatient']
    return s


# ---------------------------------------------------------------------------
# Co-analyte draw table
# ---------------------------------------------------------------------------
def popri_normalize(values, test_names, sex01):
    """(x - low) / (high - low) with the sex-specific PopRI, clipped to [-CLIP, CLIP]."""
    ri = pd.DataFrame([(t, s, *REFERENCE_INTERVALS[t]['F' if s == 1 else 'M'][:2])
                       for t in REFERENCE_INTERVALS for s in (0, 1)],
                      columns=['test_name', 'sex', 'low', 'high'])
    key = pd.DataFrame({'test_name': np.asarray(test_names), 'sex': np.asarray(sex01).astype(np.int8)})
    b = key.merge(ri, on=['test_name', 'sex'], how='left')
    assert b['low'].notna().all()
    norm = (np.asarray(values, dtype=np.float32) - b['low'].values) / (b['high'].values - b['low'].values)
    return np.clip(norm, -CLIP, CLIP).astype(np.float32)


def build_panel(df):
    """Return (draw_idx per row, panel[n_draws, K] float16 with NaN for not-drawn)."""
    norm = popri_normalize(df['numeric_value'].values, df['test_name'].values, df['sex'].values)
    draw = df.groupby(['subject_id', 'time'], sort=False).ngroup().values.astype(np.int32)
    n = int(draw.max()) + 1
    panel = np.full((n, K), np.nan, dtype=np.float16)
    panel[draw, df['test_name'].map(TEST_VOCAB).values] = norm
    return draw, panel


# ---------------------------------------------------------------------------
# Sequences (same grouping/order as process_data.create_sequences)
# ---------------------------------------------------------------------------
def is_normal(values, test_name, sex01):
    low, high, _ = REFERENCE_INTERVALS[test_name]['F' if sex01 == 1 else 'M']
    return (low <= values) & (values <= high)


def is_high_low(values, test_name, sex01):
    """-1 low, 0 normal, 1 high (same as process_data.is_high_low)."""
    low, high, _ = REFERENCE_INTERVALS[test_name]['F' if sex01 == 1 else 'M']
    return np.where(values < low, -1, np.where(values > high, 1, 0))


def create_sequences_v3(df, setting, draw_idx):
    """df must be in the order process_data.add_time_delta_columns left it
    (sorted by source, subject_id, test_name, time)."""
    src = df['source'].values
    pid = df['subject_id'].values
    tn = df['test_name'].values
    # group boundaries (consecutive runs of the same key)
    change = np.ones(len(df), dtype=bool)
    change[1:] = (src[1:] != src[:-1]) | (pid[1:] != pid[:-1]) | (tn[1:] != tn[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(df))

    x_all = df['numeric_value'].values.astype(np.float32)
    t_all = df['time_delta'].values.astype(np.float32)
    age_all = df['age'].values.astype(np.float32)
    sex_all = df['sex'].values
    setting = np.asarray(setting, dtype=np.int8)
    draw_idx = np.asarray(draw_idx, dtype=np.int32)

    seqs = []
    for a, b in zip(starts, ends):
        test_name = tn[a]
        cid = TEST_VOCAB.get(test_name)
        sex01 = sex_all[a]
        if cid is None or pd.isna(sex01) or test_name in EXCLUDE:
            continue
        x = x_all[a:b]
        seqs.append({
            'source': src[a],
            'pid': pid[a],
            'test_name': test_name,
            'cid': cid,
            'sex': sex01,
            'age': age_all[a],
            'x': x,
            't': t_all[a:b],
            's': is_normal(x, test_name, sex01),
            's3': is_high_low(x, test_name, sex01),
            'age_t': age_all[a:b],
            'setting': setting[a:b],
            'draw_idx': draw_idx[a:b],
        })
    return seqs


def align_with_v2(seqs, v2_path):
    """Verify v3 sequences are the same observations as v2 (pid, cid, x, t, age) and
    copy the v2 state labels so ablation arms share the baseline's labels even if
    process/config.py reference intervals have been edited since v2 was built.
    Returns the number of sequences whose recomputed states differed."""
    with open(v2_path, 'rb') as f:
        v2 = pickle.load(f)
    assert len(v2) == len(seqs), f'count mismatch v2={len(v2)} v3={len(seqs)}'
    n_diff = 0
    diff_tests = {}
    for i, (a, b) in enumerate(zip(v2, seqs)):
        assert a['pid'] == b['pid'] and a['cid'] == b['cid'] and a['source'] == b['source'], i
        assert np.allclose(a['x'], b['x']) and np.allclose(a['t'], b['t']), i
        assert abs(float(a['age']) - float(b['age'])) < 1e-3, i
        if not np.array_equal(a['s3'], b['s3']):
            n_diff += 1
            diff_tests[a['test_name']] = diff_tests.get(a['test_name'], 0) + 1
        b['s'] = a['s']
        b['s3'] = a['s3']
    _log(f'align: {len(seqs):,} sequences match v2 observations; state labels copied from v2 '
         f'({n_diff:,} differed under current config.py RIs: {diff_tests})')
    return n_diff


# ---------------------------------------------------------------------------
def process_source(name, csv_name, save_dir, check=False):
    _log(f'== {name}: loading {csv_name}')
    df = load_processed_df(os.path.join(save_dir, csv_name))
    _log(f'{name}: {len(df):,} rows, {df["subject_id"].nunique():,} patients')

    if name == 'MIMIC-IV':
        setting = mimic_setting(df)
    elif name == 'EHRSHOT':
        setting = ehrshot_setting(df)
    else:
        raise ValueError(name)
    dist = pd.Series(setting).map(SETTING_NAMES).value_counts(normalize=True)
    _log(f'{name}: setting distribution {dist.round(4).to_dict()}')

    draw_idx, panel = build_panel(df)
    _log(f'{name}: draw table {panel.shape}, analytes/draw mean {np.isfinite(panel.astype(np.float32)).sum(1).mean():.1f}')

    seqs = create_sequences_v3(df, setting, draw_idx)
    _log(f'{name}: {len(seqs):,} sequences')
    return seqs, panel, dist


def _save(save_dir, name, seqs, panel):
    with open(os.path.join(save_dir, f'{name}_sequences_v3.pkl'), 'wb') as f:
        pickle.dump(seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(save_dir, f'{name}_panel_v3.npy'), panel)
    _log(f'{name}: saved {len(seqs):,} sequences + panel {panel.shape}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--save-dir', default=PROCESSED_DIR)
    ap.add_argument('--check', action='store_true', help='also verify the combined list against combined_sequences_v2.pkl')
    ap.add_argument('--sources', nargs='+', default=['MIMIC-IV', 'EHRSHOT'])
    args = ap.parse_args()

    csvs = {'MIMIC-IV': 'MIMIC-IV_processed_df.csv', 'EHRSHOT': 'EHRSHOT_processed_df.csv'}
    out = {}
    for name in args.sources:
        out[name] = process_source(name, csvs[name], args.save_dir, check=args.check)

    if set(args.sources) == {'MIMIC-IV', 'EHRSHOT'}:
        # combined = mimiciv_seq + ehrshot_seq (same order as process_data.main).
        # State labels are aligned against combined_sequences_v2.pkl (the training
        # file; per-source v2 pickles do not all exist) and the per-source v3
        # pickles are slices of the aligned combined list.
        m_seqs, m_panel, m_dist = out['MIMIC-IV']
        e_seqs, e_panel, e_dist = out['EHRSHOT']
        off = m_panel.shape[0]
        comb = m_seqs + [{**s, 'draw_idx': s['draw_idx'] + off} for s in e_seqs]
        panel = np.concatenate([m_panel, e_panel], axis=0)
        v2_comb = os.path.join(args.save_dir, 'combined_sequences_v2.pkl')
        if os.path.exists(v2_comb):
            align_with_v2(comb, v2_comb)
        elif args.check:
            raise FileNotFoundError(v2_comb)
        _save(args.save_dir, 'combined', comb, panel)
        # per-source files (labels now aligned); ehrshot draw_idx back to its own panel
        n_m = len(m_seqs)
        _save(args.save_dir, 'MIMIC-IV', comb[:n_m], m_panel)
        _save(args.save_dir, 'EHRSHOT', [{**s, 'draw_idx': s['draw_idx'] - off} for s in comb[n_m:]], e_panel)
        pd.DataFrame({'MIMIC-IV': m_dist, 'EHRSHOT': e_dist}).fillna(0).to_csv(
            os.path.join(args.save_dir, 'setting_summary_v3.csv'))
    else:
        for name, (seqs, panel, _) in out.items():
            v2_path = os.path.join(args.save_dir, f'{name}_sequences_v2.pkl')
            if os.path.exists(v2_path):
                align_with_v2(seqs, v2_path)
            _save(args.save_dir, name, seqs, panel)
    _log('done')


if __name__ == '__main__':
    main()
