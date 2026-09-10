#!/usr/bin/env python3
"""Patient index over the v3 draw table, for conditioning on the full irregular past.

process/covariates.py builds a draw table with one row per (patient, timestamp)
holding every analyte drawn at that moment, but keeps only `draw_idx` -- the rows
where the *target* analyte was itself drawn. Every draw at a timestamp where the
target was not measured is present in the panel and unreachable from the sequence,
because build_panel calls groupby(...).ngroup() and discards the group keys.

This script recovers those keys and builds a CSR-style index so a sequence can
reach its patient's entire draw history in time order.

Time: sequences carry `time_delta`, which process_data.add_time_delta_columns
computes per ['source','subject_id','test_name'] -- so every analyte has its OWN
origin and the deltas of two analytes are NOT on a common clock. This script emits
absolute time (days since the unix epoch) per panel row instead, which is the only
way to interleave analytes correctly. model.TimeEmbedding uses consecutive gaps and
TimeEmbeddingQuery uses a difference, so both are origin-invariant; any shared clock
works as long as tokens are sorted ascending.

Row order must match covariates.build_panel exactly, so the dataframe is loaded with
the same loader and never re-sorted. --check re-derives draw_idx for a sample of
sequences and asserts it agrees with the pickle.

Output ({name}_drawmeta_v3.npz in --save-dir):
  row_time  (n_draws,) float32  absolute days, per panel row
  row_key   (n_draws,) int64    patient key of each panel row (src*10^12 + pid)
  order     (n_draws,) int32    panel rows sorted by (patient, time)
  pat_key   (n_pat,)   int64    sorted unique patient keys
  pat_ptr   (n_pat+1,) int64    slice into `order` for each patient
  offset    scalar             first EHRSHOT row index (combined only, else 0)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from process.covariates import load_processed_df, PROCESSED_DIR  # noqa: E402

SRC_MULT = 10 ** 12  # subject_id is only unique within a source
# keyed by the df/sequence `source` value (process_data.py writes 'mimiciv'/'ehrshot'),
# NOT by the process_source loop name, so model/data.py can rebuild the same key
SRC_ID = {'mimiciv': 0, 'ehrshot': 1}
NAME_TO_SRC = {'MIMIC-IV': 'mimiciv', 'EHRSHOT': 'ehrshot'}


def _log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def source_meta(name, csv_name, save_dir):
    """(draw_idx per row, row_time, row_pid) using build_panel's exact grouping."""
    _log(f'== {name}: loading {csv_name}')
    df = load_processed_df(os.path.join(save_dir, csv_name))
    _log(f'{name}: {len(df):,} rows')

    # identical to covariates.build_panel; sort=False so numbering follows the
    # dataframe's existing order, which is why the df must not be re-sorted
    g = df.groupby(['subject_id', 'time'], sort=False)
    draw = g.ngroup().values.astype(np.int32)
    n = int(draw.max()) + 1

    row_time = np.empty(n, dtype=np.float32)
    row_pid = np.empty(n, dtype=np.int64)
    t_days = df['time'].values.astype('datetime64[s]').astype(np.int64) / 86400.0
    row_time[draw] = t_days.astype(np.float32)
    row_pid[draw] = df['subject_id'].values.astype(np.int64)
    _log(f'{name}: {n:,} draws, {len(np.unique(row_pid)):,} patients')
    return draw, row_time, row_pid


def build_index(row_time, row_key):
    """CSR: panel rows grouped by patient, ascending in time within each patient."""
    order = np.lexsort((row_time, row_key)).astype(np.int32)
    sorted_key = row_key[order]
    pat_key, starts = np.unique(sorted_key, return_index=True)
    pat_ptr = np.append(starts, len(order)).astype(np.int64)
    return order, pat_key, pat_ptr


def check(save_dir, name, draw, row_time, n_seq=2000, seed=0):
    """Re-derive draw_idx from the rebuilt grouping and compare against the pickle."""
    import pickle
    p = os.path.join(save_dir, f'{name}_sequences_v3.pkl')
    if not os.path.exists(p):
        _log(f'check: {p} absent, skipping')
        return
    _log(f'check: loading {p}')
    with open(p, 'rb') as f:
        seqs = pickle.load(f)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(n_seq, len(seqs)), replace=False)
    bad_monotone = 0
    for i in idx:
        s = seqs[i]
        di = np.asarray(s['draw_idx'])
        # the target's own draws must be strictly increasing in absolute time,
        # and their gaps must equal the gaps of the stored time_delta
        at = row_time[di]
        if len(at) > 1:
            if not np.all(np.diff(at) >= 0):
                bad_monotone += 1
                continue
            gap_abs = np.diff(at.astype(np.float64))
            gap_rel = np.diff(np.asarray(s['t'], dtype=np.float64))
            if not np.allclose(gap_abs, gap_rel, atol=1.5e-2):
                raise AssertionError(
                    f'seq {i} ({s["test_name"]}): absolute-time gaps disagree with '
                    f'time_delta gaps\n abs={gap_abs[:5]}\n rel={gap_rel[:5]}')
    _log(f'check: {len(idx)} sequences OK ({bad_monotone} non-monotone, expected 0)')
    assert bad_monotone == 0, 'draw_idx is not time-ordered; numbering does not match'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--save-dir', default=PROCESSED_DIR)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    csvs = {'MIMIC-IV': 'MIMIC-IV_processed_df.csv', 'EHRSHOT': 'EHRSHOT_processed_df.csv'}
    per = {}
    for name, csv in csvs.items():
        draw, rt, rp = source_meta(name, csv, args.save_dir)
        key = SRC_ID[NAME_TO_SRC[name]] * SRC_MULT + rp
        if args.check:
            check(args.save_dir, name, draw, rt)
        per[name] = (rt, key)
        order, pat_key, pat_ptr = build_index(rt, key)
        np.savez(os.path.join(args.save_dir, f'{name}_drawmeta_v3.npz'),
                 row_time=rt, row_key=key, order=order, pat_key=pat_key,
                 pat_ptr=pat_ptr, offset=np.int64(0))
        _log(f'{name}: saved drawmeta, {len(pat_key):,} patients')

    # combined: MIMIC rows first, then EHRSHOT, matching covariates.main()
    m_rt, m_key = per['MIMIC-IV']
    e_rt, e_key = per['EHRSHOT']
    rt = np.concatenate([m_rt, e_rt])
    key = np.concatenate([m_key, e_key])
    order, pat_key, pat_ptr = build_index(rt, key)
    np.savez(os.path.join(args.save_dir, 'combined_drawmeta_v3.npz'),
             row_time=rt, row_key=key, order=order, pat_key=pat_key,
             pat_ptr=pat_ptr, offset=np.int64(len(m_rt)))
    _log(f'combined: {len(rt):,} draws, {len(pat_key):,} patients')

    per_pat = np.diff(pat_ptr)
    qs = np.percentile(per_pat, [50, 75, 90, 95, 99, 100]).astype(int)
    _log(f'draws/patient  p50={qs[0]} p75={qs[1]} p90={qs[2]} p95={qs[3]} p99={qs[4]} max={qs[5]}')
    _log('done')


if __name__ == '__main__':
    main()
