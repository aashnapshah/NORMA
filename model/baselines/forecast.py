"""Point-forecast baselines: mean / last / ARIMA, plain and state-informed.

  mean, last, arima          history only
  mean_state, last_state     same statistics restricted to history values whose
                             population-defined state equals the realized future
                             state (fallback to the plain version if none)
  arimax                     ARIMA(1,1,1) with one-hot state as exogenous
                             regressors, forecast with the realized future state
                             (fallback to plain arima if the history has a single
                             state or the fit fails)

The state-informed variants are handed the realized future state so they sit on
matched inputs with the oracle NORMA evaluation (Referee 3, minor 1).

Two callers, two entry points:

  forecast_pair()   one (history, next value) pair.  scripts/05_forecasting.py
                    loads this module by file path for the external cohorts, so
                    everything above __main__ stays free of project imports --
                    numpy only, statsmodels lazily.
  __main__          the development test split, from NORMA's own sequences.
                    Its `from data import ...` lives inside main() for the same
                    reason.  Predictions are raw results, so it writes
                    results/raw/<cohort>/forecast_baselines.parquet -- the same
                    basename, shape and per-cohort folder 05_forecasting uses
                    for the external cohorts, one column per model.

                    The `combined` development split pools two cohorts, so it
                    is SPLIT BY SOURCE on the way out: one parquet under
                    results/raw/ehrshot/ and one under results/raw/mimiciv/,
                    never a pooled `dev` file.  Each sequence carries its own
                    source, so this needs no pid lookup.  Read back by
                    model/evaluate.py via baselines_path() / read_baselines().

Usage:
    python forecast.py --split test --workers 16 --with_state
    python forecast.py --split test --max_sequences 3000 --suffix _smoke
"""
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ORDER = (1, 1, 1)
PLAIN = ['mean', 'last', 'arima', 'arima_raw']
STATE_INFORMED = ['mean_state', 'last_state', 'arimax', 'arimax_raw']


# ═══════════════════════════════════════════════════════════════════════════
# Estimators
# ═══════════════════════════════════════════════════════════════════════════

def arima_forecast(x, exog_h=None, exog_next=None, order=ORDER):
    from statsmodels.tsa.arima.model import ARIMA
    if len(x) < sum(order):
        return np.nan
    try:
        fit = ARIMA(x, exog=exog_h, order=order, enforce_stationarity=False,
                    enforce_invertibility=False).fit()
        return float(np.asarray(fit.forecast(1, exog=exog_next)).ravel()[0])
    except Exception:
        return np.nan


def _onehot(s, nstates=3):
    return np.eye(nstates)[np.asarray(s, dtype=int)][:, 1:]   # drop one column (intercept)


def _sane(pred, x_h):
    """Reject numerically divergent ARIMA forecasts (enforce_stationarity=False on
    4-5 point histories can return 1e50+).  A forecast is kept if it lies within
    the history range widened by 3x the range or 25% of the level."""
    if not np.isfinite(pred):
        return False
    lo, hi = np.nanmin(x_h), np.nanmax(x_h)
    slack = max(3.0 * (hi - lo), 0.25 * max(abs(hi), abs(lo)), 1e-6)
    return (lo - slack) <= pred <= (hi + slack)


def forecast_pair(x_h, s_h, s_next, with_state=False, skip_arima=False):
    """x_h: history values; s_h: history states in {0,1,2}; s_next: realized future state.

    'arima' is the guarded forecast (raw if sane, else last value); 'arima_raw'
    and 'arima_fallback' make the guard transparent.  Same for 'arimax', which
    falls back to the guarded 'arima'."""
    x_h = np.asarray(x_h, dtype=float)
    s_h = np.asarray(s_h, dtype=int)
    out = {
        'mean': float(np.nanmean(x_h)),
        'last': float(x_h[-1]),
    }
    fittable = len(x_h) >= sum(ORDER) and not skip_arima
    raw = arima_forecast(x_h) if fittable else np.nan
    ok = _sane(raw, x_h)
    out['arima_raw'] = raw
    # undefined (NaN) below the minimum history; guarded fallback to last value otherwise
    out['arima'] = (raw if ok else out['last']) if fittable else np.nan
    out['arima_fallback'] = fittable and not ok
    if with_state:
        same = x_h[s_h == s_next]
        out['mean_state'] = float(np.nanmean(same)) if same.size else out['mean']
        out['last_state'] = float(same[-1]) if same.size else out['last']
        if not fittable:
            out['arimax_raw'] = np.nan
            out['arimax'] = np.nan
        elif len(np.unique(s_h)) > 1:
            rawx = arima_forecast(x_h, _onehot(s_h), _onehot([s_next]))
            out['arimax_raw'] = rawx
            out['arimax'] = rawx if _sane(rawx, x_h) else out['arima']
        else:
            out['arimax_raw'] = raw
            out['arimax'] = out['arima']
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Development split runner
# ═══════════════════════════════════════════════════════════════════════════
# Scores NORMA's own three-state test split.  The archived Nov-2025 predictions
# (_archive/baselines_legacy_2026-09-08/) used the two-state split over a
# per-analyte subsample, so their test rows were not NORMA's test rows.

import os   # noqa: E402  runner-only; forecast_pair above needs neither
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(MODEL_DIR)
RAW_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'raw')
# The stage prefix is baked in rather than taken from datasets.stage_file(): this
# is stage 05's analysis run on the dev split, and it must land on exactly the
# name 05_forecasting.py's result_path() produces for the external cohorts.
BASELINES_FILE = '05_forecast_baselines.parquet'
EXCLUDE_CODES = {'CRP', 'GGT', 'LDH', 'PT'}   # too sparse in the dev split to summarise

# The cohorts the `combined` development split pools, as `source` is spelled in
# the sequences pickle -- and, since 2026-09-08, as results/raw/ is keyed.
DEV_COHORTS = ('ehrshot', 'mimiciv')

# One column per model in the parquet, plus these keys.
KEY_COLUMNS = ['pid', 'cid', 'code', 'split', 'x_next', 's_next', 's_last', 'n_hist']


def baselines_path(cohort, suffix='', root=RAW_RESULTS_DIR):
    """results/raw/<cohort>/forecast_baselines[<suffix>].parquet -- predictions are
    raw results, one folder per cohort.  Shared with model/evaluate.py."""
    stem, ext = os.path.splitext(BASELINES_FILE)
    return os.path.join(root, cohort, f'{stem}{suffix}{ext}')


def read_baselines(cohorts=DEV_COHORTS, suffix='', root=RAW_RESULTS_DIR):
    """Every cohort's baselines in one frame, with the cohort in a `source`
    column.  Missing cohorts are skipped; raises if none is present."""
    import pandas as pd
    frames = []
    for c in cohorts:
        path = baselines_path(c, suffix, root)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['source'] = c
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f'no forecast baselines for {list(cohorts)} under {root}; '
            f'run model/baselines/forecast.py --split test --with_state')
    return pd.concat(frames, ignore_index=True)


def score_sequence(args):
    seq, with_state = args
    from data import CODE_TO_TEST_NAME
    x = np.asarray(seq['x'], dtype=float)
    s = np.asarray(seq['s3'], dtype=int) + 1          # -1,0,1 -> 0,1,2
    x_h, x_next = x[:-1], x[-1]
    s_h, s_next = s[:-1], s[-1]
    out = {
        'pid': seq['pid'], 'cid': int(seq['cid']), 'code': CODE_TO_TEST_NAME.get(int(seq['cid'])),
        'source': seq['source'],
        'x_next': float(x_next), 's_next': int(s_next), 's_last': int(s_h[-1]), 'n_hist': int(len(x_h)),
    }
    out.update(forecast_pair(x_h, s_h, s_next, with_state=with_state))
    return out


def main():
    import argparse
    import time
    from multiprocessing import Pool

    import pandas as pd

    from data import load_and_split_data   # local: data.py pulls in torch

    p = argparse.ArgumentParser()
    p.add_argument('--source', default='combined')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--max_sequences', type=int, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--with_state', action='store_true', help='also fit state-informed baselines')
    p.add_argument('--results_root', default=RAW_RESULTS_DIR,
                   help='results/raw root; the run writes to <results_root>/<cohort>/')
    p.add_argument('--suffix', default='')
    args = p.parse_args()

    data_dir = os.path.join(ROOT_DIR, '..', 'data', 'processed')
    train_seq, val_seq, test_seq = load_and_split_data(data_dir, args.source, print_info=False, nstates=3)
    seqs = {'train': train_seq, 'val': val_seq, 'test': test_seq}[args.split]
    del train_seq, val_seq
    if args.max_sequences:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(seqs), size=min(args.max_sequences, len(seqs)), replace=False)
        seqs = [seqs[i] for i in idx]
    print(f'{len(seqs):,} {args.split} sequences, {args.workers} workers, with_state={args.with_state}', flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(score_sequence, ((s, args.with_state) for s in seqs), chunksize=200)):
            rows.append(r)
            if i % 50000 == 0:
                print(f'  {i:>8,}  {time.time() - t0:6.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df['split'] = args.split
    print(f'done in {time.time() - t0:.0f}s; arima NaN rate {df["arima"].isna().mean():.3f}; '
          f'arima guard fallback rate {df["arima_fallback"].mean():.2%}')

    models = PLAIN + (STATE_INFORMED if args.with_state else [])
    keep = [c for c in KEY_COLUMNS + models + ['arima_fallback'] if c in df.columns]
    for cohort, part in df.groupby('source', sort=True):
        path = baselines_path(cohort, args.suffix, args.results_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part[keep].to_parquet(path, index=False)
        print(f'wrote {path}  ({len(part):,} rows, {len(models)} models)')

    print('\nMAE (median across analytes) by model and realized state:')
    lines = []
    for m in [m for m in models if not m.endswith('_raw')]:
        d = df[~df['code'].isin(EXCLUDE_CODES)].dropna(subset=[m])
        err = (d['x_next'] - d[m]).abs()
        by = d.assign(err=err).groupby(['code', 's_next'])['err'].mean().unstack()
        lines.append({'model': m, 'all': d.assign(err=err).groupby('code')['err'].mean().median(),
                      **{n: by[q].median() for q, n in enumerate(['low', 'normal', 'high']) if q in by}})
    print(pd.DataFrame(lines).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
