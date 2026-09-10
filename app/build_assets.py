"""Export everything the deployed app needs into app/data/.

Run from the repo root on a machine that has the model logs, the validation
results and (for the example cache) the development sequences:

    python app/build_assets.py            # everything
    python app/build_assets.py --no-cache # skip the slow example cache

Outputs (all small, committed with the app):
    data/metrics/            NORMA bootstrap metrics + checkpoint metadata
    data/gaussian_eb_prior.json   EB prior estimated on the development cohorts
    data/cohen_m2/           Cohen m2 boosters (one JSON per analyte) + residual SDs
    data/validation/         slim copies of 12_eval / 16_benchmark results per cohort
    data/example_cache.json  six example histories per analyte with NORMA + benchmarks
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import app as A  # noqa: E402  (loads config, benchmarks; not the model)

DATA_ROOT = os.environ.get('NORMA_DATA_DIR', os.path.join(PROJECT_ROOT, '..', 'data'))
SEQS_PATH = os.path.join(DATA_ROOT, 'processed', 'combined_sequences_v2.pkl')
VAL_CACHE = os.path.join(PROJECT_ROOT, 'model', 'logs', 'baselines')
# results/ is flat now (2026-09-08): one folder per cohort, every filename unique.
VAL_RESULTS = {ds: f'results/raw/{ds}' for ds in ('inspire', 'eicu', 'chs')}
KEEP_METHODS = ['PerRI', 'Cohen_m4', 'Gaussian_eb', 'PopRI', f'NORMA_{A.NORMA_RUN_ID}']
EVAL_COLS = ['analyte', 'method', 'outcome', 'n', 'n_events', 'n_flagged',
             'sensitivity', 'specificity', 'ppv', 'npv']
BENCH_COLS = ['outcome', 'method', 'n_patients', 'n_events', 'event_rate', 'auroc',
              'native_flag_rate', 'native_ppv', 'native_lift', 'ppv_at_05', 'lift_at_05',
              'n_analytes']
N_EXAMPLES = 6
MIN_POINTS, MAX_POINTS = 6, 30


def export_metrics():
    src = os.path.join(A.LOG_DIR, A.NORMA_RUN_ID)
    os.makedirs(A.METRICS_DIR, exist_ok=True)
    for f in ['bootstrap_metrics.csv', 'bootstrap_metrics_by_code.csv']:
        shutil.copy(os.path.join(src, f), os.path.join(A.METRICS_DIR, f))
    meta = json.load(open(os.path.join(src, 'checkpoint_best.json')))
    import torch
    sd = torch.load(os.path.join(src, 'checkpoint_best.pth'), map_location='cpu')['model_state_dict']
    meta['n_params'] = int(sum(v.numel() for v in sd.values()))
    json.dump(meta, open(os.path.join(A.METRICS_DIR, 'checkpoint_best.json'), 'w'), indent=1)
    print(f'metrics: {A.NORMA_RUN_ID}, {meta["n_params"]:,} parameters')


def export_eb_prior():
    art = pickle.load(open(os.path.join(VAL_CACHE, 'gaussian_eb_prior_dev.pkl'), 'rb'))
    prior = {f'{a}|{s}': {k: float(v) for k, v in p.items()} for (a, s), p in art['prior'].items()}
    json.dump(prior, open(os.path.join(A.APP_DATA, 'gaussian_eb_prior.json'), 'w'), indent=0)
    print(f'EB prior: {len(prior)} (analyte, sex) entries')


def export_cohen_m2():
    art = pickle.load(open(os.path.join(VAL_CACHE, 'cohen_dev_models.pkl'), 'rb'))
    out_dir = os.path.join(A.APP_DATA, 'cohen_m2')
    os.makedirs(out_dir, exist_ok=True)
    sigma = {}
    for analyte, entry in art['models'].items():
        if 'm2' not in entry:
            continue
        entry['m2']['bst'].save_model(os.path.join(out_dir, f'{analyte}.json'))
        sigma[analyte] = float(entry['m2']['sigma'])
    json.dump(sigma, open(os.path.join(out_dir, 'sigma.json'), 'w'), indent=1)
    print(f'Cohen m2: {len(sigma)} analytes')


def export_validation():
    os.makedirs(A.VALIDATION_DIR, exist_ok=True)
    for ds, raw_dir in VAL_RESULTS.items():
        eval_path = os.path.join(PROJECT_ROOT, raw_dir, 'eval_pop_normal.csv')
        bench_path = os.path.join(PROJECT_ROOT, raw_dir, 'method_comparison.csv')
        if not os.path.exists(eval_path):
            print(f'{ds}: no eval results, skipped')
            continue
        df = pd.read_csv(eval_path, keep_default_na=False)
        df = df[df['method'].isin(KEEP_METHODS)].copy()
        df['method'] = df['method'].replace({f'NORMA_{A.NORMA_RUN_ID}': 'NORMA'})
        df[EVAL_COLS].to_csv(os.path.join(A.VALIDATION_DIR, f'{ds}_eval.csv'), index=False)
        msg = f'{ds}: eval {len(df)} rows'
        if os.path.exists(bench_path):
            b = pd.read_csv(bench_path, keep_default_na=False)
            b = b[(b['subset'] == 'all') & b['method'].isin(KEEP_METHODS)].copy()
            b['method'] = b['method'].replace({f'NORMA_{A.NORMA_RUN_ID}': 'NORMA'})
            b[BENCH_COLS].to_csv(os.path.join(A.VALIDATION_DIR, f'{ds}_benchmark.csv'), index=False)
            msg += f', benchmark {len(b)} rows'
        print(msg)


def build_example_cache(seed=0):
    print(f'Loading sequences from {SEQS_PATH} ...')
    with open(SEQS_PATH, 'rb') as f:
        seqs = pickle.load(f)
    by_test = {}
    for s in seqs:
        if s['test_name'] in A.COVERED_TESTS_EXCLUDE or s['test_name'] not in A.TEST_VOCAB:
            continue
        if MIN_POINTS <= len(s['x']) <= MAX_POINTS and np.all(np.isfinite(s['x'])):
            by_test.setdefault(s['test_name'], []).append(s)
    del seqs
    rng = random.Random(seed)
    cache = {}
    for test_name in sorted(by_test):
        entries = []
        for seq in rng.sample(by_test[test_name], min(N_EXAMPLES * 2, len(by_test[test_name]))):
            if len(entries) >= N_EXAMPLES:
                break
            x, t = seq['x'].tolist(), seq['t'].tolist()
            sex01, age = int(seq['sex']), float(seq['age'])
            history = [{'day': t[i], 'value': x[i]} for i in range(len(x) - 1)]
            try:
                predictions, benchmarks_ = A.predict_all(test_name, sex01, age, history, t[-1])
            except Exception as e:
                print(f'  {test_name}: skipped one example ({e})')
                continue
            if 'error' in predictions.get('Normal', {}):
                continue
            entries.append({
                'sex': sex01, 'age': round(age, 0), 'x': [round(v, 3) for v in x],
                't': [round(v, 3) for v in t],
                'predictions': predictions, 'benchmarks': benchmarks_,
            })
        if entries:
            cache[test_name] = entries
        print(f'  {test_name}: {len(entries)} examples')
    json.dump(cache, open(A.EXAMPLE_CACHE_PATH, 'w'))
    total = sum(len(v) for v in cache.values())
    print(f'Example cache: {total} examples across {len(cache)} tests '
          f'({os.path.getsize(A.EXAMPLE_CACHE_PATH) / 1024:.0f} KB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true', help='skip the example cache')
    ap.add_argument('--only-cache', action='store_true', help='only rebuild the example cache')
    args = ap.parse_args()
    os.makedirs(A.APP_DATA, exist_ok=True)
    if not args.only_cache:
        export_metrics()
        export_eb_prior()
        export_cohen_m2()
        export_validation()
    if not args.no_cache:
        build_example_cache()
