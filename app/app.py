"""NORMA web app: individualised reference intervals next to the paper's benchmarks.

Everything the deployed app needs beyond the code lives in app/data/ (built by
build_assets.py): the example cache, the NORMA bootstrap metrics, the exported
benchmark artifacts and slim copies of the validation results. The NORMA
checkpoint is read from model/logs/<run>/ when present, else from HuggingFace.
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, redirect

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model'))
sys.path.insert(0, ROOT)

# process/ moved under scripts/ on 2026-09-08, which left this import pointing at
# a directory that no longer exists; model/bootstrap.py owns the path list now.
import bootstrap  # noqa: F401,E402
from process.config import REFERENCE_INTERVALS  # noqa: E402
import benchmarks as bm                  # noqa: E402

# Vocab built directly (data.py pulls in torch at import time)
TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NORMA_RUN_ID = 'q_age_set'         # NORMA2, quantile head, age-at-draw + care-setting covariates: the main model (2026-09-03)
HF_REPO = 'aashnaps/NORMA'
LOG_DIR = os.path.join(PROJECT_ROOT, 'model', 'logs')
APP_DATA = os.path.join(ROOT, 'data')
METRICS_DIR = os.path.join(APP_DATA, 'metrics')
VALIDATION_DIR = os.path.join(APP_DATA, 'validation')
EXAMPLE_CACHE_PATH = os.path.join(APP_DATA, 'example_cache.json')

# Analytes hidden from the UI; the checkpoint still has all 34 for compatibility
COVERED_TESTS_EXCLUDE = {'CRP', 'LDH', 'GGT', 'PT'}

STATE_LABELS = {0: 'Low', 1: 'Normal', 2: 'High'}
STATE_COLORS = {k: bm.registry.STATE_COLORS[v.lower()] for k, v in STATE_LABELS.items()}

METHODS = bm.methods_json()
METHOD_COLORS = {m['key']: m['color'] for m in METHODS}

VALIDATION_DATASETS = {
    'inspire': 'INSPIRE',
    'eicu': 'eICU-CRD',
    'chs': 'CHS',
}
# PopRI is undefined inside its own normal range, so it is not a comparator here
COMPARATORS = ['PerRI', 'Cohen_m4', 'Gaussian_eb']


def _covered_tests():
    return sorted(t for t in REFERENCE_INTERVALS if t not in COVERED_TESTS_EXCLUDE)


def _bootstrap_metrics():
    path = os.path.join(METRICS_DIR, 'bootstrap_metrics.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df = df[df['Model'] == 'NORMA']
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r['Split'].lower(), {})[r['Metric']] = round(float(r['mean']), 3)
    return out


def _n_params_from_checkpoint():
    """Parameter count, for app/data/metrics/checkpoint_best.json files written
    before build_assets.py started recording n_params. Returns None on the
    deployed host, where the checkpoint is fetched from Hugging Face lazily and
    model/logs/ is absent; the template omits the stat in that case."""
    p = os.path.join(LOG_DIR, NORMA_RUN_ID, 'checkpoint_best.pth')
    if not os.path.exists(p):
        return None
    import torch
    sd = torch.load(p, map_location='cpu')['model_state_dict']
    return int(sum(v.numel() for v in sd.values()))


def _model_meta():
    meta = {}
    p = os.path.join(METRICS_DIR, 'checkpoint_best.json')
    if os.path.exists(p):
        meta = json.load(open(p))
    hp = meta.get('hyperparameters', {})
    return {
        'run_id': NORMA_RUN_ID,
        'model': hp.get('model', 'NORMA2'),
        'output_mode': hp.get('output_mode', 'quantile'),
        'd_model': hp.get('d_model', 64),
        'nlayers': hp.get('nlayers', 8),
        'nhead': hp.get('nhead', 4),
        'nstates': hp.get('nstates', 3),
        'n_params': meta.get('n_params') or _n_params_from_checkpoint(),
        'epoch': meta.get('epoch'),
        'n_tests': len(_covered_tests()),
        'metrics': _bootstrap_metrics(),
        'hf_repo': HF_REPO,
    }


MODEL_META = _model_meta()

# ---------------------------------------------------------------------------
# NORMA
# ---------------------------------------------------------------------------
_MODEL = None


def _ensure_model():
    """Load NORMA once. Returns (model, hparams, is_quantile)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import argparse
    import torch
    from utils import create_model

    device = torch.device('cpu')
    local_ckpt = os.path.join(LOG_DIR, NORMA_RUN_ID, 'checkpoint_best.pth')
    if os.path.exists(local_ckpt):
        ckpt_path = local_ckpt
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=f'{NORMA_RUN_ID}/checkpoint_best.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    hp = dict(checkpoint['hyperparameters'])
    hp['run_id'] = NORMA_RUN_ID
    hparams = argparse.Namespace(**hp)
    model = create_model(hparams, ncodes=len(TEST_VOCAB), checkpoint=checkpoint).to(device)
    model.eval()
    is_quantile = getattr(hparams, 'output_mode', 'quantile') == 'quantile'
    print(f'NORMA {NORMA_RUN_ID} loaded: {hparams.model}, quantile={is_quantile}')
    _MODEL = (model, hparams, is_quantile)
    return _MODEL


def run_inference(test_name, sex01, age, history, t_next, s_next_int):
    """NORMA prediction for one (test, patient, queried state).

    history : list of {'day': float, 'value': float}, sorted by day
    t_next  : days from the first history measurement
    Returns {mu, sigma, ci_lower, ci_upper, q25, q75}; for the quantile head
    mu is the median and sigma the 95% width / 3.92.
    """
    import torch
    model, hparams, is_quantile = _ensure_model()
    low, high, _ = REFERENCE_INTERVALS[test_name][bm.sex_str(sex01)]

    t_arr = np.array([h['day'] for h in history], dtype=np.float32)
    x_arr = np.array([h['value'] for h in history], dtype=np.float32)
    s_arr = np.array([{'Low': 0, 'Normal': 1, 'High': 2}[bm.classify(v, low, high)] for v in x_arr],
                     dtype=np.int64)

    x_h = torch.tensor(x_arr).view(1, -1, 1).float()
    s_h = torch.tensor(s_arr).view(1, -1).long()
    t_h = torch.tensor(t_arr).view(1, -1, 1).float()
    sex_t = torch.tensor([sex01]).long()
    age_t = torch.tensor([[age]]).float()
    cid_t = torch.tensor([TEST_VOCAB[test_name]]).long()
    s_next_t = torch.tensor([[s_next_int]]).long()
    t_next_t = torch.tensor([[t_next]]).float()

    with torch.no_grad():
        output = model(x_h, s_h, t_h, sex_t, age_t, cid_t, s_next_t, t_next_t, pad_mask=None)

    if is_quantile:
        q = output.squeeze(0).cpu().numpy()          # [q2.5, q25, q50, q75, q97.5]
        mu, ci_lower, ci_upper = float(q[2]), float(q[0]), float(q[4])
        sigma = (ci_upper - ci_lower) / 3.92
        q25, q75 = float(q[1]), float(q[3])
    else:
        mu_t, lv_t = output
        mu = float(mu_t.squeeze())
        sigma = float(torch.exp(0.5 * lv_t).squeeze())
        ci_lower, ci_upper = mu - 1.96 * sigma, mu + 1.96 * sigma
        q25, q75 = mu - 0.674 * sigma, mu + 0.674 * sigma
    return {
        'mu': round(mu, 3), 'sigma': round(sigma, 3),
        'ci_lower': round(ci_lower, 3), 'ci_upper': round(ci_upper, 3),
        'q25': round(q25, 3), 'q75': round(q75, 3),
    }


def predict_all(test_name, sex01, age, history, t_next):
    """NORMA in every queried state + benchmarks for one history."""
    predictions = {}
    for s_idx, s_label in STATE_LABELS.items():
        try:
            predictions[s_label] = dict(run_inference(test_name, sex01, age, history, t_next, s_idx),
                                        color=STATE_COLORS[s_idx])
        except Exception as e:
            predictions[s_label] = {'error': str(e)}
    values = [h['value'] for h in history]
    return predictions, bm.run_benchmarks(test_name, sex01, age, values)


def _result(test_name, sex01, age, predictions, benchmarks_, actual_value):
    low, high, unit = REFERENCE_INTERVALS[test_name][bm.sex_str(sex01)]
    norma = predictions.get('Normal', {})
    return {
        'norma': norma,
        'predictions': predictions,
        'benchmarks': benchmarks_,
        'classifications': bm.classify_all(actual_value, test_name, sex01, norma, benchmarks_),
        'ref_low': low, 'ref_high': high, 'unit': unit,
        'actual_value': actual_value,
        'model_id': NORMA_RUN_ID,
    }


# ---------------------------------------------------------------------------
# Example cache (built offline by build_assets.py)
# ---------------------------------------------------------------------------
_EXAMPLE_CACHE = {}
if os.path.exists(EXAMPLE_CACHE_PATH):
    with open(EXAMPLE_CACHE_PATH) as _f:
        _EXAMPLE_CACHE.update(json.load(_f))
    print(f'Example cache: {sum(len(v) for v in _EXAMPLE_CACHE.values())} examples '
          f'across {len(_EXAMPLE_CACHE)} tests')
else:
    print(f'WARNING: {EXAMPLE_CACHE_PATH} not found; run app/build_assets.py')

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')


def _ref_json():
    return {
        test: {sex: {'low': v[0], 'high': v[1], 'unit': v[2]} for sex, v in ranges.items()}
        for test, ranges in REFERENCE_INTERVALS.items()
    }


def _available_datasets():
    out = []
    for key, label in VALIDATION_DATASETS.items():
        if os.path.exists(os.path.join(VALIDATION_DIR, f'{key}_eval.csv')):
            out.append({'key': key, 'label': label})
    return out


def _page_context():
    return dict(
        tests=_covered_tests(),
        ref_json=json.dumps(_ref_json()),
        model_info=json.dumps(MODEL_META),
        model_meta=MODEL_META,
        methods_json=json.dumps(METHODS),
        datasets_json=json.dumps(_available_datasets()),
        comparators_json=json.dumps([{'key': k, 'label': bm.registry.label(k).replace('$_{RI}$', ' RI'),
                                      'html': bm._html_label(k)} for k in COMPARATORS]),
    )


@app.route('/')
def landing():
    return render_template('about.html', **_page_context())


@app.route('/app')
def index():
    return render_template('index.html', **_page_context())


@app.route('/about')
@app.route('/training')
@app.route('/validation')
def legacy_redirect():
    return redirect('/#architecture')


@app.route('/api/predict', methods=['POST'])
def predict():
    from datetime import datetime
    data = request.get_json()
    test_name = data['test_name']
    sex01 = int(data['sex'])
    age = float(data['age'])
    raw_hist = sorted(data['history'], key=lambda h: h['date'])

    if len(raw_hist) < 2:
        return jsonify({'error': 'Need at least 2 history points'}), 400
    if test_name not in TEST_VOCAB:
        return jsonify({'error': f'Unknown test: {test_name}'}), 400

    # The last point is held out as the value to classify
    actual_point = raw_hist[-1]
    actual_value = float(actual_point['value'])
    actual_date = actual_point['date']
    input_hist = raw_hist[:-1]

    first_date = datetime.strptime(input_hist[0]['date'], '%Y-%m-%d')
    history = [{'day': (datetime.strptime(h['date'], '%Y-%m-%d') - first_date).days,
                'value': float(h['value'])} for h in input_hist]
    t_next = (datetime.strptime(actual_date, '%Y-%m-%d') - first_date).days

    predictions, benchmarks_ = predict_all(test_name, sex01, age, history, t_next)
    out = _result(test_name, sex01, age, predictions, benchmarks_, actual_value)
    out.update({
        'pred_date': actual_date, 'actual_date': actual_date,
        'dates': [h['date'] for h in input_hist] + [actual_date],
    })
    return jsonify(out)


@app.route('/api/example/<test_name>')
def example(test_name):
    """A pre-computed example with bundled predictions, dated to end today."""
    if test_name not in _EXAMPLE_CACHE:
        return jsonify({'error': f'No cached examples for {test_name}'}), 404
    from datetime import datetime, timedelta

    cache = _EXAMPLE_CACHE[test_name]
    idx = request.args.get('idx', type=int)
    if idx is None:
        idx = random.randrange(len(cache))
    idx = idx % len(cache)
    entry = cache[idx]

    x, t = entry['x'], entry['t']
    today = datetime.today()
    first_day = today - timedelta(days=t[-1])
    history = [{'date': (first_day + timedelta(days=t[i])).strftime('%Y-%m-%d'),
                'value': round(x[i], 2)} for i in range(len(x) - 1)]
    actual_value = round(x[-1], 2)
    actual_date = today.strftime('%Y-%m-%d')

    out = _result(test_name, entry['sex'], entry['age'], entry['predictions'],
                  entry['benchmarks'], actual_value)
    out.update({
        'history': history,
        'pred_date': actual_date, 'actual_date': actual_date,
        'dates': [h['date'] for h in history] + [actual_date],
        'sex': entry['sex'], 'age': entry['age'],
        'example_id': idx + 1,
        'n_examples': len(cache), 'idx': idx,
    })
    return jsonify(out)


@app.route('/api/metrics')
def metrics():
    """NORMA vs history-only forecasters on the development test split (bootstrap)."""
    overall_path = os.path.join(METRICS_DIR, 'bootstrap_metrics.csv')
    bycode_path = os.path.join(METRICS_DIR, 'bootstrap_metrics_by_code.csv')
    if not os.path.exists(overall_path):
        return jsonify({'error': 'bootstrap metrics not built; run app/build_assets.py'}), 404
    rename = {'last': 'Last', 'mean': 'Mean', 'arima': 'ARIMA', NORMA_RUN_ID: 'NORMA'}
    overall = pd.read_csv(overall_path)
    overall['Model'] = overall['Model'].replace(rename)
    out = {'overall': overall.to_dict(orient='records')}
    if os.path.exists(bycode_path):
        bycode = pd.read_csv(bycode_path)
        bycode['Model'] = bycode['Model'].replace(rename)
        bycode['test_name'] = bycode['Code'].map(CODE_TO_TEST_NAME).fillna(bycode['Code'].astype(str))
        out['by_code'] = bycode.to_dict(orient='records')
    return jsonify(out)


@app.route('/api/validation')
def validation():
    """Per-analyte NORMA minus comparator deltas among PopRI-normal values.

    Source: validation/12_eval eval_pop_normal.csv (slim copy in app/data).
    """
    dataset = request.args.get('dataset', 'inspire')
    comparator = request.args.get('vs', 'PerRI')
    path = os.path.join(VALIDATION_DIR, f'{dataset}_eval.csv')
    if not os.path.exists(path):
        return jsonify({'error': f'No validation results for {dataset}'}), 404
    if comparator not in COMPARATORS:
        return jsonify({'error': f'Unknown comparator {comparator}'}), 400

    df = pd.read_csv(path, keep_default_na=False)
    outcomes = sorted(df['outcome'].unique())
    analytes = sorted(df['analyte'].unique())
    key = ['analyte', 'outcome']
    norma = df[df['method'] == 'NORMA'].set_index(key)
    comp = df[df['method'] == comparator].set_index(key)

    result = {'analytes': analytes, 'outcomes': outcomes, 'dataset': dataset, 'comparator': comparator}
    for metric in ['sensitivity', 'specificity', 'ppv']:
        result[metric] = {}
        for outcome in outcomes:
            deltas = []
            for analyte in analytes:
                k = (analyte, outcome)
                if k in norma.index and k in comp.index:
                    deltas.append(round(float(norma.loc[k, metric]) - float(comp.loc[k, metric]), 4))
                else:
                    deltas.append(None)
            result[metric][outcome] = deltas
    return jsonify(result)


@app.route('/api/benchmark')
def benchmark():
    """Cohort-level method comparison (validation/16_benchmark method_comparison.csv)."""
    dataset = request.args.get('dataset', 'inspire')
    path = os.path.join(VALIDATION_DIR, f'{dataset}_benchmark.csv')
    if not os.path.exists(path):
        return jsonify({'error': f'No benchmark results for {dataset}'}), 404
    df = pd.read_csv(path, keep_default_na=False)
    return jsonify({'dataset': dataset, 'rows': df.to_dict(orient='records'),
                    'outcomes': sorted(df['outcome'].unique())})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
