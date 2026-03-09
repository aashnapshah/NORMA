import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect
from pathlib import Path

# Load .env file if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _env_f:
        for _line in _env_f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'model'))
sys.path.insert(0, os.path.join(ROOT, 'process'))

from config import REFERENCE_INTERVALS
from data import TEST_VOCAB, CODE_TO_TEST_NAME

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUN_ID   = '167f05e8'
LOG_DIR  = os.path.join(ROOT, 'model', 'logs')

# Excluded from "covered tests" UI (dropdown, methods page); model still has 34 tests for checkpoint compatibility
COVERED_TESTS_EXCLUDE = {'CRP', 'LDH', 'GGT'}

STATE_LABELS = {0: 'Low', 1: 'Normal', 2: 'High'}
STATE_COLORS = {0: '#e74c3c', 1: '#27ae60', 2: '#e67e22'}

# Static model metadata (from checkpoint_best.json) — avoids loading torch just for display
_MODEL_META = {
    'run_id':   RUN_ID,
    'model':    'NormaLight',
    'd_model':  64,
    'nlayers':  8,
    'nstates':  3,
    'val_r2':   0.750,
    'val_loss': 5.70,
    'train_r2': 0.762,
    'train_loss': 5.97,
    'test_r2':  0.747,
    'test_loss': 5.73,
    'epoch':    19,
    'normalize': False,
    'n_tests':  len([t for t in REFERENCE_INTERVALS if t not in COVERED_TESTS_EXCLUDE]),
}

# ---------------------------------------------------------------------------
# Lazy model loading — deferred until first inference request
# ---------------------------------------------------------------------------
MODEL = None
_ckpt = None
_hparams = None
NSTATES = 3
NORMALIZE = False
DEVICE = None

HF_REPO = 'aashnaps/NORMA'

def _ensure_model():
    global MODEL, _ckpt, _hparams, NSTATES, NORMALIZE, DEVICE
    if MODEL is not None:
        return
    import torch
    from utils import load_checkpoint, create_model
    DEVICE = torch.device('cpu')
    print(f"Loading model {RUN_ID}...")
    # Try local checkpoint first, fall back to HuggingFace
    local_ckpt = os.path.join(LOG_DIR, RUN_ID, 'checkpoint_best.pth')
    if os.path.exists(local_ckpt):
        _ckpt, _hparams = load_checkpoint(LOG_DIR, RUN_ID, best=True, device=DEVICE, quiet=True)
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=f'{RUN_ID}/checkpoint_best.pth')
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        import argparse
        hparams = checkpoint['hyperparameters']
        hparams['run_id'] = RUN_ID
        _hparams = argparse.Namespace(**hparams)
        _ckpt = checkpoint
    MODEL = create_model(_hparams, ncodes=len(TEST_VOCAB), checkpoint=_ckpt).to(DEVICE)
    MODEL.eval()
    NSTATES = getattr(_hparams, 'nstates', 3)
    NORMALIZE = getattr(_hparams, 'normalize', False)
    print(f"Model loaded. nstates={NSTATES}, normalize={NORMALIZE}")

# Pre-load sequences and pre-compute example cache (6 per test) for instant example loading
_SEQS_PATH = os.path.join(ROOT, 'data', 'processed', 'combined_sequences_v2.pkl')
_SEQS_BY_TEST = {}
_EXAMPLE_CACHE = {}   # test_name -> list of precomputed dicts
N_CACHED_EXAMPLES = 6

_CACHE_JSON_PATH = os.path.join(ROOT, 'data', 'example_cache.json')

if os.path.exists(_CACHE_JSON_PATH):
    # Production mode: load pre-computed example cache directly (no sequences file needed)
    with open(_CACHE_JSON_PATH) as _f:
        _EXAMPLE_CACHE.update(json.load(_f))
    print(f"Example cache loaded from JSON: {sum(len(v) for v in _EXAMPLE_CACHE.values())} examples across {len(_EXAMPLE_CACHE)} tests")
elif os.path.exists(_SEQS_PATH):
    import pickle
    with open(_SEQS_PATH, 'rb') as _f:
        _all_seqs = pickle.load(_f)
    for _seq in _all_seqs:
        _SEQS_BY_TEST.setdefault(_seq['test_name'], []).append(_seq)
    print(f"Sequences loaded: {len(_all_seqs):,} total, {len(_SEQS_BY_TEST)} tests")
else:
    print(f"WARNING: neither sequences file nor example_cache.json found")

_WI_AGES     = [20, 30, 40, 50, 60, 70, 80]
_WI_HORIZONS = [7, 30, 60, 90, 180, 365, 730]

def _wi_point(test_name, sex01, age, history, t_last, horizon):
    """Single Normal-state prediction for what-if grid."""
    t_next = t_last + horizon
    sex_str = 'F' if sex01 == 1 else 'M'
    low, high, _ = REFERENCE_INTERVALS[test_name][sex_str]
    mu, sigma = run_inference(test_name, sex01, age, history, t_next, 1)  # Normal state
    return {
        'mu': round(mu, 3), 'sigma': round(sigma, 3),
        'ci_lower': round(mu - 1.96 * sigma, 3),
        'ci_upper': round(mu + 1.96 * sigma, 3),
        'ref_low': low, 'ref_high': high,
    }

def _build_example_cache():
    if not _SEQS_BY_TEST:
        return
    print("Pre-computing example cache…")
    for test_name, seqs in _SEQS_BY_TEST.items():
        if test_name not in TEST_VOCAB:
            continue
        candidates = [s for s in seqs if len(s['x']) >= 4]
        if not candidates:
            candidates = seqs
        picks = random.sample(candidates, min(N_CACHED_EXAMPLES, len(candidates)))
        entries = []
        for seq in picks:
            try:
                x     = seq['x'].tolist()
                t     = seq['t'].tolist()
                sex01 = int(seq['sex'])
                age   = float(seq['age'])
                sex_str = 'F' if sex01 == 1 else 'M'
                low, high, unit = REFERENCE_INTERVALS[test_name][sex_str]
                history = [{'day': t[i], 'value': x[i]} for i in range(len(x) - 1)]
                t_next  = t[-1]
                # Base horizon = gap between last two observed points
                base_horizon = int(round(t[-1] - t[-2])) if len(t) >= 2 else 30
                t_last = t[-2] if len(t) >= 2 else t[-1]
                preds = {}
                for s_idx, s_label in STATE_LABELS.items():
                    mu, sigma = run_inference(test_name, sex01, age, history, t_next, s_idx)
                    preds[s_label] = {
                        'mu': round(mu, 3), 'sigma': round(sigma, 3),
                        'ci_lower': round(mu - 1.96 * sigma, 3),
                        'ci_upper': round(mu + 1.96 * sigma, 3),
                        'color': STATE_COLORS[s_idx],
                    }
                gmm_mu, gmm_sigma = compute_gmm_setpoint([h['value'] for h in history])
                ref_mu    = (low + high) / 2
                ref_sigma = (high - low) / 3.92

                # Pre-compute what-if sweeps (instant frontend lookups, no live API calls)
                wi_sweeps = {
                    'age':     [dict(value=a, **_wi_point(test_name, sex01,  a,    history, t_last, base_horizon)) for a in _WI_AGES],
                    'sex':     [dict(value=s, **_wi_point(test_name, s,      age,  history, t_last, base_horizon)) for s in [0, 1]],
                    'horizon': [dict(value=h, **_wi_point(test_name, sex01,  age,  history, t_last, h))            for h in _WI_HORIZONS],
                }

                entries.append({
                    'pid': str(seq['pid']), 'sex': sex01, 'age': round(age, 0),
                    'x': x, 't': t, 'base_horizon': base_horizon,
                    'predictions': preds,
                    'gmm_prediction': {
                        'mu': round(gmm_mu, 3), 'sigma': round(gmm_sigma, 3),
                        'ci_lower': round(gmm_mu - 1.96 * gmm_sigma, 3),
                        'ci_upper': round(gmm_mu + 1.96 * gmm_sigma, 3),
                    },
                    'ref_prediction': {
                        'mu': round(ref_mu, 3), 'sigma': round(ref_sigma, 3),
                        'ci_lower': low, 'ci_upper': high,
                    },
                    'ref_low': low, 'ref_high': high, 'unit': unit,
                    'wi_sweeps': wi_sweeps,
                })
            except Exception:
                pass
        if entries:
            _EXAMPLE_CACHE[test_name] = entries
    total = sum(len(v) for v in _EXAMPLE_CACHE.values())
    print(f"Example cache built: {total} examples across {len(_EXAMPLE_CACHE)} tests")

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def value_to_state3(value, low, high):
    """Map a raw value to the shifted 3-state index (0=low, 1=normal, 2=high)."""
    if value < low:  return 0
    if value > high: return 2
    return 1

def run_inference(test_name, sex01, age, history, t_next, s_next_int):
    """
    Run NormaLight inference for a single (test, patient, state) combination.

    history : list of {'day': float, 'value': float}, already sorted by day
    t_next  : float – days from first history measurement
    s_next_int : 0=low, 1=normal, 2=high  (shifted scale used by model)
    Returns (mu, sigma) in original (unnormalized) units.
    """
    import torch
    _ensure_model()
    cid     = TEST_VOCAB[test_name]
    sex_str = 'F' if sex01 == 1 else 'M'
    low, high, unit = REFERENCE_INTERVALS[test_name][sex_str]

    t_arr = np.array([h['day']   for h in history], dtype=np.float32)
    x_arr = np.array([h['value'] for h in history], dtype=np.float32)

    if NORMALIZE:
        span  = high - low
        x_arr = (x_arr - low) / span

    s_arr = np.array([value_to_state3(v, low, high) for v in
                      (x_arr * (high - low) + low if NORMALIZE else x_arr)],
                     dtype=np.int64)

    x_h     = torch.tensor(x_arr).view(1, -1, 1).float()
    s_h     = torch.tensor(s_arr).view(1, -1).long()
    t_h     = torch.tensor(t_arr).view(1, -1, 1).float()
    sex_t   = torch.tensor([sex01]).long()
    age_t   = torch.tensor([[age]]).float()
    cid_t   = torch.tensor([cid]).long()
    s_next_t = torch.tensor([[s_next_int]]).long()
    t_next_t = torch.tensor([[t_next]]).float()

    with torch.no_grad():
        mu_t, lv_t = MODEL(x_h, s_h, t_h, sex_t, age_t, cid_t,
                           s_next_t, t_next_t, pad_mask=None)

    mu    = float(mu_t.squeeze())
    sigma = float(torch.exp(0.5 * lv_t).squeeze())

    if NORMALIZE:
        span  = high - low
        mu    = mu    * span + low
        sigma = sigma * span

    return mu, sigma

# ---------------------------------------------------------------------------
# GMM setpoint (personalised reference interval from patient history)
# ---------------------------------------------------------------------------
def compute_gmm_setpoint(values):
    """Fit GMM to patient history values; return (mu, sigma) of dominant component."""
    from sklearn.mixture import GaussianMixture
    x = np.array(values, dtype=float)
    if len(x) < 2:
        return float(np.mean(x)), float(max(np.std(x), 0.01))

    best_aic, best_gm = np.inf, None
    for n in range(1, min(4, len(x) + 1)):
        try:
            gm = GaussianMixture(n_components=n, covariance_type='full',
                                 random_state=0, max_iter=300, reg_covar=0.001).fit(x.reshape(-1,1))
            aic = gm.aic(x.reshape(-1,1))
            if aic < best_aic and gm.converged_:
                best_aic, best_gm = aic, gm
        except Exception:
            pass

    if best_gm is None or best_gm.n_components == 1:
        return float(np.mean(x)), float(max(np.std(x), 0.01))

    weights  = best_gm.weights_
    dom_idx  = int(np.argmax(weights))
    thresh   = {2: 0.70, 3: 0.45}.get(best_gm.n_components, 0.60)
    if weights[dom_idx] >= thresh:
        mu    = float(best_gm.means_[dom_idx, 0])
        sigma = float(np.sqrt(best_gm.covariances_[dom_idx, 0, 0]))
        return mu, max(sigma, 0.01)

    mu = float(np.dot(weights, best_gm.means_[:,0]))
    return mu, float(max(np.std(x), 0.01))

_build_example_cache()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/app')
def index():
    tests = sorted(t for t in REFERENCE_INTERVALS.keys() if t not in COVERED_TESTS_EXCLUDE)
    ref_json = {
        test: {
            sex: {'low': v[0], 'high': v[1], 'unit': v[2]}
            for sex, v in ranges.items()
        }
        for test, ranges in REFERENCE_INTERVALS.items()
    }
    model_info = dict(_MODEL_META)
    return render_template('index.html',
                           tests=tests,
                           ref_json=json.dumps(ref_json),
                           model_info=json.dumps(model_info))

@app.route('/api/predict', methods=['POST'])
def predict():
    from datetime import datetime
    data      = request.get_json()
    test_name = data['test_name']
    sex01     = int(data['sex'])
    age       = float(data['age'])
    raw_hist  = sorted(data['history'], key=lambda h: h['date'])

    if len(raw_hist) < 2:
        return jsonify({'error': 'Need at least 2 history points'}), 400
    if test_name not in TEST_VOCAB:
        return jsonify({'error': f'Unknown test: {test_name}'}), 400

    # Hold out the last data point as actual observed value
    actual_point = raw_hist[-1]
    actual_value = float(actual_point['value'])
    actual_date  = actual_point['date']
    input_hist   = raw_hist[:-1]

    # Convert dates -> days from first measurement
    first_date = datetime.strptime(input_hist[0]['date'], '%Y-%m-%d')
    history = [
        {'day': (datetime.strptime(h['date'], '%Y-%m-%d') - first_date).days,
         'value': float(h['value'])}
        for h in input_hist
    ]
    t_next = (datetime.strptime(actual_date, '%Y-%m-%d') - first_date).days

    sex_str = 'F' if sex01 == 1 else 'M'
    low, high, unit = REFERENCE_INTERVALS[test_name][sex_str]

    predictions = {}
    for s_idx, s_label in STATE_LABELS.items():
        try:
            mu, sigma = run_inference(test_name, sex01, age, history, t_next, s_idx)
            predictions[s_label] = {
                'mu':       round(mu, 3),
                'sigma':    round(sigma, 3),
                'ci_lower': round(mu - 1.96 * sigma, 3),
                'ci_upper': round(mu + 1.96 * sigma, 3),
                'color':    STATE_COLORS[s_idx],
            }
        except Exception as e:
            predictions[s_label] = {'error': str(e)}

    # GMM setpoint — personalised reference interval from patient history
    hist_values = [h['value'] for h in history]
    try:
        gmm_mu, gmm_sigma = compute_gmm_setpoint(hist_values)
        gmm_prediction = {
            'mu':       round(gmm_mu, 3),
            'sigma':    round(gmm_sigma, 3),
            'ci_lower': round(gmm_mu - 1.96 * gmm_sigma, 3),
            'ci_upper': round(gmm_mu + 1.96 * gmm_sigma, 3),
        }
    except Exception as e:
        gmm_prediction = {'error': str(e)}

    # Reference range — modelled as Gaussian with 95% CI = [low, high]
    ref_mu    = (low + high) / 2
    ref_sigma = (high - low) / 3.92
    ref_prediction = {
        'mu':       round(ref_mu, 3),
        'sigma':    round(ref_sigma, 3),
        'ci_lower': low,
        'ci_upper': high,
    }

    # Classify actual value under each paradigm
    def classify(val, lo, hi):
        if val < lo: return 'Low'
        if val > hi: return 'High'
        return 'Normal'

    norm_pred = predictions.get('Normal', {})
    classifications = {
        'population':    classify(actual_value, low, high),
        'personalized':  classify(actual_value,
                                  gmm_prediction.get('ci_lower', low),
                                  gmm_prediction.get('ci_upper', high)),
        'norma':         classify(actual_value,
                                  norm_pred.get('ci_lower', low),
                                  norm_pred.get('ci_upper', high)),
    }

    return jsonify({
        'predictions':     predictions,
        'gmm_prediction':  gmm_prediction,
        'ref_prediction':  ref_prediction,
        'ref_low':         low,
        'ref_high':        high,
        'unit':            unit,
        'pred_date':       actual_date,
        'actual_value':    actual_value,
        'actual_date':     actual_date,
        'classifications': classifications,
        'dates':           [h['date'] for h in input_hist] + [actual_date],
    })

@app.route('/api/example/<test_name>')
def example(test_name):
    """Return a pre-computed example with bundled predictions for instant rendering."""
    if test_name not in _EXAMPLE_CACHE:
        return jsonify({'error': f'No cached examples for {test_name}'}), 404

    from datetime import datetime, timedelta
    cache = _EXAMPLE_CACHE[test_name]
    idx   = request.args.get('idx', type=int)
    if idx is None:
        idx = random.randrange(len(cache))
    idx   = idx % len(cache)
    entry = cache[idx]

    x, t      = entry['x'], entry['t']
    today     = datetime.today()
    first_day = today - timedelta(days=t[-1])

    history = [
        {'date':  (first_day + timedelta(days=t[i])).strftime('%Y-%m-%d'),
         'value': round(x[i], 2)}
        for i in range(len(x) - 1)
    ]
    pred_date = today.strftime('%Y-%m-%d')

    # Include the actual last observed value for evaluation
    actual_value = round(x[-1], 2)
    actual_date  = (first_day + timedelta(days=t[-1])).strftime('%Y-%m-%d')

    # Classify actual value under each paradigm
    ref_low, ref_high = entry['ref_low'], entry['ref_high']
    norm_pred = entry['predictions'].get('Normal', {})
    gmm_pred  = entry['gmm_prediction']

    def classify(val, lo, hi):
        if val < lo: return 'Low'
        if val > hi: return 'High'
        return 'Normal'

    classifications = {
        'population': classify(actual_value, ref_low, ref_high),
        'personalized': classify(actual_value, gmm_pred.get('ci_lower', ref_low), gmm_pred.get('ci_upper', ref_high)),
        'norma': classify(actual_value, norm_pred.get('ci_lower', ref_low), norm_pred.get('ci_upper', ref_high)),
    }

    return jsonify({
        'history':        history,
        'pred_date':      actual_date,
        'actual_value':   actual_value,
        'actual_date':    actual_date,
        'classifications': classifications,
        'dates':          [h['date'] for h in history] + [actual_date],
        'pid':            entry['pid'],
        'sex':            entry['sex'],
        'age':            entry['age'],
        'predictions':    entry['predictions'],
        'gmm_prediction': entry['gmm_prediction'],
        'ref_prediction': entry['ref_prediction'],
        'ref_low':        entry['ref_low'],
        'ref_high':       entry['ref_high'],
        'unit':           entry['unit'],
        'wi_sweeps':      entry.get('wi_sweeps'),
        'n_examples':     len(cache),
        'idx':            idx,
    })

@app.route('/')
def landing():
    tests = sorted(t for t in REFERENCE_INTERVALS.keys() if t not in COVERED_TESTS_EXCLUDE)
    model_info = dict(_MODEL_META)
    ref_json = {
        test: {sex: {'low': v[0], 'high': v[1], 'unit': v[2]}
               for sex, v in ranges.items()}
        for test, ranges in REFERENCE_INTERVALS.items()
    }
    return render_template('about.html', model_info=model_info, tests=tests,
                           ref_json=json.dumps(ref_json))

def _methods_context():
    tests = sorted(t for t in REFERENCE_INTERVALS.keys() if t not in COVERED_TESTS_EXCLUDE)
    return {
        'model_info': dict(_MODEL_META),
        'tests': tests,
    }

@app.route('/about')
def about():
    return redirect('/training')

@app.route('/training')
def training():
    return redirect('/#architecture')

@app.route('/validation')
def validation():
    return redirect('/#architecture')


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

@app.route('/api/interpret', methods=['POST'])
def interpret():
    import urllib.request, json as _json

    d         = request.get_json()
    test_name = d['test_name']
    sex_str   = 'Female' if d['sex'] == 1 else 'Male'
    age       = d['age']
    history   = d['history']
    pred_date = d['pred_date']
    preds     = d['predictions']
    ref_low   = d['ref_low']
    ref_high  = d['ref_high']
    unit      = d['unit']

    norm  = preds.get('Normal', {})
    low_p = preds.get('Low',    {})
    high_p= preds.get('High',   {})
    gmm_pred = d.get('gmm_prediction', {})

    norm_mu   = norm.get('mu', '?');  norm_lo = norm.get('ci_lower', '?'); norm_hi = norm.get('ci_upper', '?')
    low_mu    = low_p.get('mu', '?')
    high_mu   = high_p.get('mu', '?')
    gmm_mu    = gmm_pred.get('mu', '?'); gmm_lo = gmm_pred.get('ci_lower', '?'); gmm_hi = gmm_pred.get('ci_upper', '?')

    hist_text = '\n'.join(f"  {h['date']}: {h['value']} {unit}" for h in history[-12:])
    prompt = (
        f"You are interpreting a lab result prediction for a {int(age)}-year-old {sex_str} patient.\n\n"
        f"Lab test: {test_name}\n\n"
        f"PATIENT HISTORY ({unit}):\n{hist_text}\n\n"
        f"THREE REFERENCE APPROACHES for {pred_date}:\n"
        f"1. Population reference range: {ref_low}–{ref_high} {unit} "
        f"(standard population-level interval by sex; does not account for individual variation)\n"
        f"2. Personalized (GMM) setpoint: {gmm_mu} {unit}, 95% CI {gmm_lo}–{gmm_hi} {unit} "
        f"(fitted to this patient's own observed history; reflects their personal physiological baseline)\n"
        f"3. NORMA model prediction: {norm_mu} {unit}, 95% CI {norm_lo}–{norm_hi} {unit} "
        f"(transformer model conditioned on longitudinal history, timing, sex, age, and clinical state; "
        f"if Low state: {low_mu} {unit}; if High state: {high_mu} {unit})\n\n"
        f"Please write a concise 3–4 sentence interpretation covering:\n"
        f"(a) What {test_name} measures and why it matters clinically.\n"
        f"(b) The recent trend in the patient's values — are they stable, rising, or falling?\n"
        f"(c) Why the Personalized range differs from the population range "
        f"(i.e. it is derived from the patient's own observed values).\n"
        f"(d) Why NORMA's prediction may be more balanced than the other two "
        f"(it integrates temporal dynamics, timing of next measurement, and clinical state conditioning "
        f"rather than just a static fit or population average).\n"
        f"Be factual, not alarmist. Do not use bullet points — write in flowing prose."
    )

    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Set the GEMINI_API_KEY environment variable.'}), 500

    try:
        url     = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'
        payload = _json.dumps({'contents': [{'parts': [{'text': prompt}]}]}).encode()
        req     = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = _json.loads(resp.read())
        text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'interpretation': text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


_CODE_TO_NAME = {
    0:'HCT', 1:'HGB', 2:'MCH', 3:'MCHC', 4:'MPV', 5:'PLT', 6:'RBC', 7:'RDW',
    8:'WBC', 9:'MCV', 10:'NA', 11:'K', 12:'CL', 13:'CO2', 14:'BUN', 15:'CRE',
    16:'GLU', 17:'A1C', 18:'CA', 19:'ALT', 21:'AST', 24:'ALP', 25:'TBIL',
    26:'DBIL', 27:'ALB', 28:'TP', 30:'TC', 31:'HDL', 32:'LDL', 33:'TGL',
}

@app.route('/api/metrics')
def metrics():
    import pandas as pd
    overall_path = os.path.join(LOG_DIR, RUN_ID, 'bootstrap_metrics.csv')
    bycode_path  = os.path.join(LOG_DIR, RUN_ID, 'bootstrap_metrics_by_code.csv')

    # Fall back to HuggingFace if local files don't exist
    if not os.path.exists(overall_path):
        from huggingface_hub import hf_hub_download
        overall_path = hf_hub_download(repo_id=HF_REPO, filename=f'{RUN_ID}/bootstrap_metrics.csv')
        bycode_path  = hf_hub_download(repo_id=HF_REPO, filename=f'{RUN_ID}/bootstrap_metrics_by_code.csv')

    overall = pd.read_csv(overall_path)
    bycode  = pd.read_csv(bycode_path)

    overall['Model'] = overall['Model'].replace({RUN_ID: 'NORMA'})
    bycode['Model']  = bycode['Model'].replace({RUN_ID: 'NORMA'})
    bycode['test_name'] = bycode['Code'].map(_CODE_TO_NAME).fillna(bycode['Code'].astype(str))

    return jsonify({
        'overall': overall.to_dict(orient='records'),
        'by_code': bycode.to_dict(orient='records'),
    })


@app.route('/api/sensitivity', methods=['POST'])
def sensitivity():
    """Sweep a single parameter (age, sex, or horizon) holding others fixed."""
    from datetime import datetime, timedelta
    data      = request.get_json()
    test_name = data['test_name']
    sex01     = int(data['sex'])
    base_age  = float(data['age'])
    history   = sorted(data['history'], key=lambda h: h['day'])
    base_t    = float(data['base_t'])   # days from first measurement to "today"
    param     = data['param']           # 'age' | 'sex' | 'horizon'
    values    = data['values']          # list of sweep values

    if test_name not in TEST_VOCAB:
        return jsonify({'error': f'Unknown test: {test_name}'}), 400

    sex_str  = 'F' if sex01 == 1 else 'M'
    low, high, unit = REFERENCE_INTERVALS[test_name][sex_str]
    results = []

    for v in values:
        sweep_sex  = (1 if v else 0) if param == 'sex'     else sex01
        sweep_age  = float(v)        if param == 'age'     else base_age
        sweep_t    = base_t + float(v) if param == 'horizon' else base_t
        sweep_sstr = 'F' if sweep_sex == 1 else 'M'
        s_low, s_high, _ = REFERENCE_INTERVALS[test_name][sweep_sstr]

        try:
            mu, sigma = run_inference(test_name, sweep_sex, sweep_age, history, sweep_t, 1)  # Normal state
            results.append({
                'value':    v,
                'mu':       round(mu, 3),
                'sigma':    round(sigma, 3),
                'ci_lower': round(mu - 1.96 * sigma, 3),
                'ci_upper': round(mu + 1.96 * sigma, 3),
                'ref_low':  s_low,
                'ref_high': s_high,
            })
        except Exception as e:
            results.append({'value': v, 'error': str(e)})

    return jsonify({'results': results, 'unit': unit})


_DISEASE_RISK_CACHE = {}

@app.route('/api/disease_risk/<test_name>')
def disease_risk(test_name):
    import urllib.request, json as _json, re

    if test_name in _DISEASE_RISK_CACHE:
        return jsonify(_DISEASE_RISK_CACHE[test_name])

    prompt = (
        f"You are a clinical reference assistant. For the lab test '{test_name}', "
        f"provide structured information in the following JSON format exactly — no markdown, no extra text:\n"
        f'{{\n'
        f'  "what_it_measures": "one sentence explaining what {test_name} measures and its physiological role",\n'
        f'  "low_risks": [\n'
        f'    {{"disease": "Disease name", "description": "brief clinical significance (1 sentence)", "citation": "Author et al., Journal Year"}}\n'
        f'  ],\n'
        f'  "high_risks": [\n'
        f'    {{"disease": "Disease name", "description": "brief clinical significance (1 sentence)", "citation": "Author et al., Journal Year"}}\n'
        f'  ]\n'
        f'}}\n'
        f'Include 3–4 entries for each of low_risks and high_risks. '
        f'Use real, specific diseases and real published citations (author, journal, year). '
        f'Return only valid JSON.'
    )

    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Set the GEMINI_API_KEY environment variable.'}), 500

    try:
        url     = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'
        payload = _json.dumps({'contents': [{'parts': [{'text': prompt}]}],
                               'generationConfig': {'temperature': 0.2}}).encode()
        req     = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=25) as resp:
            result = _json.loads(resp.read())
        raw = result['candidates'][0]['content']['parts'][0]['text']
        # Strip any markdown code fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r'```$', '', raw.strip())
        data = _json.loads(raw.strip())
        _DISEASE_RISK_CACHE[test_name] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
