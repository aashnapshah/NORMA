import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'model'))
sys.path.insert(0, os.path.join(ROOT, 'process'))

from config import REFERENCE_INTERVALS
from utils import load_checkpoint, create_model
from data import TEST_VOCAB, CODE_TO_TEST_NAME

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUN_ID   = '167f05e8'
LOG_DIR  = os.path.join(ROOT, 'model', 'logs')
DEVICE   = torch.device('cpu')

STATE_LABELS = {0: 'Low', 1: 'Normal', 2: 'High'}
STATE_COLORS = {0: '#e74c3c', 1: '#27ae60', 2: '#e67e22'}

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print(f"Loading model {RUN_ID}...")
_ckpt, _hparams = load_checkpoint(LOG_DIR, RUN_ID, best=True, device=DEVICE, quiet=True)
MODEL   = create_model(_hparams, ncodes=len(TEST_VOCAB), checkpoint=_ckpt).to(DEVICE)
MODEL.eval()
NSTATES = getattr(_hparams, 'nstates', 3)
NORMALIZE = getattr(_hparams, 'normalize', False)
print(f"Model loaded. nstates={NSTATES}, normalize={NORMALIZE}")

# Pre-load a sample of test predictions for the "load example" feature
_PREDS_PATH = os.path.join(LOG_DIR, RUN_ID, 'predictions_combined.csv')
_PREDS_DF   = None
if os.path.exists(_PREDS_PATH):
    _PREDS_DF = pd.read_csv(_PREDS_PATH, usecols=['pid', 'cid', 'code', 'x_next', 't_next', 's_next', 'split'])
    print(f"Predictions loaded: {len(_PREDS_DF):,} rows")

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
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    tests = sorted(REFERENCE_INTERVALS.keys())
    ref_json = {
        test: {
            sex: {'low': v[0], 'high': v[1], 'unit': v[2]}
            for sex, v in ranges.items()
        }
        for test, ranges in REFERENCE_INTERVALS.items()
    }
    model_info = {
        'run_id':   RUN_ID,
        'model':    getattr(_hparams, 'model',   'NormaLight'),
        'd_model':  getattr(_hparams, 'd_model', 64),
        'nlayers':  getattr(_hparams, 'nlayers', 8),
        'nstates':  NSTATES,
        'val_r2':   round(_ckpt['metrics']['val']['r2'], 3),
        'val_loss': round(_ckpt['metrics']['val']['loss'], 3),
    }
    return render_template('index.html',
                           tests=tests,
                           ref_json=json.dumps(ref_json),
                           model_info=json.dumps(model_info))

@app.route('/api/predict', methods=['POST'])
def predict():
    data      = request.get_json()
    test_name = data['test_name']
    sex01     = int(data['sex'])
    age       = float(data['age'])
    history   = sorted(data['history'], key=lambda h: h['day'])
    t_next    = float(data['t_next'])

    if len(history) < 1:
        return jsonify({'error': 'Need at least 1 history point'}), 400
    if test_name not in TEST_VOCAB:
        return jsonify({'error': f'Unknown test: {test_name}'}), 400

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

    return jsonify({
        'predictions': predictions,
        'ref_low':  low,
        'ref_high': high,
        'unit':     unit,
        't_next':   t_next,
    })

@app.route('/api/example/<test_name>')
def example(test_name):
    """Return a real patient history for a given test from the predictions CSV."""
    if _PREDS_DF is None:
        return jsonify({'error': 'No predictions data available'}), 404
    if test_name not in TEST_VOCAB:
        return jsonify({'error': 'Unknown test'}), 404

    cid  = TEST_VOCAB[test_name]
    rows = _PREDS_DF[(_PREDS_DF['cid'] == cid) & (_PREDS_DF['split'] == 'test')]
    if rows.empty:
        return jsonify({'error': 'No examples found'}), 404

    # Pick a random patient and reconstruct their history
    pid       = random.choice(rows['pid'].unique().tolist())
    pt_rows   = rows[rows['pid'] == pid].sort_values('t_next')
    history   = [{'day': round(float(r['t_next']), 1), 'value': round(float(r['x_next']), 2)}
                 for _, r in pt_rows.iterrows()]

    # t_next is the day after the last history point
    last_day  = history[-1]['day'] if history else 0
    t_next    = round(last_day + (history[-1]['day'] - history[0]['day']) / max(len(history) - 1, 1), 1) \
                if len(history) > 1 else last_day + 30.0

    return jsonify({'history': history, 't_next': t_next, 'pid': str(pid)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
