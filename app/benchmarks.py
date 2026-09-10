"""Reference-interval benchmarks the app shows next to NORMA, for one patient.

The paper compares NORMA against five families of interval; the app runs the
ones that are defined from a single analyte history:

  PopRI        fixed population interval (scripts/process/config.py)
  PerRI        personalised interval: dominant GMM component of the patient's
               own history (model/baselines/gaussian.py, gmm_setpoint)
  Gaussian_eb  empirical-Bayes Gaussian on the PopRI-normal history, prior
               estimated once on the development cohorts
               (model/baselines/gaussian.py, eb_posterior)
  Cohen_m2     Cohen et al. 2021 single-lab model: XGBoost on
               (age, sex, history mean), trained on the development cohorts
               (model/baselines/cohen.py). m3/m4 need the co-analyte panel,
               which the app does not collect.

Names and colours come from scripts/lib/models.py so the app matches the
figures. Artifacts (EB prior, Cohen boosters) are exported into app/data/ by
build_assets.py so the deployed app carries no pickle.
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
DATA_DIR = os.path.join(ROOT, 'data')

for _p in (os.path.join(PROJECT_ROOT, 'model', 'baselines'),
           os.path.join(PROJECT_ROOT, 'scripts', 'process')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import REFERENCE_INTERVALS  # noqa: E402
from gaussian import eb_posterior, gmm_setpoint   # noqa: E402  numpy/pandas/sklearn only


def _load_registry():
    spec = importlib.util.spec_from_file_location(
        'norma_model_registry', os.path.join(PROJECT_ROOT, 'scripts', 'lib', 'models.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


registry = _load_registry()

Z = 1.96

# Order the app displays methods in. NORMA first, then the population reference,
# then the personalised benchmarks from least to most model-based.
METHOD_KEYS = ['NORMA', 'PopRI', 'PerRI', 'Gaussian_eb', 'Cohen_m2']

METHOD_NOTES = {
    'NORMA': 'Transformer conditioned on the history, timing, age, sex and the normal state',
    'PopRI': 'Fixed population interval by sex',
    'PerRI': 'Dominant Gaussian-mixture component of the patient\'s own history',
    'Gaussian_eb': 'Empirical-Bayes Gaussian on the history values inside the population interval',
    'Cohen_m2': 'Cohen et al. 2021 single-lab model (age, sex, history mean)',
}


def _html_label(key):
    label = registry.label(key)
    return label.replace('$_{RI}$', '<sub>RI</sub>').replace('$', '')


def methods_json():
    """[{key, label, html, color, note}] for the templates."""
    return [{
        'key': k,
        'label': registry.label(k, short=True).replace('$_{RI}$', ' RI').replace('$', ''),
        'html': _html_label(k),
        'color': registry.color(k),
        'note': METHOD_NOTES[k],
    } for k in METHOD_KEYS]


def sex_str(sex01):
    return 'F' if int(sex01) == 1 else 'M'


def classify(value, low, high):
    if value < low:
        return 'Low'
    if value > high:
        return 'High'
    return 'Normal'


def _interval(mu, sigma, low=None, high=None, fallback=False, **extra):
    ci_lower = mu - Z * sigma if low is None else low
    ci_upper = mu + Z * sigma if high is None else high
    out = {
        'mu': round(float(mu), 3), 'sigma': round(float(sigma), 3),
        'ci_lower': round(float(ci_lower), 3), 'ci_upper': round(float(ci_upper), 3),
        'fallback': bool(fallback),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# PopRI
# ---------------------------------------------------------------------------
def pop_ri(test_name, sex01):
    low, high, _unit = REFERENCE_INTERVALS[test_name][sex_str(sex01)]
    # Modelled as a Gaussian whose 95% band is the interval, for the bell curve
    return _interval((low + high) / 2, (high - low) / (2 * Z), low=low, high=high)


# ---------------------------------------------------------------------------
# PerRI
# ---------------------------------------------------------------------------
def per_ri(values):
    """Dominant Gaussian-mixture component of the history, straight from the
    pipeline's own estimator so the app cannot drift from the figures."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return _interval(np.nan, 0.01, n=0)
    mu, sigma = gmm_setpoint(x)
    # floor sigma so a zero-variance history still renders as a band, not a spike
    return _interval(float(mu), max(float(sigma), 0.01), n=int(len(x)))


# ---------------------------------------------------------------------------
# Gaussian EB
# ---------------------------------------------------------------------------
_EB_PRIOR = None


def _eb_prior():
    global _EB_PRIOR
    if _EB_PRIOR is None:
        path = os.path.join(DATA_DIR, 'gaussian_eb_prior.json')
        _EB_PRIOR = json.load(open(path)) if os.path.exists(path) else {}
    return _EB_PRIOR


def gaussian_eb(test_name, sex01, values):
    """EB posterior-predictive band on the PopRI-normal history; PopRI if none."""
    low, high, _ = REFERENCE_INTERVALS[test_name][sex_str(sex01)]
    prior = _eb_prior().get(f'{test_name}|{sex_str(sex01)}')
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= low) & (x <= high)]
    if prior is None or len(x) < 1:
        out = pop_ri(test_name, sex01)
        out['fallback'] = True
        out['n'] = int(len(x))
        return out
    mu, sd = eb_posterior(x, prior)
    return _interval(mu, sd, n=int(len(x)))


# ---------------------------------------------------------------------------
# Cohen m2
# ---------------------------------------------------------------------------
_COHEN = {}


def _cohen_model(test_name):
    if test_name in _COHEN:
        return _COHEN[test_name]
    meta_path = os.path.join(DATA_DIR, 'cohen_m2', 'sigma.json')
    model_path = os.path.join(DATA_DIR, 'cohen_m2', f'{test_name}.json')
    entry = None
    if os.path.exists(meta_path) and os.path.exists(model_path):
        try:
            import xgboost as xgb
            bst = xgb.Booster()
            bst.load_model(model_path)
            entry = {'bst': bst, 'sigma': float(json.load(open(meta_path))[test_name])}
        except Exception as e:  # xgboost missing or model unreadable
            print(f'Cohen m2 unavailable for {test_name}: {e}')
    _COHEN[test_name] = entry
    return entry


def cohen_m2(test_name, sex01, age, values):
    """Cohen m2 personalised range: prediction ± z · residual SD. None if no model."""
    entry = _cohen_model(test_name)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if entry is None or len(x) == 0:
        return None
    import xgboost as xgb
    X = np.array([[float(age), 1.0 if int(sex01) == 1 else 0.0, float(x.mean())]])
    pred = float(entry['bst'].predict(xgb.DMatrix(X))[0])
    return _interval(pred, entry['sigma'], n=int(len(x)))


# ---------------------------------------------------------------------------
# All benchmarks for one history
# ---------------------------------------------------------------------------
def run_benchmarks(test_name, sex01, age, values):
    """{method_key: interval dict} for every benchmark that is defined."""
    out = {'PopRI': pop_ri(test_name, sex01)}
    try:
        out['PerRI'] = per_ri(values)
    except Exception as e:
        out['PerRI'] = {'error': str(e)}
    try:
        out['Gaussian_eb'] = gaussian_eb(test_name, sex01, values)
    except Exception as e:
        out['Gaussian_eb'] = {'error': str(e)}
    try:
        c = cohen_m2(test_name, sex01, age, values)
        if c is not None:
            out['Cohen_m2'] = c
    except Exception as e:
        out['Cohen_m2'] = {'error': str(e)}
    return out


def classify_all(actual_value, test_name, sex01, norma, benchmarks):
    """Classification of the held-out value under every interval.

    Values outside the population interval are abnormal under every method
    (the personalised intervals reclassify values inside it, as in the
    PopRI-normal analyses); each method's own interval decides the rest.
    """
    low, high, _ = REFERENCE_INTERVALS[test_name][sex_str(sex01)]
    pop_class = classify(actual_value, low, high)
    out = {'PopRI': pop_class}
    for key, iv in [('NORMA', norma)] + list(benchmarks.items()):
        if key == 'PopRI' or not iv or 'error' in iv:
            continue
        own = classify(actual_value, iv['ci_lower'], iv['ci_upper'])
        out[key] = pop_class if pop_class != 'Normal' else own
    return out
