"""
Leak-free priors over the future population-defined state, p(s_next | .).

NORMA is a conditional model p(x_next | H, s_next).  To compare it with
baselines that only see the history H, we need p(x_next | H), which requires
a prior over s_next that uses nothing the baselines do not also see.  Two
priors are estimated from the *training* split only:

  marginal    p(s_next | analyte)
  transition  p(s_next | s_last, analyte), s_last = state of the most recent
              history value (derived from the population reference interval,
              so it is part of H for every method)

Both use additive (Laplace) smoothing of 1 count per cell.

Usage:
    python state_prior.py                      # fits and writes state_priors_combined.json
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'process'))

STATE_NAMES = ['low', 'normal', 'high']

# Weights that produced each run's published predictions_combined.csv.  train.py
# writes that file from the in-memory model after the last epoch (checkpoint_latest),
# but 167f05e8's file was regenerated later from checkpoint_best; verified by
# re-scoring the test split against the published file (2026-08-26).
PUBLISHED_CHECKPOINT = {'334f7e21': 'latest', '167f05e8': 'best', 'q_age_set': 'latest'}
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions', 'state_priors_combined.json')


def _states(seq, nstates):
    s = np.asarray(seq['s'] if nstates == 2 else seq['s3'], dtype=np.int64)
    return s + 1 if nstates == 3 else s


def fit_state_priors(train_seq, nstates=3, alpha=1.0):
    """Count (analyte, s_last, s_next) over training sequences and normalise."""
    counts = {}
    for seq in train_seq:
        s = _states(seq, nstates)
        if len(s) < 2:
            continue
        c = counts.setdefault(int(seq['cid']), np.zeros((nstates, nstates)))
        c[int(s[-2]), int(s[-1])] += 1
    priors = {}
    for cid, c in counts.items():
        marg = c.sum(axis=0) + alpha
        trans = c + alpha
        priors[cid] = {
            'n': int(c.sum()),
            'marginal': (marg / marg.sum()).tolist(),
            'transition': (trans / trans.sum(axis=1, keepdims=True)).tolist(),
        }
    # pooled fallback for analytes absent from the training split
    pooled = sum(counts.values()) + alpha
    priors['_pooled'] = {
        'n': int(sum(c.sum() for c in counts.values())),
        'marginal': (pooled.sum(axis=0) / pooled.sum()).tolist(),
        'transition': (pooled / pooled.sum(axis=1, keepdims=True)).tolist(),
    }
    return priors


def save_state_priors(priors, path=DEFAULT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in priors.items()}, f, indent=1)


def load_state_priors(path=DEFAULT_PATH):
    with open(path) as f:
        raw = json.load(f)
    return {(k if k.startswith('_') else int(k)): v for k, v in raw.items()}


def prior_weights(priors, cid, s_last, kind='transition'):
    """Vectorised p(s_next) for arrays cid (N,) and s_last (N,). Returns (N, nstates)."""
    cid = np.asarray(cid).astype(int)
    s_last = np.asarray(s_last).astype(int)
    out = np.empty((len(cid), len(priors['_pooled']['marginal'])))
    for i, (c, sl) in enumerate(zip(cid, s_last)):
        p = priors.get(c, priors['_pooled'])
        out[i] = p['transition'][sl] if kind == 'transition' else p['marginal']
    return out


if __name__ == '__main__':
    from data import load_and_split_data, CODE_TO_TEST_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed'))
    parser.add_argument('--source', default='combined')
    parser.add_argument('--nstates', type=int, default=3)
    parser.add_argument('--out', default=DEFAULT_PATH)
    args = parser.parse_args()

    train_seq, _, _ = load_and_split_data(args.data_dir, args.source, print_info=False, nstates=args.nstates)
    priors = fit_state_priors(train_seq, nstates=args.nstates)
    save_state_priors(priors, args.out)
    print(f'wrote {args.out} ({len(priors) - 1} analytes, n={priors["_pooled"]["n"]:,})')
    print('\npooled marginal  ', np.round(priors['_pooled']['marginal'], 3))
    print('pooled transition (rows = s_last, cols = s_next)')
    print(np.round(priors['_pooled']['transition'], 3))
    print('\nper-analyte P(stay in state):')
    for cid in sorted(k for k in priors if not isinstance(k, str)):
        t = np.array(priors[cid]['transition'])
        print(f"  {CODE_TO_TEST_NAME.get(cid, cid):<6s} n={priors[cid]['n']:>8,}  "
              f"marginal={np.round(priors[cid]['marginal'], 2)}  stay={np.round(np.diag(t), 2)}")
