"""Priors over the future state, and mixing the state-conditional predictions.

NORMA is a conditional model p(x_next | H, s_next). Comparing it with baselines
that only see the history H needs p(x_next | H), which takes two pieces -- a
prior over s_next that uses nothing the baselines do not also see, and the
machinery to marginalise over it:

    p(x | H) = sum_s p(s | H) p(x | H, s)

p(s_next | .), estimated from the *training* split only, with additive
(Laplace) smoothing of one count per cell:

  marginal    p(s_next | analyte)
  transition  p(s_next | s_last, analyte), s_last = state of the most recent
              history value (derived from the population reference interval,
              so it is part of H for every method)

The mixing, given those weights:

  Gaussian head  each state gives (mu_s, sigma_s). The mixture mean and
                 variance are exact; quantiles come from the mixture CDF (a sum
                 of normal CDFs) inverted by bisection.
  Quantile head  each state gives five quantiles. Each becomes a
                 piecewise-linear CDF with linear tails (extrapolated at the
                 slope of the outermost segment), the CDFs are mixed, and the
                 mixture CDF is inverted on the union of knots. The mixture
                 median is therefore NOT the weighted average of the state
                 medians.

Was state_prior.py and state_mixture.py; they were only ever imported together,
by model/inference.py and scripts/04_refs.py. Not to be confused with
priors.py, which builds the state-conditional prior over *values* that the
prior-anchored heads and losses read.

Usage:
    python states.py               # fit and write state_priors_combined.json
    python states.py --selftest    # check the mixing math
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import norm

import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path

STATE_NAMES = ['low', 'normal', 'high']
QUANTILE_LEVELS = np.array([0.025, 0.25, 0.50, 0.75, 0.975])
QUANTILE_COLS = ['q025', 'q25', 'q50', 'q75', 'q975']

# Weights that produced each run's published predictions_combined.csv.  train.py
# writes that file from the in-memory model after the last epoch (checkpoint_latest),
# but 167f05e8's file was regenerated later from checkpoint_best; verified by
# re-scoring the test split against the published file (2026-08-26).
PUBLISHED_CHECKPOINT = {'334f7e21': 'latest', '167f05e8': 'best', 'q_age_set': 'latest'}
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'predictions', 'state_priors_combined.json')


# ═══════════════════════════════════════════════════════════════════════════
# p(s_next | .) -- the leak-free state prior
# ═══════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════
# Mixing the state-conditional predictions into p(x_next | H)
# ═════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------- Gaussian head

def mix_gaussian(mu, log_var, w, levels=QUANTILE_LEVELS, n_iter=60):
    """mu, log_var: (N, S); w: (N, S) rows summing to 1.

    Returns dict with mixture 'mu', 'log_var' and one entry per quantile level."""
    sd = np.exp(0.5 * log_var)
    m = (w * mu).sum(1)
    v = (w * (sd ** 2 + mu ** 2)).sum(1) - m ** 2
    v = np.clip(v, 1e-12, None)

    # invert the mixture CDF by bisection on a bracket covering all components
    lo = (mu - 8 * sd).min(1)
    hi = (mu + 8 * sd).max(1)
    out = {'mu': m, 'log_var': np.log(v)}
    for lev, col in zip(levels, QUANTILE_COLS):
        a, b = lo.copy(), hi.copy()
        for _ in range(n_iter):
            mid = 0.5 * (a + b)
            cdf = (w * norm.cdf((mid[:, None] - mu) / sd)).sum(1)
            go_right = cdf < lev
            a = np.where(go_right, mid, a)
            b = np.where(go_right, b, mid)
        out[col] = 0.5 * (a + b)
    return out


# ---------------------------------------------------------------- quantile head

def _pl_knots(q, levels=QUANTILE_LEVELS):
    """Piecewise-linear CDF knots for quantile arrays q: (N, S, K).

    Returns x (N, S, K+2) and F (K+2,) with F[0]=0, F[-1]=1; the outer knots are
    extrapolated at the slope of the outermost observed segment."""
    q = np.sort(q, axis=-1)
    # enforce strict monotonicity so slopes are finite
    eps = 1e-6 * np.maximum(np.abs(q).max(-1), 1.0)
    for k in range(1, q.shape[-1]):
        q[..., k] = np.maximum(q[..., k], q[..., k - 1] + eps)
    lo_slope = (levels[1] - levels[0]) / (q[..., 1] - q[..., 0])       # dF/dx
    hi_slope = (levels[-1] - levels[-2]) / (q[..., -1] - q[..., -2])
    x0 = q[..., 0] - levels[0] / lo_slope
    x1 = q[..., -1] + (1 - levels[-1]) / hi_slope
    x = np.concatenate([x0[..., None], q, x1[..., None]], axis=-1)
    F = np.concatenate([[0.0], levels, [1.0]])
    return x, F


def _pl_cdf(x_eval, knots, F):
    """Evaluate piecewise-linear CDFs. x_eval: (N, M); knots: (N, S, K); F: (K,).
    Returns (N, S, M)."""
    N, S, K = knots.shape
    out = np.empty((N, S, x_eval.shape[1]))
    for i in range(N):
        for s in range(S):
            out[i, s] = np.interp(x_eval[i], knots[i, s], F)
    return out


def mix_quantiles(q, w, levels=QUANTILE_LEVELS):
    """q: (N, S, 5) state-conditional quantiles; w: (N, S).

    Returns dict with the five mixture quantiles plus 'mu' (= mixture median)
    and 'log_var' (from the 95% width, as predict.py does)."""
    N, S, K = q.shape
    knots, F = _pl_knots(q.astype(float), levels)
    # union of all knots per row; the mixture CDF is piecewise-linear between them
    grid = np.sort(knots.reshape(N, -1), axis=1)                       # (N, S*(K+2))
    cdf_s = _pl_cdf(grid, knots, F)                                    # (N, S, M)
    cdf = (w[:, :, None] * cdf_s).sum(1)                               # (N, M)
    out = {}
    for lev, col in zip(levels, QUANTILE_COLS):
        out[col] = np.array([np.interp(lev, cdf[i], grid[i]) for i in range(N)])
    out['mu'] = out['q50']
    sigma = (out['q975'] - out['q025']) / 3.92
    out['log_var'] = 2.0 * np.log(sigma + 1e-8)
    return out


def mix_quantiles_batched(q, w, batch=20000):
    parts = [mix_quantiles(q[i:i + batch], w[i:i + batch]) for i in range(0, len(q), batch)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mix_gaussian_batched(mu, log_var, w, batch=50000):
    parts = [mix_gaussian(mu[i:i + batch], log_var[i:i + batch], w[i:i + batch]) for i in range(0, len(mu), batch)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

# ═══════════════════════════════════════════════════════════════════════════
# Self-test for the mixing math
# ═══════════════════════════════════════════════════════════════════════════

def selftest():
        # sanity checks
        rng = np.random.default_rng(0)
        # 1. one-hot weights recover the component exactly
        mu = rng.normal(size=(5, 3)); lv = rng.normal(size=(5, 3)) * 0.3
        w = np.eye(3)[rng.integers(0, 3, 5)]
        g = mix_gaussian(mu, lv, w)
        assert np.allclose(g['mu'], (w * mu).sum(1)) and np.allclose(g['log_var'], (w * lv).sum(1), atol=1e-6)
        assert np.allclose(g['q50'], (w * mu).sum(1), atol=1e-6)
        # 2. bimodal mixture median lies between the components, not at either
        mu = np.array([[0.0, 10.0, 20.0]]); lv = np.zeros((1, 3)); w = np.array([[0.5, 0.0, 0.5]])
        g = mix_gaussian(mu, lv, w)
        assert 3 < g['q50'][0] < 17 and abs(g['mu'][0] - 10.0) < 1e-9   # CDF is flat at 0.5 across the gap
        # 3. quantile-head: one-hot recovers component quantiles; 50/50 split gives median between
        qs = np.array([[[-2, -0.7, 0, 0.7, 2], [8, 9.3, 10, 10.7, 12], [18, 19.3, 20, 20.7, 22]]], dtype=float)
        r = mix_quantiles(qs, np.array([[0, 1, 0]]))
        assert np.allclose([r[c][0] for c in QUANTILE_COLS], qs[0, 1], atol=1e-6), r
        r = mix_quantiles(qs, np.array([[0.5, 0, 0.5]]))
        assert 2 < r['q50'][0] < 18 and r['q025'][0] < -0.7 and r['q975'][0] > 20.7, r
        # 4. quantile-head mixture agrees with Gaussian mixture when components are Gaussian quantiles
        mu = np.array([[1.0, 3.0, 4.0]]); sd = np.array([[0.5, 1.0, 0.7]]); w = np.array([[0.2, 0.5, 0.3]])
        qs = mu[..., None] + sd[..., None] * norm.ppf(QUANTILE_LEVELS)
        rq = mix_quantiles(qs, w); rg = mix_gaussian(mu, 2 * np.log(sd), w)
        print('gaussian vs pl-quantile mixture medians:', rg['q50'][0], rq['q50'][0])
        assert abs(rq['q50'][0] - rg['q50'][0]) < 0.1
        print('all mixture checks passed')


if __name__ == '__main__':
    from data import load_and_split_data, CODE_TO_TEST_NAME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--selftest', action='store_true',
                        help='check the mixing math instead of fitting priors')
    parser.add_argument('--data_dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed'))
    parser.add_argument('--source', default='combined')
    parser.add_argument('--nstates', type=int, default=3)
    parser.add_argument('--out', default=DEFAULT_PATH)
    args = parser.parse_args()

    if args.selftest:
        selftest()
        raise SystemExit(0)
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
