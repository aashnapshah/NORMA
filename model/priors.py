"""State-conditional population prior for the prior-aware NORMA heads and losses."""
import numpy as np
import torch
from scipy.stats import norm

import bootstrap  # noqa: F401  -- puts the pipeline's import roots on sys.path
from process.config import REFERENCE_INTERVALS
from data import TEST_VOCAB

QUANTILES = (0.025, 0.25, 0.50, 0.75, 0.975)
MIN_N = 50


def _gauss_q(m, sd):
    return np.array([m + norm.ppf(q) * sd for q in QUANTILES], dtype=np.float32)


def build_prior_table(train_seq, nstates=3, ncodes=None):
    if nstates != 3:
        raise ValueError('prior table assumes the 3-state coding (0 low, 1 normal, 2 high)')
    ncodes = ncodes or len(TEST_VOCAB)
    S = nstates
    q = np.zeros((ncodes, S, 2, len(QUANTILES)), dtype=np.float32)
    mu = np.zeros((ncodes, S, 2), dtype=np.float32)
    var = np.ones((ncodes, S, 2), dtype=np.float32)
    rho = np.full(ncodes, 0.5, dtype=np.float32)

    # empirical next values by (code, state, sex) and within/total variance by code
    vals = {}
    within, total = {}, {}
    for seq in train_seq:
        cid = int(seq['cid'])
        sex = 1 if (seq['sex'] == 'F' or seq['sex'] == 1) else 0
        x = np.asarray(seq['x'], dtype=np.float64)
        s3 = int(np.asarray(seq['s3'])[-1]) + 1
        vals.setdefault((cid, s3, sex), []).append(float(x[-1]))
        if len(x) >= 2:
            within.setdefault(cid, []).append(float(x.var(ddof=1)))
        total.setdefault(cid, []).extend(x.tolist())

    code_of = {i: name for name, i in TEST_VOCAB.items()}
    for cid in range(ncodes):
        name = code_of.get(cid)
        if name is None or name not in REFERENCE_INTERVALS:
            continue
        for sex, key in ((0, 'M'), (1, 'F')):
            lo, hi, _ = REFERENCE_INTERVALS[name][key]
            lo, hi = float(lo), float(hi)
            m, sd = 0.5 * (lo + hi), (hi - lo) / 3.92
            # normal: the reference interval itself
            q[cid, 1, sex] = _gauss_q(m, sd); mu[cid, 1, sex] = m; var[cid, 1, sex] = sd ** 2
            # abnormal states: empirical, else a Gaussian beyond the bound
            for st, fallback_m in ((0, lo - sd), (2, hi + sd)):
                v = vals.get((cid, st, sex), [])
                if len(v) < MIN_N:
                    v = vals.get((cid, st, 0), []) + vals.get((cid, st, 1), [])
                if len(v) >= MIN_N:
                    v = np.asarray(v)
                    q[cid, st, sex] = np.quantile(v, QUANTILES).astype(np.float32)
                    mu[cid, st, sex] = v.mean(); var[cid, st, sex] = max(v.var(), 1e-6)
                else:
                    q[cid, st, sex] = _gauss_q(fallback_m, sd)
                    mu[cid, st, sex] = fallback_m; var[cid, st, sex] = sd ** 2
        if cid in within and len(total.get(cid, [])) > 1:
            tv = float(np.var(total[cid]))
            if tv > 0:
                rho[cid] = float(np.clip(np.mean(within[cid]) / tv, 0.05, 0.95))

    return {'q': torch.from_numpy(q), 'mu': torch.from_numpy(mu),
            'var': torch.from_numpy(var), 'rho': torch.from_numpy(rho)}


def save_prior_table(table, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in table.items()}, path)
