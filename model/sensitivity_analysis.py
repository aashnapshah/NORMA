import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import REFERENCE_INTERVALS
from utils import load_checkpoint, create_model
from data import TEST_VOCAB

# Module-level model state (set by init_model or __main__)
MODEL = None
DEVICE = torch.device('cpu')
NSTATES = 3
NORMALIZE = False
IS_QUANTILE = False


def init_model(model=None, device=None, hparams=None, run_id=None, log_dir=None):
    """Initialize module-level model state. Either pass model/device/hparams directly,
    or pass run_id/log_dir to load from checkpoint."""
    global MODEL, DEVICE, NSTATES, NORMALIZE, IS_QUANTILE

    if model is not None:
        MODEL = model
        DEVICE = device or torch.device('cpu')
        MODEL.to(DEVICE)
        MODEL.eval()
        NSTATES = getattr(hparams, 'nstates', 3)
        NORMALIZE = getattr(hparams, 'normalize', False)
        IS_QUANTILE = (getattr(hparams, 'output_mode', 'gaussian') == 'quantile'
                       and getattr(hparams, 'model', '') == 'NORMA2')
    elif run_id is not None:
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'logs')
        DEVICE = device or torch.device('cpu')
        ckpt, hparams = load_checkpoint(log_dir, run_id, best=True, device=DEVICE, quiet=True)
        MODEL = create_model(hparams, ncodes=len(TEST_VOCAB), checkpoint=ckpt).to(DEVICE)
        MODEL.eval()
        NSTATES = getattr(hparams, 'nstates', 3)
        NORMALIZE = getattr(hparams, 'normalize', False)
        IS_QUANTILE = (getattr(hparams, 'output_mode', 'gaussian') == 'quantile'
                       and getattr(hparams, 'model', '') == 'NORMA2')
    else:
        raise ValueError("Provide either (model, device, hparams) or (run_id, log_dir)")

    print(f'Sensitivity model ready  nstates={NSTATES}  normalize={NORMALIZE}  quantile={IS_QUANTILE}')

# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def value_to_state3(value, low, high):
    if value < low:  return 0
    if value > high: return 2
    return 1


def predict(test_name, sex01, age, t_arr, x_arr, t_next, state=1):
    """Run model. Returns (mu, sigma) for Gaussian or (median, ci_width) for quantile."""
    cid     = TEST_VOCAB[test_name]
    sex_str = 'F' if sex01 == 1 else 'M'
    low, high, unit = REFERENCE_INTERVALS[test_name][sex_str]

    t = np.array(t_arr, dtype=np.float32)
    x = np.array(x_arr, dtype=np.float32)

    if NORMALIZE:
        span = high - low
        x    = (x - low) / span

    s_arr = np.array([value_to_state3(v, low, high) for v in
                      (x * (high - low) + low if NORMALIZE else x)], dtype=np.int64)

    x_h = torch.tensor(x).view(1, -1, 1).float()
    s_h = torch.tensor(s_arr).view(1, -1).long()
    t_h = torch.tensor(t).view(1, -1, 1).float()
    sex_t    = torch.tensor([sex01]).long()
    age_t    = torch.tensor([[age]]).float()
    cid_t    = torch.tensor([cid]).long()
    s_next_t = torch.tensor([[state]]).long()
    t_next_t = torch.tensor([[t_next]]).float()

    with torch.no_grad():
        output = MODEL(x_h, s_h, t_h, sex_t, age_t, cid_t, s_next_t, t_next_t, pad_mask=None)

    if IS_QUANTILE:
        # output is (1, 5) — quantiles [q2.5, q25, q50, q75, q97.5]
        q = output.squeeze(0).cpu().numpy()  # (5,)
        median = float(q[2])
        ci_width = float(q[4] - q[0])  # q97.5 - q2.5
        return median, ci_width
    else:
        mu_t, lv_t = output
        mu    = float(mu_t.squeeze())
        sigma = float(torch.exp(0.5 * lv_t).squeeze())
        if NORMALIZE:
            span  = high - low
            mu    = mu * span + low
            sigma = sigma * span
        return mu, sigma

BASELINE_N_HIST  = 10
BASELINE_SPACING = 90       # days between measurements
BASELINE_AGE     = 50
BASELINE_SEX     = 0        # M
BASELINE_HORIZON = 30       # days ahead
N_SAMPLES        = 30       # samples per history_std level

SWEEPS = {
    'age':            np.arange(20, 81, 5),
    'sex':            [0, 1],
    'history_length': [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300],
    'horizon':        [7, 14, 30, 60, 90, 180, 365, 730, 1095, 1825, 2555, 3650],
    'history_std':    np.linspace(0.0, 3.0, 13),  # multiplier of (ref_high-ref_low)/10
}


def make_flat_history(midpoint, n_hist, spacing):
    """Deterministic flat history at the midpoint."""
    t = np.arange(n_hist, dtype=float) * spacing
    x = np.full(n_hist, midpoint, dtype=float)
    return t, x


def make_noisy_history(midpoint, n_hist, spacing, std_abs, rng):
    """One random history drawn from N(midpoint, std_abs)."""
    t = np.arange(n_hist, dtype=float) * spacing
    x = rng.normal(loc=midpoint, scale=std_abs, size=n_hist)
    return t, x


def predict_mu_and_ci(test_name, sex, age, t_hist, x_hist, horizon):
    """Run model, return (mu, ci_width)."""
    t_next = t_hist[-1] + horizon
    mu, ci_or_sigma = predict(test_name, sex, age, t_hist.tolist(), x_hist.tolist(), t_next, state=1)
    if IS_QUANTILE:
        return mu, ci_or_sigma  # predict already returns (median, ci_width)
    else:
        return mu, 1.96 * 2 * ci_or_sigma  # Gaussian: sigma → 95% CI width


EXCLUDE_LABS = {'LDH', 'GGT', 'CRP', 'PT'}

def run_sweeps():
    """Run all sweeps across all labs. Returns a DataFrame."""
    all_test_names = sorted([k for k in REFERENCE_INTERVALS.keys() if k not in EXCLUDE_LABS])

    records = []
    for test_name in tqdm(all_test_names, desc='Labs'):
        sex_str = 'F' if BASELINE_SEX == 1 else 'M'
        low, high, _ = REFERENCE_INTERVALS[test_name][sex_str]
        midpoint = (low + high) / 2.0
        ref_span = high - low

        # Baseline: flat history, all defaults
        t0, x0 = make_flat_history(midpoint, BASELINE_N_HIST, BASELINE_SPACING)
        try:
            baseline_mu, baseline_ci = predict_mu_and_ci(test_name, BASELINE_SEX, BASELINE_AGE, t0, x0, BASELINE_HORIZON)
        except Exception:
            continue

        baseline_ci_norm = baseline_ci / ref_span * 100  # as % of reference range

        for feature, vals in SWEEPS.items():

            if feature == 'history_std':
                for v in vals:
                    std_abs = v * ref_span / 10.0
                    mus, widths = [], []
                    for i in range(N_SAMPLES):
                        rng = np.random.default_rng(i)
                        if std_abs > 0:
                            t_h, x_h = make_noisy_history(midpoint, BASELINE_N_HIST, BASELINE_SPACING, std_abs, rng)
                        else:
                            t_h, x_h = make_flat_history(midpoint, BASELINE_N_HIST, BASELINE_SPACING)
                        try:
                            m, w = predict_mu_and_ci(test_name, BASELINE_SEX, BASELINE_AGE, t_h, x_h, BASELINE_HORIZON)
                            mus.append(m)
                            widths.append(w)
                        except Exception:
                            pass
                    if len(widths) < 2:
                        continue
                    ci_arr = np.array(widths)
                    mu_arr = np.array(mus)
                    ci_norm = ci_arr / ref_span * 100
                    records.append({
                        'test_name': test_name, 'feature': feature, 'value': v,
                        'ref_span': ref_span,
                        'ci_mean': ci_arr.mean(), 'ci_std': ci_arr.std(),
                        'ci_lo': np.percentile(ci_arr, 2.5), 'ci_hi': np.percentile(ci_arr, 97.5),
                        'ci_norm_mean': ci_norm.mean(),
                        'ci_norm_lo': np.percentile(ci_norm, 2.5),
                        'ci_norm_hi': np.percentile(ci_norm, 97.5),
                        'baseline_ci': baseline_ci,
                        'baseline_ci_norm': baseline_ci_norm,
                        'pct_change': (ci_arr.mean() - baseline_ci) / baseline_ci * 100 if baseline_ci > 1e-10 else 0.0,
                        'mu_mean': mu_arr.mean(), 'mu_std': mu_arr.std(),
                        'mu_lo': np.percentile(mu_arr, 2.5), 'mu_hi': np.percentile(mu_arr, 97.5),
                        'midpoint': midpoint,
                        'mu_pct_dev': (mu_arr.mean() - midpoint) / midpoint * 100,
                    })
            else:
                for v in vals:
                    age, sex, n_hist, horizon = BASELINE_AGE, BASELINE_SEX, BASELINE_N_HIST, BASELINE_HORIZON
                    if   feature == 'age':            age = v
                    elif feature == 'sex':            sex = int(v)
                    elif feature == 'history_length': n_hist = int(v)
                    elif feature == 'horizon':        horizon = v

                    t_h, x_h = make_flat_history(midpoint, n_hist, BASELINE_SPACING)
                    try:
                        mu_val, ci_w = predict_mu_and_ci(test_name, sex, age, t_h, x_h, horizon)
                    except Exception:
                        continue
                    ci_norm_val = ci_w / ref_span * 100
                    records.append({
                        'test_name': test_name, 'feature': feature, 'value': v,
                        'ref_span': ref_span,
                        'ci_mean': ci_w, 'ci_std': 0.0,
                        'ci_lo': ci_w, 'ci_hi': ci_w,
                        'ci_norm_mean': ci_norm_val,
                        'ci_norm_lo': ci_norm_val,
                        'ci_norm_hi': ci_norm_val,
                        'baseline_ci': baseline_ci,
                        'baseline_ci_norm': baseline_ci_norm,
                        'pct_change': (ci_w - baseline_ci) / baseline_ci * 100 if baseline_ci > 1e-10 else 0.0,
                        'mu_mean': mu_val, 'mu_std': 0.0,
                        'mu_lo': mu_val, 'mu_hi': mu_val,
                        'midpoint': midpoint,
                        'mu_pct_dev': (mu_val - midpoint) / midpoint * 100,
                    })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_sweep_curves(results_df, all_test_names, save_dir='../figures'):
    """Per-feature sweep curves: 2 rows (CI width change, mu deviation from midpoint)."""
    sns.set_context('paper', font_scale=1.3)
    sns.set_style('white')

    features = list(SWEEPS.keys())
    n_feat = len(features)
    labs = all_test_names

    palette = sns.color_palette('tab20', n_colors=len(labs))
    lab_cmap = {lab: palette[i] for i, lab in enumerate(labs)}

    fig, axes = plt.subplots(2, n_feat, figsize=(4.5 * n_feat, 7), sharey='row')

    # Row 0: CI width % change
    for ax, feature in zip(axes[0], features):
        sub = results_df[results_df['feature'] == feature]
        for lab in labs:
            d = sub[sub['test_name'] == lab].sort_values('value')
            if d.empty:
                continue
            ax.plot(d['value'], d['pct_change'], 'o-', color=lab_cmap[lab],
                    lw=1.5, ms=3, alpha=0.7)
            if feature == 'history_std':
                ax.fill_between(d['value'], d['pct_lo'], d['pct_hi'],
                                color=lab_cmap[lab], alpha=0.1)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        if ax == axes[0][0]:
            ax.set_ylabel('CI width change (%)')

    # Row 1: mu % deviation from midpoint
    for ax, feature in zip(axes[1], features):
        sub = results_df[results_df['feature'] == feature]
        for lab in labs:
            d = sub[sub['test_name'] == lab].sort_values('value')
            if d.empty:
                continue
            ax.plot(d['value'], d['mu_pct_dev'], 'o-', color=lab_cmap[lab],
                    lw=1.5, ms=3, alpha=0.7)
            if feature == 'history_std' and 'mu_lo' in d.columns:
                mu_lo_pct = (d['mu_lo'] - d['midpoint']) / d['midpoint'] * 100
                mu_hi_pct = (d['mu_hi'] - d['midpoint']) / d['midpoint'] * 100
                ax.fill_between(d['value'], mu_lo_pct, mu_hi_pct,
                                color=lab_cmap[lab], alpha=0.1)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel(feature.replace('_', ' ').title())
        if ax == axes[1][0]:
            ax.set_ylabel('Predicted mu deviation\nfrom midpoint (%)')

    handles = [plt.Line2D([0], [0], color=lab_cmap[l], lw=2, label=l) for l in labs]
    fig.legend(handles=handles, title='Lab', bbox_to_anchor=(1.01, 0.5),
               loc='center left', frameon=False, fontsize=9, title_fontsize=10)
    sns.despine(fig)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(os.path.join(save_dir, 'sensitivity_sweep_curves.png'), dpi=200, bbox_inches='tight')
    plt.show()
    return lab_cmap


def plot_effect_sizes(results_df, all_test_names, lab_cmap, save_dir='../figures'):
    """Effect-size summary: range of % change per (lab, feature)."""
    sns.set_context('paper', font_scale=1.3)
    sns.set_style('white')

    features = list(SWEEPS.keys())
    n_feat = len(features)

    effect_df = (results_df
        .groupby(['test_name', 'feature'])
        .agg(pct_min=('pct_change', 'min'),
             pct_max=('pct_change', 'max'),
             pct_range=('pct_change', lambda x: x.max() - x.min()))
        .reset_index())

    fig, axes = plt.subplots(1, n_feat, figsize=(4.5 * n_feat, 5), sharey=True)

    for ax, feature in zip(axes, features):
        sub = effect_df[effect_df['feature'] == feature].sort_values('pct_range', ascending=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.barh(i, row['pct_max'] - row['pct_min'], left=row['pct_min'],
                   linewidth=0.5,
                    height=0.7, alpha=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['test_name'])
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('CI width change (%)')
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')

    fig.suptitle('Sensitivity of 95% CI Width to Each Feature (% change from baseline)',
                 fontsize=13, fontweight='bold', y=1.02)
    sns.despine(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_effect_sizes.png'), dpi=200, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

NORMA_RUNS = {
    '334f7e21': 'NORMA-Quantile',
    '167f05e8': 'NORMA-Gaussian',
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=list(NORMA_RUNS.keys()))
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(__file__), '..', 'results', 'prediction', 'raw'))
    args = parser.parse_args()

    LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for run_id in args.runs:
        label = NORMA_RUNS.get(run_id, run_id)
        print(f'\n{"="*60}')
        print(f'Running sensitivity for {label} ({run_id})')
        print(f'{"="*60}')

        init_model(run_id=run_id, log_dir=LOG_DIR)
        df = run_sweeps()
        df.insert(0, 'model', label)

        print(f'{len(df)} rows | {df.test_name.nunique()} labs | '
              f'{df.feature.nunique()} features')
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(args.output_dir, 'sensitivity.csv')
    combined.to_csv(out_path, index=False)
    print(f'\nSaved {len(combined)} rows to {out_path}')
