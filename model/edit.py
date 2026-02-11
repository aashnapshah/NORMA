
import pandas as pd
import numpy as np
import os
import sys
import argparse
import torch
from math import sqrt, pi
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/home/aas926/miniconda3/envs/normal/bin/ffmpeg'

sys.path.append('../../NORMA/')
from process.config import REFERENCE_INTERVALS 
from ARIMA.setpoints import *
from helpers.plots import *
from data import *
from utils import *
from model import *

DATA_DIR = '../../NORMA/data/processed/'
LOG_DIR = '../../NORMA/model/logs/'
DATA_PATH = 'EHRSHOT_processed_df.csv'
FILE_PATH = os.path.join(DATA_DIR, DATA_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_seq(df, pid=None, data_dir=DATA_DIR, csv_name=DATA_PATH):
    if pid is None:
        pid = df['subject_id'].sample(1).values[0]
    return df.query('subject_id == @pid')

def pred(model, seq, nstates, normal=False):
    batch = TimeSeriesDataset([seq], nstates)[0]
    batch = collate_fn([batch])
    batch = to_device_batch(batch, device)
    if normal:
        batch['s_next'] = torch.ones_like(batch['s_next'])
    else:
        batch['s_next'] = torch.zeros_like(batch['s_next'])
        
    model.eval()
    with torch.no_grad():
        mu, log_var = forward_model(model, batch)
        mu = float(mu.cpu().numpy().squeeze())
        log_var = float(log_var.cpu().numpy().squeeze())
        var = np.exp(log_var)
        std = np.sqrt(var)
    return mu, std

def predict_cf(model, device, train_loader, val_loader, test_loader):
    all_predictions = []
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"Evaluating {split_name} set...")
        predictions_df = get_predictions_cf(model, device, loader, split_name)
        predictions_df['split'] = split_name
        all_predictions.append(predictions_df)
    return pd.concat(all_predictions)

def get_predictions_cf(model, device, loader, split_name):
    model.eval()
    model.to(device)

    all_records = []

    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(loader, desc=f"{split_name} (CF predict)", leave=False):
            batch = to_device_batch(batch, device)
            pids = batch['pids']  
            ones_snext = torch.ones_like(batch['s_next'])
            zeros_snext = torch.zeros_like(batch['s_next'])

            out = {}
            for state, snext in zip([True, False], [ones_snext, zeros_snext]):
                mu, log_var = forward_model(model, batch, s_next=snext)
                mu = mu.detach()
                log_var = log_var.detach()
                std = torch.exp(0.5 * log_var)  # numerically stable std

                # grab fields needed for row, move to cpu only once
                cid = batch['cid'].detach().cpu().numpy()
                x_next = batch['x_next'].detach().cpu().numpy()
                t_next = batch['t_next'].detach().cpu().numpy()
                s_next = batch['s_next'].detach().cpu().numpy()
                mu_np = mu.detach().cpu().numpy()
                std_np = std.detach().cpu().numpy()

                for i in range(len(mu_np)):
                    cid_val = int(cid[i].item() if hasattr(cid[i], 'item') else cid[i])
                    x_next_val = float(x_next[i].item() if hasattr(x_next[i], 'item') else x_next[i])
                    t_next_val = float(t_next[i].item() if hasattr(t_next[i], 'item') else t_next[i])
                    s_next_val = int(s_next[i].item() if hasattr(s_next[i], 'item') else s_next[i])
                    mu_val = float(mu_np[i].item() if hasattr(mu_np[i], 'item') else mu_np[i])
                    std_val = float(std_np[i].item() if hasattr(std_np[i], 'item') else std_np[i])

                    all_records.append({
                        'pid': pids[i],
                        'cid': cid_val,
                        'code': CODE_TO_TEST_NAME[cid_val],
                        'x_next': x_next_val,
                        't_next': t_next_val,
                        's_next': s_next_val,
                        'state': state,
                        'mu': mu_val,
                        'std': std_val,
                    })

    return pd.DataFrame(all_records)


    
def pop_interval(code, sex):
    sex = 'F' if sex == 1 else 'M'
    low, high, _ = REFERENCE_INTERVALS[code][sex]
    std = (high - low) / 4
    mu = (low + high) / 2
    return mu, std

def _row(run_id, state, source, code, mu, std):
    return {
        "run_id": run_id,
        "state": state,
        "source": source,
        "pid": pid,
        "code": code,
        "mu": float(mu),
        "std": float(std),
    }

def _build_models(LOG_DIR, run_ids, device):
    out = []
    for run_id in run_ids:
        ckpt, hparams = load_checkpoint(LOG_DIR, run_id, best=False, device=device, quiet=True)
        model = create_model(hparams, 34, ckpt).to(device)
        out.append({"run_id": run_id, "source": hparams.train.upper(), "model": model})
    return out

def predict_intervals(seqs, run_ids, LOG_DIR, device, add_baselines=True):
    if isinstance(seqs, pd.DataFrame):
        seqs = create_sequences(seqs, quiet=True)

    models = _build_models(LOG_DIR, run_ids, device)
    data_records = []

    for seq in seqs:
        base_row = {
            'pid': seq['pid'],
            'cid': seq['cid'],
            'sex': seq['sex'],
            'age': seq['age'],
            'state': True,
        }

        for m in models:
            for state in [True, False]:
                row = base_row.copy()
                mu, std = pred(m["model"], seq, normal=state)
                row.update({
                    'run_id': m["run_id"],
                    'state': state,
                    'mu': mu,
                    'std': std,
                })
                data_records.append(row)

        if add_baselines:
            row = base_row.copy()
            mu, var = calculate_guassians(seq)  
            std = np.sqrt(var)*2
            row.update({
                'run_id': 'Personalized',
                'mu': mu,
                'std': std,
            })
            data_records.append(row)

            row = base_row.copy()
            mu, std = pop_interval(seq['test_name'], seq['sex'])
            row.update({
                'run_id': 'Population',
                'mu': mu,
                'std': std,
            })
            data_records.append(row)

    return pd.DataFrame(data_records)

def load_counterfactual_predictions(run_ids, log_dir, split='test'):
    results = []
    for id in run_ids:
        counterfactual_path = os.path.join(log_dir, id, "counterfactual_predictions_combined.csv") 
        predictions_path = os.path.join(log_dir, id, "predictions_combined.csv") 
        path = counterfactual_path if os.path.exists(counterfactual_path) else predictions_path
        state_col = 'state' if 'state' in pd.read_csv(path).columns else 's_next'

        subresults = pd.read_csv(path).query(f"split == '{split}'")
        if 'log_var' not in subresults.columns:
            subresults['std_pred'] = np.exp(subresults['std'])
            subresults['var_pred'] = subresults['std']**2
            subresults['log_var'] = np.log(subresults['var_pred'])
            
        subresults['var_pred'] = np.exp(subresults['log_var']).fillna(0)
        subresults['std_pred'] = np.sqrt(subresults['var_pred'])
        subresults['code'] = subresults['code'].fillna('NA') 
        subresults['Run ID'] = id
        results.append(subresults)
    return pd.concat(results), state_col

def _gaussian_pdf(y, mu, sigma):
    s = max(float(sigma), 1e-6)
    z = (y - float(mu)) / s
    return np.exp(-0.5 * z * z) / (s * sqrt(2 * pi))

def plot_sequence_and_intervals(code, seq, df):
    y = seq["x"]
    x = seq["t"]

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 1.2), width_ratios=[3.3, 1], dpi=300, gridspec_kw={'wspace': 0.02}
    )
    ax, ax_density = axes
    ax.plot(x, y, marker="o", color="black", linewidth=1.1, label=str(code))
    
    sub = df[df["code"] == code]

    for i, (_, r) in enumerate(sub.iterrows()):
        mu, std = float(r["mu"]), float(r["std"])
        if r["run_id"] == "Population":
            ax.hlines(mu - std, x.min(), x.max(), linestyles=(0, (2, 2)), color=scheme_colors["Population"], linewidth=1.1, label="Population")
            ax.hlines(mu + std, x.min(), x.max(), linestyles=(0, (2, 2)), color=scheme_colors["Population"], linewidth=1.1)
        elif r['run_id'] == "Personalized" and len(y) > 4:
            label = 'Personalized'
           # ax.axhspan(mu - std, mu + std, alpha=0.18, color=scheme_colors["Personalized"], label=label)
        else:
            if r['state'] == True:
                label = 'Normal'
                ax.axhspan(mu - std, mu + std, alpha=0.18, color=scheme_colors["Hybrid"], label=label)
            if r['state'] == False:
                label = 'Abnormal'
                ax.axhspan(mu - std, mu + std, alpha=0.18, color=scheme_colors["Personalized"], label=label)
                #ax.axhline(mu, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel("Days")
    ax.set_ylabel(f"{code}")
    ax.set_xlim(x.min(), x.max())

    lows = [y.min()] + list((sub["mu"] - 2*sub["std"]).to_numpy())
    highs = [y.max()] + list((sub["mu"] + 2*sub["std"]).to_numpy())
    y_lo, y_hi = float(np.min(lows)), float(np.max(highs))
    y_grid = np.linspace(y_lo, y_hi, 500)

    max_peak = 0.0
    pdfs = []
    max_pdf_y_lo, max_pdf_y_hi = y_lo, y_hi  # For expanding y-limits as needed
    for _, r in sub.iterrows():
        mu, std = float(r["mu"]), float(r["std"])
        pdf = _gaussian_pdf(y_grid, mu, std)
        pdfs.append(pdf)
        max_peak = max(max_peak, pdf.max())
        # Find nonzero pdf span for all PDFs, to help y-lim determination
        nonzero = y_grid[pdf > (pdf.max() * 1e-4)]
        if len(nonzero) > 0:
            max_pdf_y_lo = min(max_pdf_y_lo, nonzero[0])
            max_pdf_y_hi = max(max_pdf_y_hi, nonzero[-1])

    # Expand y-axis limits for both axes so PDFs fit fully
    pad_lower = (max_pdf_y_lo - y_lo) * 0.2 if max_pdf_y_lo > y_lo else 0
    pad_upper = (y_hi - max_pdf_y_hi) * 0.2 if max_pdf_y_hi < y_hi else 0
    plot_y_lo = min(y_lo, max_pdf_y_lo - pad_lower)
    plot_y_hi = max(y_hi, max_pdf_y_hi + pad_upper)

    ax.set_ylim(plot_y_lo, plot_y_hi)
    ax_density.set_ylim(plot_y_lo, plot_y_hi)
    ax_density.yaxis.set_visible(False)
    ax_density.xaxis.set_visible(False)
    ax_density.set_frame_on(False)

    total_width = 0.9
    num_pdfs = len(pdfs)
    density_offset = 0.0
    total_span = total_width * (num_pdfs - 1)
    left_starts = np.linspace(0, total_span, num_pdfs)
  
    height_scale = 1.22
    y_mid = (plot_y_lo + plot_y_hi) / 2
    prev_right = 0.0
    for i, ((_, r), pdf) in enumerate(zip(sub.iterrows(), pdfs)):
        if r["run_id"] == "Population" or r["run_id"] == "Personalized":
            continue
        if r['state'] == True:
            label = 'Normal'
            color = scheme_colors["Hybrid"]
        else:
            label = 'Abnormal'
            color = scheme_colors["Personalized"]
        scaled = (pdf / max_peak) * total_width
        scaled_y_grid = (y_grid - y_mid) * height_scale + y_mid
        left = prev_right + 0.1
        #color = scheme_colors[r["run_id"]] if r["run_id"] in scheme_colors else scheme_colors['Hybrid']
        ax_density.fill_betweenx(scaled_y_grid, left, left + scaled, alpha=0.30, color=color)
        ax_density.plot(left + scaled, scaled_y_grid, linewidth=1, color=color, label=f"{label}")

    handles, labels = [], []
    seen = set()
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)

    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    return fig, ax

def plot_reference_intervals(results, reference_intervals, n_cols=9, figsize=(18, 2), dpi=300):
    unique_codes = results['code'].sort_values().unique()
    ncodes = len(unique_codes)
    n_rows = (ncodes + n_cols - 1) // n_cols
    fig_height = figsize[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height), dpi=dpi)
    axes = axes.flatten()
    
    # Prepare for legend handles/labels
    demo_ax = axes[0]

    # Only create legend lines once for legend above
    normal_line = demo_ax.plot([], [], color='#2166AC', label='Normal', linewidth=1)[0]
    abnormal_line = demo_ax.plot([], [], color='#B2182B', label='Abnormal', linewidth=1)[0]
    reference_patch = plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=8, label='Reference Range')
    # Will leave density color not filled to match axvspan

    for i, code in enumerate(unique_codes):
        ax = axes[i]
        code_data = results[results['code'] == code]

        sns.kdeplot(
            data=code_data[code_data['state'] == True],
            x='mu', label='Normal', ax=ax, color='#2166AC', linewidth=1)
        sns.kdeplot(
            data=code_data[code_data['state'] == False],
            x='mu', label='Abnormal', ax=ax, color='#B2182B', linewidth=1)

        low_m, high_m, unit = reference_intervals[code]['M']
        low_f, high_f, unit = reference_intervals[code]['F']
        low = min(low_m, low_f)
        high = max(high_m, high_f)

        ax.axvspan(low, high, alpha=0.1, color='gray', label='Reference Range')
        ax.set_xlabel(f'{code} ({unit})')
        ax.tick_params(width=0.5, length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i % n_cols != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Density')

        ax.legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add one legend above
    handles = [normal_line, abnormal_line, reference_patch]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, columnspacing=1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def animate_sequence_and_intervals(code, seq, df, axes, *, intermediate=None):
    # axes = (main, density). No new figure.
    ax, ax_density = axes
    ax.cla(); ax_density.cla()

    y = seq["x"]
    x = seq["t"]

    handles = []
    labels = []

    # Plot observed
    line_obs, = ax.plot(x, y, marker="o", color="black", linewidth=1.1, label='Observed')
    handles.append(line_obs)
    labels.append('Observed')

    sub = df[df["code"] == code]

    max_peak = 0.0
    pdfs = []
    info = []
    mins = []
    maxs = []
    population_handle = None
    personalized_handle = None
    norma_handle = None

    for i, (_, r) in enumerate(sub.iterrows()):
        mu, std = float(r["mu"]), float(r["std"])
        if r["run_id"] == "Population":
            h1 = ax.hlines(mu - std, x.min(), x.max(), linestyles=(0, (2, 2)), color=scheme_colors["Population"], linewidth=1.1)
            h2 = ax.hlines(mu + std, x.min(), x.max(), linestyles=(0, (2, 2)), color=scheme_colors["Population"], linewidth=1.1)
            if population_handle is None:
                population_handle = plt.Rectangle((0, 0), 1, 1, fc=scheme_colors['Population'], alpha=0.50, label='Population')
            mins.append(mu - 2.5*std)
            maxs.append(mu + 2.5*std)
        elif r['run_id'] == "Personalized":
            # mark for legend, but do not plot if commented out
            if personalized_handle is None:
                personalized_handle = plt.Rectangle((0, 0), 1, 1, fc=scheme_colors['Personalized'], alpha=0.50, label='Personalized')
            mins.append(mu - 2.5*std)
            maxs.append(mu + 2.5*std)
        elif r["run_id"] != "Population" and r["run_id"] != "Personalized":
            if r.get("state", False):
                h = ax.axhspan(mu - std, mu + std, alpha=0.5, color=scheme_colors["Hybrid"], label='NORMA')
                if norma_handle is None:
                    norma_handle = h

    mins = [intermediate[-1][0]["x"].min()] + mins
    maxs = [intermediate[-1][0]["x"].max()] + maxs

    y_lo = min(mins) + 0.05*(intermediate[-1][0]["x"].max() - intermediate[-1][0]["x"].min())
    y_hi = max(maxs) + 0.05*(intermediate[-1][0]["x"].max() - intermediate[-1][0]["x"].min())
    y_grid = np.linspace(y_lo, y_hi, 500)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"{code}")
    ax.set_xlim(x.min(), x.max())

    # Collect PDFs and find max peak for normalization
    for _, r in sub.iterrows():
        mu, std = float(r["mu"]), float(r["std"])
        if len(y) < 4 and r["run_id"] == "Personalized":
            continue
        pdf = _gaussian_pdf(y_grid, mu, std)
        pdfs.append(pdf)
        info.append(r)
        max_peak = max(max_peak, pdf.max())

    ax.set_ylim(y_lo, y_hi)
    ax_density.set_ylim(y_lo, y_hi)
    ax_density.yaxis.set_visible(False)
    ax_density.xaxis.set_visible(False)
    ax_density.set_frame_on(False)

    # Plot all density curves (PDFs) on the same line (not stacked)
    density_handles = []
    density_labels = []
    if max_peak > 0:
        for i, (r, pdf) in enumerate(zip(info, pdfs)):
            color = scheme_colors[r["run_id"]] if r["run_id"] in scheme_colors else scheme_colors['Hybrid']
            # rescale for comparability
            scaled = pdf / max_peak
            label = f"{r['run_id'].title()}" if r["run_id"] in scheme_colors else "NORMA"
            line, = ax_density.plot(scaled, y_grid, linewidth=1, color=color)
            ax_density.fill_betweenx(y_grid, 0, scaled, alpha=0.30, color=color)
            # Only add one handle per label
            #if label not in density_labels:
                #density_handles.append(line)
                #density_labels.append(label)
        ax_density.set_xlim(0, 1.05)

    # Prepare handles/labels for legend
    # Prefer order: Observed, NORMA, Personalized, Population, then densities (if not redundant)
    if norma_handle is not None:
        handles.append(norma_handle)
        labels.append("NORMA")
    if personalized_handle is not None:
        handles.append(personalized_handle)
        labels.append("Personalized")
    if population_handle is not None:
        handles.append(population_handle)
        labels.append("Population")
    # Now add unique density handles not already in handles
    for h, l in zip(density_handles, density_labels):
        # Avoid duplicating NORMA, Personalized, Population
        if l not in labels:
            handles.append(h)
            labels.append(l)

    # Defensive: only draw legend for lines that exist
    ax.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2))
   # ax_density.legend(density_handles, density_labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2))
    if ax.get_figure():
        ax.get_figure().legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2))


def make_partial(seq, i):
    sp = dict(seq)  # shallow copy
    for k in ("x", "t", "s"):
        if k in sp:
            sp[k] = sp[k][:i]
    return sp
