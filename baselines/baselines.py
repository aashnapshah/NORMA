import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

sys.path.append("../model/")
from utils import *
from data import *
from edit import *

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source = "combined"
train_seq, val_seq, test_seq = load_and_split_data(DATA_DIR, source, print_info=False)
print(len(test_seq))

def _split_xy(seq):
    x = np.asarray(seq["x"], dtype=float)
    if x.size < 2:
        return x[:-1], np.nan
    return x[:-1], x[-1]

def pred_mean(x_hist):
    return float(np.nanmean(x_hist)) if x_hist.size else np.nan

def pred_last(x_hist):
    return float(x_hist[-1]) if x_hist.size else np.nan

def pred_arima(x_hist, order=(1,1,1)):
    if x_hist.size < sum(order):
        return np.nan
    try:
        fit = ARIMA(x_hist, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        return float(fit.forecast(1)[0])
    except Exception:
        return np.nan

def run_baseline(sequences, split_name, model_name, predictor):
    rows = []
    for seq in tqdm(sequences, desc=f"Running {model_name} on {split_name}"):
        x_hist, x_next = _split_xy(seq)
        pred = predictor(x_hist)
        rows.append({
            "subject_id": seq["pid"],
            "cid": seq["cid"],
            "split": split_name,
            "model": model_name,
            "x_next": float(x_next),
            "x_pred": float(pred) if np.isfinite(pred) else np.nan,
        })
    return pd.DataFrame(rows)

def run_all(train_seq, val_seq, test_seq, baselines):
    pred_dir = "predictions"
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for name, fn in baselines.items():
        parts = [
            run_baseline(train_seq, "train", name, fn),
            run_baseline(val_seq, "val", name, fn),
            run_baseline(test_seq, "test", name, fn),
        ]
        sub_df = pd.concat(parts, ignore_index=True)
        sub_df.to_csv(f"{pred_dir}/{name.replace(' ','_')}_baseline_{source}.csv", index=False)
        print(f"Created {len(sub_df)} rows for {name}")
        all_dfs.append(sub_df)
    return pd.concat(all_dfs, ignore_index=True)

baselines = {
    "mean": pred_mean,
    "last": pred_last,
    "arima": lambda x: pred_arima(x, (1,1,1)),
}

df = run_all(train_seq, val_seq, test_seq, baselines)

pred_dir = "predictions"
Path(pred_dir).mkdir(parents=True, exist_ok=True)
for name in df["model"].unique():
    df[df["model"] == name].to_csv(f"{pred_dir}/{name.replace(' ','_')}_baseline_{source}.csv", index=False)
df.to_csv(f"{pred_dir}/all_baselines_{source}.csv", index=False)

print(
    "Created:",
    {name: int((df['model'] == name).sum()) for name in df['model'].unique()},
    "| total:", len(df)
)
