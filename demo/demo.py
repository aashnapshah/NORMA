"""
NORMA terminal demo — generate individualized reference intervals for a few
synthetic patients, without any training or private data.

What this does:
  - Loads the public NORMA checkpoint from HuggingFace (no training needed).
  - Reads a small synthetic lab-history file (demo_patients.csv).
  - For each patient/analyte, predicts a NORMA reference interval at the next
    follow-up (90 days after the last measurement) and compares it to the
    fixed population reference interval (Pop_RI).

Run:
    python demo.py                      # uses demo_patients.csv, 90-day horizon
    python demo.py --input my.csv       # your own history file (same columns)
    python demo.py --horizon 180        # predict further out

Input columns: patient_id, sex (M/F), age, analyte, day, value
  - day: days since the patient's first measurement (day 0)
  - value: the lab result in the analyte's standard unit
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

# Make the NORMA model + config importable regardless of where you run from.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "process"))

from config import REFERENCE_INTERVALS          # noqa: E402  population ranges + vocab
from utils import create_model                   # noqa: E402  builds NORMA from checkpoint

# Public model weights. run_id 167f05e8 = NormaLight (Gaussian parameterization).
HF_REPO = "aashnaps/NORMA"
RUN_ID = "167f05e8"

TEST_VOCAB = {name: i for i, name in enumerate(REFERENCE_INTERVALS.keys())}


def value_to_state3(value, low, high):
    """Map a raw value to NORMA's 3-state code: 0=low, 1=normal, 2=high."""
    if value < low:
        return 0
    if value > high:
        return 2
    return 1


def load_norma(device):
    """Download the public checkpoint and build the model. Returns (model, is_quantile)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=f"{RUN_ID}/checkpoint_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)

    hp = checkpoint["hyperparameters"]
    hp["run_id"] = RUN_ID
    hparams = argparse.Namespace(**hp)

    model = create_model(hparams, ncodes=len(TEST_VOCAB), checkpoint=checkpoint).to(device)
    model.eval()
    is_quantile = getattr(hparams, "output_mode", "gaussian") == "quantile"
    return model, is_quantile


def norma_interval(model, is_quantile, analyte, sex01, age, history, t_next):
    """
    NORMA reference interval for one patient/analyte at query time t_next.

    history : list of (day, value), sorted by day. We condition the query on the
              'normal' state so the interval is the expected healthy range given
              this patient's history.
    Returns (mu, ci_lower, ci_upper).
    """
    sex_str = "F" if sex01 == 1 else "M"
    low, high, _unit = REFERENCE_INTERVALS[analyte][sex_str]

    t_arr = np.array([d for d, _ in history], dtype=np.float32)
    x_arr = np.array([v for _, v in history], dtype=np.float32)
    s_arr = np.array([value_to_state3(v, low, high) for v in x_arr], dtype=np.int64)

    x_h = torch.tensor(x_arr).view(1, -1, 1).float()
    s_h = torch.tensor(s_arr).view(1, -1).long()
    t_h = torch.tensor(t_arr).view(1, -1, 1).float()
    sex_t = torch.tensor([sex01]).long()
    age_t = torch.tensor([[age]]).float()
    cid_t = torch.tensor([TEST_VOCAB[analyte]]).long()
    s_next_t = torch.tensor([[1]]).long()          # 1 = normal state
    t_next_t = torch.tensor([[t_next]]).float()

    with torch.no_grad():
        output = model(x_h, s_h, t_h, sex_t, age_t, cid_t, s_next_t, t_next_t, pad_mask=None)

    if is_quantile:
        q = output.squeeze(0).cpu().numpy()        # [q2.5, q25, q50, q75, q97.5]
        return float(q[2]), float(q[0]), float(q[4])
    mu_t, lv_t = output
    mu = float(mu_t.squeeze())
    sigma = float(torch.exp(0.5 * lv_t).squeeze())
    return mu, mu - 1.96 * sigma, mu + 1.96 * sigma


def main():
    ap = argparse.ArgumentParser(description="NORMA reference-interval demo")
    ap.add_argument("--input", default=os.path.join(HERE, "demo_patients.csv"),
                    help="CSV with columns: patient_id, sex, age, analyte, day, value")
    ap.add_argument("--horizon", type=int, default=90,
                    help="days after last measurement to predict the interval for")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Loading NORMA ({HF_REPO}, run {RUN_ID}) on {device}...")
    model, is_quantile = load_norma(device)
    print("Model loaded.\n")

    df = pd.read_csv(args.input)

    print(f"{'patient':<8}{'analyte':<9}{'sex':<5}{'Pop_RI':>16}{'NORMA_RI':>18}{'last obs':>10}")
    print("-" * 66)
    for (pid, analyte), g in df.groupby(["patient_id", "analyte"], sort=False):
        g = g.sort_values("day")
        sex_str = str(g["sex"].iloc[0]).upper()
        sex01 = 1 if sex_str == "F" else 0
        age = float(g["age"].iloc[0])
        history = list(zip(g["day"].astype(float), g["value"].astype(float)))
        t_next = float(g["day"].max()) + args.horizon
        last_val = history[-1][1]

        low, high, unit = REFERENCE_INTERVALS[analyte][sex_str]
        _mu, lo, hi = norma_interval(model, is_quantile, analyte, sex01, age, history, t_next)

        pop = f"{low:g}-{high:g}"
        norma = f"{lo:.2f}-{hi:.2f}"
        print(f"{pid:<8}{analyte:<9}{sex_str:<5}{pop:>16}{norma:>18}{last_val:>10g}  {unit}")

    print("\nPop_RI   = fixed population reference interval (same for everyone of that sex).")
    print("NORMA_RI = individualized 95% interval given this patient's trajectory.")


if __name__ == "__main__":
    main()
