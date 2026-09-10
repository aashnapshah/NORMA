#!/bin/bash
#SBATCH --job-name=NORMA_eval
#SBATCH --output=jobs/logs/eval_forecasting_%x_%j.log
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -t 05:59:00
#SBATCH -p short
#
# Score forecasting predictions with bootstrap CIs (model/evaluate.py).
#   dev          held-out test split, all rows, plus EHRSHOT / MIMIC-IV separately (--by_source)
#   dev_common   same, rows where every model is defined
#   <cohort>     external cohort (eicu | inspire | chs), all rows
#   <cohort>_common
# Afterwards: python nature-comm-submission-2026/scripts/summarize_forecasting_variants.py
#
# Usage:  for m in dev dev_common eicu eicu_common; do sbatch jobs/run_eval_forecasting.sh $m; done
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python

set -euo pipefail
MODE=${1:?usage: sbatch jobs/run_eval_forecasting.sh dev|dev_common|eicu|eicu_common|inspire|inspire_common}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=../scripts:../scripts/process:.
RAW=../results/raw/dev
case "$MODE" in
  dev)
    python -u evaluate.py --variants oracle normal marginal marginal_freq --by_state --state_baselines --by_source \
        --output_dir "$RAW" --suffix state_variants ;;
  dev_common)
    python -u evaluate.py --variants oracle normal marginal marginal_freq --by_state --state_baselines --by_source \
        --common_rows --output_dir "$RAW" --suffix state_variants_common ;;
  *_common)
    python -u evaluate.py --cohorts "${MODE%_common}" --common_rows --output_dir "$RAW" --suffix common ;;
  *)
    python -u evaluate.py --cohorts "$MODE" --output_dir "$RAW" ;;
esac
