#!/usr/bin/env bash
#SBATCH --job-name=NORMA_devfc
#SBATCH --output=jobs/logs/devfc_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Development cohorts (NORMA test split of EHRSHOT / MIMIC-IV) through the same
# pipeline as the validation cohorts, so every cohort carries the same models:
# split -> NORMA predictions + intervals -> Pop/Per/Gaussian/Cohen intervals ->
# forecasting evaluation (history baselines + scoring).
#
# Usage: sbatch jobs/run_dev_forecast.sh {ehrshot|mimiciv} [--max_patients 40000]
#        (the optional args go to refs.py (both steps) and forecasting.py alike,
#         so all three cover the same patients)

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e

SRC="${1:?usage: run_dev_forecast.sh ehrshot-or-mimiciv [--max_patients N]}"; shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== process/dev_cohort.py --source $SRC ===";                   python "../../process/dev_cohort.py" --source "$SRC"
echo "=== 02_index_labs ($SRC) ===";                 python 02_index_labs.py --dataset "$SRC"
echo "=== 04_refs.py --only norma ($SRC) $* ===";           python 04_refs.py --only norma --dataset "$SRC" "$@"
echo "=== 04_refs.py --only baselines ($SRC) $* ===";       python 04_refs.py --only baselines --dataset "$SRC" "$@"
echo "=== 05_forecasting ($SRC) $* ===";             python 05_forecasting.py --dataset "$SRC" --workers 8 "$@"
echo "Dev forecasting complete for $SRC."
