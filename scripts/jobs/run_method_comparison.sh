#!/usr/bin/env bash
#SBATCH --job-name=NORMA_methcmp
#SBATCH --output=jobs/logs/methcmp_%x_%j.log
#SBATCH --mem=180G
#SBATCH -c 4
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Cross-method comparison of every reference-interval model:
#   16_benchmark.py --only comparison — threshold-free AUROC + equal-alert-budget PPV/lift
#   05_forecasting.py     — every model as a point forecast of the first index
#                             measurement (interval centres, history baselines, NORMA)
#
# Usage: sbatch jobs/run_method_comparison.sh {eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
set -e

DATASET="${1:?usage: run_method_comparison.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 16_benchmark ($DATASET) ==="
python 16_benchmark.py --only comparison --dataset "$DATASET"

echo "=== 05_forecasting ($DATASET) ==="
python 05_forecasting.py --dataset "$DATASET" --workers 8

echo "=== 16_benchmark ($DATASET) ==="
python 16_benchmark.py --only burden --dataset "$DATASET"

echo "=== 06_calibration ($DATASET) ==="
python 06_calibration.py --only coverage --dataset "$DATASET"

echo "Method comparison complete for $DATASET."
