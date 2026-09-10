#!/usr/bin/env bash
#SBATCH --job-name=NORMA_norma
#SBATCH --output=jobs/logs/norma_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#
# Step 05 only: recompute NORMA reference intervals (corrected input convention:
# offsets in days anchored per pair, query at the actual time of the first index
# measurement, low/normal/high history states)
# on a GPU. Every other saved row (base / pop / per GMM / gaussian_* / cohen_*) is kept.
# Follow with jobs/run_downstream.sh (06 -> 13 + 16 + 05_forecasting).
#
# Usage: sbatch jobs/run_norma_only.sh {eicu|inspire} [extra refs.py args]
#   covariate arms: sbatch jobs/run_norma_only.sh eicu --runs 334f7e21 q_age q_set q_co q_age_set
#   (keep the baseline run in --runs: the norma step writes the columns for the runs it is
#   given, so dropping it would strip the published NORMA_334f7e21 intervals)

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
DATASET="${1:?usage: run_norma_only.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "=== 04_refs.py --only norma --device cuda ($DATASET) ${*:2} ==="
python -u 04_refs.py --only norma --dataset "$DATASET" --device cuda "${@:2}"
echo "NORMA predictions complete for $DATASET."
