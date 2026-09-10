#!/bin/bash
#SBATCH --job-name=NORMA_sens_methods
#SBATCH --output=jobs/logs/sens_methods_%j.log
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -t 03:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Sensitivity of every reference-interval method (PopRI, PerRI, Gaussian x3,
# Cohen m4, NORMA + covariate-ablation arms) to history length / horizon /
# within-person SD on the same synthetic histories ->
# 06_calibration/results/prediction/sensitivity_methods.csv, then rebuild the
# sensitivity figures (all-methods overlay + NORMA-arms-only).
#
# Usage:  sbatch jobs/run_sensitivity_methods.sh [extra args]
#   arms:  sbatch jobs/run_sensitivity_methods.sh --norma_runs q_age q_set q_co q_age_set
#   (q_age_set_co is excluded while it is still training -- checkpoint_latest is epoch 6)

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=8

python -u 06_sensitivity.py "$@"
python -W ignore make_figures.py --only 06_calibration
