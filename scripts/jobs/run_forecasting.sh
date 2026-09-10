#!/bin/bash
#SBATCH --job-name=NORMA_fcst
#SBATCH --output=jobs/logs/forecasting_%x_%j.log
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH -t 11:59:00
#SBATCH -p short
#
# Forecasting on an external cohort: NORMA at every query state (04_refs.py --only norma,
# also the source of NORMA's reference intervals), then the history-only (and
# state-informed) baselines and the scoring against the interval centres.
# the baselines step of 04_refs.py must have run for the cohort (the centres come from ref_intervals).
#
# Usage:  sbatch jobs/run_forecasting.sh eicu
#         sbatch jobs/run_forecasting.sh inspire
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python

set -euo pipefail
DATASET=${1:?usage: sbatch jobs/run_forecasting.sh eicu-or-inspire-or-chs}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python 04_refs.py --only norma --dataset "$DATASET" --batch_size 2048
python 05_forecasting.py --dataset "$DATASET" --workers 16 --with_state --force_baselines
