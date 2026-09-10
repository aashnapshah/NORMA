#!/usr/bin/env bash
#SBATCH --job-name=NORMA_down
#SBATCH --output=jobs/logs/down_%x_%j.log
#SBATCH --mem=180G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Everything downstream of the reference intervals (04_refs.py): 07 classify
# -> 06 calibration -> 09 age_ri -> 10 mortality -> 11 lead_time -> 12 eval ->
# 13 cox -> 14 patient_level -> 16 benchmark -> 05 forecasting -> 17 outcomes.
# 08_variability is skipped (reads only the unchanged `per` rows).
#
# Usage: sbatch [--dependency=afterok:<refs job>] jobs/run_downstream.sh {eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
DATASET="${1:?usage: run_downstream.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo ""; echo "=== $(date +%H:%M) $* ==="; python -u "$@"; }

run 07_classify.py --dataset "$DATASET" --force
run 06_calibration.py --dataset "$DATASET"
run 09_age_ri.py --dataset "$DATASET"
run 10_mortality.py --dataset "$DATASET"
run 11_lead_time.py --dataset "$DATASET"
run 12_eval.py --dataset "$DATASET"
run 13_cox.py --dataset "$DATASET" --n_jobs 8
run 14_patient_level.py --dataset "$DATASET" --panels all --no-gbm --no-bootstrap --common-analytes
run 16_benchmark.py --dataset "$DATASET"
run 05_forecasting.py --dataset "$DATASET" --workers 8
run 17_outcomes.py --dataset "$DATASET"
echo ""; echo "Downstream complete for $DATASET."
