#!/usr/bin/env bash
#SBATCH --job-name=NORMA_REFRESH
#SBATCH --output=logs/full_refresh_%j.log
#SBATCH -e logs/full_refresh_%j.log
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -p medium
#
# Regenerate everything from the NORMA reference intervals onward for eICU and
# INSPIRE, e.g. after the main model changes (lib/datasets.py NORMA_RUN_ID).  NOT
# re-run: the baselines step of 04_refs (PopRI/PerRI/Gaussian/Cohen unchanged),
# 08_variability (per rows only), CHS (needs the Clalit bundle rebuilt).
#
# The norma step runs with the default run list (= every arm in dataset.run_ids)
# rather than --runs <main>: it OVERWRITES norma_predictions.parquet with the runs
# it computes, so a single-arm call would drop the other arms' forecasts.
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo -e "\n=== $(date +%H:%M) $* ==="; python "$@" 2>&1; echo "  -- ok --"; }

for DS in eicu inspire; do
  echo -e "\n########## $DS ##########"
  run 04_refs.py --dataset $DS --only norma                 # all arms incl. main
  run 07_classify.py --dataset $DS --force
  run 06_calibration.py --dataset $DS
  run 09_age_ri.py --dataset $DS
  run 10_mortality.py --dataset $DS
  run 11_lead_time.py --dataset $DS
  run 12_eval.py --dataset $DS
  run 13_cox.py --dataset $DS --n_jobs 8
  run 16_benchmark.py --dataset $DS
  run 14_patient_level.py --dataset $DS --panels all --no-gbm --no-bootstrap --common-analytes
  run 05_forecasting.py --dataset $DS --workers 8
  run 17_outcomes.py --dataset $DS
done
echo -e "\nFull refresh complete."
