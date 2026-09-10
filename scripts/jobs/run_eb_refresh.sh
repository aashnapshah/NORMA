#!/usr/bin/env bash
#SBATCH --job-name=EB_REFRESH
#SBATCH --output=logs/eb_refresh_%j.log
#SBATCH -e logs/eb_refresh_%j.log
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 2-12:00:00
#SBATCH -p medium
#
# Empirical-Bayes benchmark redefined (2026-09-03): the mean prior is the
# population reference interval itself (--gaussian_prior popri). Recompute the
# gaussian_* rows, then everything downstream of the classification, for both
# cohorts (only the EB rows are recomputed: MLE and the truncated fit use no
# prior, so their rows are unchanged). Also runs the lead-time and extras stages (their separate jobs were
# cancelled and folded in here so nothing reads a half-updated table).
# Runs after the main refresh (--dependency=afterok) so ref_intervals is stable.
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo -e "\n=== $(date +%H:%M) $* ==="; python "$@" 2>&1; echo "  -- ok --"; }
LM="0 24 72 168"; HZ="24 72 168 720"
for DS in eicu inspire; do
  echo -e "\n########## $DS ##########"
  run 04_refs.py --only baselines --dataset $DS --gaussian_methods eb --gaussian_prior popri --gaussian_force
  run 07_classify.py --only classify   --dataset $DS --force
  run 07_classify.py --only prevalence --dataset $DS --force
  run 06_calibration.py --only coverage --dataset $DS
  run 06_calibration.py --only conformal       --dataset $DS
  run 09_age_ri.py       --dataset $DS
  run 10_mortality.py --dataset $DS
  run 11_lead_time.py --only future_abnormal --dataset $DS
  run 11_lead_time.py       --dataset $DS --only lead_time
  run 12_eval.py --only metrics   --dataset $DS --force
  run 12_eval.py --only auroc          --dataset $DS
  run 13_cox.py      --dataset $DS
  run 16_benchmark.py --only comparison       --dataset $DS
  run 16_benchmark.py --only operating_point --dataset $DS
  run 16_benchmark.py --only burden         --dataset $DS
  run 16_benchmark.py --only significance            --dataset $DS
  run 14_patient_level.py --only refit --dataset $DS --panels all --no-gbm --no-bootstrap --common-analytes --min-coverage 0.75
  run 14_patient_level.py --only swap    --dataset $DS --common-analytes
  run 05_forecasting.py   --dataset $DS --workers 8
  for OC in $( [ "$DS" = eicu ] && echo "aki mortality sepsis prolonged_los" || echo "mortality unplanned_icu periop_infection prolonged_los" ); do
    run 17_outcomes.py --dataset $DS --outcomes $OC
    run 17_outcomes.py --dataset $DS --outcomes $OC
  done
done
echo -e "\nEB refresh complete."
