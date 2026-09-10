#!/usr/bin/env bash
#SBATCH --job-name=COHEN_REFRESH
#SBATCH --output=logs/cohen_refresh_%j.log
#SBATCH -e logs/cohen_refresh_%j.log
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -p medium
#
# Regenerate everything that depends on the Cohen interval WIDTH after the
# sigma fix in model/baselines/cohen.py (s.d. of predicted values, not of
# residuals; Cohen et al. Fig. 4c). The Cohen *centre* is unchanged, so
# 04_refs.py --only norma and 05_forecasting are not re-run -- forecasting scores
# interval centres only.
#
# Backup of the pre-fix results: validation/backup_pre_cohen_fix_20260902/
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() { echo -e "\n=== $* ==="; python "$@" 2>&1; echo "  -- ok --"; }

for DS in eicu inspire; do
  echo -e "\n########## $DS ##########"
  # 1x  rebuild the Cohen artifact (sigma is cached inside it) and its ref rows
  run 04_refs.py --only baselines --dataset $DS --cohen_models m2 m3 m4 \
      --cohen_retrain --cohen_force
  # 2x  reclassify, then every measurement-level analysis
  # --force is REQUIRED: classify.py:246 skips when classification.parquet exists,
  # so without it the new Cohen widths never reach the classification and every
  # downstream stage silently re-reads the old ones.
  run 07_classify.py --only classify   --dataset $DS --force
  run 07_classify.py --only prevalence --dataset $DS --force
  run 06_calibration.py --only coverage --dataset $DS
  run 11_lead_time.py --only future_abnormal --dataset $DS
  run 12_eval.py --only metrics   --dataset $DS --force
  run 12_eval.py --only auroc          --dataset $DS
  run 13_cox.py      --dataset $DS
  # 3x  benchmark + patient level
  run 16_benchmark.py --only comparison      --dataset $DS
  run 16_benchmark.py --only operating_point --dataset $DS
  run 16_benchmark.py --only burden         --dataset $DS
  run 16_benchmark.py --only significance            --dataset $DS
  run 14_patient_level.py --only refit --dataset $DS --panels all --no-gbm \
      --no-bootstrap --common-analytes --min-coverage 0.75
  run 14_patient_level.py --only swap    --dataset $DS --common-analytes
  # Cohen Fig. 5c-f: KM cumulative incidence at a matched operating point
  for OC in $( [ "$DS" = eicu ] && echo "aki mortality sepsis" || echo "mortality unplanned_icu periop_infection" ); do
    run 17_outcomes.py --dataset $DS --outcomes $OC
  done
done
echo -e "\nCohen refresh complete."
