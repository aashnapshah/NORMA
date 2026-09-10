#!/usr/bin/env bash
#SBATCH --job-name=NORMA_eb_popri
#SBATCH --output=jobs/logs/eb_popri_%x_%j.log
#SBATCH --mem=180G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#
# gaussian_eb changed on 2026-09-08: the prior is now process/config.py's
# REFERENCE_INTERVALS (mu = PopRI midpoint, tau = half-width / z) with no fitted
# artifact, and the patient's latent (mean, sd) come from the truncated-normal
# MLE capped at the population variance, so eb is trunc with shrinkage rather
# than mle with shrinkage.  CHS can then run the same estimator inside Clalit,
# where no dev cohort or artifact exists.
#
# Stage 04 touches eb only: mle and trunc never load a prior, so
# --gaussian_methods eb --gaussian_force drops and rebuilds exactly those rows.
# --cohen_models (empty) skips augment_cohen so the 157 MB artifact is not read.
#
# Everything below 04 scores gaussian_eb through the classification, so each
# stage gets --force -- without it the already_done() guards would reuse the
# results computed under the old prior.  08_variability and 10_mortality are
# skipped: they read `per` and `base` rows, which do not change.
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo ""; echo "=== $(date +%H:%M) $* ==="; python -u "$@"; }

# 1. reference intervals, every cohort that has them
for DS in ehrshot mimiciv inspire eicu; do
  run 04_refs.py --dataset "$DS" --only baselines \
      --gaussian_methods eb --gaussian_force --cohen_models
done

# 2. everything that scores them, on the two validation cohorts
for DS in inspire eicu; do
  run 07_classify.py     --dataset "$DS" --force
  run 06_calibration.py  --dataset "$DS" --force
  run 09_age_ri.py       --dataset "$DS" --force
  run 11_lead_time.py    --dataset "$DS" --force
  run 12_eval.py         --dataset "$DS" --force
  run 13_cox.py          --dataset "$DS" --force --n_jobs 8
  # --no-gbm but bootstrap on: matches what produced the current 14_concordance
  # (model_type penalized_cox/swap, CIs on 3,200 of 3,272 rows), so the rerun
  # differs only by the estimator and not by the configuration.
  run 14_patient_level.py --dataset "$DS" --force --panels all --no-gbm --common-analytes
  run 16_benchmark.py    --dataset "$DS" --force
  run 05_forecasting.py  --dataset "$DS" --force --workers 8
  run 17_outcomes.py     --dataset "$DS" --force
done

# 3. dev-cohort forecasting reads the interval centres
for DS in ehrshot mimiciv; do
  run 05_forecasting.py --dataset "$DS" --force --workers 8
done

run make_figures.py --all
run make_tables.py --all
echo ""; echo "=== $(date +%H:%M) done ==="
