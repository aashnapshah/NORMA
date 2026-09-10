#!/usr/bin/env bash
#SBATCH --job-name=NORMA_abl_refs
#SBATCH --output=jobs/logs/abl_refs_%x_%j.log
#SBATCH --mem=140G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#
# NORMA reference intervals for the covariate-ablation arms alongside the published
# baseline, so every downstream analysis can compare them (Aashna 2026-08-31:
# "make versions that just compare the different norma ablations").
#
# 04_refs.py --only norma loads each run, reads its use_age_t/use_setting/use_coanalytes
# flags, and builds the matching per-measurement covariates from index_labs
# (lib/forecast_pairs.build_pairs(covariates=...)).  It upserts only the
# norma_<run_id> rows; base / pop / per / gaussian_* / cohen_* are untouched.
#
# Needs the `setting` column in index_labs: eICU has it from 01_process, INSPIRE was
# patched in place by process/inspire.py --attach_setting.  q_age_set_co is omitted while it
# is still training.
#
# Usage: sbatch jobs/run_ablation_refs.sh {eicu|inspire}

module load gcc/14.2.0
module load cuda/12.8
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATASET="${1:?usage: run_ablation_refs.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNS="334f7e21 q_age q_set q_co q_age_set"
echo "=== $DATASET: NORMA reference intervals for runs: $RUNS ==="
python -u 04_refs.py --only norma --dataset "$DATASET" --runs $RUNS --device cuda --batch_size 1024
echo "=== done $(date) ==="
