#!/usr/bin/env bash
#SBATCH --job-name=NORMA_refscpu
#SBATCH --output=jobs/logs/refscpu_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Step 05 only: recompute NORMA reference intervals (corrected input convention:
# offsets in days anchored per pair, query at the actual time of the first index
# measurement, low/normal/high history states)
# on a GPU. Every other saved row (base / pop / per GMM / gaussian_* / cohen_*) is kept.
# Follow with jobs/run_downstream.sh (06 -> 13 + 16 + 05_forecasting).
#
# Usage: sbatch jobs/run_refs_norma.sh {eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
# one BLAS/OpenMP thread per worker: the loky GMM workers otherwise spawn ~47 threads each and thrash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Reference intervals FROM SCRATCH (PopRI / PerRI GMM / Gaussian / Cohen + NORMA slice from
# 04_refs/cache/<ds>/norma_predictions.parquet, which jobs/run_norma_only.sh must have written).
# Any existing ref_intervals.* is moved aside as .bak-<stamp> so nothing stale is reused.
# Usage: sbatch [--dependency=afterok:<norma job>] jobs/run_refs_scratch.sh {eicu|inspire}
DATASET="${1:?usage: run_refs_scratch.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "$DATASET" in eicu) DATA_DIR=data/eICU ;; inspire) DATA_DIR=data/INSPIRE/25-75 ;; *) echo "unknown dataset"; exit 1 ;; esac
STAMP=$(date +%Y%m%d-%H%M)
for f in "$DATA_DIR"/ref_intervals.csv "$DATA_DIR"/ref_intervals.parquet; do
    [[ -f "$f" ]] && { mv "$f" "$f.bak-$STAMP"; echo "  $f -> $f.bak-$STAMP"; }
done
echo "=== 04_refs.py --only baselines from scratch ($DATASET) ==="
python -u 04_refs.py --only baselines --dataset "$DATASET"
echo "Fresh refs complete for $DATASET."
