#!/usr/bin/env bash
#SBATCH --job-name=NORMA_refs
#SBATCH --output=jobs/logs/refs_%x_%j.log
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
# Usage: sbatch jobs/run_refs_norma.sh {eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
# one BLAS/OpenMP thread per worker: the loky GMM workers otherwise spawn ~47 threads each and thrash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATASET="${1:?usage: run_refs_norma.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "$DATASET" in eicu) DATA_DIR=data/e-icu ;; inspire) DATA_DIR=data/inspire/25-75 ;; *) echo "unknown dataset"; exit 1 ;; esac

echo "=== Backup ref_intervals (pre time-unit fix) ==="
for f in "$DATA_DIR"/ref_intervals.csv "$DATA_DIR"/ref_intervals.parquet; do
    if [[ -f "$f" && ! -f "$f.bak-pretimefix" ]]; then cp -p "$f" "$f.bak-pretimefix"; echo "  $f -> $f.bak-pretimefix"; fi
done
echo "=== 04_refs.py --only norma --device cuda ($DATASET) ==="
python -u 04_refs.py --only norma --dataset "$DATASET" --device cuda
echo "=== 04_refs.py --only baselines ($DATASET) ==="
python -u 04_refs.py --dataset "$DATASET" --only baselines
echo "NORMA refs complete for $DATASET."
