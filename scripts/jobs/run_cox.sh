#!/usr/bin/env bash
#SBATCH --job-name=NORMA_cox
#SBATCH --output=jobs/logs/cox_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Landmark Cox models, all exposures x encodings x subsets (13_cox.py).
# Re-marks the exposure rows first so classification.parquet carries them.
# Usage: sbatch jobs/run_cox.sh {eicu|inspire}
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
DATASET="${1:?usage: run_cox.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "=== 07_classify --force ($DATASET): store exposure markers ==="
python 07_classify.py --only classify --dataset "$DATASET" --force
echo "=== 13_cox ($DATASET) ==="
python 13_cox.py --dataset "$DATASET" --n_jobs 8
echo "Cox complete for $DATASET."
