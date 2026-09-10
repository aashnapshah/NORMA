#!/usr/bin/env bash
#SBATCH --job-name=EXTRAS
#SBATCH --output=logs/extras_%j.log
#SBATCH -e logs/extras_%j.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -t 8:00:00
#SBATCH -p short
# Two analyses whose figures the registry expects but that no job script ran:
# (width at matched coverage). Both read the classification table, so they run
# after the full refresh (--dependency=afterok).
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo -e "\n=== $(date +%H:%M) $* ==="; python "$@" 2>&1; echo "  -- ok --"; }
for DS in eicu inspire; do
  run 06_calibration.py --only conformal --dataset $DS
done
echo -e "\nExtras complete."
