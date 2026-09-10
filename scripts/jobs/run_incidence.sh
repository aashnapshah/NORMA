#!/usr/bin/env bash
#SBATCH --job-name=INCIDENCE
#SBATCH --output=logs/incidence_%j.log
#SBATCH -e logs/incidence_%j.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -p short
#
# 17_outcomes re-run after the fixes of 2026-09-03:
#   - outcome time columns read in ds.outcome_time_unit (minutes), not hours
#   - standard of care = latest value AVAILABLE AT the landmark (no look-ahead)
#   - NaN != 1 no longer counts unmeasured patients as SOC-flagged
#   - every method emitted at two anchors: matched sensitivity (Cohen) and
#     matched-to-SOC alert rate (fair head-to-head with current practice)
#   - landmarks swept in one pass
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo -e "\n=== $* ==="; python "$@" 2>&1; echo "  -- ok --"; }

LM="0 24 72 168"
HZ="24 72 168 720"
for OC in aki mortality sepsis prolonged_los; do
  run 17_outcomes.py --dataset eicu --outcomes $OC
done
for OC in mortality unplanned_icu periop_infection prolonged_los; do
  run 17_outcomes.py --dataset inspire --outcomes $OC
done
echo -e "\nIncidence re-run complete."
