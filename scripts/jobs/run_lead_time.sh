#!/usr/bin/env bash
#SBATCH --job-name=LEAD_TIME
#SBATCH --output=logs/lead_time_%j.log
#SBATCH -e logs/lead_time_%j.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
#SBATCH -p short
# 11_lead_time.py: outcome-agnostic lead time (default endpoint = the
# value leaving Pop_RI) for every analyte, then the clinical-endpoint variants
# the incidence figures draw on. Submitted with --dependency=afterok:<full
# refresh> so it reads the refreshed classification for BOTH cohorts.
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() { echo -e "\n=== $(date +%H:%M) $* ==="; python "$@" 2>&1; echo "  -- ok --"; }
for DS in eicu inspire; do
  run 11_lead_time.py --dataset $DS --only lead_time
  for OC in $( [ "$DS" = eicu ] && echo "aki mortality sepsis prolonged_los" || echo "mortality unplanned_icu periop_infection prolonged_los" ); do
    run 17_outcomes.py --dataset $DS --outcomes $OC
  done
done
echo -e "\nLead time complete."
