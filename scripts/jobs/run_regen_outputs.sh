#!/usr/bin/env bash
#SBATCH --job-name=REGEN_FIGS
#SBATCH --output=logs/regen_outputs_%j.log
#SBATCH -e logs/regen_outputs_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -t 4:00:00
#SBATCH -p short
# Regenerate EVERY figure and table once the full refresh (new main NORMA =
# q_age_set_co, new Cohen sigma) and the lead-time job have both finished.
# Submitted with --dependency=afterok:<lead-time job>, which itself depends on
# the refresh, so this is the first output set that is consistent end to end.
set -euo pipefail
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "=== $(date) make_figures ==="; python make_figures.py 2>&1 | grep -vE "Warning|from pandas"
echo "=== $(date) make_tables ===";  python make_tables.py  2>&1 | grep -vE "Warning|from pandas"
echo "=== $(date) placeholders remaining ==="; python make_figures.py --check 2>&1 | grep -vE "Warning|from pandas" | tail -40
echo "Regeneration complete."
