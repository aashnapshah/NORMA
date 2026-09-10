#!/usr/bin/env bash
#SBATCH --job-name=NORMA_cohort
#SBATCH --output=jobs/logs/cohort_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 4
#SBATCH -t 05:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# 03_cohort_summary for one cohort: cohort.csv / demographics.csv (all measurements)
# and cohort_by_split.csv / demographics_by_split.csv (baseline / index; plus NORMA's
# train / val / test for ehrshot and mimiciv).  CHS is produced inside Clalit.
#
# Usage: sbatch jobs/run_cohort_summary.sh {ehrshot|mimiciv|eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e

DS="${1:?usage: run_cohort_summary.sh ehrshot-or-mimiciv-or-eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "=== 03_cohort_summary.py --dataset $DS ==="
python 03_cohort_summary.py --dataset "$DS"
