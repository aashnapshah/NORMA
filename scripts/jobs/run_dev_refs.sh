#!/usr/bin/env bash
#SBATCH --job-name=NORMA_devrefs
#SBATCH --output=jobs/logs/devrefs_%x_%j.log
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# NORMA predictions + reference intervals + forecasting evaluation for a development
# cohort, once 02_index_labs has written index_labs.parquet (i.e. run_dev_forecast.sh
# without the two processing steps).
#
# Usage: sbatch jobs/run_dev_refs.sh {ehrshot|mimiciv} [--max_patients 40000]

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
SRC="${1:?usage: run_dev_refs.sh ehrshot-or-mimiciv [--max_patients N]}"; shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -f "data/$SRC/index_labs.parquet" || { echo "no index_labs for $SRC yet"; exit 1; }
echo "=== 04_refs.py --only norma ($SRC) $* ===";           python 04_refs.py --only norma --dataset "$SRC" "$@"
echo "=== 04_refs.py --only baselines ($SRC) $* ===";       python 04_refs.py --only baselines --dataset "$SRC" "$@"
echo "=== 05_forecasting ($SRC) $* ===";             python 05_forecasting.py --dataset "$SRC" --workers 8 "$@"
echo "Dev refs complete for $SRC."
