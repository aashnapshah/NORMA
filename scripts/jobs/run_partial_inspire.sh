#!/bin/bash
#SBATCH --job-name=NORMA_partial
#SBATCH --output=logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -p medium
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Step 0: Convert checkpoint → ref_intervals.parquet (complete pairs only) ==="
python jobs/checkpoint_to_parquet.py

echo ""
echo "=== Step 1: Age RI ==="
python 09_age_ri.py --dataset inspire

echo ""
echo "=== Step 2: Mortality ==="
python 10_mortality.py --dataset inspire

echo ""
echo "=== Step 3: Classify ==="
python 07_classify.py --only classify --dataset inspire --force

echo ""
echo "=== Step 4: Prevalence ==="
python 07_classify.py --only prevalence --dataset inspire --force

echo ""
echo "=== Step 5: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset inspire

echo ""
echo "=== Step 6: Eval metrics ==="
python 12_eval.py --only metrics --dataset inspire --force

echo ""
echo "=== Step 7: Cox models ==="
python 13_cox.py --dataset inspire

echo ""
echo "=== Step 8: Patient Cox / ROC ==="
python 14_patient_level.py --only refit --dataset inspire

echo ""
echo "Partial pipeline complete (PopRI + PerRI only, no NORMA)."
echo "Re-run full pipeline after 04_refs.py --only norma adds the NORMA intervals."
