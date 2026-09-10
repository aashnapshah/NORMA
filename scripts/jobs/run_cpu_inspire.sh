#!/bin/bash
#SBATCH --job-name=NORMA_CPU
#SBATCH --output=logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 50:00:00
#SBATCH -p medium
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# No CUDA / old gcc modules (RHEL9 environment)
module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# echo "=== Step 1: Process + split ==="
# python process/inspire.py
# python 02_index_labs.py --dataset inspire

# echo ""
# echo "=== Step 2: Cohort summary ==="
# python 03_cohort_summary.py --dataset inspire

# echo ""
# echo "=== Step 3: Variability ==="
# python 08_variability.py --dataset inspire

echo ""
echo "=== Step 4: NORMA predictions + ref intervals ==="
python 04_refs.py --only norma --dataset inspire --runs 334f7e21
python 04_refs.py --only baselines --dataset inspire

echo ""
echo "=== Step 5: Age RI ==="
python 09_age_ri.py --dataset inspire

echo ""
echo "=== Step 6: Mortality ==="
python 10_mortality.py --dataset inspire

echo ""
echo "=== Step 7: Classify ==="
python 07_classify.py --only classify --dataset inspire --force

echo ""
echo "=== Step 8: Prevalence ==="
python 07_classify.py --only prevalence --dataset inspire --force

echo ""
echo "=== Step 9: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset inspire

echo ""
echo "=== Step 10: Eval metrics ==="
python 12_eval.py --only metrics --dataset inspire --force

echo ""
echo "=== Step 11: Cox models ==="
python 13_cox.py --dataset inspire

echo ""
echo "=== Step 12: ROC curves ==="
python 14_patient_level.py --only refit --dataset inspire

echo ""
echo "CPU pipeline complete."
