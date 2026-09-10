#!/bin/bash
#SBATCH --job-name=NORMA_SWEEP
#SBATCH --output=logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 50:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load cuda/11.7

# Re-run INSPIRE pipeline with visit-based split
set -e

cd "$(dirname "$0")"

# echo "=== Step 1: Split (visit-based) ==="
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
python 04_refs.py --only norma --dataset inspire --device cuda
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
echo "=== Step 13: Patient-specific Cox models ==="
python 14_patient_level.py --only refit --dataset inspire

echo ""
echo "=== Done ==="
