#!/bin/bash
#SBATCH --job-name=NORMA_refs_inspire
#SBATCH --output=jobs/logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 50:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
module load cuda/11.7

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Step 1: NORMA predictions + ref intervals ==="
python 04_refs.py --only norma --dataset inspire --device cuda
python 04_refs.py --only baselines --dataset inspire

echo ""
echo "=== Step 2: Age RI ==="
python 09_age_ri.py --dataset inspire

echo ""
echo "=== Step 3: Mortality ==="
python 10_mortality.py --dataset inspire

echo ""
echo "=== Step 4: Classify ==="
python 07_classify.py --only classify --dataset inspire --force

echo ""
echo "=== Step 5: Prevalence ==="
python 07_classify.py --only prevalence --dataset inspire --force

echo ""
echo "=== Step 6: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset inspire

echo ""
echo "=== Step 7: Eval metrics ==="
python 12_eval.py --only metrics --dataset inspire --force

echo ""
echo "=== Step 8: Cox models ==="
python 13_cox.py --dataset inspire

echo ""
echo "=== Step 9: Patient Cox / NRI ==="
python 14_patient_level.py --only refit --dataset inspire --no-gbm

echo ""
echo "INSPIRE pipeline complete."
