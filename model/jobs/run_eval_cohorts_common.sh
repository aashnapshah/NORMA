#!/bin/bash
#SBATCH --job-name=NORMA_evalc
#SBATCH --output=logs/eval_cohorts_common_%j.log
#SBATCH --mem=32G
#SBATCH -c 2
#SBATCH -t 6:00:00
#SBATCH -p short
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=../process:.
python evaluate.py --cohorts eicu inspire --common_rows --output_dir ../validation/results/prediction/raw/cohorts_common
