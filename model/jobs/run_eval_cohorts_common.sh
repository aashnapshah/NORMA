#!/bin/bash
#SBATCH --job-name=NORMA_evalc
#SBATCH --output=logs/eval_cohorts_common_%j.log
#SBATCH --mem=32G
#SBATCH -c 2
#SBATCH -t 6:00:00
#SBATCH -p short
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# output_dir pointed into the validation/ tree that was removed on 2026-09-08;
# the dev raw results live under results/raw/dev/ now. PYTHONPATH is handled by
# model/bootstrap.py.
python evaluate.py --cohorts eicu inspire --common_rows --output_dir ../results/raw/dev --suffix cohorts_common
