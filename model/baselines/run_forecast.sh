#!/bin/bash
#SBATCH --job-name=NORMA_BASE
#SBATCH --output=../logs/baselines_%j.log
#SBATCH --mem=48G
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -p short
# Rebuild forecasting baselines on NORMA's three-state test split, with the
# state-informed variants.   sbatch run_forecast.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
python forecast.py --split test --workers 16 --with_state
