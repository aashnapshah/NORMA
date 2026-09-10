#!/bin/bash
#SBATCH --job-name=NORMA_STATES
#SBATCH --output=logs/predict_states_%j.log
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
# Run NORMA over the test split with every query state and derive the
# normal-fixed / marginalised forecasting variants (Referee 3, minor 1).
#   sbatch run_inference.sh 334f7e21
#   sbatch run_inference.sh 167f05e8
module load gcc/9.2.0
module load cuda/11.7
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# model/bootstrap.py handles sys.path; predict_states.py merged into inference.py
python inference.py --run_id "$1" --batch_size 2048
