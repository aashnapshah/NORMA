#!/bin/bash
#SBATCH --job-name=NORMA_MODELS
#SBATCH --output=logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 50:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1       # use 1 GPU per experiment (change to :2 if you really need 2)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu

module load gcc/9.2.0
module load cuda/11.7

# # conda env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate norma   # or your env

# mkdir -p logs
python train.py --nstates 3 --train combined --test combined
python train.py --nstates 3 --train ehrshot --test ehrshot

python train.py --nstates 2
