#!/bin/bash
#SBATCH --job-name=NORMA_SWEEP
#SBATCH --output=logs/sweeps/%j.log
#SBATCH --mem=64G
#SBATCH -t 50:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1       # use 1 GPU per experiment (change to :2 if you really need 2)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu

module load gcc/9.2.0
module load cuda/11.7

python train.py --run_id 167f05e8 
python evaluate.py --model 167f05e8

sleep infinity


# # conda env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate norma   # or your env

# mkdir -p logs
#python train.py --train merged --test merged
#wandb agent aashna-shah/NORMA/nhnk48gn
# python train.py --loss_type GaussianNLLLoss --source MIMIC-IV --n_la
# python train.py --loss_type GaussianNLLLoss --source merged 

#python train.py --loss_type GaussianNLLLoss --mse_warmup_epochs 0 --source merged 

#python train.py --loss_type GaussianNLLLoss --mse_warmup_epochs 5 --source MIMIC-IV
#python train.py --loss_type GaussianNLLLoss --mse_warmup_epochs 5 --source merged 

#python train.py --loss_type MSELoss --mse_warmup_epochs 0 --source merged
#python train.py --loss_type MSELoss --mse_warmup_epochs 0 --source EHRSHOT
#python train.py --loss_type MSELoss --mse_warmup_epochs 0 --source MIMIC-IV





