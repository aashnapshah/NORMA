#!/bin/bash
# Covariate ablation for the Nature Comms revision (R1-1 / R2-4 / R2-3).
#
# Every arm is identical to run 334f7e21 (NORMA2, quantile head, nstates=3, seed 42,
# same split) except for the per-measurement covariates switched on. Run ids are the
# arm names (--run_name), so logs live in logs/<name>/ and W&B shows them by name
# inside group "covariate-ablation".
#
#   name          covariates                              family
#   334f7e21      none (baseline, already trained)        -
#   q_age         age at each draw                        single / additive step 1
#   q_set         care setting                            single
#   q_co          same-draw co-analytes + their states    single
#   q_age_set     age + setting                           additive step 2
#   q_age_set_co  age + setting + co-analytes             additive step 3 (= all)
#   q_co_q        co-analytes, query sees last panel too  symmetry fix for q_co
#   q_age_co      age + co-analytes                       leave-one-out from full: -setting
#   q_set_co      setting + co-analytes                   leave-one-out from full: -age
# Since 2026-09-04 q_age_set is the MAIN model (scripts/lib/datasets.NORMA_RUN_ID);
# the ablation is read as leave-one-out from it.
#
# Requires ../../data/processed/combined_{sequences,panel}_v3.* (process/covariates.py).
# Memory: the v3 sequences are ~2.4M Python dicts and the 4 forked DataLoader workers
# copy-on-write the whole list -> peak ~5x the in-memory dataset; 64G OOM-killed every arm
# at epoch 5 (2026-08-27), peak ~67G, hence 128G. Any GPU type; gpu_requeue only ("gpu_quad,gpu_requeue"
# fails the QoS check since 2026-08-28 pm) with --requeue:
# gpu_requeue may preempt, but the RESUME path restarts from checkpoint_latest (<=1 epoch lost).
# Model is tiny (d_model 64) so GPU type barely matters; the loader is the bottleneck.
#
# Usage:  bash run_covariate_ablation.sh                 # submit all five
#         bash run_covariate_ablation.sh q_age q_co      # subset
#         RESUME=1 bash run_covariate_ablation.sh q_age  # resume an interrupted arm
#         EVAL=1 bash run_covariate_ablation.sh q_age    # eval only (predict/evaluate/sensitivity
#                                                        #  from checkpoint_latest; training done)
#         Compare afterwards with:  python compare_ablation.py
set -euo pipefail
# these scripts moved into model/jobs/; train.py is one level up, and the
# sbatch body below inherits this working directory
cd "$(dirname "$0")/.."
mkdir -p logs/ablation

declare -A FLAGS=(
  [q_age]="--use_age_t"
  [q_set]="--use_setting"
  [q_co]="--use_coanalytes"
  [q_age_set]="--use_age_t --use_setting"
  [q_age_set_co]="--use_age_t --use_setting --use_coanalytes"
  [q_co_q]="--use_coanalytes --query_coanalytes"
  [q_age_co]="--use_age_t --use_coanalytes"
  [q_set_co]="--use_setting --use_coanalytes"
)
ORDER=(q_age q_set q_co q_age_set q_age_set_co q_co_q q_age_co q_set_co)
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

COMMON="--model NORMA2 --loss QuantileLoss --output_mode quantile --d_model 64 --nhead 4 --nlayers 8 \
--nstates 3 --batch_size 32 --lr 0.0001 --epochs 50 --patience 10 --train combined --test combined \
--seed 42 --data_version v3 --data_dir ../../data/processed/ --wandb_group covariate-ablation"

for arm in "${ARMS[@]}"; do
  [[ -v FLAGS[$arm] ]] || { echo "unknown arm: $arm (choose from ${ORDER[*]})"; exit 1; }
  if [ "${EVAL:-0}" = "1" ]; then
    ident="--run_id ${arm}"   # no --resume: skip training, run predict/evaluate/sensitivity only
  elif [ "${RESUME:-0}" = "1" ]; then
    ident="--run_id ${arm} --resume"
  else
    [ -e "logs/${arm}/checkpoint_latest.pth" ] && { echo "logs/${arm} exists; use RESUME=1"; exit 1; }
    ident="--run_name ${arm}"
  fi
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=NORMA_${arm}
#SBATCH --output=logs/ablation/${arm}_%j.log
#SBATCH --mem=128G
#SBATCH -c 6
#SBATCH -t 4-12:00:00
#SBATCH -p gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
module load gcc/14.2.0
module load cuda/12.8
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate normal
ulimit -n 65536 2>/dev/null || ulimit -n hard 2>/dev/null || true
echo "arm=${arm} flags='${FLAGS[$arm]}' host=\$(hostname) start=\$(date)"
python train.py ${COMMON} ${FLAGS[$arm]} ${ident} --wandb_tags ablation ${arm} --description "covariate ablation: ${arm} (${FLAGS[$arm]})"
echo "end=\$(date)"
EOF
done
