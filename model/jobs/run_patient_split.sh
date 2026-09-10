#!/bin/bash
# Patient-level split runs (R3 comment 11).
#
# The published dev split is at the patient-analyte level, so a patient's other
# analytes can sit across the train/test boundary. That is tolerable while the
# model only sees the target analyte, and NOT tolerable once it conditions on
# co-analytes. These arms re-run under --split_by patient, where whole patients
# are held out.
#
# The patient split changes the test set, so NONE of the covariate-ablation arms
# (334f7e21, q_age, q_set, q_co, q_age_set, q_age_set_co) are valid comparators.
# Compare only within this group, against p_base.
#
#   name        flags                                   answers
#   p_base      --split_by patient                      new reference point; gap vs 334f7e21
#                                                       = how much the sequence split inflated dev
#   p_co        + --use_coanalytes                      was the q_co null contaminated by the split?
#   p_causal    + --causal_memory                       effect of masking the cross-attention block
#   p_full      + --use_full_panel --causal_memory     THE multivariate arm: every draw in the
#                                                       patient's past, all 34 analytes, at their
#                                                       own irregular times
#
# p_full differs from p_base in two ways, so read the chain p_base -> p_causal -> p_full:
# p_causal isolates the mask, p_full adds the data on top of it. Causal masking is not
# optional here -- with a time-sorted multi-analyte stream, an unmasked cross-attention
# block would let a co-analyte drawn at the query timestamp reach the query token.
#
# p_full needs {source}_drawmeta_v3.npz (python process/draw_meta.py). It implies
# --use_coanalytes. Sequence length grows to <= --max_draws tokens (p50 7, p90 38,
# p99 140 draws/patient, so 128 covers ~99%).
#
# Caveat on p_co: co-analytes are still same-timestamp-only (data.py indexes
# panel[draw_idx], i.e. rows where the target analyte was itself drawn). This
# re-tests the old question cleanly; it does not test the irregular multivariate
# past, which needs the panel row -> (subject_id, time) map that build_panel
# currently discards.
#
# Memory: as in run_covariate_ablation.sh, v3 sequences are ~2.4M dicts and the 4
# forked workers copy-on-write the whole list -> 128G, not 64G.
#
# Usage:  bash run_patient_split.sh                  # submit all three
#         bash run_patient_split.sh p_base           # subset
#         RESUME=1 bash run_patient_split.sh p_base  # resume an interrupted arm
#         EVAL=1   bash run_patient_split.sh p_base  # eval only, from checkpoint_latest
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs/patient_split

declare -A FLAGS=(
  [p_base]=""
  [p_co]="--use_coanalytes"
  [p_causal]="--causal_memory"
  [p_full]="--use_full_panel --causal_memory --max_draws 128"
)
ORDER=(p_base p_co p_causal p_full)
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

# Identical to run_covariate_ablation.sh COMMON except for --split_by patient.
# --data_version v3 is kept on every arm (not just p_co) so all three read the
# same sequence file and differ only in the flag under test.
COMMON="--model NORMA2 --loss QuantileLoss --output_mode quantile --d_model 64 --nhead 4 --nlayers 8 \
--nstates 3 --batch_size 32 --lr 0.0001 --epochs 50 --patience 10 --train combined --test combined \
--seed 42 --data_version v3 --data_dir ../../data/processed/ --split_by patient \
--wandb_group patient-split"

for arm in "${ARMS[@]}"; do
  [[ -v FLAGS[$arm] ]] || { echo "unknown arm: $arm (choose from ${ORDER[*]})"; exit 1; }
  if [ "${EVAL:-0}" = "1" ]; then
    ident="--run_id ${arm}"
  elif [ "${RESUME:-0}" = "1" ]; then
    ident="--run_id ${arm} --resume"
  else
    [ -e "logs/${arm}/checkpoint_latest.pth" ] && { echo "logs/${arm} exists; use RESUME=1"; exit 1; }
    ident="--run_name ${arm}"
  fi
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=NORMA_${arm}
#SBATCH --output=logs/patient_split/${arm}_%j.log
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
python train.py ${COMMON} ${FLAGS[$arm]} ${ident} --wandb_tags patient-split ${arm} --description "patient-level split: ${arm} (${FLAGS[$arm]:-baseline})"
echo "end=\$(date)"
EOF
done
