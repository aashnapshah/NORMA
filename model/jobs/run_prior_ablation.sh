#!/bin/bash
# Prior-anchored NORMA arms: does the interval revert to the population reference
# when the history is short, noisy or stale? (2026-09-08; the quantile model covers
# ~77% at nominal 95% and its width tracks history dispersion with no prior floor:
# logs/ablation/summary.csv, validation/06_calibration sensitivity_methods.csv.)
#
# Every arm = the main model (q_age_set: NORMA2, 64/4/8, --use_age_t --use_setting,
# seed 42, sequence split) with a different loss or output head. Compare against
# q_age_set with:  python shrinkage.py --runs q_age_set pa_k5 ...   (coverage and
# width by history length, plus the post-hoc blend) and python compare_ablation.py.
#
#   name      head      loss                        prior weight          idea
#   pa_k5     quantile  pinball + anchor, k=5       k/(n+k)               prior pseudo-observations: the
#   pa_k20    quantile  pinball + anchor, k=20      k/(n+k)               expected pinball under the Pop_RI
#                                                                         prior (cross-entropy to the prior
#                                                                         quantiles), normal-state queries
#   pa_tau    quantile  pinball + anchor, k=5       k/(n_eff+k),          same, n decays with time since each
#                                                   n_eff=sum e^{-dt/365d} draw (OU-process setpoint drift)
#   pf_k5     quantile  pinball + width floor, k=5  k/(n+k)               soft floor on width only, all states
#   pg_k5     gate      pinball on g*own+(1-g)*prior learned g, KL to      the model learns how much to trust
#                       + Bernoulli KL(g || n/(n+k)) n/(n+k)              the patient; g is written out
#   pn_k5     nig       Student-t NLL               conjugate: kappa0=    normal-inverse-gamma posterior
#                                                   rho/(1-rho), nu0=5    predictive; n_eff=0 gives Pop_RI
#   gk_k5     gaussian  Gaussian NLL + KL to Pop_RI k/(n+k)               the existing NORMALoss with a
#                       (NORMALoss), lambda 0.1                           history-dependent weight
#
# Usage:  bash run_prior_ablation.sh                 # all arms
#         bash run_prior_ablation.sh pa_k5 pn_k5     # subset
#         RESUME=1 bash run_prior_ablation.sh pa_k5
#         EVAL=1 bash run_prior_ablation.sh pa_k5    # predict/evaluate only
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs/prior_ablation

declare -A FLAGS=(
  [pa_k5]="--loss QuantilePriorLoss --output_mode quantile --prior_mode anchor --prior_k 5 --prior_lambda 1"
  [pa_k20]="--loss QuantilePriorLoss --output_mode quantile --prior_mode anchor --prior_k 20 --prior_lambda 1"
  [pa_tau]="--loss QuantilePriorLoss --output_mode quantile --prior_mode anchor --prior_k 5 --prior_lambda 1 --prior_tau 365"
  [pf_k5]="--loss QuantilePriorLoss --output_mode quantile --prior_mode floor --prior_k 5 --prior_lambda 1"
  [pg_k5]="--loss QuantilePriorLoss --output_mode gate --prior_mode gate --prior_k 5 --prior_lambda 1"
  [pn_k5]="--loss StudentTNLLLoss --output_mode nig --nig_nu0 5"
  [gk_k5]="--loss NORMALoss --output_mode gaussian --align_by_n --prior_k 5 --lambda_align 0.1"
)
ORDER=(pa_k5 pa_k20 pa_tau pf_k5 pg_k5 pn_k5 gk_k5)
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

COMMON="--model NORMA2 --d_model 64 --nhead 4 --nlayers 8 --nstates 3 --batch_size 32 --lr 0.0001 \
--epochs 50 --patience 10 --train combined --test combined --seed 42 --data_version v3 \
--use_age_t --use_setting --data_dir ../../data/processed/ --wandb_group prior-ablation"

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
  sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=NORMA_${arm}
#SBATCH --output=logs/prior_ablation/${arm}_%j.log
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
python train.py ${COMMON} ${FLAGS[$arm]} ${ident} --wandb_tags prior ${arm} --description "prior ablation: ${arm} (${FLAGS[$arm]})"
echo "end=\$(date)"
EOS
done
