#!/bin/bash
# Prior-anchored NORMA arms: does the interval revert to the population reference
# when the history is short, noisy or stale? (2026-09-08; the quantile model covers
# ~77% at nominal 95% and its width tracks history dispersion with no prior floor:
# logs/ablation/summary.csv, validation/06_calibration sensitivity_methods.csv.)
#
# Two input families, one loss list. Each arm is some reference model with a
# different loss or output head, and nothing else changed:
#
#   m_* (run first)  p_full's inputs and split plus age at each draw:
#                    --use_full_panel --causal_memory --max_draws 128 --use_age_t
#                    --split_by patient. NOTE p_full itself carries no --use_age_t,
#                    so it is not an exact control; a p_full_age arm on QuantileLoss
#                    is needed for the loss to be the only difference.
#                    This is the family the prior question is actually about: a
#                    width floor matters most on the arm with the most context to
#                    be overconfident from, and p_full is both the multivariate
#                    arm and the worst calibrated of its group (val coverage 0.741
#                    at nominal 0.95, against 0.814 for p_base).
#   the rest         q_age_set's inputs: --use_age_t --use_setting, sequence split.
#                    Compare against q_age_set.
#
# Compare within a family only -- the two are scored on different test rows.
#   python shrinkage.py --runs p_full m_pa_k5 ...        (multivariate family)
#   python shrinkage.py --runs q_age_set pa_k5 ...       (sex/age/setting family)
# plus python compare_ablation.py for coverage and width by history length.
#
# 2026-09-11: of the sex/age/setting family only gk_k5 ever finished; pa_k5 and
# pg_k5 have no checkpoint, pa_k20/pa_tau/pf_k5 died at epoch 0 and pn_k5 at 24,
# so that family needs rerunning whatever happens to the m_* arms.
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
# The loss table below names the seven losses; each exists twice, as m_<name> on
# the multivariate inputs and <name> on the sex/age/setting inputs.
#
# Usage:  bash run_prior_ablation.sh                    # all 14, m_* first
#         bash run_prior_ablation.sh m_pf_k5 m_pn_k5    # subset
#         (resubmitting an arm resumes it from its latest checkpoint)
#         EVAL=1 bash run_prior_ablation.sh m_pf_k5     # predict/evaluate only
set -euo pipefail
# these scripts moved into model/jobs/; train.py is one level up, and the
# sbatch body below inherits this working directory
cd "$(dirname "$0")/.."
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
# The m_ arms carry the same loss flags; only the inputs and the split differ, so
# the loss list is not duplicated.
for arm in pa_k5 pa_k20 pa_tau pf_k5 pg_k5 pn_k5 gk_k5; do
  FLAGS[m_${arm}]="${FLAGS[$arm]}"
done

# Multivariate family first: bash run_prior_ablation.sh with no arguments submits
# the m_* arms before the sex/age/setting ones.
ORDER=(m_pa_k5 m_pa_k20 m_pa_tau m_pf_k5 m_pg_k5 m_pn_k5 m_gk_k5
       pa_k5 pa_k20 pa_tau pf_k5 pg_k5 pn_k5 gk_k5)
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

BASE="--model NORMA2 --d_model 64 --nhead 4 --nlayers 8 --nstates 3 --batch_size 32 --lr 0.0001 \
--epochs 50 --patience 10 --train combined --test combined --seed 42 --data_version v3 \
--data_dir ../../data/processed/ --wandb_group prior-ablation"
# Inputs per family. The m_ line is run_patient_split.sh's p_full arm verbatim, so
# p_full is the control for every m_ arm and the loss is the only difference.
# --use_age_t is supported on full-panel arms (data.py rebuilds age_h on the panel
# clock); --use_setting is not and train.py rejects the pair, so these arms are
# "sex, age, every past draw" rather than the main model's age+setting.
INPUTS_MULTI="--split_by patient --use_full_panel --causal_memory --max_draws 128 --use_age_t"
INPUTS_MAIN="--use_age_t --use_setting"

for arm in "${ARMS[@]}"; do
  [[ -v FLAGS[$arm] ]] || { echo "unknown arm: $arm (choose from ${ORDER[*]})"; exit 1; }
  case "$arm" in
    m_*) INPUTS="$INPUTS_MULTI"; FAMILY="multivariate";;
    *)   INPUTS="$INPUTS_MAIN";  FAMILY="sex/age/setting";;
  esac
  COMMON="$BASE $INPUTS"
  # The run's identity is decided when the job STARTS, not when it is submitted:
  # gpu_requeue preempts and requeues jobs, and a requeued job re-runs this same
  # body. Deciding at submit time restarted every preempted arm from epoch 0
  # (2026-09-08: pa_k20, pa_tau, pf_k5 died mid-epoch 1; pa_k5, pg_k5 before a
  # checkpoint). Now a job resumes whenever a checkpoint exists.
  if [ "${EVAL:-0}" = "1" ]; then
    ident_rule="IDENT='--run_id ${arm}'"
  else
    ident_rule="if [ -e logs/${arm}/checkpoint_latest.pth ]; then IDENT='--run_id ${arm} --resume'; else IDENT='--run_name ${arm}'; fi"
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
${ident_rule}
echo "arm=${arm} flags='${FLAGS[$arm]}' ident='\$IDENT' host=\$(hostname) start=\$(date)"
python train.py ${COMMON} ${FLAGS[$arm]} \$IDENT --wandb_tags prior ${arm} --description "prior ablation (${FAMILY}): ${arm} (${FLAGS[$arm]} ${INPUTS})"
echo "end=\$(date)"
EOS
done
