#!/usr/bin/env bash
#SBATCH --job-name=patient_cox
#SBATCH --output=jobs/logs/%j_patient_cox.log
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 11:00:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Patient-level multi-analyte survival models (13) + feature-swap NRI (14)
# across ALL reference-interval methods:
#   PopRI, PerRI, Gaussian_{mle,trunc,eb}, Cohen_{m2,m3,m4}, NORMA
#
# --common-analytes restricts to analytes every method covers (Cohen skips
# analytes with too few healthy training pairs), so the comparison is fair.
# --min-coverage 0.75 matches 14_patient_level; requiring 100% of the panel measured
# within 48h leaves zero eICU patients (rare labs: CRP, LDH, PT) and 30 then
# silently produced "No results produced" for every outcome.
#
# Usage:
#   sbatch run_patient_cox.sh          # both datasets
#   sbatch run_patient_cox.sh eicu     # eICU only
#   sbatch run_patient_cox.sh inspire  # INSPIRE only

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p jobs/logs

run() {
    echo ""
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
    python "$@" 2>&1
    echo "  -- done --"
}

WHICH="${1:-all}"

for DS in eicu inspire; do
    if [[ "$WHICH" == "all" || "$WHICH" == "$DS" ]]; then
        run 14_patient_level.py --only refit --dataset "$DS" --panels all --no-gbm \
            --common-analytes --min-coverage 0.75
        run 14_patient_level.py --only swap   --dataset "$DS" --common-analytes
    fi
done

echo ""
echo "Patient-level Cox + NRI swap complete."
