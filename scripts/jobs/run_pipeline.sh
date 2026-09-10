#!/usr/bin/env bash
#SBATCH --job-name=NORMA_VALIDATE
#SBATCH --output=logs/%j.log
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL
#
# NORMA validation pipeline, one script per stage (see PIPELINE.md).
#
# Usage:
#   bash jobs/run_pipeline.sh              # eICU + INSPIRE + sandbox CHS
#   bash jobs/run_pipeline.sh eicu         # eICU only
#   bash jobs/run_pipeline.sh inspire      # INSPIRE only
#   bash jobs/run_pipeline.sh chs          # sandbox CHS only
#   real CHS runs inside Clalit: python jobs/run_clalit.py

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -e
mkdir -p logs

run() {
    echo ""
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
    python "$@" 2>&1
    echo "  -- done --"
}

# Every stage after 01-03, for one cohort.  Order: 04 refs -> 05 forecasting (refs
# only) -> 07 classify (everything else reads the classification) -> the rest.
downstream() {   # $1 = dataset
    run 04_refs.py --dataset "$1" --device cuda
    run 05_forecasting.py --dataset "$1" --workers 8
    run 07_classify.py --dataset "$1" --force
    run 06_calibration.py --dataset "$1"
    run 08_variability.py --dataset "$1"
    run 09_age_ri.py --dataset "$1"
    run 10_mortality.py --dataset "$1"
    run 11_lead_time.py --dataset "$1"
    run 12_eval.py --dataset "$1"
    run 13_cox.py --dataset "$1" --n_jobs 8
    run 14_patient_level.py --dataset "$1" --panels all --no-gbm --no-bootstrap --common-analytes
    run 16_benchmark.py --dataset "$1"
    run 17_outcomes.py --dataset "$1"
}

WHICH="${1:-all}"

if [[ "$WHICH" == "all" || "$WHICH" == "eicu" ]]; then
    echo "########## eICU ##########"
    run ../../process/eicu.py
    run 02_index_labs.py --dataset eicu
    run 03_cohort_summary.py --dataset eicu
    downstream eicu
fi

if [[ "$WHICH" == "all" || "$WHICH" == "inspire" ]]; then
    echo "########## INSPIRE ##########"
    run ../../process/inspire.py
    run 02_index_labs.py --dataset inspire
    run 03_cohort_summary.py --dataset inspire
    downstream inspire
fi

if [[ "$WHICH" == "all" || "$WHICH" == "chs" ]]; then
    echo "########## CHS (sandbox) ##########"
    run ../../process/clalit.py --sandbox --force
    run 02_index_labs.py --dataset chs --force
    run 03_cohort_summary.py --dataset chs
    run 04_refs.py --dataset chs
    run 05_forecasting.py --dataset chs
    run 07_classify.py --dataset chs --force
    run 06_calibration.py --dataset chs
    run 08_variability.py --dataset chs
    run 09_age_ri.py --dataset chs
    run 10_mortality.py --dataset chs
    run 11_lead_time.py --dataset chs --only future_abnormal
    run 12_eval.py --dataset chs --only metrics auroc
    run 13_cox.py --dataset chs --only models
    run 16_benchmark.py --dataset chs
fi

echo ""
echo "Pipeline complete."
