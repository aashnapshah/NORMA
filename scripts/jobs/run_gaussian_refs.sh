#!/bin/bash
#SBATCH --job-name=NORMA_gauss
#SBATCH --output=jobs/logs/gaussian_%x_%j.log
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Append the PopRI-normal-history Gaussian baselines (gaussian_mle/trunc/eb)
# to an existing ref_intervals file, then re-run classification and the
# downstream method-level analyses so the new methods appear in results.
#
# Usage:  sbatch jobs/run_gaussian_refs.sh eicu [START_STEP]
#         sbatch jobs/run_gaussian_refs.sh inspire [START_STEP]
# START_STEP (default 1) skips earlier steps, e.g. 5 = re-run only eval + Cox.
#
# The Cohen step of 05 is skipped here (--cohen_models with no values) so this
# job never triggers Cohen training; run that separately.

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
set -e

DATASET="${1:?usage: run_gaussian_refs.sh eicu-or-inspire [START_STEP]}"
START="${2:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "$DATASET" in
    eicu)    DATA_DIR=data/e-icu ;;
    inspire) DATA_DIR=data/inspire/25-75 ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

if (( START <= 1 )); then
echo "=== Backup ref_intervals ==="
for f in "$DATA_DIR"/ref_intervals.csv "$DATA_DIR"/ref_intervals.parquet; do
    if [[ -f "$f" && ! -f "$f.bak-pregaussian" ]]; then
        cp -p "$f" "$f.bak-pregaussian"; echo "  $f -> $f.bak-pregaussian"
    fi
done

echo ""
echo "=== Step 1: Append Gaussian baselines (04) ==="
python 04_refs.py --only baselines --dataset "$DATASET" --cohen_models \
    --gaussian_methods mle trunc eb --gaussian_prior cohort --gaussian_force
fi

if (( START <= 2 )); then
echo ""
echo "=== Step 2: Classify ==="
python 07_classify.py --only classify --dataset "$DATASET" --force
fi

if (( START <= 3 )); then
echo ""
echo "=== Step 3: Prevalence ==="
python 07_classify.py --only prevalence --dataset "$DATASET" --force
fi

if (( START <= 4 )); then
echo ""
echo "=== Step 4: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset "$DATASET"
fi

if (( START <= 5 )); then
echo ""
echo "=== Step 5: Eval metrics ==="
python 12_eval.py --only metrics --dataset "$DATASET" --force
fi

if (( START <= 6 )); then
echo ""
echo "=== Step 6: Cox models ==="
python 13_cox.py --dataset "$DATASET"
fi

echo ""
echo "Gaussian baselines complete for $DATASET."
