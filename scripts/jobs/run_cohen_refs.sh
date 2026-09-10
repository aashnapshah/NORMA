#!/bin/bash
#SBATCH --job-name=NORMA_cohen
#SBATCH --output=jobs/logs/cohen_%x_%j.log
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Append the Cohen et al. 2021 benchmark (cohen_m2/m3/m4) to an existing
# ref_intervals file, then re-run classification and the downstream
# method-level analyses so the new methods appear in results.
#
# Models are trained ONCE on the NORMA dev split (EHRSHOT, Cohen healthy-cohort
# criteria) and cached at cache/cohen_dev_models.pkl; pass "retrain" to rebuild.
#
# Usage:  sbatch jobs/run_cohen_refs.sh eicu retrain
#         sbatch --dependency=afterok:<eicu_job> jobs/run_cohen_refs.sh inspire
#
# Gaussian baselines are left at their defaults: 05 skips them when the rows
# are already present, otherwise it appends them too.

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal   # sksurv/lifelines live here, not in the default python
set -e

DATASET="${1:?usage: run_cohen_refs.sh eicu-or-inspire [retrain|force]}"
# retrain = retrain dev models AND recompute cohen_* rows; force = recompute
# rows with the cached models (needed when cohen_* rows already exist).
RETRAIN=""
[[ "${2:-}" == "retrain" ]] && RETRAIN="--cohen_retrain --cohen_force"
[[ "${2:-}" == "force" ]] && RETRAIN="--cohen_force"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "$DATASET" in
    eicu)    DATA_DIR=data/e-icu ;;
    inspire) DATA_DIR=data/inspire/25-75 ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

echo "=== Backup ref_intervals ==="
for f in "$DATA_DIR"/ref_intervals.csv "$DATA_DIR"/ref_intervals.parquet; do
    if [[ -f "$f" && ! -f "$f.bak-precohen" ]]; then
        cp -p "$f" "$f.bak-precohen"; echo "  $f -> $f.bak-precohen"
    fi
done

echo ""
echo "=== Step 1: Append Cohen benchmark (04) $RETRAIN ==="
python 04_refs.py --only baselines --dataset "$DATASET" \
    --cohen_models m2 m3 m4 --cohen_train_sources mimiciv ehrshot $RETRAIN

echo ""
echo "=== Step 2: Classify ==="
python 07_classify.py --only classify --dataset "$DATASET" --force

echo ""
echo "=== Step 3: Prevalence ==="
python 07_classify.py --only prevalence --dataset "$DATASET" --force

echo ""
echo "=== Step 4: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset "$DATASET"

echo ""
echo "=== Step 5: Eval metrics ==="
python 12_eval.py --only metrics --dataset "$DATASET" --force

echo ""
echo "=== Step 6: Cox models ==="
python 13_cox.py --dataset "$DATASET"

echo "=== Step 7: benchmark comparison, burden, significance ==="
python 16_benchmark.py --only comparison --dataset "$DATASET"
python 16_benchmark.py --only burden --dataset "$DATASET"
python 16_benchmark.py --only significance --dataset "$DATASET"
python 05_forecasting.py --dataset "$DATASET" --workers 8

echo ""
echo "Cohen benchmark complete for $DATASET."
