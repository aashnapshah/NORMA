#!/usr/bin/env bash
#SBATCH --job-name=NORMA_timefix
#SBATCH --output=jobs/logs/timefix_%x_%j.log
#SBATCH --mem=180G
#SBATCH -c 16
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Recompute NORMA reference intervals after the 04_refs input-convention
# fix (history offsets in DAYS anchored at the pair's first measurement, query
# horizon in days, 3-state history coding low/normal/high), then re-run every
# downstream analysis so the benchmark compares like with like.
#
# Diagnostic that motivated this (validation/tests/diag_norma_time_units.py, INSPIRE,
# 1500 pairs): the old convention made intervals 61% too wide and flipped 24% of
# abnormal/normal calls.
#
# Usage: sbatch jobs/run_norma_timefix.sh {eicu|inspire}

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e

DATASET="${1:?usage: run_norma_timefix.sh eicu-or-inspire}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "$DATASET" in
    eicu)    DATA_DIR=data/e-icu ;;
    inspire) DATA_DIR=data/inspire/25-75 ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

echo "=== Backup ref_intervals + results (pre time-unit fix) ==="
for f in "$DATA_DIR"/ref_intervals.csv "$DATA_DIR"/ref_intervals.parquet; do
    if [[ -f "$f" && ! -f "$f.bak-pretimefix" ]]; then
        cp -p "$f" "$f.bak-pretimefix"; echo "  $f -> $f.bak-pretimefix"
    fi
done
RES=$(python -c "
import sys; sys.argv=['x','--dataset','$DATASET']
import argparse
from datasets import add_dataset_args, get_dataset
p=argparse.ArgumentParser(); add_dataset_args(p)
print(get_dataset(p.parse_args()).setup_output()[1])" 2>/dev/null | tail -1)
if [[ -n "$RES" && -d "$RES" ]]; then
    mkdir -p "$RES/pre_timefix"
    for f in prevalence.csv prevalence_per_normal.csv eval_all.csv eval_pop_normal.csv \
             eval_per_normal.csv eval_per_normal_pop_normal.csv cox.csv cox_pop_normal.csv \
             cox_per_normal.csv lead_time.csv lead_time_per_normal.csv \
             method_comparison.csv method_comparison_analyte.csv \
             forecast_by_analyte.csv forecast_summary.csv \
             multi_analyte_metrics.csv multi_analyte_importance.csv nri.csv \
             nri_swap.csv swap_metrics.csv; do
        [[ -f "$RES/$f" && ! -f "$RES/pre_timefix/$f" ]] && cp -p "$RES/$f" "$RES/pre_timefix/$f"
    done
    echo "  results snapshot -> $RES/pre_timefix/"
fi

echo ""
echo "=== Step 1: Recompute NORMA predictions + reference intervals (04, corrected inputs) ==="
python 04_refs.py --only norma --dataset "$DATASET"
python 04_refs.py --only baselines --dataset "$DATASET"

echo ""
echo "=== Step 2: Classify ==="
python 07_classify.py --only classify --dataset "$DATASET" --force

echo "=== Step 3: Prevalence ==="
python 07_classify.py --only prevalence --dataset "$DATASET" --force

echo "=== Step 4: Lead time ==="
python 11_lead_time.py --only future_abnormal --dataset "$DATASET"

echo "=== Step 5: Eval metrics ==="
python 12_eval.py --only metrics --dataset "$DATASET" --force

echo "=== Step 6: Cox models ==="
python 13_cox.py --dataset "$DATASET"

echo "=== Step 7: Method comparison (threshold-free + budget-matched) ==="
python 16_benchmark.py --only comparison --dataset "$DATASET"

echo "=== Step 8: Forecasting evaluation ==="
python 05_forecasting.py --dataset "$DATASET" --workers 8

echo "=== Step 9: Abnormal-burden stratification ==="
python 16_benchmark.py --only burden --dataset "$DATASET"

echo "=== Step 10: Significance testing vs each comparator (R2.2) ==="
python 16_benchmark.py --only significance --dataset "$DATASET"

echo ""
echo "NORMA time-unit fix rerun complete for $DATASET."
