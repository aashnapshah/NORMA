#!/bin/bash
#SBATCH --job-name=NORMA_VALIDATE
#SBATCH --output=logs/%j.log
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu
#
# NORMA Validation Pipeline
#
# Usage:
#   bash run_pipeline.sh                                    # run all steps
#   bash run_pipeline.sh --steps 5 6 7 10                   # run specific steps
#   bash run_pipeline.sh --steps 5 6 --recompute 5 6        # recompute steps 5 & 6
#   bash run_pipeline.sh --run_ids 334f7e21 167f05e8        # specify NORMA models
#   bash run_pipeline.sh --skip_process                     # skip step 1
#
# Steps:
#    1  Process eICU          (01_process_eicu.py)
#    2  Split data            (02_split_data.py)
#    3  Compute refs          (03_compute_refs.py)
#    4  Classify              (04_classify.py)
#    5  Cox models            (05_cox_models.py)
#    6  Eval metrics          (06_eval_metrics.py)
#    7  Plot prevalence       (07_plot_prevalence.py)
#    8  Variability           (08_variability.py)
#    9  Summary stats         (09_summary_stats.py)
#   10  Paper analyses        (paper_analyses.py)

set -e

# Modules
module load gcc/9.2.0 2>/dev/null || true
module load cuda/11.7 2>/dev/null || true

# ── Defaults ────────────────────────────────────────────────────
RUN_IDS=""
N_PER_CODE=""
DEVICE="cpu"
BASELINE_PCT="0.75"
MIN_BASELINE_COUNT="5"
MIN_BASELINE_DAYS="14"
SKIP_PROCESS=false
STEPS=()
RECOMPUTE=()

# ── Parse args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                RUN_IDS="$RUN_IDS $1"; shift
            done ;;
        --steps)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STEPS+=("$1"); shift
            done ;;
        --recompute)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                RECOMPUTE+=("$1"); shift
            done ;;
        --n_per_code)       N_PER_CODE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --baseline_pct)     BASELINE_PCT="$2"; shift 2 ;;
        --min_baseline_count) MIN_BASELINE_COUNT="$2"; shift 2 ;;
        --min_baseline_days)  MIN_BASELINE_DAYS="$2"; shift 2 ;;
        --skip_process)     SKIP_PROCESS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default: all steps (skip 1 if --skip_process)
if [ ${#STEPS[@]} -eq 0 ]; then
    if [ "$SKIP_PROCESS" = true ]; then
        STEPS=(2 3 4 5 6 7 8 9 10)
    else
        STEPS=(1 2 3 4 5 6 7 8 9 10)
    fi
fi

cd "$(dirname "$0")"
mkdir -p logs results/eicu figures

# ── Recompute: delete cached outputs ────────────────────────────
delete_if_exists() {
    for f in "$@"; do
        if [ -f "$f" ]; then
            echo "  Removing $f (recompute)"
            rm "$f"
        fi
    done
}

for step in "${RECOMPUTE[@]}"; do
    case "$step" in
        1)  delete_if_exists data/eicu_processed.pkl ;;
        2)  delete_if_exists data/split_df.pkl results/attrition_table.csv ;;
        3)  delete_if_exists data/ref_intervals.csv ;;
        4)  delete_if_exists results/classification_detail.pkl results/classification_detail.csv ;;
        5)  delete_if_exists results/cox_results.csv results/cox_results_normal_setpoint.csv ;;
        6)  delete_if_exists results/eval_metrics.csv results/eval_metrics_pop_restricted.csv results/eval_metrics_pop_restricted_normal_setpoint.csv ;;
        7)  delete_if_exists figures/abnormality_prevalence.pdf ;;
        8)  delete_if_exists results/variability.csv ;;
        9)  delete_if_exists results/summary_stats_table2.csv results/quintile_mortality.csv ;;
        10) delete_if_exists results/eicu/report.txt results/eicu/lead_time_by_lab.csv results/eicu/discordance_by_lab.csv results/eicu/nns_by_outcome.csv ;;
    esac
done

# ── Info ────────────────────────────────────────────────────────
echo "============================================================"
echo "NORMA Validation Pipeline"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-interactive}"
echo "  Device:    $DEVICE"
echo "  Run IDs:   ${RUN_IDS:-none}"
echo "  Steps:     ${STEPS[*]}"
echo "  Recompute: ${RECOMPUTE[*]:-none}"
echo ""

# ── Helper ──────────────────────────────────────────────────────
should_run() {
    local target="$1"
    for s in "${STEPS[@]}"; do
        [[ "$s" == "$target" ]] && return 0
    done
    return 1
}

# ── Pipeline ────────────────────────────────────────────────────

if should_run 1; then
    echo ">>> Step 1: Processing raw eICU data..."
    python3 01_process_eicu.py
    echo ""
fi

if should_run 2; then
    echo ">>> Step 2: Splitting data..."
    SPLIT_ARGS="--baseline_pct $BASELINE_PCT --min_baseline_count $MIN_BASELINE_COUNT --min_baseline_days $MIN_BASELINE_DAYS"
    [ -n "$N_PER_CODE" ] && SPLIT_ARGS="$SPLIT_ARGS --n_per_code $N_PER_CODE"
    python3 02_split_data.py $SPLIT_ARGS
    echo ""
fi

if should_run 3; then
    echo ">>> Step 3: Computing reference intervals..."
    REF_ARGS="--device $DEVICE"
    [ -n "$RUN_IDS" ] && REF_ARGS="$REF_ARGS --run_ids $RUN_IDS"
    python3 03_compute_refs.py $REF_ARGS
    echo ""
fi

if should_run 4; then
    echo ">>> Step 4: Classifying index measurements..."
    CLS_ARGS="--device $DEVICE"
    [ -n "$RUN_IDS" ] && CLS_ARGS="$CLS_ARGS --run_ids $RUN_IDS"
    python3 04_classify.py $CLS_ARGS
    echo ""
fi

if should_run 5; then
    echo ">>> Step 5: Cox models..."
    python3 05_cox_models.py --time_windows
    echo ""
fi

if should_run 6; then
    echo ">>> Step 6: Eval metrics..."
    python3 06_eval_metrics.py --time_windows
    echo ""
fi

if should_run 7; then
    echo ">>> Step 7: Plotting prevalence..."
    python3 07_plot_prevalence.py
    echo ""
fi

if should_run 8; then
    echo ">>> Step 8: Variability..."
    python3 08_variability.py
    echo ""
fi

if should_run 9; then
    echo ">>> Step 9: Summary stats..."
    python3 09_summary_stats.py
    echo ""
fi

if should_run 10; then
    echo ">>> Step 10: Paper analyses (lead time, discordance, NNS)..."
    NORMA_RUN="${RUN_IDS%% *}"  # first run ID
    NORMA_RUN="${NORMA_RUN:-334f7e21}"
    python3 paper_analyses.py --dataset eicu --norma_run "$NORMA_RUN"
    echo ""
fi

echo "============================================================"
echo "Pipeline complete!"
echo "Results: results/"
echo "Figures: figures/"
echo "============================================================"
