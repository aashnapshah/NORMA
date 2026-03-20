#!/bin/bash
#SBATCH --job-name=NORMA_FULL
#SBATCH --output=logs/%j.log
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu
#
# NORMA: Full pipeline from raw data to manuscript-ready outputs.
#
# Usage:
#   bash run_all.sh                    # everything
#   bash run_all.sh --from validation  # skip processing + training
#   bash run_all.sh --from format      # just formatting + manuscript
#   bash run_all.sh --from manuscript  # just compile manuscript
#
# Stages:
#   1. process     — ETL: raw EHR → processed data
#   2. train       — Train NORMA model
#   3. predict     — Generate predictions + baselines
#   4. validation  — Clinical validation pipeline (11 steps)
#   5. format      — Tables, figures, sync to manuscript/
#   6. manuscript  — Compile supplement PDF
#

set -e
cd "$(dirname "$0")"

# ── Parse args ────────────────────────────────────────────────
FROM_STAGE="process"
DEVICE="cpu"
VALIDATION_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)         FROM_STAGE="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        *)              VALIDATION_ARGS="$VALIDATION_ARGS $1"; shift ;;
    esac
done

STAGES=(process train predict validation format manuscript)

should_run() {
    local target="$1"
    local started=false
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$FROM_STAGE" ]] && started=true
        [[ "$started" == true && "$s" == "$target" ]] && return 0
    done
    return 1
}

echo "════════════════════════════════════════════════════════════"
echo "  NORMA — Full Pipeline"
echo "════════════════════════════════════════════════════════════"
echo "  Starting from: $FROM_STAGE"
echo "  Device:        $DEVICE"
echo ""

# ── Stage 1: Data Processing ──────────────────────────────────
if should_run process; then
    echo "━━━ Stage 1: Data Processing ━━━"
    python process/data_process.py
    echo ""
fi

# ── Stage 2: Model Training ──────────────────────────────────
if should_run train; then
    echo "━━━ Stage 2: Model Training ━━━"
    echo "  Run manually — training configs vary per experiment."
    echo "  Example: python model/train.py --config model/sweep.yaml"
    echo "  Skipping (use --from predict to continue after training)."
    echo ""
fi

# ── Stage 3: Predictions + Baselines ─────────────────────────
if should_run predict; then
    echo "━━━ Stage 3: Predictions + Baselines ━━━"
    python model/predict.py
    python baselines/baselines.py
    echo ""
fi

# ── Stage 4: Clinical Validation ─────────────────────────────
if should_run validation; then
    echo "━━━ Stage 4: Clinical Validation ━━━"
    cd validation
    bash run_pipeline.sh --device "$DEVICE" $VALIDATION_ARGS
    cd ..
    echo ""
fi

# ── Stage 5: Formatting + Figures ─────────────────────────────
if should_run format; then
    echo "━━━ Stage 5: Formatting + Figures ━━━"
    python scripts/format.py
    echo ""
fi

# ── Stage 6: Compile Manuscript ──────────────────────────────
if should_run manuscript; then
    echo "━━━ Stage 6: Compile Manuscript ━━━"
    cd manuscript
    if command -v tectonic &>/dev/null; then
        tectonic supplement.tex && echo "  → supplement.pdf"
        tectonic manuscript.tex && echo "  → manuscript.pdf"
    else
        echo "  tectonic not found — install with: cargo install tectonic"
    fi
    cd ..
    echo ""
fi

echo "════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo ""
echo "  Results:    results/"
echo "  Manuscript: manuscript/"
echo "════════════════════════════════════════════════════════════"
