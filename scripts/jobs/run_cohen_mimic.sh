#!/usr/bin/env bash
#SBATCH --job-name=cohen_mimic
#SBATCH --output=jobs/logs/cohen_mimic_%j.log
#SBATCH --mem=180G
#SBATCH -c 8
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=END,FAIL
#
# Retrain the Cohen benchmark on BOTH dev cohorts (MIMIC-IV + EHRSHOT) so its
# training data matches NORMA's, and so the analytes MIMIC carries in bulk
# (A1C, LDL, TC, HDL, TGL, LDH, CRP) are no longer skipped for want of training
# pairs. EHRSHOT alone left 4 analytes untrained even at within-norm >= 0.8.
#
# Step 1 builds the healthy-cohort event tables for MIMIC (ICD/med/pregnancy/
# hospitalisation windows); EHRSHOT's are already cached.
#
# Usage: sbatch jobs/run_cohen_mimic.sh

module load gcc/14.2.0
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate normal
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== Step 1: build MIMIC healthy-filter event tables ==="
python model/baselines/cohen_healthy.py --build --sources mimiciv

echo ""
echo "=== Step 2: attrition — how much of MIMIC survives the healthy filter? ==="
python - <<'PYEOF'
import os, pandas as pd
CACHE = "validation/artifacts/cohen_healthy"
for src in ("ehrshot", "mimiciv"):
    have = [f for f in os.listdir(CACHE) if f.startswith(src)]
    print(f"  {src}: {sorted(have)}")
    for f in sorted(have):
        n = len(pd.read_parquet(os.path.join(CACHE, f)))
        print(f"      {f}: {n:,} rows")
PYEOF

echo ""
echo "Event tables built. Retrain with:"
echo "  sbatch jobs/run_cohen_refs.sh eicu retrain   (after setting --cohen_train_sources)"
