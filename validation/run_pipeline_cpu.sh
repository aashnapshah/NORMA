#!/bin/bash
#SBATCH --job-name=NORMA_CPU
#SBATCH --output=logs/%j.log
#SBATCH --mem=64G
#SBATCH -t 11:59:00
#SBATCH -p short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu

module load gcc/9.2.0

RUN_IDS="334f7e21 167f05e8"
DEVICE="cpu"

# echo ">>> Step 1: Processing raw eICU data..."
# python 01_process_eicu.py
# echo ""

# echo ">>> Step 2: Splitting data..."
# python 02_split_data.py --baseline_pct 0.75 --min_baseline_count 5 --min_baseline_days 14
# echo ""

echo ">>> Step 3: Computing reference intervals..."
python 03_compute_refs.py --run_ids $RUN_IDS --device $DEVICE
echo ""

echo ">>> Step 4: Classification..."
python 04_classify.py --run_ids $RUN_IDS --device $DEVICE
echo ""

echo ">>> Step 5: Cox models..."
python 05_cox_models.py
echo ""

echo ">>> Step 6: Eval metrics..."
python 06_eval_metrics.py
echo ""

echo ">>> Step 7: Prevalence plots..."
python 07_plot_prevalence.py
echo ""

echo ">>> Step 8: Variability..."
python 08_variability.py
echo ""

echo ">>> Step 9: Summary stats..."
python 09_summary_stats.py
echo ""

echo "Done!"
