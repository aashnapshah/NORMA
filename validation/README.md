# Validation Pipeline

## Compute Scripts

All scripts take `--dataset eicu` or `--dataset chs` and output to `results/{dataset}/raw/`.

| # | Script | Depends on | Output | Notes |
|---|---|---|---|---|
| 01 | `01_process_eicu.py` | raw eICU data | processed eICU | eICU only |
| 02 | `02_split_data.py` | 01 | `split_df.pkl` | baseline/index split |
| 03 | `03_compute_refs.py` | 02 | `ref_intervals.csv` | PopRI, PerRI, NORMA intervals |
| 04 | `04_variability.py` | 03 | `variability.csv` | intra/inter CV, bootstrap CIs |
| 05 | `05_mortality.py` | 02 | `mortality_by_quintile.csv` | mortality by analyte value quintile |
| 06 | `06_classify.py` | 02, 03 | `classification.csv` | classify index measurements |
| 07 | `07_prevalence.py` | 06 | `prevalence.csv` | abnormality prevalence + reclassification rate |
| 08 | `08_eval_metrics.py` | 06 | `eval_metrics.csv`, `eval_metrics_restricted.csv` | PPV, sensitivity, specificity, accuracy per analyte |
| 09 | `09_cox_models.py` | 06 | `cox_results.csv` | Cox PH hazard ratios + concordance |
| 10 | `10_lead_time.py` | 06 | `lead_time.csv` | How much earlier NORMA flags vs PopRI |

## Example Run (eICU)

```bash
cd validation
python 02_split_data.py --dataset eicu
python 03_compute_refs.py --dataset eicu
python 04_variability.py --dataset eicu --bootstrap 1000
python 05_mortality.py --dataset eicu
python 06_classify.py --dataset eicu
python 07_prevalence.py --dataset eicu
python 08_eval_metrics.py --dataset eicu
python 09_cox_models.py --dataset eicu
python 10_lead_time.py --dataset eicu
```

## Formatting

One script formats all validation results (tables + figures):

```bash
python scripts/format_validation.py
```

Auto-detects which datasets have data. Outputs to:

- Per-dataset tables: `results/{dataset}/tables/`
- Combined figures: `results/validation/figures/`

| Table/Figure | Source CSV | Description |
|---|---|---|
| `variability` | `variability.csv` | CV intra/inter + index of individuality |
| `mortality` | `mortality_by_quintile.csv` | mortality rate by analyte quintile (faceted grid) |
| `prevalence` | `prevalence.csv` | abnormality prevalence + reclassification rate (RR) |

## Config

- `config.py`: `NORMA_RUN_ID` sets which NORMA model to use (default: `334f7e21`, quantile loss)
- `datasets/`: dataset definitions (eICU, CHS) with outcome columns and loading logic

## Column Conventions

All output CSVs use standardized column names:
- `analyte` (not `lab_code`)
- `patient_id` (not `uniquepid`)
- `value` (not `labresult`)
