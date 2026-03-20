# Pipeline Reference

## Overview

```
RAW DATA ──► process/ ──► PROCESSED DATA ──► model/ ──► PREDICTIONS
                                                           │
                                              validation/ ◄─┘
                                                  │
                                           RAW RESULTS (CSVs)
                                                  │
                                              scripts/ ◄─┘
                                                  │
                                      ┌───────────┼───────────┐
                                   TABLES      FIGURES    MANUSCRIPT
```

---

## Stage 1: Data Processing (`process/`)

| Script | Depends On | Output |
|---|---|---|
| `process/data_process.py` | Raw EHR data in `data/raw/` | Processed measurements in `data/processed/` |
| `process/process_eicu.py` | `data/raw/` eICU files | Processed eICU data |
| `process/process_data.py` | `data/raw/` | Flexible wrapper for any dataset |
| `process/config.py` | — | Reference intervals, lab code mappings (imported by many scripts) |

---

## Stage 2: Model Training & Prediction (`model/`)

| Script | Depends On | Output |
|---|---|---|
| `model/train.py` | `data/processed/`, `model/model.py`, `model/loss.py` | Trained model checkpoints in `model/logs/` |
| `model/predict.py` | Trained model, `data/processed/` | Predictions in `model/predictions/` |
| `model/evaluate.py` | Predictions | R², MAE, MAPE metrics |
| `model/sensitivity_analysis.py` | Trained model | Sensitivity sweep CSVs |
| `baselines/baselines.py` | `data/processed/` | ARIMA/mean/last baseline predictions in `baselines/predictions/` |

---

## Stage 3: Clinical Validation (`validation/`)

Scripts run in order. Each reads from `data/` or `model/predictions/` and writes raw CSVs to `results/{dataset}/raw/`.

| Script | Depends On | Output |
|---|---|---|
| `01_process_eicu.py` | Raw eICU data | Processed patient data |
| `02_split_data.py` | Processed data | Train/test splits |
| `03_compute_refs.py` | Splits, model predictions | PopRI, PerRI, NORMA-RI per patient |
| `04_variability.py` | Processed data | `variability.csv` — CV intra/inter, individuality index |
| `05_mortality.py` | Processed data, splits | `mortality_by_quintile.csv` — mortality by analyte quintile |
| `06_classify.py` | Reference intervals | `classifications.csv` — normal/abnormal per method |
| `07_prevalence.py` | Classifications | `prevalence.csv` — abnormality rates per method/analyte |
| `08_eval_metrics.py` | Classifications, outcomes | `eval_metrics_restricted.csv` — PPV, sensitivity, specificity, accuracy |
| `09_cox_models.py` | Classifications, outcomes | `cox_results.csv` — HR, C-index per analyte/outcome |
| `10_lead_time.py` | Classifications | `lead_time.csv` — hours earlier NORMA flags vs PopRI |
| `11_age_ri.py` | Reference intervals | `age_ri.csv` — age-stratified reference bounds |

**Run all:** `bash validation/run_pipeline.sh`

---

## Stage 4: Formatting & Figures (`scripts/`)

**Single command:** `python scripts/format.py`

This runs all formatters, builds composite figures, syncs to `manuscript/`, and compiles the supplement PDF.

### Formatting scripts (`scripts/helpers/`)

Convert raw CSVs from Stage 3 into publication-ready LaTeX tables and matplotlib figures.

| Script | Reads From | Writes To |
|---|---|---|
| `helpers/format_cohort.py` | `results/{dataset}/raw/cohort_*.csv` | `results/tables/{csv,latex}/` |
| `helpers/format_prediction.py` | `results/prediction/raw/*.csv` (from model eval) | `results/prediction/{tables,figures}/` |
| `helpers/format_validation.py` | `results/{dataset}/raw/*.csv` (from validation) | `results/{dataset}/{tables,figures}/`, `results/validation/{tables,figures}/` |

### Figure scripts (`scripts/`)

| Script | Reads From | Writes To |
|---|---|---|
| `make_figures.py` | Individual plots in `results/*/figures/` | Composite PDFs in `results/manuscript_figures/` |
| `schematic_architecture.py` | — (generates from code) | `results/manuscript_figures/figure3a_architecture.pdf` |
| `schematic_cohort.py` | — (generates from code) | `results/manuscript_figures/figure2a_cohort.pdf` |
| `schematic_token_example.py` | — (generates from code) | `results/prediction/figures/token_example.pdf` |

### Shared utilities

| File | Purpose |
|---|---|
| `plots.py` | Matplotlib styling (Work Sans, Set2, PDF), color schemes, `save_fig()`, `save_table()`, `compile_latex()` |
| `cohort_summary.py` | Compute cohort demographics CSVs (run before format_cohort) |

---

## Stage 5: Manuscript Sync

`format.py` automatically syncs outputs to `manuscript/` for Overleaf:

```
manuscript/
├── manuscript.tex          # main text
├── figures.tex             # main figures (1-5)
├── supp_figures.tex        # supplemental figures (S1-S6)
├── tables.tex              # supplemental tables
├── supplement.tex          # standalone compilable supplement
├── figures/                # all figure PDFs (synced from results/)
└── tables/                 # all table .tex files (synced from results/)
```

The sync mapping is defined in `FIGURE_MAP` and `TABLE_MAP` in `format.py`.

---

## Output Reference

### Tables in supplement

| Table | Source Script | Raw CSV |
|---|---|---|
| Analyte Reference | `format_cohort.py` | `process/config.py` (reference intervals) |
| Cohort Summary | `format_cohort.py` | `results/{dataset}/raw/cohort_summary.csv` |
| Design Choices | `format_cohort.py` | `results/tables/csv/design_choices.csv` |
| Prediction Performance | `format_prediction.py` | `results/prediction/raw/forecasting_overall.csv` |
| Analyte Performance | `format_prediction.py` | `results/prediction/raw/forecasting_by_analyte.csv` |
| Sensitivity Summary | `format_prediction.py` | `results/prediction/raw/sensitivity.csv` |
| Sensitivity (per model) | `format_prediction.py` | `results/prediction/raw/sensitivity.csv` |
| Variability | `format_validation.py` | `results/{dataset}/raw/variability.csv` |
| Prevalence | `format_validation.py` | `results/{dataset}/raw/prevalence.csv` |
| Eval Summary | `format_validation.py` | `results/{dataset}/raw/eval_metrics_restricted.csv` |
| Eval per-outcome | `format_validation.py` | `results/{dataset}/raw/eval_metrics_restricted.csv` |
| Lead Time | `format_validation.py` | `results/eicu/raw/lead_time.csv` |

### Figures in supplement

| Figure | Source Script | Data Source |
|---|---|---|
| Figure 1 (Overview) | Extracted from external PDF | — |
| Figure 2 (Variability) | `make_figures.py` | variability + mortality plots |
| Figure 3 (Architecture) | `make_figures.py` | prediction performance plots |
| Figure 4 (CHS) | `make_figures.py` | CHS validation plots |
| Figure 5 (eICU) | `make_figures.py` | eICU validation plots |
| S1: Token Example | `schematic_token_example.py` | — |
| S2: Gaussian Performance | `make_figures.py` | Gaussian prediction plots |
| S3: Prevalence (combined) | `format_validation.py` | prevalence.csv |
| S4: Cox C-index | `format_validation.py` | cox_results.csv |
| S5: Mortality by Deviation | `format_validation.py` | mortality data |
| S6: Age-dependent RI | `format_validation.py` | age_ri.csv |

---

## Quick Start

```bash
# Run the full validation pipeline (Stage 3)
bash validation/run_pipeline.sh

# Generate all tables, figures, sync to manuscript, compile supplement (Stages 4-5)
python scripts/format.py

# Or run individual formatters
python scripts/format.py cohort
python scripts/format.py prediction
python scripts/format.py validation
```
