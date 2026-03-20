Below is a complete, structured Markdown document covering:

* Home Page
* About NORMA (including Training & Evaluation + Sensitivity Analysis)
* Try NORMA Page

This is written to appeal simultaneously to academics, clinicians, and venture investors.

---

# NORMA Website Content (Full Draft)

---

# HOME PAGE

---

## Hero Section

# Redefining “Normal” in Medicine

AI-powered personalized laboratory interpretation that detects meaningful biological change earlier than static reference ranges.

**Validated on 4.5 million adults in a national health system.**

[ Request Pilot Access ]
[ See NORMA in Action ]

---

## The Problem

## Most Clinical Decisions Rely on a Flawed Assumption

Modern medicine defines “normal” using population reference intervals — typically the central 95% of values from ostensibly healthy individuals.

This assumes:

* One range fits everyone.
* Individual biology is interchangeable.
* Disease begins outside the population range.

Evidence shows the opposite.

Many biomarkers fluctuate tightly around narrow individual setpoints.
Clinically meaningful deterioration often occurs *within* the population “normal” range.

---

## The Goldilocks Problem in Lab Interpretation

### Too Cold — Population Reference Intervals

* One-size-fits-all thresholds
* Ignores individual biological baselines
* Delays detection of early disease
* Provides false reassurance

Used universally in clinical medicine.

---

### Too Hot — Pure Personalization

* Learns from sparse personal history
* May normalize undiagnosed chronic disease
* Overfits biological noise
* Increases false alarms

Explored in academic setpoint models.

---

### Just Right — NORMA

* Anchored in large-scale population biology
* Personalized to individual longitudinal dynamics
* Predicts expected next value under a healthy counterfactual state
* Improves both sensitivity and specificity
* Designed for real-world clinical deployment

---

## Built and Validated at Health-System Scale

* 2.8M longitudinal training sequences
* 34 laboratory biomarkers modeled
* 4.5M adults in external validation
* 1.7B laboratory results evaluated

NORMA is not a prototype.
It is infrastructure-grade clinical AI.

---

## Clinical Impact

## What Changes When “Normal” Is Personalized

* Earlier detection of metabolic disease
* Improved mortality risk stratification
* Reduced false reassurance from static thresholds
* Reduced unnecessary alerts from overfitting
* Designed for integration into EHR workflows

NORMA upgrades how every lab result is interpreted.

---

## Interactive Preview

Explore how NORMA interprets real longitudinal laboratory trajectories.

[ Launch Interactive Demo ]

---

## Closing Statement

NORMA is not another alert system.
It is a new interpretive layer for laboratory medicine.

[ Request Pilot Access ]

---

# ABOUT NORMA

---

# Training & Evaluation

---

## Overview

# Training at Population Scale. Evaluated in the Real World.

NORMA is a contextual autoregressive sequence model that learns patient-specific laboratory dynamics while remaining anchored to large-scale population structure.

It reframes lab interpretation as a counterfactual prediction problem:

> Given a patient’s history, what value would be expected at the next measurement under a healthy state?

---

## Data Sources

### Training Cohorts

* EHRSHOT (Stanford Medicine)
* MIMIC-IV (Beth Israel Deaconess Medical Center)

Each training example consists of a patient’s ordered sequence of measurements for a single laboratory test.
The model observes all but the final value and learns to predict the next measurement as a probabilistic distribution.

Data characteristics:

* Adult patients
* Irregular sampling intervals
* Real-world ordering patterns
* Clinical state variation

---

## Modeling Framework

### Architecture

* Autoregressive transformer
* Causal self-attention masking (no future leakage)
* Per-biomarker modeling
* State-conditioned predictions

### Objective

* Gaussian Negative Log-Likelihood
* Joint estimation of predicted mean (μ) and uncertainty (σ)

---

## Training Procedure

* Optimizer: AdamW
* Early stopping on validation loss
* Regularization to prevent overfitting
* Separate models trained per biomarker

Technical details available in expandable appendix.

---

## Predictive Performance

Evaluation metrics:

| Metric                | Result                         |
| --------------------- | ------------------------------ |
| Mean Validation R²    | 0.91                           |
| 95% Interval Coverage | Approximately nominal          |
| Calibration           | Well-calibrated across deciles |

Predicted uncertainty intervals are empirically calibrated.

---

## External Validation

NORMA was externally validated on:

**4.5 million adults in Clalit Health Services**

Validation included:

* Outpatient testing environments
* Diverse demographic representation
* Longitudinal follow-up
* Real-world disease progression

No retraining was performed during validation.

---

## Clinical Outcome Evaluation

We evaluated whether deviations defined by NORMA predict:

* Incident metabolic disease
* All-cause mortality
* Clinical deterioration

Compared against:

* Population reference interval deviations
* Purely individualized setpoint deviations

NORMA improved both sensitivity and specificity relative to static thresholds and isolated personalization.

---

## Robustness Analysis

NORMA maintains performance across:

* Age strata
* Sex
* Measurement density
* Sampling irregularity
* Chronic disease presence
* Temporal drift

Predictions remain stable and calibrated across heterogeneous clinical contexts.

---

# Sensitivity & Uncertainty Analysis

---

## How Uncertainty Adapts to Context

NORMA predicts a full probability distribution (μ, σ) for each future laboratory value.

The width of the prediction interval reflects:

* Biological variability
* Data density
* Demographic priors
* Forecast horizon

Personalization should narrow intervals when confidence is high — and widen them when uncertainty increases.

---

## Interactive Sensitivity Explorer

Users can:

### Select Biomarker

A1C, PLT, LDL, Creatinine, etc.

### Adjust Patient Characteristics

* Age
* Sex
* Observed variance
* Number of prior measurements
* Time since last measurement

### Adjust Forecast Horizon

* 1 month
* 3 months
* 6 months
* 12 months
* 24 months

---

## Observed Sensitivity Behaviors

### Age

Predicted mean may shift according to demographic priors.
Interval width adjusts modestly when history is sparse.

---

### Sex

Sex-conditioned population structure influences mean predictions in relevant biomarkers (e.g., HGB, HDL).
Personal history dominates when dense.

---

### Observed Variance

High within-person variability widens predicted intervals.
Stable trajectories narrow them.

This demonstrates adaptive personalization.

---

### Forecast Horizon

Longer prediction horizons increase uncertainty.
Interval width expands appropriately over time.

Uncertainty propagation is controlled and calibrated.

---

### Measurement Density

Sparse histories yield wider intervals.
Dense longitudinal data narrows predictions.

Model confidence increases with depth of data.

---

## Quantitative Outputs

Displayed dynamically:

* Predicted mean (μ)
* Predicted uncertainty (σ)
* 95% interval width
* Relative change in width (%)

This allows direct measurement of model stability.

---

## Interpretation

Personalization without calibration creates noise.
Calibration without personalization creates blindness.
NORMA adapts uncertainty to context.

---

# TRY NORMA

---

# See How NORMA Interprets Your Data

Upload longitudinal laboratory values to generate personalized expected ranges and deviation analysis.

---

## Data Input Options

* Upload CSV
* Paste structured values
* Use example synthetic patient

Clear formatting instructions provided.

---

## Visualization

The system displays:

* Historical measurements
* Population reference band
* Personalized baseline band
* NORMA predicted mean
* 95% predictive interval

Users can toggle:

* Population vs Personalized vs NORMA

---

## Interpretation Output

For each prediction:

* Expected value
* Confidence interval
* Deviation magnitude
* Clinical signal interpretation

Designed for clarity, not alarmism.

---

## Example Scenarios

Explore:

* Gradual metabolic drift
* Hematologic decline
* Inflammatory marker escalation

Observe how deviation appears earlier under contextual modeling.

---

## Integration Pathways

For clinicians and health systems:

* API-ready deployment
* EHR integration
* Decision support compatibility

For researchers:

* Methods transparency
* Calibration evaluation
* Robustness analysis

For investors:

* Infrastructure-layer positioning
* Scalable across health systems
* Defensible longitudinal modeling framework

---

## Call to Action

[ Request Clinical Pilot ]
[ Explore Full Evaluation ]
[ Contact Us ]

---

# Final Positioning Statement

NORMA transforms laboratory interpretation from static classification to contextual prediction.

It bridges population biology and individual trajectory modeling — at national scale.

---

If you would like, I can next provide:

* A refined typography hierarchy (H1/H2 sizing and spacing rules)
* A disciplined color palette with hex codes
* Or a more aggressive VC-facing narrative variant of this same structure
