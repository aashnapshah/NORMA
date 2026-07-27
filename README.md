# NORMA: Normal Outcome Range Modeling with Attention

Blood-based biomarkers underpin clinical diagnosis and management, yet their interpretation relies largely on fixed population reference intervals that ignore stable, intra-patient variability.
As such, population-based interpretation can mask meaningful deviation from an individual's baseline, risking delayed disease detection.
To remedy this, there have been increasing efforts to personalize blood biomarker interpretation using individual testing histories.
However, these methods may overfit to sparse data, inflating false-positive rates and unnecessary follow-up, and can also unwittingly include unrecognized or subclinical disease.
Here, we leverage nearly 2 billion longitudinal laboratory measurements from over 1.6 million individuals across North America, the Middle East, and East Asia, to show that while laboratory values are highly individual, purely personalized intervals routinely overfit, classifying up to 68% of measurements as abnormal, without corresponding associations with adverse clinical outcomes.
We then introduce NORMA, a conditional transformer-based framework that generates reference intervals by conditioning on both a patient's history and population-level data about "normal" variation.
NORMA-derived intervals achieve higher precision for predicting outcomes, including mortality, acute kidney injury, and chronic disease.
These findings caution against over-personalization in laboratory medicine and demonstrate that anchoring individual trajectories to population-level priors outperforms either approach alone.
To promote transparency, we publicly release the model, code, and an interactive user interface for accessible, individualized laboratory interpretation.

- Interactive web app: [norma-tpy0.onrender.com](https://norma-tpy0.onrender.com/)
- Model weights: [huggingface.co/aashnaps/NORMA](https://huggingface.co/aashnaps/NORMA)
- The full paper is available at [arXiv:2605.18701](https://arxiv.org/abs/2605.18701).

## 1. System requirements

Software dependencies (pinned minimums in [`requirements.txt`](requirements.txt)):

| Package | Version | Used for |
|---|---|---|
| Python | 3.9-3.11 | tested on 3.9 |
| torch | >= 2.0 | model |
| numpy | >= 1.24 | arrays |
| pandas | >= 1.5 | data handling |
| scikit-learn | >= 1.3 | splits, metrics |
| huggingface_hub | >= 0.20 | download model weights |
| flask, gunicorn | >= 3.0, >= 21.0 | interactive web app |
| matplotlib, seaborn | >= 3.7, >= 0.13 | figures |

Operating systems tested:

- macOS 15 (Darwin 24.6), Apple Silicon and Intel
- Linux (Ubuntu 22.04) — the hosted app runs here

Hardware:

- No non-standard hardware required. The model and demo run on a standard CPU.
- A CUDA GPU is optional and only speeds up full model training. Inference and the demo do not need one.

## 2. Installation guide

```bash
git clone https://github.com/aashnapshah/NORMA.git
cd NORMA
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
```

## 3. Demo

Two ways to run the demo.

### Online (no install)

Open [norma-tpy0.onrender.com](https://norma-tpy0.onrender.com/), go to **Try NORMA**, pick a lab test, and load one of the built-in example patients (or type your own history).
The app shows the NORMA interval next to the population interval and flags the value as low / normal / high.
Note: the free hosting tier sleeps when idle, so the first load can take ~30 seconds to wake.

### Terminal

Runs a small synthetic dataset ([`demo/demo_patients.csv`](demo/demo_patients.csv), 4 patients) through the public checkpoint:

```bash
cd demo
python demo.py
```

Expected output:

```
patient analyte  sex            Pop_RI          NORMA_RI  last obs
------------------------------------------------------------------
P1      HGB      F               12-16       12.11-14.56      12.9  g/dL
P2      CRE      M             0.7-1.3         0.90-1.39      1.31  mg/dL
P3      A1C      F               4-5.6         5.07-5.77       5.7  %
P4      PLT      M             150-450     140.78-323.64       215  10³/µL
```

`NORMA_RI` is the individualized 95% interval; `Pop_RI` is the fixed population range.
For example, patient P4's platelet interval tightens to 141-324 (vs. the population 150-450) given a steadily declining trajectory.

Expected run time on a normal desktop: about 10 seconds on CPU (plus a one-time checkpoint download of a few MB on the first run).

## 4. Instructions for use

### Run on your own data

`demo.py` accepts any history file with the same columns (`patient_id, sex, age, analyte, day, value`), where `day` is days since the patient's first measurement:

```bash
python demo/demo.py --input my_patients.csv       # your data
python demo/demo.py --horizon 180                 # predict 180 days out
```

Covered analytes are the keys of `REFERENCE_INTERVALS` in [`process/config.py`](process/config.py) (30+ common CBC, metabolic, liver, and lipid tests plus HbA1c).

### Run the web app locally

```bash
gunicorn app.app:app          # then open http://127.0.0.1:8000
```

### Train the model (needs processed sequence data)

Training uses longitudinal sequences derived from MIMIC-IV and EHRSHOT, which are access-restricted and not distributed here.
With processed sequences in `data/processed/`, the two parameterizations reported in the paper are:

```bash
# Gaussian parameterization
python model/train.py --model NormaLight --loss GaussianNLLLoss --output_mode gaussian --epochs 50

# Quantile parameterization
python model/train.py --model NORMA2 --loss QuantileLoss --output_mode quantile --epochs 50
```

See `python model/train.py --help` for the full list of flags (architecture, batch size, learning rate, etc.).

### Reproduce the paper results

The validation pipeline (external cohorts, outcome analyses, figures) lives in [`validation/`](validation/); see [`validation/README.md`](validation/README.md) and [`validation/PIPELINE.md`](validation/PIPELINE.md).
These scripts require the access-restricted clinical cohorts (CHS, eICU-CRD, INSPIRE) and are provided for transparency rather than one-command reproduction.

## Repository structure

```
model/         NORMA architecture, training, inference, loss functions
process/       raw EHR processing and reference-interval config
validation/    external-cohort validation pipeline (results + figures)
app/           interactive Flask web app (deployed on Render)
manuscript/    figure and table generation for the paper
demo/          small synthetic dataset + terminal demo script
requirements.txt
```

## Citation

```bibtex
@article{shah2026norma,
  title   = {Learning Normal Representations for Blood Biomarkers},
  author  = {Shah, Aashna P. and Li, Michelle M. and Lal, Yash and Cohen, Seffi and
             Antwarg, Liat F. and Sanchez, Morgan and Diao, James A. and Patel, Chirag J. and
             Reis, Ben Y. and Balicer, Ran D. and Dagan, Noa and Manrai, Arjun K.},
  journal = {arXiv preprint arXiv:2605.18701},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
