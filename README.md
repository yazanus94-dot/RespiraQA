# RespiraQA

### Respiratory Sound Classification & AI Quality Assurance

<!-- After the first push, enable: [![Tests](https://github.com/YOUR_USERNAME/RespiraQA/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/RespiraQA/actions) -->

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Overview

RespiraQA is an educational biomedical engineering and software QA project
for respiratory audio preprocessing, annotated respiratory-cycle
classification, machine-learning evaluation, and robustness testing. It is
built as a reproducible pipeline: every result in `reports/` is generated
by committed code from a configuration file.

## Research Problem

Respiratory recordings vary in length, sampling rate, recording device,
noise, and acoustic characteristics. This project studies a reproducible
processing and evaluation pipeline for annotated respiratory sound cycles,
with software-quality-assurance practices (validation reports, unit tests,
CI, robustness probes) applied throughout.

## Objectives

- Validate respiratory audio and annotations.
- Extract annotated respiratory cycles.
- Preprocess and visualize acoustic signals.
- Extract interpretable acoustic features.
- Train and compare baseline classifiers.
- Evaluate performance with patient-independent splits.
- Test software behavior and model robustness.
- Produce reproducible reports through automated QA.

## Dataset

ICBHI 2017 Respiratory Sound Database.
Official source: https://bhichallenge.med.auth.gr/

The dataset must be downloaded separately and placed under
`data/raw/ICBHI2017/` following `docs/dataset_documentation.md`. Raw audio
and annotations are **never committed** to this repository; `data/raw/`,
`data/interim/`, and `data/processed/` are git-ignored. A fully synthetic,
deterministic fixture (`tests/fixtures/synthetic/`) with no real patient
data is committed for tests and for the demo pipeline.

## Architecture

See `docs/architecture.md` for the full data-flow diagram. In short:

```text
audio + annotations -> validation -> manifest -> cycles -> preprocessing
-> features -> patient-independent split -> training -> evaluation
-> metrics/plots + robustness tests -> reproducible reports -> CI
```

## Technology Stack

Python, NumPy, SciPy, Librosa, SoundFile, Pandas, Scikit-learn,
Matplotlib, Joblib, PyYAML, Pytest, Ruff, and GitHub Actions.

## Installation

```powershell
git clone https://github.com/YOUR_USERNAME/RespiraQA.git
cd RespiraQA

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

(Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.)

## Usage

### Run the full pipeline on the synthetic demo (no download needed)

```powershell
python -m respiraqa run-all --config config/config.synthetic.yaml
```

This runs all seven stages — validation, manifest, cycles, features,
training, evaluation, robustness — and writes everything under `reports/`,
`data/interim/`, `data/processed/`, and `models/`.

### Run the pipeline on the real ICBHI dataset

```powershell
# 1. Download the ICBHI 2017 database from https://bhichallenge.med.auth.gr/
#    and organize it as described in docs/dataset_documentation.md.
# 2. Point config/config.yaml dataset paths at your local layout and add
#    the official split file path (dataset.split_file).
# 3. Run each stage (or use the orchestrator):

python scripts/01_validate_dataset.py --config config/config.yaml
python scripts/02_build_manifest.py --config config/config.yaml
python scripts/03_extract_cycles.py --config config/config.yaml
python scripts/04_extract_features.py --config config/config.yaml
python scripts/05_train_baseline.py --config config/config.yaml
python scripts/06_evaluate_model.py --config config/config.yaml
python scripts/07_robustness_tests.py --config config/config.yaml
```

### Tests and lint

```powershell
pytest -v
pytest --cov=respiraqa --cov-report=term-missing
ruff check .
```

## Results

Results below come from the committed implementation running on the
**synthetic demo fixture** (8 synthetic recordings, 4 synthetic patients,
32 annotated cycles; patient-independent split). They demonstrate that the
pipeline works end-to-end; they are **not** performance results on real
respiratory data.

| Metric (synthetic test split) | RandomForest | DummyClassifier |
|---|---|---|
| Accuracy | 0.3750 | 0.3750 |
| Macro F1 | 0.2262 | 0.1364 |
| Balanced accuracy | 0.3333 | — |
| Event sensitivity | 1.0000 | — |
| Average score | 0.5000 | — |

Class distribution over all 32 synthetic cycles: both=12, crackles=10,
wheezes=8, normal=2 (heavily imbalanced by fixture design). Robustness
probes on development cycles showed no degradation under amplitude
changes, −0.19 accuracy under 10 dB noise, and no failures or crashes in
any condition (see `reports/robustness/`).

To generate real results, run the pipeline on the ICBHI database as
described under Usage; all reports and figures are regenerated from the
actual run.

## Software Quality Assurance

- **Unit and integration tests:** 57 tests covering validation rules,
  audio loading, preprocessing, feature sanity, model behavior, metric
  math, split-integrity/leakage checks, and a synthetic end-to-end run.
- **CI:** GitHub Actions runs Ruff + the full test suite on Python 3.11
  for every push/PR to `main`, using only committed synthetic fixtures.
- **Robustness tests:** controlled corruption experiments (noise at
  documented SNRs, amplitude changes, truncation) reported with baseline
  deltas in `reports/robustness/`.

## Limitations

See `docs/limitations.md`. Highlights: small/imbalanced database,
subjective annotations, high variance of single-split evaluation with few
patients, uncalibrated probabilities, and the fact that controlled signal
corruption is not clinical validation.

## Future Work

- Compare additional classical models (gradient boosting, SVM).
- Evaluate a CNN or other audio model on spectrograms.
- Add explainability (feature importance, error analysis by patient).
- Evaluate on an additional appropriately licensed dataset.
- Build an optional interactive visualization interface.

## Medical Disclaimer

This project is for education and research. It is not a validated medical
device and must not be used for diagnosis or treatment decisions.

## References

- ICBHI Challenge — https://bhichallenge.med.auth.gr/
- Rocha et al., ICBHI 2017 database paper — https://eden.dei.uc.pt/~ruipedro/publications/Conferences/ICBHI2017a.pdf
- Technical references: `docs/research_references.md`

## Author

Yazan Aljabari — Biomedical Engineering | Software QA | AI Evaluation
