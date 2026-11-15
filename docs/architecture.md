# Architecture

## Data-flow diagram

```text
ICBHI audio + annotation files
              |
              v
      Dataset validation           scripts/01_validate_dataset.py
              |                    -> reports/metrics/dataset_validation.{json,md}
              v
       Recording manifest          scripts/02_build_manifest.py
              |                    -> data/interim/manifest.csv
              v
  Annotated respiratory cycles     scripts/03_extract_cycles.py
              |                    -> data/interim/cycles.csv
              v
      Audio preprocessing          src/respiraqa/preprocessing/*
              |
              v
      Acoustic features            scripts/04_extract_features.py
              |                    -> data/processed/features.csv
              v
  Patient-independent data split   official split file / patient grouping
              |
              v
       Model training              scripts/05_train_baseline.py
              |                    -> models/respiraqa_baseline.joblib + metadata
              v
       Model evaluation            scripts/06_evaluate_model.py
              |                    -> reports/metrics/* + reports/figures/*
        +------+------+
        |             |
        v             v
  Metrics/plots   Robustness tests   scripts/07_robustness_tests.py
        |             |              -> reports/robustness/*
        +------+------+
               |
               v
        Reproducible reports
               |
               v
        GitHub Actions QA           .github/workflows/ci.yml
```

## Stage inputs and outputs

| Stage | Input | Output |
|---|---|---|
| 1. Validation | WAV + TXT files under the configured dataset dirs | JSON + Markdown validation report; non-zero exit on critical errors |
| 2a. Manifest | Paired audio/annotation files, optional split file | One row per recording: ids, paths, sample rate, channels, duration, official partition |
| 2b. Cycles | Manifest + annotation rows | One row per valid annotated cycle: times, labels, four-class label, sample rate, partition |
| 3. Preprocessing | Cycle audio | Mono float64 signal, optional resampling, optional band-pass, optional normalization; parameters recorded in config and metadata |
| 4. Features | Preprocessed cycles | Feature table: 13 MFCC (mean/std) + RMS, ZCR, centroid, bandwidth, rolloff, spectral entropy |
| 5. Training | Feature table (train split only) | Fitted Dummy and RandomForest pipelines; model artifact + metadata JSON |
| 6. Evaluation | Test split features + model | Accuracy, per-class P/R/F1, macro F1, balanced accuracy, event metrics, confusion matrix, predictions, figures |
| 7. Robustness | Train (development) cycles + model | Per-condition accuracy/F1 deltas under noise, amplitude, truncation conditions |

## Key design decisions

- **Patient independence first.** Every split operation is grouped by patient ID; an automated check raises on any patient appearing in two partitions.
- **Configuration-driven.** All dataset paths, processing settings, and feature settings live in `config/config.yaml`; a second file (`config/config.synthetic.yaml`) drives the committed synthetic demo.
- **Validation reports distinguish errors from warnings.** Missing files, unreadable audio, and invalid annotation rows are surfaced, never silently skipped.
- **Learned transformations inside pipelines.** Feature scaling lives inside the scikit-learn `Pipeline`, fit on training data only.
- **Single evaluation on the test split.** Robustness and tuning work happens on training/validation patients only.
