# Real ICBHI validation — reproducible workflow (NOT EXECUTED)

**Status: no real ICBHI audio or patient-level labels have been downloaded or evaluated here.**
All committed metrics come from the committed synthetic fixture and are labelled
`SYNTHETIC DEMO - not clinical validation` in `reports/` and `models/model_metadata.json`.

## Principles

- The synthetic demo remains the offline test: `python -m respiraqa run-all --config config/config.synthetic.yaml`.
- Real ICBHI data must be obtained by you under the challenge's published terms; do not commit
  audio, annotations, patient identifiers or archives.
- Record the download date, release and licence in `docs/dataset_documentation.md` before use.
- Patient-independent validation is mandatory: the official `patient_id` -> partition mapping
  must be supplied via `dataset.split_file`; no random record-level split is acceptable.

## Expected local layout

```text
data/raw/ICBHI2017/
  audio/          # *.wav recordings
  annotations/    # *.txt event annotations
  metadata/       # patient/partition metadata
```

`config/config.yaml` is already parameterized for this layout and uses
`filter_enabled: false` / `normalization: none`, matching the feature code path actually used.
Set `dataset.split_file` to your official train/test mapping before running.

## Reproduction commands (run only after obtaining the licensed dataset)

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .

# 1. Validate the real dataset against the documented schema/labels
python scripts/01_validate_dataset.py --config config/config.yaml

# 2. Build manifest and patient-indexed cycles
python scripts/02_build_manifest.py --config config/config.yaml
python scripts/03_extract_cycles.py --config config/config.yaml

# 3. Features, patient-independent training and evaluation
python scripts/04_extract_features.py --config config/config.yaml
python scripts/05_train_baseline.py --config config/config.yaml
python scripts/06_evaluate_model.py --config config/config.yaml --split test
python scripts/07_robustness_tests.py --config config/config.yaml
```

The evaluation writes `data_provenance` into `classification_report.json` and a
`REAL-DATA RESULTS` banner into `evaluation_summary.md` when the configured dataset root is not
the synthetic fixture. Each JSON/Markdown artefact therefore states which dataset produced it.

## Verification checklist before claiming real results

- [ ] Official train/test `split_file` configured and verified patient-disjoint.
- [ ] Download date, version and licence recorded in `docs/dataset_documentation.md`.
- [ ] `classification_report.json` shows `"synthetic": false` and the real dataset root.
- [ ] Per-class support and confusion matrix reviewed; no class omitted from macro F1.
- [ ] No patient-level audio, annotations or identifiers tracked by Git.

## Current blockers

1. The licensed ICBHI dataset is not present locally; no download was performed.
2. Terms acceptance and dataset acquisition must be completed by you.
3. Therefore **no real-data accuracy, F1 or robustness result is claimed** in this repository.
