# Final independent audit — RespiraQA

## A. Implementation summary

Fixed: Python 3.12 CI alignment (install failure), pytest invocation/path, macro-F1 label universe,
stereo resampling, synthetic provenance labelling, then replaced path-substring provenance with
explicit validated `dataset.provenance` fields (`dataset_id`, `source`, `version`, `license`,
optional `terms_accepted_by_owner`). Real-data evidence checks fail closed (manifest, audio/annotation
dirs, official patient split, owner terms). Split helper now raises when patients/classes are
insufficient instead of reporting training performance as validation.

Files changed: `src/respiraqa/utils/config.py`, `src/respiraqa/evaluation/metrics.py`,
`src/respiraqa/data/audio_loader.py`, `scripts/05_train_baseline.py`, `scripts/06_evaluate_model.py`,
config YAMLs, regenerated reports/model metadata, `tests/*` (including `test_provenance.py`,
`test_validation_split.py`), `docs/REAL_DATA_VALIDATION.md`, README.

## B. Verification evidence (final)

- Branch `main`; working tree clean; CI [45368939552](https://github.com/yazanus94-dot/RespiraQA/actions/runs/45368939552) success (69 tests at that commit; README corrected to 69).
- `pytest -q` → **69 passed** (latest verified run).
- `python -m respiraqa run-all --config config/config.synthetic.yaml` → all 7 stages succeed.
- Ruff check/format clean across the repository.
- Synthetic reports contain `data_provenance: synthetic` and a `SYNTHETIC DEMO - not clinical validation` banner.

## C. Scientific status

Synthetic implementation and tests: complete. Real dataset: not acquired; no real EDF/audio, no real
training, no real metrics claimed. Workflow is documented in `docs/REAL_DATA_VALIDATION.md`.

## D. Remaining blockers

Licensed ICBHI data acquisition and personal terms acceptance; official patient-partition mapping;
set `dataset.terms_accepted_by_owner: true` only after acceptance. No claim of clinical validation.
