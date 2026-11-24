# Testing Strategy

## Principles

- Tests must be deterministic and fast: no network, no full ICBHI download.
- Synthetic fixtures (`tests/fixtures_helpers.py`,
  `tests/fixtures/synthetic/`) stand in for real data.
- Failures and edge cases are tested, not just the happy path.

## Test suite

| Test file | Coverage |
|---|---|
| `test_annotations.py` | Valid rows, missing columns, invalid binary labels, invalid time intervals, non-numeric rows, beyond-duration intervals, empty files |
| `test_audio_loader.py` | Valid WAV, missing file, unreadable file, empty file, multichannel mono conversion, true resampling (length changes), invalid target rate |
| `test_preprocessing.py` | Band-pass output/attenuation, invalid cutoffs vs Nyquist, empty/non-finite input, all normalization modes, cycle extraction valid/invalid/too-short |
| `test_features.py` | Feature schema shape, no NaN/Inf, determinism, short/empty/non-finite input rejection, class-discriminative sanity check |
| `test_model.py` | Fit/predict, supported classes only, feature-order mismatch rejection, NaN feature rejection, artifact + metadata roundtrip |
| `test_metrics.py` | Metric math on known small examples, event-metric definition, distributions, error cases |
| `test_pipeline.py` | Manifest uniqueness, patient-split integrity + leakage detection, cycle table validity, end-to-end synthetic run (manifest → cycles → features → train → evaluate) |

## Commands

```powershell
pytest -v
pytest --cov=respiraqa --cov-report=term-missing
ruff check .
```

Coverage targets meaningful logic (validation rules, split integrity,
feature sanity, metric math) rather than a percentage goal.

## Continuous integration

`.github/workflows/ci.yml` runs Ruff and the full test suite on Python
3.12 for pushes and pull requests to `main`. Python 3.12 is required by the
pinned NumPy, SciPy and librosa versions. CI never needs the real
dataset: all checks use the committed synthetic fixtures.
