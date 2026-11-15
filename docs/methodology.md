# Methodology

## Data handling

1. Audio and annotation files are matched by recording basename (stem).
2. Patient IDs are derived from the ICBHI recording naming convention
   (leading digits before the first underscore, e.g. `101_1b1_Al_sc_Litt3200` -> `101`).
3. The official patient-independent partition is read from a configured
   split file (`patient_id <partition>` per line). Integrity is verified:
   a patient may never appear in more than one partition.
4. Respiratory cycles are extracted per annotation row using rounded
   sample indices; cycles beyond the recording duration or below the
   minimum length are skipped and reported.

## Signal preprocessing

- Loading via `soundfile` to float64; channels averaged to mono with the
  original channel count preserved in metadata.
- Resampling uses `librosa.resample` (actual signal resampling, never a
  re-label of the sample-rate metadata).
- Optional Butterworth band-pass (zero-phase, `sosfiltfilt`) with cutoffs
  validated against the Nyquist frequency of each recording's actual rate.
- Optional amplitude normalization (`peak`, `zscore`, `minmax`).

Every preprocessing setting is recorded in the configuration and in the
model metadata JSON. The default settings are starting points, not proven
optimal choices; comparisons should be made experimentally.

## Feature extraction

Per annotated cycle, computed at the cycle's native sample rate:

- 13 MFCC coefficients; mean and standard deviation over frames
  (26 columns). Frame/hop sizes come from configuration and are clamped to
  the cycle length.
- RMS energy (mean frame level)
- Zero-crossing rate
- Spectral centroid, bandwidth, rolloff (mean over frames)
- Spectral entropy of the normalized power spectrum

All features are deterministic for identical input/settings, and the
extraction pipeline asserts that no NaN/Inf values are produced; cycles
that fail are skipped and reported, not silently imputed.

## Modeling

- **DummyClassifier** (most-frequent) as the performance floor.
- **RandomForestClassifier** (200 trees, `class_weight="balanced_subsample"`,
  seeded) on standardized features inside a `Pipeline`; the scaler is fit
  on training data only.
- A patient-grouped, stratified validation split (`StratifiedGroupKFold`)
  carved from training patients supports tuning; the official test split
  is evaluated exactly once after development is complete.

## Evaluation metrics

- Accuracy, macro F1, balanced accuracy, per-class precision/recall/F1.
- Challenge-style event metrics with an explicit definition: cycles
  annotated with crackles/wheezes/both are event-positive; `normal`
  cycles are normal-negative. `average_score = (sensitivity + specificity)/2`.
  This detection metric is reported separately from four-class macro F1.
- Reported model probabilities are not calibrated and are flagged as such.

## Robustness protocol

Controlled corruption experiments (additive noise at documented SNRs,
amplitude scaling, truncation) are run on development (train/validation)
cycles only. Results record the reference-condition baseline, the metric
under each condition, and deltas. These experiments characterize software
and model behavior under signal changes; they are not clinical validation
and are never used to tune against the official test split.
