# Limitations

## Dataset and annotation limitations

- The ICBHI 2017 database is modest in size and imbalanced across the
  four cycle classes; per-class metrics are correspondingly noisy.
- Annotations were produced by human experts; labeling respiratory cycles
  is inherently subjective, and inter-annotator agreement is imperfect.
- Recording devices, sample rates, and acoustic environments vary widely
  across the database; features tuned to one device may not transfer.
- Class labels describe sounds heard in a cycle, not disease status.
  Conflating crackle/wheeze presence with diagnosis is an error.

## Split and evaluation limitations

- With few patients per class, patient-independent evaluation on a single
  held-out partition has high variance. Results should be read as a
  baseline demonstration, not a performance claim.
- The synthetic demo run (`config/config.synthetic.yaml`) uses 8 synthetic
  recordings and 4 synthetic patients; its numbers say nothing about real
  respiratory-sound classification performance.

## Model limitations

- The Random Forest on hand-crafted acoustic features is a deliberately
  simple baseline. Published deep-learning systems report substantially
  higher scores on the same database.
- Model probabilities are not calibrated; confidence values are indicative
  only.
- No failure-mode analysis beyond the confusion matrix and the robustness
  suite has been performed yet.

## Robustness-testing limitations

- The robustness suite applies controlled signal corruptions (noise,
  amplitude changes, truncation). Passing/failing these probes is **not**
  clinical validation and says nothing about performance on real patients,
  real devices, or real clinical noise.
- Robustness results are collected on development data and must never be
  used to tune against the official test split.

## Safety

RespiraQA is an educational/research project. It is not a medical device,
is not clinically validated, and must not be used for diagnosis or
treatment decisions.
