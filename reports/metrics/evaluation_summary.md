# Evaluation Summary (test split)

> **SYNTHETIC DEMO RESULTS - NOT CLINICAL VALIDATION**
> Dataset: `tests\fixtures\synthetic`

- Cycles evaluated: **16**
- Accuracy: **0.3750**
- Macro F1: **0.2262**
- Balanced accuracy: **0.3333**

## Per-class metrics

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| normal | 0.000 | 0.000 | 0.000 | 1 |
| crackles | 0.000 | 0.000 | 0.000 | 5 |
| wheezes | 0.400 | 1.000 | 0.571 | 4 |
| both | 0.333 | 0.333 | 0.333 | 6 |

## Event metrics (challenge-style definition)

- Event sensitivity: **1.0000** (15 event cycles)
- Normal specificity: **0.0000** (1 normal cycles)
- Average score: **0.5000**
- Definition: Event-positive = crackles/wheezes/both; normal-negative = normal. Detection metric, distinct from four-class macro F1.

## Comparison with DummyClassifier

- Model accuracy 0.3750 vs dummy 0.3750
- Model macro F1 0.2262 vs dummy 0.1364

## Notes

- Probabilities reported by the model are not calibrated.
- This evaluation was run once on the patient-independent 'test' split; no tuning was performed against it.