"""Classification metrics computed from actual predictions.

Includes standard metrics (accuracy, macro F1, balanced accuracy,
per-class precision/recall/F1) and a challenge-style event metric in
which non-normal cycles are event-positive and normal cycles are
normal-negative. The event metric counts detection of annotated sounds;
it is distinct from four-class macro F1 and the two must not be mixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from respiraqa import SUPPORTED_CLASSES


def compute_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float] | None = None,
) -> dict[str, Any]:
    """Compute the full evaluation metric set.

    Args:
        true_labels: Ground-truth class labels.
        predicted_labels: Predicted class labels.
        confidences: Optional predicted-class probabilities (not calibrated).

    Returns:
        Dict with accuracy, macro F1, balanced accuracy, per-class report,
        class distribution, event sensitivity/specificity, average score,
        and evaluated cycle counts.

    Raises:
        ValueError: If inputs are empty or lengths differ.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    if not true_labels:
        raise ValueError("Metric inputs are empty")

    classes = list(SUPPORTED_CLASSES)

    accuracy = float(accuracy_score(true_labels, predicted_labels))
    macro_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=classes,
            average="macro",
            zero_division=0,
        )
    )
    balanced_accuracy = float(balanced_accuracy_score(true_labels, predicted_labels))

    precision, recall, f1, support = precision_recall_fscore_support(        true_labels, predicted_labels, labels=classes, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    }

    class_distribution = {
        name: int(list(true_labels).count(name)) for name in SUPPORTED_CLASSES
    }

    event_metrics = compute_event_metrics(true_labels, predicted_labels)

    payload: dict[str, Any] = {
        "n_cycles": len(true_labels),
        "classes": classes,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
        "class_distribution": class_distribution,
        "event_metrics": event_metrics,
        "probabilities_calibrated": False,
        "note": (
            "Probabilities are reported as produced by the model and are not "
            "calibrated; treat confidence values with care."
        ),
    }
    if confidences is not None:
        payload["mean_confidence"] = float(np.mean(confidences)) if confidences else None
    return payload


def accuracy_macro_f1(
    true_labels: list[str],
    predicted_labels: list[str],
) -> tuple[float, float]:
    """Convenience pair (accuracy, macro F1) used by the robustness suite."""
    if len(true_labels) != len(predicted_labels) or not true_labels:
        raise ValueError("Metric inputs are empty or lengths differ")
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    macro_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=list(SUPPORTED_CLASSES),
            average="macro",
            zero_division=0,
        )
    )
    return accuracy, macro_f1


def compute_event_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, Any]:
    """Challenge-style event sensitivity/specificity.

    Definition used in this project:

    - Event-positive: any annotated cycle with crackles, wheezes, or both.
    - Normal-negative: ``normal`` cycles.

    ``sensitivity`` = fraction of event-positive cycles detected as
    event-positive; ``specificity`` = fraction of normal cycles detected as
    normal. ``average_score = (sensitivity + specificity) / 2``.
    """
    true_positive_flags = np.array([label != "normal" for label in true_labels], dtype=bool)
    pred_positive_flags = np.array([label != "normal" for label in predicted_labels], dtype=bool)

    n_events = int(true_positive_flags.sum())
    n_normals = int((~true_positive_flags).sum())

    event_hits = int((true_positive_flags & pred_positive_flags).sum())
    normal_hits = int((~true_positive_flags & ~pred_positive_flags).sum())

    sensitivity = event_hits / n_events if n_events else 0.0
    specificity = normal_hits / n_normals if n_normals else 0.0
    return {
        "definition": (
            "Event-positive = crackles/wheezes/both; normal-negative = normal. "
            "Detection metric, distinct from four-class macro F1."
        ),
        "n_event_cycles": n_events,
        "n_normal_cycles": n_normals,
        "event_sensitivity": float(sensitivity),
        "normal_specificity": float(specificity),
        "average_score": float((sensitivity + specificity) / 2.0),
    }


def classification_report_dict(
    true_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, Any]:
    """Return sklearn's classification report as a dict keyed by class."""
    return classification_report(
        true_labels,
        predicted_labels,
        labels=list(SUPPORTED_CLASSES),
        zero_division=0,
        output_dict=True,
    )
