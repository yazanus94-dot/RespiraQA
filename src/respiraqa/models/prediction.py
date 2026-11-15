"""Prediction helpers with feature-order validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from respiraqa import SUPPORTED_CLASSES
from respiraqa.models.baseline import _to_feature_matrix


def predict_with_confidence(
    model: Any,
    features: Any,
    feature_names: list[str],
) -> dict[str, Any]:
    """Predict class labels and (if available) probabilities.

    Args:
        model: Fitted sklearn estimator/pipeline.
        features: Feature matrix or DataFrame matching ``feature_names``.
        feature_names: Ordered feature names used during training.

    Returns:
        Dict with ``labels`` (list of str) and ``confidence`` (list of float
        or None when the model cannot produce probabilities). Probabilities
        from tree-based models are not calibrated and are reported as-is.
    """
    matrix = _to_feature_matrix(features, feature_names)
    predictions = model.predict(matrix)
    labels = [str(value) for value in predictions]

    unsupported = set(labels) - set(SUPPORTED_CLASSES)
    if unsupported:
        raise ValueError(f"Model produced unsupported classes: {sorted(unsupported)}")

    confidence: list[float] | None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(matrix))
            class_index = {name: i for i, name in enumerate(model.classes_)}
            confidence = [
                float(probabilities[row, class_index[label]])
                for row, label in enumerate(labels)
            ]
        except (AttributeError, KeyError, IndexError, ValueError):
            confidence = None
    else:
        confidence = None

    return {"labels": labels, "confidence": confidence}
