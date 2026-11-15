"""Baseline model construction, training, and persistence.

Two baselines are provided:

1. ``DummyClassifier`` (most-frequent strategy) as the reference floor.
2. ``RandomForestClassifier`` on acoustic features inside a scikit-learn
   ``Pipeline`` so any learned transformation is fit on training data only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from respiraqa import SUPPORTED_CLASSES
from respiraqa.utils.reproducibility import get_environment_versions

MODEL_CHOICES = ("dummy", "random_forest")


class ModelError(ValueError):
    """Raised for unsupported model choices or feature-order mismatches."""


def build_model(
    model_type: str = "random_forest",
    random_state: int = 42,
    n_estimators: int = 200,
) -> Pipeline:
    """Build a scikit-learn Pipeline for the requested model type.

    Args:
        model_type: ``dummy`` or ``random_forest``.
        random_state: Seed passed to the estimator.
        n_estimators: Tree count for the random forest.

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        ModelError: For an unsupported model type.
    """
    if model_type == "dummy":
        estimator = DummyClassifier(strategy="most_frequent")
        return Pipeline([("classifier", estimator)])
    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
    raise ModelError(f"Unsupported model type '{model_type}'; expected one of {MODEL_CHOICES}")


def train_model(
    model: Pipeline,
    features: Any,
    labels: Iterable[str],
    feature_names: list[str],
) -> Pipeline:
    """Fit a pipeline on the training split only.

    Args:
        model: Pipeline from :func:`build_model`.
        features: 2-D array/DataFrame of shape (n_samples, n_features) whose
            columns match ``feature_names`` order.
        labels: Class-label strings for each row.
        feature_names: Ordered feature names; validated before fitting.

    Returns:
        The fitted pipeline.
    """
    matrix = _to_feature_matrix(features, feature_names)
    targets = list(labels)
    if len(targets) != matrix.shape[0]:
        raise ModelError(
            f"Feature/label count mismatch: {matrix.shape[0]} features vs {len(targets)} labels"
        )
    unknown = set(targets) - set(SUPPORTED_CLASSES)
    if unknown:
        raise ModelError(f"Labels contain unsupported classes: {sorted(unknown)}")

    model.fit(matrix, targets)
    return model


def _to_feature_matrix(features: Any, feature_names: list[str]) -> np.ndarray:
    """Convert feature input to a validated 2-D float array."""
    if isinstance(features, pd.DataFrame):
        if list(features.columns) != list(feature_names):
            raise ModelError(
                "Feature-column order mismatch: model was built with a different "
                "feature order than the input frame"
            )
        matrix = features.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ModelError("Feature matrix must be 2-D")
    if matrix.shape[1] != len(feature_names):
        raise ModelError(
            f"Feature-count mismatch: got {matrix.shape[1]} columns, expected {len(feature_names)}"
        )
    if not np.isfinite(matrix).all():
        raise ModelError("Feature matrix contains NaN or infinite values")
    return matrix


def save_model(
    model: Pipeline,
    model_path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the fitted model and a metadata JSON sidecar.

    Metadata always records feature names, supported class names, package
    versions, and the supplied training details. Test labels are never saved.
    """
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    payload: dict[str, Any] = {
        "feature_names": getattr(model, "feature_names_in_", None) or metadata.get(
            "feature_names"
        )
        if metadata
        else None,
        "classes": list(SUPPORTED_CLASSES),
        "versions": get_environment_versions(),
    }
    if metadata:
        payload.update(metadata)
        payload["classes"] = list(SUPPORTED_CLASSES)
        payload["feature_names"] = list(
            getattr(model, "feature_names_in_", None) or metadata.get("feature_names", [])
        )

    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_model(
    model_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[Pipeline, dict[str, Any] | None]:
    """Load a model (and optional metadata) in a fresh process."""
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    metadata: dict[str, Any] | None = None
    if metadata_path is not None:
        metadata_file = Path(metadata_path)
        if metadata_file.is_file():
            with metadata_file.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
    return model, metadata
