"""Tests for baseline model training, prediction, and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.fixtures_helpers import make_recording

from respiraqa import SUPPORTED_CLASSES
from respiraqa.features.acoustic_features import extract_features, feature_names
from respiraqa.models.baseline import (
    ModelError,
    build_model,
    load_model,
    save_model,
    train_model,
)
from respiraqa.models.prediction import predict_with_confidence


@pytest.fixture(scope="module")
def training_table():
    """Small feature table built from synthetic recordings of all classes."""
    rows = []
    for class_index, class_label in enumerate(SUPPORTED_CLASSES):
        for repeat in range(3):
            audio = make_recording(class_label, seed=class_index * 10 + repeat)
            features = extract_features(audio, 8000)
            features["class_label"] = class_label
            rows.append(features)
    frame = pd.DataFrame(rows)
    names = feature_names()
    return frame, names


def test_build_model_supported_types():
    for model_type in ("dummy", "random_forest"):
        model = build_model(model_type)
        assert model is not None
    with pytest.raises(ModelError, match="Unsupported model type"):
        build_model("neural_magic")


def test_fit_and_predict(training_table):
    frame, names = training_table
    model = build_model("random_forest", random_state=0, n_estimators=10)
    model = train_model(model, frame[names], frame["class_label"], names)
    predictions = predict_with_confidence(model, frame[names], names)
    assert len(predictions["labels"]) == len(frame)
    assert set(predictions["labels"]).issubset(set(SUPPORTED_CLASSES))


def test_model_returns_supported_class(training_table):
    frame, names = training_table
    model = build_model("dummy")
    model = train_model(model, frame[names], frame["class_label"], names)
    predictions = predict_with_confidence(model, frame[names], names)
    assert set(predictions["labels"]).issubset(set(SUPPORTED_CLASSES))


def test_feature_column_mismatch_rejected(training_table):
    frame, names = training_table
    model = build_model("random_forest", n_estimators=10)
    model = train_model(model, frame[names], frame["class_label"], names)
    shuffled = frame[names[::-1]]
    with pytest.raises(ModelError, match="mismatch"):
        predict_with_confidence(model, shuffled, names)


def test_nan_features_rejected(training_table):
    frame, names = training_table
    broken = frame[names].copy()
    broken.iloc[0, 0] = np.nan
    model = build_model("dummy")
    with pytest.raises(ModelError, match="NaN"):
        train_model(model, broken, frame["class_label"], names)


def test_model_and_metadata_roundtrip(tmp_path, training_table):
    frame, names = training_table
    model = build_model("random_forest", random_state=0, n_estimators=10)
    model = train_model(model, frame[names], frame["class_label"], names)
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(
        model,
        model_path,
        metadata_path,
        metadata={"feature_names": names, "model_type": "random_forest"},
    )
    assert model_path.is_file()
    assert metadata_path.is_file()

    loaded, metadata = load_model(model_path, metadata_path)
    predictions = predict_with_confidence(loaded, frame[names], names)
    assert set(predictions["labels"]).issubset(set(SUPPORTED_CLASSES))
    assert metadata["classes"] == list(SUPPORTED_CLASSES)
    assert metadata["feature_names"] == names
