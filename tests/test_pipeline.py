"""End-to-end pipeline test on a synthetic dataset + split integrity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from respiraqa.data.dataset import (
    DatasetError,
    build_cycle_table,
    build_manifest,
    read_split_file,
    verify_patient_split_integrity,
)
from respiraqa.evaluation.metrics import compute_metrics
from respiraqa.features.acoustic_features import extract_features_table, feature_names
from respiraqa.models.baseline import build_model, train_model
from respiraqa.models.prediction import predict_with_confidence


def test_manifest_one_row_per_recording(synthetic_dataset):
    manifest, issues = build_manifest(
        synthetic_dataset / "audio",
        synthetic_dataset / "annotations",
        synthetic_dataset / "split_file.txt",
    )
    assert len(manifest) == 8
    assert manifest["recording_id"].is_unique
    assert manifest["sample_rate"].notna().all()
    assert (manifest["official_partition"].isin(["train", "test"])).all()


def test_patient_does_not_appear_in_multiple_splits(synthetic_dataset):
    manifest, _ = build_manifest(
        synthetic_dataset / "audio",
        synthetic_dataset / "annotations",
        synthetic_dataset / "split_file.txt",
    )
    # Should not raise: each patient has exactly one partition.
    verify_patient_split_integrity(manifest)
    grouped = manifest.groupby("patient_id")["official_partition"].nunique()
    assert (grouped == 1).all()


def test_patient_leakage_is_detected():
    manifest = pd.DataFrame(
        {
            "recording_id": ["a", "b"],
            "patient_id": ["1", "1"],
            "official_partition": ["train", "test"],
        }
    )
    with pytest.raises(DatasetError, match="leakage"):
        verify_patient_split_integrity(manifest)


def test_split_file_multi_partition_patient_rejected(tmp_path):
    path = tmp_path / "split.txt"
    path.write_text("101\ttrain\n101\ttest\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="multiple partitions"):
        read_split_file(path)


def test_cycles_table_valid(synthetic_dataset):
    manifest, _ = build_manifest(
        synthetic_dataset / "audio",
        synthetic_dataset / "annotations",
        synthetic_dataset / "split_file.txt",
    )
    cycles, issues = build_cycle_table(manifest, min_cycle_seconds=0.2)
    assert len(cycles) == 32  # 8 recordings x 4 cycles
    assert cycles["class_label"].isin(["normal", "crackles", "wheezes", "both"]).all()
    assert cycles["cycle_id"].is_unique
    within = cycles.apply(
        lambda row: row["end_time"] <= manifest.set_index("recording_id")
        .loc[row["recording_id"], "duration_seconds"],
        axis=1,
    )
    assert within.all()
    assert not any(issue["severity"] == "error" for issue in issues)


def test_missing_annotation_file_is_reported(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "101_normal.wav").write_bytes(b"\x00" * 16)
    manifest, issues = build_manifest(audio, audio, None)
    assert manifest.empty
    assert any(
        issue["code"] == "audio_without_annotations" for issue in issues
    )


def test_end_to_end_pipeline_on_synthetic_data(synthetic_dataset):
    """Small end-to-end run: manifest -> cycles -> features -> train -> predict."""
    manifest, _ = build_manifest(
        synthetic_dataset / "audio",
        synthetic_dataset / "annotations",
        synthetic_dataset / "split_file.txt",
    )
    cycles, _ = build_cycle_table(manifest, min_cycle_seconds=0.2)

    features, issues = extract_features_table(cycles)
    assert not features.empty
    assert not any(issue["severity"] == "error" for issue in issues)
    names = feature_names()
    assert list(features.columns) == ["recording_id", "patient_id", "cycle_id",
                                      "class_label", "official_partition"] + names
    numeric = features[names].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()

    train = features[features["official_partition"] == "train"]
    test = features[features["official_partition"] == "test"]
    assert not train.empty and not test.empty
    assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))

    model = build_model("random_forest", random_state=0, n_estimators=20)
    model = train_model(model, train[names], train["class_label"], names)
    predictions = predict_with_confidence(model, test[names], names)
    metrics = compute_metrics(test["class_label"].tolist(), predictions["labels"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_cycles"] == len(test)
