"""Tests for annotation parsing and validation."""

from __future__ import annotations

import pytest

from respiraqa.data.annotations import (
    AnnotationError,
    class_from_labels,
    read_annotation_file,
    validate_annotation_dataframe,
)


def test_valid_annotation_file(tmp_path):
    path = tmp_path / "valid.txt"
    path.write_text("0.000\t0.900\t0\t0\n1.000\t1.900\t1\t0\n", encoding="utf-8")
    frame = read_annotation_file(path)
    assert list(frame.columns[:4]) == ["start_time", "end_time", "crackle_label", "wheeze_label"]
    assert frame["class_label"].tolist() == ["normal", "crackles"]
    assert len(frame) == 2


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "bad_columns.txt"
    path.write_text("0.000\t0.900\t0\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="expected 4 columns"):
        read_annotation_file(path)


def test_invalid_binary_label_is_rejected(tmp_path):
    path = tmp_path / "bad_label.txt"
    path.write_text("0.000\t0.900\t2\t0\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="crackle_label must be 0 or 1"):
        read_annotation_file(path)


def test_invalid_time_interval_is_rejected(tmp_path):
    path = tmp_path / "bad_interval.txt"
    path.write_text("0.900\t0.000\t0\t0\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="end_time"):
        read_annotation_file(path)


def test_negative_start_time_is_rejected(tmp_path):
    path = tmp_path / "negative.txt"
    path.write_text("-0.100\t0.900\t0\t0\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="negative start_time"):
        read_annotation_file(path)


def test_non_numeric_row_is_rejected(tmp_path):
    path = tmp_path / "non_numeric.txt"
    path.write_text("zero\t0.900\t0\t0\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="non-numeric"):
        read_annotation_file(path)


def test_missing_annotation_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_annotation_file(tmp_path / "does_not_exist.txt")


def test_class_mapping_covers_all_pairs():
    assert class_from_labels(0, 0) == "normal"
    assert class_from_labels(1, 0) == "crackles"
    assert class_from_labels(0, 1) == "wheezes"
    assert class_from_labels(1, 1) == "both"
    with pytest.raises(AnnotationError):
        class_from_labels(1, 2)


def test_interval_beyond_duration_reported(tmp_path):
    path = tmp_path / "beyond.txt"
    path.write_text("0.000\t0.900\t0\t0\n", encoding="utf-8")
    frame = read_annotation_file(path)
    issues = validate_annotation_dataframe(frame, recording_duration=0.5)
    assert any(issue["code"] == "interval_beyond_duration" for issue in issues)
    assert all(issue["severity"] == "error" for issue in issues)


def test_empty_annotation_file_warns(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    frame = read_annotation_file(path)
    issues = validate_annotation_dataframe(frame)
    assert frame.empty
    assert any(issue["code"] == "no_annotations" for issue in issues)
