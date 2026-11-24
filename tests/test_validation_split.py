"""Insufficient patients/classes must fail clearly, not fake validation."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

split_module = importlib.import_module("scripts.05_train_baseline")
split_train_validation = split_module.split_train_validation


def _frame(patients: list[str], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": patients, "class_label": labels})


def test_single_patient_rejected():
    with pytest.raises(ValueError, match="at least two patients"):
        split_train_validation(_frame(["101"] * 8, ["normal", "crackles"] * 4), 0.2, 42)


def test_single_class_rejected():
    with pytest.raises(ValueError, match="two classes"):
        split_train_validation(_frame(["101", "102", "201", "202"], ["normal"] * 4), 0.2, 42)


def test_valid_split_is_patient_disjoint():
    train, validation = split_train_validation(
        _frame(
            ["101", "101", "102", "102", "201", "201", "202", "202"], ["normal", "crackles"] * 4
        ),
        0.2,
        42,
    )
    assert train and validation
    frame = _frame(
        ["101", "101", "102", "102", "201", "201", "202", "202"], ["normal", "crackles"] * 4
    )
    train_patients = set(frame.iloc[train]["patient_id"])
    validation_patients = set(frame.iloc[validation]["patient_id"])
    assert not train_patients & validation_patients
