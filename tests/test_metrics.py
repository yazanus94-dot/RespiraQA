"""Metric calculations against known small examples."""

from __future__ import annotations

import pytest

from respiraqa.evaluation.metrics import (
    accuracy_macro_f1,
    compute_event_metrics,
    compute_metrics,
)


def test_accuracy_perfect():
    true = ["normal", "crackles", "wheezes", "both"]
    pred = ["normal", "crackles", "wheezes", "both"]
    metrics = compute_metrics(true, pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_accuracy_known_example():
    true = ["normal", "normal", "crackles", "wheezes"]
    pred = ["normal", "crackles", "crackles", "wheezes"]
    accuracy, macro_f1 = accuracy_macro_f1(true, pred)
    assert accuracy == pytest.approx(0.75)
    # Per-class F1: normal (1 tp, 1 fn -> 2/3), crackles (1 tp, 1 fp -> 2/3),
    # wheezes 1.0, both 0.0 (absent from this example but part of the label set).
    assert macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1.0 + 0.0) / 4)


def test_event_metrics_definition():
    true = ["normal", "crackles", "wheezes", "both", "normal"]
    pred = ["normal", "wheezes", "normal", "both", "crackles"]
    event = compute_event_metrics(true, pred)
    # Event-positive: crackles, wheezes, both -> 3 events; hits = crackles->wheezes, both->both = 2
    # Normal-negative: 2 normals; hits = 1
    assert event["n_event_cycles"] == 3
    assert event["n_normal_cycles"] == 2
    assert event["event_sensitivity"] == pytest.approx(2 / 3)
    assert event["normal_specificity"] == pytest.approx(0.5)
    assert event["average_score"] == pytest.approx((2 / 3 + 0.5) / 2)


def test_class_distribution_counts():
    true = ["normal", "normal", "both"]
    pred = ["normal", "normal", "both"]
    metrics = compute_metrics(true, pred)
    assert metrics["class_distribution"] == {
        "normal": 2,
        "crackles": 0,
        "wheezes": 0,
        "both": 1,
    }


def test_per_class_support():
    true = ["normal", "crackles", "crackles", "both"]
    pred = ["normal", "crackles", "normal", "both"]
    metrics = compute_metrics(true, pred)
    assert metrics["per_class"]["crackles"]["support"] == 2
    assert metrics["per_class"]["crackles"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["wheezes"]["support"] == 0


def test_macro_f1_includes_absent_classes():
    """Macro F1 must average over the same four classes as the per-class table."""
    true = ["normal", "both", "normal"]
    pred = ["normal", "both", "both"]
    metrics = compute_metrics(true, pred)
    expected = sum(v["f1_score"] for v in metrics["per_class"].values()) / len(
        metrics["per_class"]
    )
    assert metrics["macro_f1"] == pytest.approx(expected)
    accuracy, robustness_f1 = accuracy_macro_f1(true, pred)
    assert robustness_f1 == pytest.approx(metrics["macro_f1"])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(["normal"], [])


def test_empty_inputs_raise():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])
