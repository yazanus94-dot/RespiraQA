"""Tests for acoustic feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from respiraqa.features.acoustic_features import (
    DEFAULT_N_MFCC,
    MIN_CYCLE_SAMPLES,
    extract_features,
    feature_names,
)


def _cycle(seconds: float = 1.0, sample_rate: int = 8000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * 200.0 * t) + 0.1 * rng.normal(size=t.size)


def test_feature_vector_shape_and_names():
    features = extract_features(_cycle(), 8000)
    names = feature_names()
    assert set(features.keys()) == set(names)
    assert len(names) == 2 * DEFAULT_N_MFCC + 6


def test_feature_vector_contains_no_nan_or_inf():
    for seed in (1, 2, 3):
        features = extract_features(_cycle(seed=seed), 8000)
        assert all(np.isfinite(value) for value in features.values())


def test_features_reproducible_for_same_input():
    cycle = _cycle()
    first = extract_features(cycle, 8000)
    second = extract_features(cycle.copy(), 8000)
    assert first.keys() == second.keys()
    assert all(
        first[name] == pytest.approx(second[name], rel=1e-9) for name in first
    )


def test_short_input_rejected():
    with pytest.raises(ValueError, match="too short"):
        extract_features(_cycle(seconds=0.005), 8000)


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        extract_features(np.array([]), 8000)


def test_nonfinite_input_rejected():
    cycle = _cycle()
    cycle[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_features(cycle, 8000)


def test_invalid_sample_rate_rejected():
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        extract_features(_cycle(), 0)


def test_minimum_cycle_samples_constant():
    assert MIN_CYCLE_SAMPLES >= 64


def test_different_classes_produce_different_features():
    from tests.fixtures_helpers import make_recording

    normal = extract_features(make_recording("normal", seed=5), 8000)
    wheeze = extract_features(make_recording("wheezes", seed=5), 8000)
    assert normal["spectral_centroid"] != pytest.approx(wheeze["spectral_centroid"])
