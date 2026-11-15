"""Tests for filtering, normalization, and cycle segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from respiraqa.preprocessing.filtering import bandpass_filter
from respiraqa.preprocessing.normalization import normalize
from respiraqa.preprocessing.segmentation import extract_cycle


def _signal(seconds: float = 1.0, sample_rate: int = 8000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=int(seconds * sample_rate))


def test_bandpass_filter_output_finite_and_shape():
    signal = _signal()
    filtered = bandpass_filter(signal, 8000, 100.0, 2000.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_bandpass_filter_attenuates_out_of_band():
    sample_rate = 8000
    t = np.arange(sample_rate) / sample_rate
    low_signal = np.sin(2 * np.pi * 20.0 * t)  # below 100 Hz cutoff
    in_band = np.sin(2 * np.pi * 500.0 * t)  # inside 100-2000 Hz
    filtered_low = bandpass_filter(low_signal, sample_rate, 100.0, 2000.0)
    filtered_band = bandpass_filter(in_band, sample_rate, 100.0, 2000.0)
    assert np.abs(filtered_low).mean() < np.abs(filtered_band).mean() / 10


def test_invalid_cutoff_raises():
    signal = _signal()
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_filter(signal, 8000, 100.0, 4000.0)
    with pytest.raises(ValueError, match="greater than low_hz"):
        bandpass_filter(signal, 8000, 2000.0, 100.0)
    with pytest.raises(ValueError, match="low_hz must be positive"):
        bandpass_filter(signal, 8000, 0.0, 2000.0)
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        bandpass_filter(signal, 0, 100.0, 2000.0)


def test_empty_and_nonfinite_audio_rejected():
    with pytest.raises(ValueError, match="empty"):
        bandpass_filter(np.array([]), 8000, 100.0, 2000.0)
    bad = _signal()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bandpass_filter(bad, 8000, 100.0, 2000.0)


def test_normalization_methods():
    signal = _signal() * 5.0 + 2.0
    peak = normalize(signal, method="peak")
    assert np.max(np.abs(peak)) == pytest.approx(1.0)
    zscore = normalize(signal, method="zscore")
    assert zscore.mean() == pytest.approx(0.0, abs=1e-9)
    assert zscore.std() == pytest.approx(1.0, abs=1e-9)
    minmax = normalize(signal, method="minmax")
    assert minmax.min() == pytest.approx(0.0, abs=1e-9)
    assert minmax.max() == pytest.approx(1.0, abs=1e-9)
    unchanged = normalize(signal, method="none")
    assert np.allclose(unchanged, signal)


def test_normalization_invalid_method_and_inputs():
    signal = _signal()
    with pytest.raises(ValueError, match="Unsupported normalization"):
        normalize(signal, method="magic")
    with pytest.raises(ValueError, match="empty"):
        normalize(np.array([]))
    bad = signal.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        normalize(bad, method="peak")


def test_extract_cycle_valid():
    sample_rate = 8000
    signal = _signal(seconds=2.0, sample_rate=sample_rate)
    cycle = extract_cycle(signal, sample_rate, 0.5, 1.2)
    expected_start = round(0.5 * sample_rate)
    expected_end = round(1.2 * sample_rate)
    assert np.allclose(cycle, signal[expected_start:expected_end])
    assert cycle.size == expected_end - expected_start


def test_invalid_cycle_duration():
    sample_rate = 8000
    signal = _signal(seconds=1.0, sample_rate=sample_rate)
    with pytest.raises(ValueError, match="Invalid cycle time interval"):
        extract_cycle(signal, sample_rate, 0.9, 0.9)
    with pytest.raises(ValueError, match="Invalid cycle time interval"):
        extract_cycle(signal, sample_rate, 0.9, 0.5)
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        extract_cycle(signal, 0, 0.1, 0.5)


def test_cycle_interval_exceeds_recording():
    sample_rate = 8000
    signal = _signal(seconds=1.0, sample_rate=sample_rate)
    with pytest.raises(ValueError, match="exceeds recording length"):
        extract_cycle(signal, sample_rate, 0.5, 2.0)


def test_cycle_too_short_rejected():
    sample_rate = 8000
    signal = _signal(seconds=1.0, sample_rate=sample_rate)
    with pytest.raises(ValueError, match="too short"):
        extract_cycle(signal, sample_rate, 0.0, 0.001)
