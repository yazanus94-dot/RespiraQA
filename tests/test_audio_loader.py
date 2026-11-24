"""Tests for audio loading, resampling, and error handling."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from respiraqa.data.audio_loader import AudioLoadError, load_audio


def test_load_valid_wav(make_audio):
    path = make_audio("valid", class_label="normal", sample_rate=8000, seconds=2.0)
    audio, info = load_audio(path)
    assert audio.ndim == 1
    assert audio.dtype == np.float64
    assert info["sample_rate"] == 8000
    assert info["native_sample_rate"] == 8000
    assert info["duration_seconds"] == pytest.approx(2.0, rel=1e-3)
    assert np.isfinite(audio).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is not audio data")
    with pytest.raises(AudioLoadError):
        load_audio(path)


def test_empty_audio_file_raises(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float64), 8000)
    with pytest.raises(AudioLoadError):
        load_audio(path)


def test_multichannel_input_averaged_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    audio = np.column_stack(
        [
            np.sin(2 * np.pi * 220 * np.linspace(0, 1, 8000)),
            -np.sin(2 * np.pi * 220 * np.linspace(0, 1, 8000)),
        ]
    )
    sf.write(path, audio, 8000, subtype="PCM_16")
    signal, info = load_audio(path, mono=True)
    assert signal.ndim == 1
    assert info["native_channels"] == 2
    assert info["channels"] == 1


def test_resampling_changes_sample_rate_and_length(make_audio):
    path = make_audio("resample", class_label="normal", sample_rate=8000, seconds=2.0)
    original, _ = load_audio(path)
    resampled, info = load_audio(path, target_sample_rate=4000)
    assert info["resampled"] is True
    assert info["sample_rate"] == 4000
    assert len(resampled) == pytest.approx(len(original) / 2, rel=1e-2)
    assert info["duration_seconds"] == pytest.approx(2.0, rel=1e-2)


def test_invalid_target_sample_rate_raises(make_audio):
    path = make_audio("any.wav".replace(".wav", ""), class_label="normal")
    with pytest.raises(ValueError, match="target_sample_rate"):
        load_audio(path, target_sample_rate=0)


def test_stereo_resampling_preserves_two_channels(tmp_path):
    """mono=False with resampling must return a 2-D array, never crash."""
    path = tmp_path / "stereo.wav"
    signal = np.zeros((8000, 2), dtype=np.float32)
    signal[:, 0] = 0.1
    signal[:, 1] = -0.1
    sf.write(path, signal, 8000)
    audio, info = load_audio(path, mono=False, target_sample_rate=4000)
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert info["channels"] == 2
    assert info["sample_rate"] == 4000
    assert len(audio) == pytest.approx(4000, rel=1e-2)


def test_sample_rate_label_not_faked(make_audio):
    """Resampling must actually change sample count, not just the label."""
    path = make_audio("check_len", class_label="normal", sample_rate=8000, seconds=1.0)
    audio, _ = load_audio(path)
    resampled, _ = load_audio(path, target_sample_rate=16000)
    assert len(resampled) == pytest.approx(len(audio) * 2, rel=1e-2)
