"""Audio loading and resampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be opened or is empty."""


def load_audio(
    path: str | Path,
    target_sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a WAV file as floating-point samples.

    Args:
        path: Path to the audio file.
        target_sample_rate: If given and different from the native rate, the
            signal is properly resampled to this rate (never just re-labeled).
        mono: If True, average channels to mono; original channel count is
            preserved in the returned metadata.

    Returns:
        Tuple ``(audio, info)`` where ``audio`` is a 1-D float64 NumPy array
        and ``info`` records native sample rate, channel count, duration,
        sample count, and whether resampling occurred.

    Raises:
        FileNotFoundError: If the file does not exist.
        AudioLoadError: If the file cannot be read or contains no samples.
        ValueError: If target_sample_rate is not positive.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if target_sample_rate is not None and target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    try:
        audio, sample_rate = sf.read(file_path, dtype="float64", always_2d=True)
    except (sf.LibsndfileError, ValueError, RuntimeError) as exc:
        raise AudioLoadError(f"Cannot read audio file {file_path}: {exc}") from exc

    native_sample_rate = int(sample_rate)
    native_channels = audio.shape[1]
    native_samples = audio.shape[0]
    if native_samples == 0:
        raise AudioLoadError(f"Audio file {file_path} contains no samples")
    if not np.isfinite(audio).all():
        raise AudioLoadError(f"Audio file {file_path} contains non-finite samples")

    if mono:
        signal = audio.mean(axis=1)
    else:
        signal = audio.reshape(-1) if audio.shape[1] == 1 else audio

    resampled = False
    if target_sample_rate is not None and target_sample_rate != native_sample_rate:
        signal = _resample(signal, native_sample_rate, target_sample_rate)
        resampled = True
        effective_rate = target_sample_rate
    else:
        effective_rate = native_sample_rate

    if mono and signal.ndim != 1:
        signal = signal.mean(axis=1)

    duration_seconds = signal.shape[0] / effective_rate if effective_rate else 0.0
    info: dict[str, Any] = {
        "path": str(file_path),
        "native_sample_rate": native_sample_rate,
        "native_channels": native_channels,
        "native_num_samples": native_samples,
        "sample_rate": effective_rate,
        "channels": signal.shape[1] if signal.ndim == 2 else 1,
        "num_samples": signal.shape[0],
        "duration_seconds": float(duration_seconds),
        "resampled": resampled,
        "mono": bool(mono),
    }
    return signal, info


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample using librosa's high-quality resampler.

    Args:
        audio: 1-D mono signal or 2-D ``(samples, channels)`` array.
        source_rate: Native sample rate (Hz).
        target_rate: Desired sample rate (Hz).

    Returns:
        Resampled float64 signal with the same number of channels.

    Raises:
        ValueError: If rates are not positive or the array is not 1-D/2-D.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    import librosa

    array = np.asarray(audio, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError("Resampling expects a 1-D or 2-D (samples, channels) signal")
    return librosa.resample(array, orig_sr=source_rate, target_sr=target_rate, axis=0)
