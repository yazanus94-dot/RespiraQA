"""Band-pass filtering with validated cutoff frequencies."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Args:
        audio: 1-D floating-point signal.
        sample_rate: Sampling rate in Hz (must be positive).
        low_hz: Low cutoff in Hz (must be > 0 and < high_hz).
        high_hz: High cutoff in Hz (must be below the Nyquist frequency).
        order: Filter order (default 4).

    Returns:
        Filtered signal with the same shape and dtype as the input.

    Raises:
        ValueError: For invalid sample rate, cutoffs at or beyond Nyquist,
            non-finite input, or empty input.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("bandpass_filter expects a 1-D mono signal")
    if signal.size == 0:
        raise ValueError("bandpass_filter received empty audio")
    if not np.isfinite(signal).all():
        raise ValueError("bandpass_filter received non-finite audio values")
    if low_hz <= 0:
        raise ValueError(f"low_hz must be positive, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})")
    nyquist = sample_rate / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz ({high_hz} Hz) must be below the Nyquist frequency ({nyquist} Hz) "
            f"for sample rate {sample_rate} Hz"
        )

    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float64)
