"""Optional amplitude normalization methods."""

from __future__ import annotations

import numpy as np

SUPPORTED_NORMALIZATIONS = ("none", "peak", "zscore", "minmax")


def normalize(audio: np.ndarray, method: str = "none") -> np.ndarray:
    """Normalize a signal amplitude.

    Args:
        audio: 1-D floating-point signal.
        method: One of ``none``, ``peak`` (scale max abs to 1),
            ``zscore`` (zero mean, unit variance), ``minmax`` (scale to [0, 1]).

    Returns:
        Normalized copy of the signal. For ``none`` (or a silent signal
        under zscore/minmax), the input cast to float64 is returned unchanged.

    Raises:
        ValueError: On empty or non-finite input or unsupported method.
    """
    if method not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported normalization '{method}'; expected one of {SUPPORTED_NORMALIZATIONS}"
        )
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("normalize expects a 1-D mono signal")
    if signal.size == 0:
        raise ValueError("normalize received empty audio")
    if not np.isfinite(signal).all():
        raise ValueError("normalize received non-finite audio values")

    if method == "none":
        return signal

    if method == "peak":
        peak = np.max(np.abs(signal))
        if peak == 0:
            return signal
        return signal / peak

    if method == "zscore":
        std = signal.std()
        if std == 0:
            return signal
        return (signal - signal.mean()) / std

    minimum, maximum = signal.min(), signal.max()
    span = maximum - minimum
    if span == 0:
        return signal
    return (signal - minimum) / span
