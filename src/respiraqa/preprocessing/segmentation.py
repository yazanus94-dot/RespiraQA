"""Respiratory-cycle segmentation from annotated time intervals."""

from __future__ import annotations

import numpy as np

DEFAULT_MIN_CYCLE_SAMPLES = 64


def extract_cycle(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    min_cycle_samples: int = DEFAULT_MIN_CYCLE_SAMPLES,
) -> np.ndarray:
    """Extract one annotated respiratory cycle from a recording.

    Times are converted to sample indices with rounding, per the project plan.

    Args:
        audio: 1-D floating-point signal.
        sample_rate: Positive sampling rate in Hz.
        start_time: Cycle start in seconds (must be >= 0).
        end_time: Cycle end in seconds (must be > start_time).
        min_cycle_samples: Reject cycles shorter than this many samples.

    Returns:
        1-D NumPy array containing the cycle samples.

    Raises:
        ValueError: On invalid sample rate, invalid time interval,
            non-finite times, intervals beyond the recording length,
            or cycles below the minimum length.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("extract_cycle expects a 1-D mono signal")
    if signal.size == 0:
        raise ValueError("extract_cycle received empty audio")

    if not (
        np.isfinite(start_time)
        and np.isfinite(end_time)
        and float(start_time) == start_time
        and float(end_time) == end_time
    ):
        raise ValueError("Cycle times must be finite numeric values")
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Invalid cycle time interval")

    start_index = round(start_time * sample_rate)
    end_index = round(end_time * sample_rate)

    if end_index > len(signal):
        raise ValueError("Cycle interval exceeds recording length")

    cycle = signal[start_index:end_index]
    if cycle.size == 0:
        raise ValueError("Extracted cycle is empty")
    if cycle.size < min_cycle_samples:
        raise ValueError(
            f"Extracted cycle is too short ({cycle.size} samples; "
            f"minimum {min_cycle_samples})"
        )
    return cycle
