"""Interpretable acoustic feature extraction for respiratory cycles.

Features per cycle (summary statistics over frames):

- 13 MFCC coefficients (mean and standard deviation)
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral entropy

All features are deterministic for the same input and settings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_CYCLE_SAMPLES = 128
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 512


def extract_features(
    cycle: np.ndarray,
    sample_rate: int,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> dict[str, float]:
    """Extract acoustic features from one respiratory cycle.

    Args:
        cycle: 1-D floating-point mono signal.
        sample_rate: Positive sampling rate in Hz.
        n_mfcc: Number of MFCC coefficients (default 13).
        n_fft: FFT size for frame-based features (clamped to cycle length).
        hop_length: Hop size between frames (default 512).

    Returns:
        Mapping of feature name to float value. All values are finite.

    Raises:
        ValueError: On empty, non-finite, or too-short cycles, or invalid
            sample rate / feature settings.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_mfcc <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_mfcc, n_fft and hop_length must be positive")

    signal = np.asarray(cycle, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("extract_features expects a 1-D mono signal")
    if signal.size == 0:
        raise ValueError("extract_features received empty audio")
    if not np.isfinite(signal).all():
        raise ValueError("extract_features received non-finite audio values")
    if signal.size < MIN_CYCLE_SAMPLES:
        raise ValueError(
            f"Cycle too short for feature extraction ({signal.size} samples; "
            f"minimum {MIN_CYCLE_SAMPLES})"
        )

    import librosa

    effective_n_fft = int(min(n_fft, signal.size))
    effective_hop = int(min(hop_length, max(1, signal.size // 2)))

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=effective_n_fft, hop_length=effective_hop
    )
    rms = librosa.feature.rms(y=signal, frame_length=effective_n_fft, hop_length=effective_hop)
    zero_crossing = librosa.feature.zero_crossing_rate(
        signal, frame_length=effective_n_fft, hop_length=effective_hop
    )
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sample_rate, n_fft=effective_n_fft, hop_length=effective_hop
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sample_rate, n_fft=effective_n_fft, hop_length=effective_hop
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sample_rate, n_fft=effective_n_fft, hop_length=effective_hop
    )

    features: dict[str, float] = {}
    for index in range(n_mfcc):
        features[f"mfcc_{index + 1:02d}_mean"] = float(np.mean(mfcc[index]))
        features[f"mfcc_{index + 1:02d}_std"] = float(np.std(mfcc[index]))
    features["rms_energy"] = float(np.mean(rms))
    features["zero_crossing_rate"] = float(np.mean(zero_crossing))
    features["spectral_centroid"] = float(np.mean(centroid))
    features["spectral_bandwidth"] = float(np.mean(bandwidth))
    features["spectral_rolloff"] = float(np.mean(rolloff))
    features["spectral_entropy"] = _spectral_entropy(signal, effective_n_fft)

    _assert_finite(features)
    return features


def _spectral_entropy(signal: np.ndarray, n_fft: int) -> float:
    """Shannon entropy of the normalized power spectrum."""
    import scipy.fft

    spectrum = np.abs(scipy.fft.rfft(signal, n=n_fft)) ** 2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    power = spectrum / total
    power = power[power > 0]
    return float(-(power * np.log2(power)).sum())


def _assert_finite(features: dict[str, float]) -> None:
    """Raise if any extracted feature is NaN or infinite."""
    bad = [name for name, value in features.items() if not np.isfinite(value)]
    if bad:
        raise ValueError(f"Non-finite feature values produced: {', '.join(bad)}")


def feature_names(n_mfcc: int = DEFAULT_N_MFCC) -> list[str]:
    """Return the ordered feature column names used for model input."""
    names: list[str] = []
    for index in range(n_mfcc):
        names.extend([f"mfcc_{index + 1:02d}_mean", f"mfcc_{index + 1:02d}_std"])
    names.extend(
        [
            "rms_energy",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_entropy",
        ]
    )
    return names


def extract_features_table(
    cycles: Any,
    sample_rate_key: str = "sample_rate",
    progress: bool = False,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[Any, list[dict[str, Any]]]:
    """Extract features for a cycles DataFrame produced by the pipeline.

    Args:
        cycles: DataFrame with columns ``cycle_id``, ``start_time``,
            ``end_time``, ``audio_path``, ``sample_rate`` and label columns.
        sample_rate_key: Column holding the native sample rate.
        progress: If True, print progress every 25 cycles.

    Returns:
        Tuple ``(features_frame, issues)``. ``features_frame`` contains the
        metadata columns plus every feature column; ``issues`` records
        cycles that were skipped.
    """
    import pandas as pd

    from respiraqa.data.audio_loader import load_audio
    from respiraqa.preprocessing.segmentation import extract_cycle

    metadata_columns = [
        "recording_id",
        "patient_id",
        "cycle_id",
        "class_label",
        "official_partition",
    ]
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    cache: dict[str, Any] = {}
    for index, row in cycles.iterrows():
        try:
            audio_path = row["audio_path"]
            if audio_path not in cache:
                cache.clear()
                audio, info = load_audio(audio_path, target_sample_rate=None, mono=True)
                cache[audio_path] = (audio, info)
            audio, info = cache[audio_path]
            sample_rate = int(row[sample_rate_key])
            cycle_signal = extract_cycle(audio, sample_rate, row["start_time"], row["end_time"])
            features = extract_features(
                cycle_signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            issues.append(
                {
                    "severity": "warning",
                    "code": "feature_extraction_failed",
                    "message": f"{row['cycle_id']}: {exc}",
                }
            )
            continue

        record: dict[str, Any] = {key: row[key] for key in metadata_columns}
        record.update(features)
        records.append(record)

        if progress and (index + 1) % 25 == 0:
            print(f"  processed {index + 1}/{len(cycles)} cycles", flush=True)

    columns = metadata_columns + feature_names(n_mfcc)
    frame = pd.DataFrame(records, columns=columns)
    return frame, issues
