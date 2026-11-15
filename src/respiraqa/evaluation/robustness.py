"""Model robustness experiments under controlled signal changes.

Experiments are signal-corruption probes, not clinical validation. They are
run on validation (or development) data and never used to tune against the
official test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from respiraqa import SUPPORTED_CLASSES


def add_background_noise(
    audio: np.ndarray,
    sample_rate: int,
    snr_db: float,
    seed: int = 0,
) -> np.ndarray:
    """Add white Gaussian noise at a documented SNR.

    Args:
        audio: 1-D floating-point signal.
        sample_rate: Sampling rate in Hz (used only for validation).
        snr_db: Desired signal-to-noise ratio in dB.
        seed: RNG seed for reproducibility.

    Returns:
        Noisy copy of the signal.

    Raises:
        ValueError: On empty/silent input, invalid sample rate, or
            non-finite SNR.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("add_background_noise received empty audio")
    if not np.isfinite(signal).all():
        raise ValueError("add_background_noise received non-finite audio")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    signal_power = float(np.mean(signal**2))
    if signal_power <= 0:
        raise ValueError("add_background_noise received a silent signal")

    rng = np.random.default_rng(seed)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def change_amplitude(audio: np.ndarray, gain: float) -> np.ndarray:
    """Scale a signal by a multiplicative gain (e.g. 0.5 or 2.0)."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("change_amplitude received empty audio")
    if not np.isfinite(gain):
        raise ValueError("gain must be finite")
    return signal * gain


def truncate_audio(audio: np.ndarray, keep_seconds: float, sample_rate: int) -> np.ndarray:
    """Truncate a signal to the first ``keep_seconds`` of audio."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(audio, dtype=np.float64)
    if keep_seconds <= 0:
        raise ValueError("keep_seconds must be positive")
    return signal[: int(round(keep_seconds * sample_rate))]


def run_robustness_suite(
    model: Any,
    cycles: Any,
    load_cycle_signal: Any,
    feature_extractor: Any,
    conditions: Any | None = None,
    feature_names: Any | None = None,
) -> list[dict]:
    """Evaluate a model under a suite of controlled corruption conditions.

    Args:
        model: Fitted estimator exposing ``predict`` on feature matrices.
        cycles: DataFrame of test/validation cycles with ``start_time``,
            ``end_time``, ``audio_path``, ``sample_rate``, ``class_label``.
        load_cycle_signal: Callable ``fn(audio_path) -> (audio, info)``.
        feature_extractor: Callable ``fn(cycle, sample_rate) -> dict``.
        conditions: Optional list of condition dicts, each with ``name``,
            ``transform`` (callable ``fn(signal, sr) -> signal``) and
            ``description``. Defaults to the built-in suite.
        feature_names: Ordered feature names expected by the model. Falls
            back to ``model.feature_names_in_`` when available.

    Returns:
        List of result dicts: condition name, parameters, number of cycles
        tested, accuracy, macro F1, baseline difference, errors/warnings.
    """
    from respiraqa.evaluation.metrics import accuracy_macro_f1

    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError("feature_names must be provided for the robustness suite")

    if conditions is None:
        conditions = [
            {
                "name": "reference",
                "description": "Unmodified signal (reference condition)",
                "transform": lambda signal, sr: signal,
                "parameters": {},
            },
            {
                "name": "noise_20db",
                "description": "Additive white noise at 20 dB SNR",
                "parameters": {"snr_db": 20},
                "transform": lambda signal, sr: add_background_noise(signal, sr, 20.0, seed=0),
            },
            {
                "name": "noise_10db",
                "description": "Additive white noise at 10 dB SNR",
                "parameters": {"snr_db": 10},
                "transform": lambda signal, sr: add_background_noise(signal, sr, 10.0, seed=0),
            },
            {
                "name": "amplitude_half",
                "description": "Amplitude scaled to 50%",
                "parameters": {"gain": 0.5},
                "transform": lambda signal, sr: change_amplitude(signal, 0.5),
            },
            {
                "name": "amplitude_double",
                "description": "Amplitude scaled to 200%",
                "parameters": {"gain": 2.0},
                "transform": lambda signal, sr: change_amplitude(signal, 2.0),
            },
            {
                "name": "truncated_1s",
                "description": "Signal truncated to its first 1 second",
                "parameters": {"keep_seconds": 1.0},
                "transform": lambda signal, sr: truncate_audio(signal, 1.0, sr),
            },
        ]

    results: list[dict] = []
    baseline_accuracy = None
    baseline_macro_f1 = None

    for condition in conditions:
        true_labels: list[str] = []
        predicted_labels: list[str] = []
        errors: list[str] = []

        for _, row in cycles.iterrows():
            audio, info = load_cycle_signal(row["audio_path"])
            signal = condition["transform"](audio, int(info["sample_rate"]))
            if signal.size == 0:
                errors.append(f"{row['cycle_id']}: signal empty after transform")
                continue
            try:
                features = feature_extractor(signal, int(info["sample_rate"]))
            except ValueError as exc:
                errors.append(f"{row['cycle_id']}: {exc}")
                continue
            matrix = np.array([[features[name] for name in feature_names]])
            prediction = str(model.predict(matrix)[0])
            if prediction not in SUPPORTED_CLASSES:
                errors.append(f"{row['cycle_id']}: unsupported prediction {prediction}")
                continue
            true_labels.append(str(row["class_label"]))
            predicted_labels.append(prediction)

        tested = len(true_labels)
        if tested == 0:
            results.append(
                {
                    "condition": condition["name"],
                    "description": condition["description"],
                    "parameters": condition["parameters"],
                    "n_cycles": 0,
                    "accuracy": None,
                    "macro_f1": None,
                    "accuracy_difference": None,
                    "macro_f1_difference": None,
                    "errors": errors,
                }
            )
            continue

        accuracy, macro_f1 = accuracy_macro_f1(true_labels, predicted_labels)
        if condition["name"] == "reference":
            baseline_accuracy, baseline_macro_f1 = accuracy, macro_f1

        results.append(
            {
                "condition": condition["name"],
                "description": condition["description"],
                "parameters": condition["parameters"],
                "n_cycles": tested,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "accuracy_difference": None
                if baseline_accuracy is None
                else accuracy - baseline_accuracy,
                "macro_f1_difference": None
                if baseline_macro_f1 is None
                else macro_f1 - baseline_macro_f1,
                "errors": errors,
            }
        )
    return results
