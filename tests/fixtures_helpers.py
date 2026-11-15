"""Deterministic synthetic respiratory-sound dataset builders for tests.

These fixtures contain no real patient data and are safe to commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# (recording_id, patient_id, class_label, partition)
RECORDING_PLAN: tuple[tuple[str, str, str, str], ...] = (
    ("101_1b1_Al_sc_Litt3200", "101", "normal", "train"),
    ("101_1b2_Pr_sc_Litt3200", "101", "crackles", "train"),
    ("102_1b1_Ar_Litt3200", "102", "wheezes", "train"),
    ("102_1b2_Tc_Litt3200", "102", "both", "train"),
    ("201_1b1_Ll_sc_Litt3200", "201", "normal", "test"),
    ("201_1b2_Mc_sc_Litt3200", "201", "crackles", "test"),
    ("202_1b1_Cr_Litt3200", "202", "wheezes", "test"),
    ("202_1b2_Ls_Litt3200", "202", "both", "test"),
)


def synth_normal(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Low-frequency breathing-like noise."""
    t = np.arange(n_samples) / 8000.0
    breath = 0.25 * np.sin(2 * np.pi * 0.35 * t) * np.sin(2 * np.pi * 0.11 * t)
    noise = rng.normal(0.0, 0.02, size=n_samples)
    signal = breath + noise
    return signal / max(1e-9, np.max(np.abs(signal)))


def synth_crackles(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Breath sound with scattered exp-decay clicks (crackles)."""
    signal = synth_normal(rng, n_samples)
    for _ in range(10):
        start = int(rng.integers(0, n_samples - 400))
        length = int(rng.integers(200, 400))
        t = np.arange(length) / 8000.0
        click = np.exp(-t * 120) * np.sin(2 * np.pi * 900 * t) * 0.8
        signal[start : start + length] += click
    return signal / max(1e-9, np.max(np.abs(signal)))


def synth_wheezes(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Breath sound with a continuous 350 Hz wheeze tone."""
    signal = synth_normal(rng, n_samples)
    t = np.arange(n_samples) / 8000.0
    wheeze = 0.35 * np.sin(2 * np.pi * 350 * t)
    signal = signal + wheeze
    return signal / max(1e-9, np.max(np.abs(signal)))


def synth_both(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Wheeze tone plus scattered crackle clicks."""
    signal = synth_wheezes(rng, n_samples)
    for _ in range(8):
        start = int(rng.integers(0, n_samples - 300))
        length = int(rng.integers(150, 300))
        t = np.arange(length) / 8000.0
        click = np.exp(-t * 140) * np.sin(2 * np.pi * 700 * t) * 0.7
        signal[start : start + length] += click
    return signal / max(1e-9, np.max(np.abs(signal)))


SYNTHESIZERS: dict[str, object] = {
    "normal": synth_normal,
    "crackles": synth_crackles,
    "wheezes": synth_wheezes,
    "both": synth_both,
}


def make_recording(
    class_label: str,
    seed: int,
    sample_rate: int = 8000,
    seconds: float = 4.0,
) -> np.ndarray:
    """Synthesize one respiratory recording for a given class."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(seconds * sample_rate))
    synthesizer = SYNTHESIZERS[class_label]
    return synthesizer(rng, n_samples)  # type: ignore[operator]


def build_synthetic_dataset(
    base_dir: Path,
    sample_rate: int = 8000,
    seconds: float = 4.0,
) -> Path:
    """Create a full synthetic dataset: audio, annotations, split file.

    Layout matches the configurable ICBHI-style layout used by the project:
    ``audio/``, ``annotations/``, ``metadata/`` and ``split_file.txt``.

    Cycles: each recording carries 4 annotated cycles with the recording's
    class label, except both-class recordings which include normal cycles
    too, so all four classes appear in every partition.
    """
    audio_dir = base_dir / "audio"
    annotation_dir = base_dir / "annotations"
    metadata_dir = base_dir / "metadata"
    for directory in (audio_dir, annotation_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, (recording_id, patient_id, class_label, partition) in enumerate(RECORDING_PLAN):
        audio = make_recording(class_label, seed=1000 + index, sample_rate=sample_rate,
                               seconds=seconds)
        sf.write(
            audio_dir / f"{recording_id}.wav",
            audio,
            sample_rate,
            subtype="PCM_16",
        )

        # Four cycles of ~0.9 s with gaps, staying within the recording.
        rows = []
        for cycle_index in range(4):
            start = cycle_index * 1.0
            end = start + 0.9
            if class_label == "both" and cycle_index == 3:
                rows.append(f"{start:.3f}\t{end:.3f}\t0\t0")
            elif class_label == "normal" and cycle_index == 3:
                rows.append(f"{start:.3f}\t{end:.3f}\t1\t0")
            elif class_label == "crackles":
                rows.append(f"{start:.3f}\t{end:.3f}\t1\t0")
            elif class_label == "wheezes":
                rows.append(f"{start:.3f}\t{end:.3f}\t0\t1")
            else:
                rows.append(f"{start:.3f}\t{end:.3f}\t1\t1")
        (annotation_dir / f"{recording_id}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

        (metadata_dir / f"{recording_id}.txt").write_text(
            f"recording_id={recording_id}\npatient_id={patient_id}\n"
            f"partition={partition}\nsynthetic=true\n",
            encoding="utf-8",
        )

    split_lines = []
    seen = set()
    for _, patient_id, _, partition in RECORDING_PLAN:
        if patient_id not in seen:
            split_lines.append(f"{patient_id}\t{partition}")
            seen.add(patient_id)
    (base_dir / "split_file.txt").write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    (base_dir / "README.md").write_text(
        "# Synthetic respiratory-sound fixture\n\n"
        "Fully synthetic, deterministic audio with no real patient data. "
        "Generated by tests/fixtures_helpers.py (build_synthetic_dataset). "
        "Used by the automated test-suite and the synthetic pipeline demo.\n",
        encoding="utf-8",
    )
    return base_dir
