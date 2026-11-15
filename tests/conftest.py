"""Shared pytest fixtures: deterministic synthetic respiratory audio."""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
from tests.fixtures_helpers import build_synthetic_dataset, make_recording


@pytest.fixture
def make_audio(tmp_path: Path):
    """Factory writing synthetic WAV files with configurable class/rate."""

    def _make(
        name: str,
        class_label: str = "normal",
        sample_rate: int = 8000,
        seconds: float = 2.0,
        directory: Path | None = None,
        seed: int = 0,
    ) -> Path:
        target_dir = directory or tmp_path
        target_dir.mkdir(parents=True, exist_ok=True)
        audio = make_recording(class_label, seed=seed, sample_rate=sample_rate, seconds=seconds)
        path = target_dir / f"{name}.wav"
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        return path

    return _make


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped full synthetic dataset in a temporary directory."""
    base_dir = tmp_path_factory.mktemp("synthetic_dataset")
    return build_synthetic_dataset(base_dir)
