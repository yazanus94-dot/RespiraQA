"""Reproducibility helpers: seeding and environment recording."""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seed(seed: int) -> None:
    """Seed Python's and NumPy's RNGs for reproducible runs."""
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))


def get_environment_versions() -> dict[str, Any]:
    """Return package and platform versions for reproducibility records."""
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "librosa",
        "soundfile",
        "sklearn",
        "matplotlib",
        "joblib",
    ):
        try:
            module = __import__(package)
            versions["packages"][package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions["packages"][package] = "not installed"
    return versions


def write_json(payload: dict[str, Any], path: str) -> None:
    """Write a JSON file, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
