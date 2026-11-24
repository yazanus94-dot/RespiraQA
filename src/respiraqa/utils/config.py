"""Configuration loading for RespiraQA pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and resolve relative paths.

    Relative dataset/report paths are resolved against the repository root,
    defined as the parent directory of the directory containing the config.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Parsed configuration as a dict.

    Raises:
        ConfigError: If the file does not exist or is not valid YAML mapping.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Configuration root must be a mapping, got {type(config).__name__}")

    return config


def repo_root(config_path: str | Path) -> Path:
    """Return the repository root inferred from the config file location."""
    return Path(config_path).resolve().parent.parent


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve a path from the config against the repository root."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root(config_path) / candidate


def dataset_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Return explicit synthetic/real provenance for the configured dataset.

    Results generated from the committed fixtures must never be mistaken for
    real-data or clinical validation, so every report records this field.
    """
    root_value = str(config.get("dataset", {}).get("root", ""))
    try:
        project_root = Path(__file__).resolve().parents[3]
        root = str(Path(root_value).resolve().relative_to(project_root))
    except (ValueError, OSError):
        root = root_value
    synthetic = "synthetic" in root.casefold()
    return {
        "dataset_root": root,
        "synthetic": synthetic,
        "label": (
            "SYNTHETIC DEMO - not clinical validation" if synthetic else "REAL DATA - verify licence and cohort before use"
        ),
    }


def resolve_config_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Return a copy of the config with known path entries made absolute."""
    resolved = dict(config)
    dataset = dict(resolved.get("dataset", {}))
    for key in (
        "root",
        "audio_dir",
        "annotation_dir",
        "metadata_dir",
        "split_file",
        "manifest_path",
        "cycles_path",
        "features_path",
    ):
        if key in dataset and dataset[key]:
            dataset[key] = str(resolve_path(config_path, dataset[key]))
    resolved["dataset"] = dataset

    training = dict(resolved.get("training", {}))
    for key in ("model_output", "metadata_output"):
        if key in training and training[key]:
            training[key] = str(resolve_path(config_path, training[key]))
    resolved["training"] = training

    evaluation = dict(resolved.get("evaluation", {}))
    for key in evaluation:
        if key.endswith("_path"):
            evaluation[key] = str(resolve_path(config_path, evaluation[key]))
    resolved["evaluation"] = evaluation

    if "output_dir" in resolved.get("reports", {}):
        reports = dict(resolved["reports"])
        reports["output_dir"] = str(resolve_path(config_path, reports["output_dir"]))
        resolved["reports"] = reports

    return resolved
