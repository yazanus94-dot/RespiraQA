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


VALID_PROVENANCE = ("synthetic", "real")


def _provenance_path(value: str | Path, config_path: str | Path | None) -> Path:
    if config_path is not None:
        return resolve_path(config_path, value)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _nonempty_directory(path: Path, label: str) -> None:
    if not path.is_dir() or not any(path.iterdir()):
        raise ConfigError(f"{label} must be an existing non-empty directory: {path}")


def _validate_official_split(split_file: Path) -> None:
    """Require an official patient partition with disjoint train/test sets."""
    from respiraqa.data.dataset import DatasetError, read_split_file

    try:
        mapping = read_split_file(split_file)
    except (FileNotFoundError, DatasetError) as exc:
        raise ConfigError(f"invalid official patient split file: {exc}") from exc
    partitions = set(mapping.values())
    if "train" not in partitions or "test" not in partitions:
        raise ConfigError("official patient split must contain both 'train' and 'test' partitions")
    patients = list(mapping)
    if len(set(patients)) != len(patients):
        raise ConfigError("patient assigned to multiple partitions")


def dataset_provenance(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    *,
    require_real_evidence: bool = False,
) -> dict[str, Any]:
    """Return validated, explicit dataset provenance.

    Provenance is NEVER inferred from a directory name. The configuration must
    declare ``dataset.provenance`` plus a dataset id, source, version and
    licence. For real research data, evidence of acquisition and an official
    patient-independent split are required before real-data results may be
    generated. Neither mode constitutes clinical validation.
    """
    dataset = config.get("dataset") or {}
    provenance = str(dataset.get("provenance", "")).strip().lower()
    if provenance not in VALID_PROVENANCE:
        raise ConfigError(
            f"dataset.provenance must be one of {', '.join(VALID_PROVENANCE)}; got {provenance!r}"
        )
    for key in ("dataset_id", "source", "version", "license"):
        value = dataset.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"dataset.{key} must be a non-empty string")

    root_value = str(dataset.get("root", ""))
    root_path = _provenance_path(root_value, config_path)
    try:
        project_root = Path(__file__).resolve().parents[3]
        root_display = str(root_path.resolve().relative_to(project_root))
    except (ValueError, OSError):
        root_display = root_value

    synthetic = provenance == "synthetic"
    result: dict[str, Any] = {
        "dataset_root": root_display,
        "provenance": provenance,
        "dataset_id": dataset["dataset_id"],
        "source": dataset["source"],
        "version": dataset["version"],
        "license": dataset["license"],
        "synthetic": synthetic,
        "label": (
            "SYNTHETIC DEMO - not clinical validation"
            if synthetic
            else "REAL RESEARCH DATA - not clinical validation"
        ),
    }

    audio_dir = _provenance_path(str(dataset.get("audio_dir", "")), config_path)
    annotation_dir = _provenance_path(str(dataset.get("annotation_dir", "")), config_path)
    split_file = _provenance_path(str(dataset.get("split_file", "")), config_path)
    if synthetic:
        _nonempty_directory(audio_dir, "synthetic audio_dir")
        _nonempty_directory(annotation_dir, "synthetic annotation_dir")
        if not split_file.is_file():
            raise ConfigError(f"synthetic split_file not found: {split_file}")
    elif require_real_evidence:
        _nonempty_directory(root_path, "real dataset root")
        _nonempty_directory(audio_dir, "real audio_dir")
        _nonempty_directory(annotation_dir, "real annotation_dir")
        manifest = _provenance_path(str(dataset.get("manifest_path", "")), config_path)
        if not manifest.is_file() or manifest.stat().st_size == 0:
            raise ConfigError(
                f"real-data results require a generated manifest at {manifest}; "
                "run scripts/02_build_manifest.py first"
            )
        if not str(dataset.get("split_file", "")).strip():
            raise ConfigError(
                "real-data results require dataset.split_file (official patient partition)"
            )
        _validate_official_split(split_file)
        if dataset.get("terms_accepted_by_owner") is not True:
            raise ConfigError(
                "real-data results require the dataset owner to accept the applicable "
                "terms and set dataset.terms_accepted_by_owner: true in the config; "
                "this tool never accepts terms on your behalf"
            )
        result["terms_accepted_by_owner"] = True
    return result


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
