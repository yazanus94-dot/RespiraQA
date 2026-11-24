"""Provenance labelling must be explicit, validated and never path-inferred."""

from __future__ import annotations

from pathlib import Path

import pytest

from respiraqa.utils.config import ConfigError, dataset_provenance, load_config

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG = ROOT / "config/config.synthetic.yaml"
REAL_CONFIG = ROOT / "config/config.yaml"


def test_synthetic_config_is_explicitly_labelled():
    config = load_config(SYNTHETIC_CONFIG)
    provenance = dataset_provenance(config, SYNTHETIC_CONFIG)
    assert provenance["synthetic"] is True
    assert provenance["provenance"] == "synthetic"
    assert provenance["dataset_id"] == "synthetic_respiraqa_fixture"
    assert "SYNTHETIC" in provenance["label"]
    assert "clinical" in provenance["label"].lower()


def test_real_config_has_explicit_identity_without_evidence():
    config = load_config(REAL_CONFIG)
    provenance = dataset_provenance(config, REAL_CONFIG)
    assert provenance["provenance"] == "real"
    assert provenance["dataset_id"] == "icbhi_2017_respiratory_sound_database"
    assert "RESEARCH DATA" in provenance["label"]
    assert "clinical" in provenance["label"].lower()


def test_missing_provenance_is_rejected(tmp_path):
    config = {"dataset": {"root": "tests/fixtures/synthetic"}}
    with pytest.raises(ConfigError, match="provenance"):
        dataset_provenance(config, tmp_path / "config.yaml")


def test_path_name_is_never_used_to_infer_provenance(tmp_path):
    config = {
        "dataset": {
            "root": "tests/fixtures/synthetic",
            "dataset_id": "x",
            "source": "y",
            "version": "1",
            "license": "z",
        }
    }
    with pytest.raises(ConfigError, match="provenance"):
        dataset_provenance(config, tmp_path / "config.yaml")


def test_invalid_provenance_value_rejected(tmp_path):
    config = {
        "dataset": {
            "provenance": "probably-synthetic",
            "dataset_id": "x",
            "source": "y",
            "version": "1",
            "license": "z",
        }
    }
    with pytest.raises(ConfigError, match="provenance"):
        dataset_provenance(config, tmp_path / "config.yaml")


def test_real_evidence_requirements_fail_closed(tmp_path):
    config = {
        "dataset": {
            "provenance": "real",
            "dataset_id": "icbhi",
            "source": "physionet",
            "version": "2017",
            "license": "terms",
            "root": str(tmp_path / "missing"),
            "audio_dir": str(tmp_path / "missing"),
            "annotation_dir": str(tmp_path / "missing"),
            "manifest_path": str(tmp_path / "missing.csv"),
            "split_file": "",
        }
    }
    with pytest.raises(ConfigError):
        dataset_provenance(config, None, require_real_evidence=True)


def test_real_evidence_requires_owner_terms_acceptance(tmp_path):
    audio = tmp_path / "audio"
    annotations = tmp_path / "annotations"
    audio.mkdir()
    annotations.mkdir()
    (audio / "a.wav").touch()
    (annotations / "a.txt").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("patient_id\n101\n", encoding="utf-8")
    split = tmp_path / "split.txt"
    split.write_text("101 train\n201 test\n", encoding="utf-8")
    config = {
        "dataset": {
            "provenance": "real",
            "dataset_id": "icbhi",
            "source": "physionet",
            "version": "2017",
            "license": "terms",
            "root": str(tmp_path),
            "audio_dir": str(audio),
            "annotation_dir": str(annotations),
            "manifest_path": str(manifest),
            "split_file": str(split),
            "terms_accepted_by_owner": False,
        }
    }
    with pytest.raises(ConfigError, match="terms_accepted_by_owner"):
        dataset_provenance(config, None, require_real_evidence=True)
    config["dataset"]["terms_accepted_by_owner"] = True
    result = dataset_provenance(config, None, require_real_evidence=True)
    assert result["synthetic"] is False
    assert result["terms_accepted_by_owner"] is True
