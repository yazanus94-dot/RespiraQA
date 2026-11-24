"""Provenance labelling must distinguish synthetic demos from real data."""

from respiraqa.utils.config import dataset_provenance


def test_synthetic_config_is_explicitly_labelled():
    provenance = dataset_provenance({"dataset": {"root": "tests/fixtures/synthetic"}})
    assert provenance["synthetic"] is True
    assert "SYNTHETIC" in provenance["label"]
    assert "clinical" in provenance["label"].lower()


def test_real_config_is_not_labelled_synthetic():
    provenance = dataset_provenance({"dataset": {"root": "data/raw/icbhi"}})
    assert provenance["synthetic"] is False
    assert "REAL" in provenance["label"]
