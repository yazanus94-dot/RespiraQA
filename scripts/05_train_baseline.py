"""Stage 5 — Train baseline models (DummyClassifier and RandomForest).

Training uses the training split only. A patient-level validation split is
derived from training patients for tuning; the official test split is
never touched here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from respiraqa.features.acoustic_features import feature_names
from respiraqa.models.baseline import build_model, save_model, train_model
from respiraqa.models.prediction import predict_with_confidence
from respiraqa.utils.config import dataset_provenance, load_config, resolve_config_paths
from respiraqa.utils.reproducibility import get_environment_versions, set_random_seed


def split_train_validation(
    features: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a patient-grouped, stratified validation split from train rows.

    Returns:
        (train_indices, validation_indices) as positional indices.
    """
    if validation_fraction <= 0.0:
        return list(range(len(features))), []

    labels = features["class_label"].to_numpy()
    groups = features["patient_id"].to_numpy()
    unique_groups = len(set(groups))
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        # Too few patients for a grouped split; train on all rows.
        return list(range(len(features))), []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    validation_indices: list[int] = []
    train_indices: list[int] = []
    # Take the first fold as validation so labels are represented.
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    if not folds:
        return list(range(len(features))), []
    validation_indices = list(folds[0][1])
    train_indices = list(folds[0][0])
    return train_indices, validation_indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    training = config.get("training", {})
    features_path = dataset["features_path"]
    model_output = training.get("model_output", "models/respiraqa_baseline.joblib")
    metadata_output = training.get(
        "metadata_output", "models/model_metadata.json"
    )
    seed = int(training.get("random_seed", 42))
    validation_fraction = float(training.get("validation_fraction_by_patient", 0.2))

    set_random_seed(seed)
    features = pd.read_csv(features_path)
    train_rows = features[features["official_partition"] == "train"]
    test_rows = features[features["official_partition"] == "test"]

    if train_rows.empty:
        print("No training rows found (official_partition == 'train'). "
              "Check the dataset split file.")
        return 2
    if test_rows.empty:
        print("Warning: no official test rows found; evaluation will be limited.")

    names = feature_names(n_mfcc=int(config.get("features", {}).get("n_mfcc", 13)))
    train_indices, validation_indices = split_train_validation(
        train_rows, validation_fraction, seed
    )
    train_fit = train_rows.iloc[train_indices]
    validation_rows = train_rows.iloc[validation_indices]

    results: dict[str, dict] = {}
    for model_type in ("dummy", "random_forest"):
        model = build_model(model_type, random_state=seed)
        train_model(model, train_fit[names], train_fit["class_label"], names)
        predictions = predict_with_confidence(model, train_fit[names], names)
        train_accuracy = float(np.mean(
            [
                a == b
                for a, b in zip(
                    train_fit["class_label"], predictions["labels"], strict=True
                )
            ]
        ))
        validation_accuracy = None
        if not validation_rows.empty:
            validation_predictions = predict_with_confidence(
                model, validation_rows[names], names
            )
            validation_accuracy = float(np.mean(
                [
                    a == b
                for a, b in zip(
                    validation_rows["class_label"],
                    validation_predictions["labels"],
                    strict=True,
                )
                ]
            ))
        results[model_type] = {
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "n_train": len(train_fit),
            "n_validation": len(validation_rows),
        }
        print(
            f"{model_type}: train accuracy {train_accuracy:.4f}"
            + (f", validation accuracy {validation_accuracy:.4f}" if validation_accuracy is not None else "")
        )

    # Persist the random forest as the primary baseline artifact.
    primary = build_model("random_forest", random_state=seed)
    train_model(primary, train_fit[names], train_fit["class_label"], names)
    save_model(primary, model_output, metadata_output, metadata={
        "model_type": "random_forest",
        "feature_names": names,
        "classes": ["normal", "crackles", "wheezes", "both"],
        "random_seed": seed,
        "training": {
            "n_train_cycles": len(train_fit),
            "n_validation_cycles": len(validation_rows),
            "validation_fraction_by_patient": validation_fraction,
            "official_partitions_used_for_training": ["train"],
            "patient_independent": True,
        },
        "data_provenance": dataset_provenance(config),
        "preprocessing": config.get("processing", {}),
        "features_config": config.get("features", {}),
        "results": results,
        "versions": get_environment_versions(),
    })

    print(f"Model saved to {model_output}")
    print(f"Metadata saved to {metadata_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
