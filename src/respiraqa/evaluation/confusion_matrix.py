"""Confusion-matrix computation, CSV export, and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    classes: list[str],
) -> np.ndarray:
    """Confusion matrix with rows=true, cols=predicted for given class order."""
    from sklearn.metrics import confusion_matrix

    if not classes:
        raise ValueError("classes list must not be empty")
    return confusion_matrix(true_labels, predicted_labels, labels=classes)


def confusion_matrix_to_records(matrix: np.ndarray, classes: list[str]) -> list[dict[str, Any]]:
    """Convert a confusion matrix to row dicts for CSV export."""
    records: list[dict[str, Any]] = []
    for i, true_class in enumerate(classes):
        record: dict[str, Any] = {"true_class": true_class}
        for j, predicted_class in enumerate(classes):
            record[f"pred_{predicted_class}"] = int(matrix[i, j])
        records.append(record)
    return records


def write_confusion_csv(matrix: np.ndarray, classes: list[str], path: str | Path) -> None:
    """Write the confusion matrix to CSV, creating parents as needed."""
    import pandas as pd

    records = confusion_matrix_to_records(matrix, classes)
    frame = pd.DataFrame.from_records(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render the confusion matrix to a PNG file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes, rotation=30, ha="right")
    axis.set_yticklabels(classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
