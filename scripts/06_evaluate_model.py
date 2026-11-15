"""Stage 6 — Evaluate the trained model on the official test split.

Runs once on the untouched, patient-independent test split after model
development. Writes metrics JSON/CSV/MD, predictions, and figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

from respiraqa.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    write_confusion_csv,
)
from respiraqa.evaluation.metrics import compute_metrics
from respiraqa.features.acoustic_features import feature_names
from respiraqa.models.baseline import build_model, load_model, train_model
from respiraqa.models.prediction import predict_with_confidence
from respiraqa.utils.config import load_config, resolve_config_paths
from respiraqa.utils.reproducibility import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def evaluate_split(
    rows: pd.DataFrame,
    names: list[str],
    model,
) -> pd.DataFrame:
    """Produce a per-cycle prediction table for one data split."""
    predictions = predict_with_confidence(model, rows[names], names)
    table = rows[["recording_id", "patient_id", "cycle_id"]].copy()
    table["true_label"] = rows["class_label"].to_numpy()
    table["predicted_label"] = predictions["labels"]
    table["confidence"] = predictions["confidence"]
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train"],
        help="Which official partition to evaluate (default: test)",
    )
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    training = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    features_path = dataset["features_path"]
    model_output = training.get("model_output", "models/respiraqa_baseline.joblib")
    metadata_output = training.get("metadata_output", "models/model_metadata.json")

    features = pd.read_csv(features_path)
    rows = features[features["official_partition"] == args.split]
    if rows.empty:
        print(f"No rows found for partition '{args.split}'.")
        return 2

    model, metadata = load_model(model_output, metadata_output)
    names = list(metadata.get("feature_names", [])) or feature_names()
    if list(rows[names].columns) != names:
        print("Feature-column mismatch between data and trained model.")
        return 2

    predictions = evaluate_split(rows, names, model)

    # Reference comparison: a fresh DummyClassifier fitted on the same split
    # inputs (its training data equals the split's training rows if present).
    train_rows = features[features["official_partition"] == "train"]
    dummy = build_model("dummy")
    if not train_rows.empty:
        train_model(dummy, train_rows[names], train_rows["class_label"], names)
    dummy_predictions = evaluate_split(rows, names, dummy)

    reports_dir = Path(config["reports"]["output_dir"]) / "metrics"
    figures_dir = Path(config["reports"]["output_dir"]) / "figures"

    predictions_path = evaluation_cfg.get("predictions_path", str(reports_dir / "predictions.csv"))
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    metrics = compute_metrics(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
        predictions["confidence"].tolist()
        if predictions["confidence"].notna().any()
        else None,
    )
    dummy_metrics = compute_metrics(
        dummy_predictions["true_label"].tolist(),
        dummy_predictions["predicted_label"].tolist(),
    )
    metrics["dummy_comparison"] = {
        "dummy_accuracy": dummy_metrics["accuracy"],
        "dummy_macro_f1": dummy_metrics["macro_f1"],
        "model_accuracy": metrics["accuracy"],
        "model_macro_f1": metrics["macro_f1"],
    }

    write_json(metrics, evaluation_cfg.get("report_json_path", str(reports_dir / "classification_report.json")))

    classes = list(metrics["classes"])
    matrix = compute_confusion_matrix(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
        classes,
    )
    confusion_csv = evaluation_cfg.get(
        "confusion_csv_path", str(reports_dir / "confusion_matrix.csv")
    )
    write_confusion_csv(matrix, classes, confusion_csv)
    plot_confusion_matrix(
        matrix,
        classes,
        figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix ({args.split} split)",
    )

    # Class distribution figure.
    distribution = metrics["class_distribution"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(list(distribution.keys()), list(distribution.values()), color="#4a7fb5")
    axis.set_ylabel("Cycles")
    axis.set_title(f"Class Distribution ({args.split} split)")
    figure.tight_layout()
    figure.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(figure)

    # Performance comparison figure.
    figure, axis = plt.subplots(figsize=(6, 4))
    labels_plot = ["Accuracy", "Macro F1"]
    model_values = [metrics["accuracy"], metrics["macro_f1"]]
    dummy_values = [dummy_metrics["accuracy"], dummy_metrics["macro_f1"]]
    x = range(len(labels_plot))
    width = 0.35
    axis.bar([i - width / 2 for i in x], model_values, width, label="Random Forest")
    axis.bar([i + width / 2 for i in x], dummy_values, width, label="DummyClassifier")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels_plot)
    axis.set_ylim(0, 1)
    axis.legend()
    axis.set_title("Model vs Dummy Baseline")
    figure.tight_layout()
    figure.savefig(figures_dir / "performance_comparison.png", dpi=150)
    plt.close(figure)

    summary_md = render_summary_md(metrics, args.split, config)
    summary_path = evaluation_cfg.get("summary_md_path", str(reports_dir / "evaluation_summary.md"))
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(summary_md, encoding="utf-8")

    print(f"Evaluation on '{args.split}' split: accuracy {metrics['accuracy']:.4f}, "
          f"macro F1 {metrics['macro_f1']:.4f}")
    print(f"Predictions written to {predictions_path}")
    print(f"Summary written to {summary_path}")
    return 0


def render_summary_md(metrics: dict, split: str, config: dict) -> str:
    """Render the Markdown evaluation summary."""
    event = metrics["event_metrics"]
    dummy = metrics.get("dummy_comparison", {})
    per_class_lines = "\n".join(
        f"| {name} | {values['precision']:.3f} | {values['recall']:.3f} | "
        f"{values['f1_score']:.3f} | {values['support']} |"
        for name, values in metrics["per_class"].items()
    )
    return "\n".join(
        [
            f"# Evaluation Summary ({split} split)",
            "",
            f"- Cycles evaluated: **{metrics['n_cycles']}**",
            f"- Accuracy: **{metrics['accuracy']:.4f}**",
            f"- Macro F1: **{metrics['macro_f1']:.4f}**",
            f"- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**",
            "",
            "## Per-class metrics",
            "",
            "| class | precision | recall | F1 | support |",
            "|---|---|---|---|---|",
            per_class_lines,
            "",
            "## Event metrics (challenge-style definition)",
            "",
            f"- Event sensitivity: **{event['event_sensitivity']:.4f}** "
            f"({event['n_event_cycles']} event cycles)",
            f"- Normal specificity: **{event['normal_specificity']:.4f}** "
            f"({event['n_normal_cycles']} normal cycles)",
            f"- Average score: **{event['average_score']:.4f}**",
            f"- Definition: {event['definition']}",
            "",
            "## Comparison with DummyClassifier",
            "",
            f"- Model accuracy {dummy.get('model_accuracy', float('nan')):.4f} vs "
            f"dummy {dummy.get('dummy_accuracy', float('nan')):.4f}",
            f"- Model macro F1 {dummy.get('model_macro_f1', float('nan')):.4f} vs "
            f"dummy {dummy.get('dummy_macro_f1', float('nan')):.4f}",
            "",
            "## Notes",
            "",
            "- Probabilities reported by the model are not calibrated.",
            "- This evaluation was run once on the patient-independent "
            f"'{split}' split; no tuning was performed against it.",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
