"""Stage 7 — Robustness tests under controlled signal changes.

Runs the robustness suite on validation (or train) cycles only — never the
official test split — and writes summary JSON/Markdown and a comparison
figure. Signal corruption is not clinical validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

from respiraqa.evaluation.robustness import run_robustness_suite
from respiraqa.features.acoustic_features import extract_features
from respiraqa.models.baseline import load_model
from respiraqa.utils.config import load_config, resolve_config_paths
from respiraqa.utils.reproducibility import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_cycle_signal(audio_path: str):
    """Load a mono recording (native rate) for robustness transforms."""
    from respiraqa.data.audio_loader import load_audio

    return load_audio(audio_path, target_sample_rate=None, mono=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run robustness experiments")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Cycle partition to probe (default: train/development data)",
    )
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    training = config.get("training", {})
    model_output = training.get("model_output", "models/respiraqa_baseline.joblib")
    metadata_output = training.get("metadata_output", "models/model_metadata.json")

    cycles = pd.read_csv(dataset["cycles_path"])
    if cycles.empty:
        print("No cycles table found; run scripts/03_extract_cycles.py first.")
        return 2
    probe_rows = cycles[cycles["official_partition"] == args.split]
    if probe_rows.empty:
        probe_rows = cycles
        print(f"Warning: no '{args.split}' cycles found; using all cycles as development data.")

    model, metadata = load_model(model_output, metadata_output)
    names = list(metadata.get("feature_names", []))

    results = run_robustness_suite(
        model=model,
        cycles=probe_rows,
        load_cycle_signal=load_cycle_signal,
        feature_extractor=extract_features,
        feature_names=names or None,
    )

    robustness_dir = Path(config["reports"]["output_dir"]) / "robustness"
    figures_dir = Path(config["reports"]["output_dir"]) / "figures"

    payload = {
        "probe_split": args.split,
        "n_probe_cycles": len(probe_rows),
        "note": (
            "Signal-corruption experiments; not clinical validation. "
            "Never used to tune against the official test split."
        ),
        "results": results,
    }
    write_json(payload, str(robustness_dir / "robustness_summary.json"))

    markdown_lines = [
        "# Robustness Summary",
        "",
        f"- Probe split: `{args.split}` ({len(probe_rows)} cycles)",
        "",
        "| condition | n tested | accuracy | macro F1 | d accuracy | d macro F1 | errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        accuracy = result["accuracy"]
        macro_f1 = result["macro_f1"]
        d_accuracy = result["accuracy_difference"]
        d_f1 = result["macro_f1_difference"]
        markdown_lines.append(
            f"| {result['condition']} | {result['n_cycles']} | "
            f"{'-' if accuracy is None else f'{accuracy:.4f}'} | "
            f"{'-' if macro_f1 is None else f'{macro_f1:.4f}'} | "
            f"{'-' if d_accuracy is None else f'{d_accuracy:+.4f}'} | "
            f"{'-' if d_f1 is None else f'{d_f1:+.4f}'} | "
            f"{len(result['errors'])} |"
        )
    markdown_lines.append("")
    (robustness_dir / "robustness_summary.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )

    # Comparison figure across conditions.
    names = [result["condition"] for result in results]
    accuracies = [result["accuracy"] for result in results]
    macro_f1s = [result["macro_f1"] for result in results]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    axis.plot(list(x), accuracies, marker="o", label="Accuracy")
    axis.plot(list(x), macro_f1s, marker="s", label="Macro F1")
    axis.set_xticks(list(x))
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Score")
    axis.legend()
    axis.set_title("Model Robustness Across Conditions")
    figure.tight_layout()
    figure.savefig(figures_dir / "robustness_comparison.png", dpi=150)
    plt.close(figure)

    print(f"Robustness report written to {robustness_dir / 'robustness_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
