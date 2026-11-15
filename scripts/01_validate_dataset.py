"""Stage 1 — Dataset validation.

Validates audio/annotation pairing, readability, sample metadata, and
annotation contents. Writes machine-readable JSON and human-readable
Markdown reports and exits non-zero on critical errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from respiraqa.data.annotations import (
    read_annotation_file,
    validate_annotation_dataframe,
)
from respiraqa.data.audio_loader import load_audio
from respiraqa.data.dataset import discover_dataset_files
from respiraqa.utils.config import load_config, resolve_config_paths, resolve_path
from respiraqa.utils.reproducibility import write_json


def validate_dataset(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Validate the configured dataset and return the report payload."""
    dataset = config.get("dataset", {})
    audio_dir = resolve_path(config_path, dataset.get("audio_dir", ""))
    annotation_dir = (
        resolve_path(config_path, dataset["annotation_dir"])
        if dataset.get("annotation_dir")
        else audio_dir
    )

    paired_audio, paired_annotations, unpaired_audio, unpaired_annotations = (
        discover_dataset_files(audio_dir, annotation_dir)
    )

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    recordings: list[dict[str, Any]] = []

    for wav in unpaired_audio:
        warnings.append(
            {"code": "audio_without_annotations", "message": f"{wav.name}: no matching annotation"}
        )
    for annotation in unpaired_annotations:
        warnings.append(
            {"code": "annotations_without_audio", "message": f"{annotation.name}: no matching audio"}
        )

    for wav, annotation in zip(paired_audio, paired_annotations, strict=True):
        record: dict[str, Any] = {"recording_id": wav.stem, "annotation_file": annotation.name}
        try:
            audio, info = load_audio(wav)
            record.update(
                {
                    "sample_rate": info["native_sample_rate"],
                    "channels": info["native_channels"],
                    "duration_seconds": info["duration_seconds"],
                    "num_samples": info["native_num_samples"],
                }
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            errors.append({"code": "unreadable_audio", "message": f"{wav.name}: {exc}"})
            recordings.append(record)
            continue

        try:
            annotations = read_annotation_file(annotation)
        except (ValueError, FileNotFoundError) as exc:
            errors.append({"code": "invalid_annotations", "message": f"{annotation.name}: {exc}"})
            recordings.append(record)
            continue

        issues = validate_annotation_dataframe(annotations, info["duration_seconds"])
        for issue in issues:
            target = errors if issue["severity"] == "error" else warnings
            target.append(
                {"code": issue["code"], "message": f"{wav.name}: {issue['message']}"}
            )

        record["annotation_rows"] = int(len(annotations))
        recordings.append(record)

    return {
        "audio_dir": str(audio_dir),
        "annotation_dir": str(annotation_dir),
        "n_audio_files": len(paired_audio) + len(unpaired_audio),
        "n_annotation_files": len(paired_annotations) + len(unpaired_annotations),
        "n_paired_recordings": len(paired_audio),
        "n_unpaired_audio": len(unpaired_audio),
        "n_unpaired_annotations": len(unpaired_annotations),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "recordings": recordings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown validation report."""
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Audio directory: `{report['audio_dir']}`",
        f"- Annotation directory: `{report['annotation_dir']}`",
        f"- Audio files: {report['n_audio_files']}",
        f"- Annotation files: {report['n_annotation_files']}",
        f"- Paired recordings: {report['n_paired_recordings']}",
        f"- Unpaired audio: {report['n_unpaired_audio']}",
        f"- Unpaired annotations: {report['n_unpaired_annotations']}",
        f"- Errors: {report['n_errors']}",
        f"- Warnings: {report['n_warnings']}",
        "",
    ]
    if report["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in report["errors"]:
            lines.append(f"- **{error['code']}**: {error['message']}")
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- *{warning['code']}*: {warning['message']}")
        lines.append("")
    lines.append("## Recordings")
    lines.append("")
    lines.append(
        "| recording | sample rate | channels | duration (s) | annotation rows |"
    )
    lines.append("|---|---|---|---|---|")
    for record in report["recordings"]:
        lines.append(
            f"| {record['recording_id']} | {record.get('sample_rate', '-')} | "
            f"{record.get('channels', '-')} | {record.get('duration_seconds', '-')} | "
            f"{record.get('annotation_rows', '-')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the respiratory-sound dataset")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    reports_dir = Path(config["reports"]["output_dir"]) / "metrics"
    json_path = reports_dir / "dataset_validation.json"
    markdown_path = reports_dir / "dataset_validation.md"

    report = validate_dataset(config, args.config)
    write_json(report, str(json_path))
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Validation report written to {json_path}")
    print(f"Errors: {report['n_errors']}, Warnings: {report['n_warnings']}")
    return 1 if report["n_errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
