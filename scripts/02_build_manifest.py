"""Stage 2a — Build the recording manifest.

One row per recording: recording_id, patient_id, paths, audio metadata,
and the official patient-independent partition (from the configured
split file when available).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from respiraqa.data.dataset import build_manifest
from respiraqa.utils.config import load_config, resolve_config_paths, resolve_path
from respiraqa.utils.reproducibility import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the recording manifest")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    audio_dir = resolve_path(args.config, dataset["audio_dir"])
    annotation_dir = (
        resolve_path(args.config, dataset["annotation_dir"])
        if dataset.get("annotation_dir")
        else None
    )
    split_file = (
        resolve_path(args.config, dataset["split_file"]) if dataset.get("split_file") else None
    )
    manifest_path = Path(dataset["manifest_path"])

    manifest, issues = build_manifest(audio_dir, annotation_dir, split_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    print(f"Manifest written to {manifest_path} ({len(manifest)} recordings)")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
    write_json({"issues": issues, "n_recordings": len(manifest)}, str(manifest_path) + ".report.json")

    unassigned = (
        int((manifest["official_partition"] == "unassigned").sum()) if not manifest.empty else 0
    )
    if split_file and unassigned:
        print(f"Warning: {unassigned} recordings have no official partition assignment")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
