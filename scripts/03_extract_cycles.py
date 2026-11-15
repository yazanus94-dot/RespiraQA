"""Stage 2b — Extract annotated respiratory cycles into a cycle table.

Reads the manifest, converts annotation intervals to per-cycle rows with
the four-class label mapping, and verifies the patient-independent split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from respiraqa.data.dataset import build_cycle_table, verify_patient_split_integrity
from respiraqa.utils.config import load_config, resolve_config_paths
from respiraqa.utils.reproducibility import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract annotated respiratory cycles")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest_path"])
    cycles_path = Path(dataset["cycles_path"])
    min_cycle_seconds = float(dataset.get("min_cycle_seconds", 0.0))

    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}. Run scripts/02_build_manifest.py first.")
        return 2

    manifest = pd.read_csv(manifest_path)
    cycles, issues = build_cycle_table(manifest, min_cycle_seconds=min_cycle_seconds)
    verify_patient_split_integrity(manifest)

    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(cycles_path, index=False)

    class_counts = cycles["class_label"].value_counts().to_dict() if not cycles.empty else {}
    print(f"Cycles written to {cycles_path} ({len(cycles)} cycles)")
    print(f"Class distribution: {class_counts}")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")

    partitions = sorted(set(manifest["official_partition"]))
    write_json(
        {
            "n_cycles": len(cycles),
            "class_counts": {k: int(v) for k, v in class_counts.items()},
            "partitions": partitions,
            "issues": issues,
        },
        str(cycles_path) + ".report.json",
    )
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
