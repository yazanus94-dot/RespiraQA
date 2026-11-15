"""RespiraQA command-line orchestrator.

``python -m respiraqa run-all --config config/config.yaml`` runs the full
pipeline in order. Individual stages remain runnable via scripts/01..07.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    ("validate", "01_validate_dataset.py"),
    ("manifest", "02_build_manifest.py"),
    ("cycles", "03_extract_cycles.py"),
    ("features", "04_extract_features.py"),
    ("train", "05_train_baseline.py"),
    ("evaluate", "06_evaluate_model.py"),
    ("robustness", "07_robustness_tests.py"),
]


def run_stage(script: Path, config: str) -> int:
    """Run one pipeline script with the given config."""
    command = [sys.executable, str(script), "--config", config]
    print(f"\n=== Running {script.name} ===")
    result = subprocess.run(command, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="respiraqa", description="RespiraQA pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_all = subparsers.add_parser("run-all", help="Run the full pipeline in order")
    run_all.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args(argv)

    if args.command == "run-all":
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        for _, script_name in STAGES:
            exit_code = run_stage(scripts_dir / script_name, args.config)
            if exit_code != 0:
                print(f"Stage {script_name} exited with code {exit_code}; aborting run-all.")
                return exit_code
        print("\nAll pipeline stages completed successfully.")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
