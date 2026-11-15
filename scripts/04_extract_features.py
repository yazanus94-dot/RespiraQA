"""Stage 4 — Acoustic feature extraction.

Loads every cycle's recording, extracts interpretable acoustic features,
and writes the feature table used by training and evaluation.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from respiraqa.features.acoustic_features import extract_features_table
from respiraqa.utils.config import load_config, resolve_config_paths
from respiraqa.utils.reproducibility import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract acoustic features from cycles")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args(argv)

    config = resolve_config_paths(load_config(args.config), args.config)
    dataset = config["dataset"]
    cycles_path = dataset["cycles_path"]
    features_path = dataset["features_path"]
    feature_cfg = config.get("features", {})

    cycles = pd.read_csv(cycles_path)
    if cycles.empty:
        print(f"No cycles found in {cycles_path}. Run scripts/03_extract_cycles.py first.")
        return 2

    features, issues = extract_features_table(
        cycles,
        progress=not args.quiet,
        n_mfcc=int(feature_cfg.get("n_mfcc", 13)),
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 512)),
    )

    output = pd.DataFrame(features)
    output.to_csv(features_path, index=False)

    feature_columns = [c for c in output.columns if c not in
                       {"recording_id", "patient_id", "cycle_id", "class_label",
                        "official_partition"}]
    numeric = output[feature_columns].to_numpy(dtype=float)
    non_finite = int((~np.isfinite(numeric)).sum())

    print(f"Features written to {features_path} ({len(output)} rows x {len(feature_columns)} features)")
    print(f"Class distribution: {output['class_label'].value_counts().to_dict()}")
    print(f"Non-finite feature values: {non_finite}")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")

    write_json(
        {
            "n_rows": len(output),
            "n_features": len(feature_columns),
            "feature_names": feature_columns,
            "n_skipped": len(issues),
            "issues": issues,
        },
        features_path + ".report.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
