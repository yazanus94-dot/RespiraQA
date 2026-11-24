"""Reading and validating respiratory-sound annotation files.

Each annotation row records four values:

    1. Respiratory-cycle start time in seconds
    2. Respiratory-cycle end time in seconds
    3. Crackle indicator (0 or 1)
    4. Wheeze indicator (0 or 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from respiraqa import SUPPORTED_CLASSES

ANNOTATION_COLUMNS = ("start_time", "end_time", "crackle_label", "wheeze_label")

CRACKLE_WHEEZE_TO_CLASS = {
    (0, 0): "normal",
    (1, 0): "crackles",
    (0, 1): "wheezes",
    (1, 1): "both",
}

CLASS_TO_CRACKLE_WHEEZE = {v: k for k, v in CRACKLE_WHEEZE_TO_CLASS.items()}


class AnnotationError(ValueError):
    """Raised when annotation content is invalid."""


def class_from_labels(crackle: int, wheeze: int) -> str:
    """Map a (crackle, wheeze) pair to one of the four supported classes."""
    key = (int(crackle), int(wheeze))
    if key not in CRACKLE_WHEEZE_TO_CLASS:
        raise AnnotationError(f"Unsupported label pair: crackle={crackle}, wheeze={wheeze}")
    return CRACKLE_WHEEZE_TO_CLASS[key]


def read_annotation_file(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-separated annotation file into a validated DataFrame.

    Args:
        path: Path to the annotation text file.

    Returns:
        DataFrame with columns ``start_time``, ``end_time``,
        ``crackle_label``, ``wheeze_label``.

    Raises:
        FileNotFoundError: If the file does not exist.
        AnnotationError: If a row does not have exactly four numeric
            columns, times are invalid, or labels are not 0/1.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise AnnotationError(
                    f"{file_path.name}:{line_number}: expected 4 columns, got {len(fields)}"
                )
            try:
                start_time, end_time, crackle, wheeze = (float(field) for field in fields)
            except ValueError as exc:
                raise AnnotationError(
                    f"{file_path.name}:{line_number}: non-numeric value ({exc})"
                ) from exc

            for name, value in (
                ("start_time", start_time),
                ("end_time", end_time),
            ):
                if value != value or value in (float("inf"), float("-inf")):
                    raise AnnotationError(f"{file_path.name}:{line_number}: {name} is not finite")
            if start_time < 0:
                raise AnnotationError(
                    f"{file_path.name}:{line_number}: negative start_time ({start_time})"
                )
            if end_time <= start_time:
                raise AnnotationError(
                    f"{file_path.name}:{line_number}: end_time ({end_time}) must be "
                    f"greater than start_time ({start_time})"
                )
            for name, value in (("crackle", crackle), ("wheeze", wheeze)):
                if value not in (0.0, 1.0):
                    raise AnnotationError(
                        f"{file_path.name}:{line_number}: {name}_label must be 0 or 1, got {value}"
                    )

            records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "crackle_label": int(crackle),
                    "wheeze_label": int(wheeze),
                }
            )

    if not records:
        return pd.DataFrame(columns=list(ANNOTATION_COLUMNS))

    frame = pd.DataFrame.from_records(records)
    frame["class_label"] = [
        class_from_labels(c, w)
        for c, w in zip(frame["crackle_label"], frame["wheeze_label"], strict=True)
    ]
    return frame


def validate_annotation_dataframe(
    frame: pd.DataFrame | None,
    recording_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Run extra cross-checks on parsed annotations.

    Args:
        frame: DataFrame returned by :func:`read_annotation_file` (or None).
        recording_duration: Optional recording length in seconds; intervals
            beyond this are reported as errors.

    Returns:
        List of issue dictionaries with keys ``severity`` (``error`` or
        ``warning``), ``code``, and ``message``. An empty list means the
        annotations passed all checks.
    """
    issues: list[dict[str, Any]] = []

    if frame is None or frame.empty:
        issues.append(
            {
                "severity": "warning",
                "code": "no_annotations",
                "message": "Annotation file contains no rows.",
            }
        )
        return issues

    beyond = frame[frame["end_time"] > recording_duration] if recording_duration else []
    for index, row in beyond.iterrows():
        issues.append(
            {
                "severity": "error",
                "code": "interval_beyond_duration",
                "message": (
                    f"Row {index}: end_time {row['end_time']:.3f}s exceeds recording "
                    f"duration {recording_duration:.3f}s."
                ),
            }
        )

    if set(frame["class_label"]) - set(SUPPORTED_CLASSES):
        issues.append(
            {
                "severity": "error",
                "code": "invalid_class_label",
                "message": "One or more rows map to an unsupported class label.",
            }
        )

    return issues
