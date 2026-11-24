"""Recording manifest and respiratory-cycle table construction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from respiraqa.data.annotations import (
    CRACKLE_WHEEZE_TO_CLASS,
    AnnotationError,
    read_annotation_file,
)
from respiraqa.data.audio_loader import AudioLoadError

MANIFEST_COLUMNS = (
    "recording_id",
    "patient_id",
    "audio_path",
    "annotation_path",
    "sample_rate",
    "channels",
    "duration_seconds",
    "official_partition",
)

CYCLE_COLUMNS = (
    "recording_id",
    "patient_id",
    "cycle_id",
    "start_time",
    "end_time",
    "crackle_label",
    "wheeze_label",
    "class_label",
    "audio_path",
    "sample_rate",
    "official_partition",
)

VALID_PARTITIONS = {"train", "test", "validation"}


class DatasetError(RuntimeError):
    """Raised when the dataset layout prevents building a manifest."""


def infer_patient_id(recording_id: str) -> str:
    """Extract the patient identifier from an ICBHI-style recording name.

    ICBHI recording names look like ``101_1b1_Al_sc_Litt3200``; the patient
    id is the part before the first underscore.
    """
    match = re.match(r"(\d+)", recording_id)
    if not match:
        raise DatasetError(f"Cannot infer patient id from recording id '{recording_id}'")
    return match.group(1)


def read_split_file(path: str | Path) -> dict[str, str]:
    """Read a patient->partition mapping file.

    Each non-empty line contains ``patient_id`` and ``partition`` separated
    by whitespace. Partition must be one of train/test/validation.
    """
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    mapping: dict[str, str] = {}
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise DatasetError(
                    f"{split_path.name}:{line_number}: expected 'patient_id partition'"
                )
            patient_id, partition = fields
            if partition not in VALID_PARTITIONS:
                raise DatasetError(
                    f"{split_path.name}:{line_number}: unknown partition '{partition}'"
                )
            if patient_id in mapping and mapping[patient_id] != partition:
                raise DatasetError(
                    f"{split_path.name}:{line_number}: patient {patient_id} assigned to "
                    f"multiple partitions"
                )
            mapping[patient_id] = partition
    return mapping


def verify_patient_split_integrity(manifest: pd.DataFrame) -> None:
    """Raise if any patient appears in more than one partition.

    Args:
        manifest: Manifest DataFrame with ``patient_id`` and
            ``official_partition`` columns.

    Raises:
        DatasetError: If patient leakage across partitions is detected.
    """
    if "patient_id" not in manifest.columns or "official_partition" not in manifest.columns:
        raise DatasetError("Manifest lacks patient_id/official_partition columns")

    leaked = manifest.groupby("patient_id")["official_partition"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        patients = ", ".join(sorted(leaked.index.astype(str)))
        raise DatasetError(
            f"Patient leakage detected: patients assigned to multiple partitions: {patients}"
        )


def discover_dataset_files(
    audio_dir: str | Path,
    annotation_dir: str | Path | None = None,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Match audio and annotation files by recording basename.

    Args:
        audio_dir: Directory containing ``.wav`` files.
        annotation_dir: Directory containing ``.txt`` annotation files. If
            None, annotations are expected next to the audio files.

    Returns:
        Tuple ``(paired_audio, paired_annotations, unpaired_audio,
        unpaired_annotations)``.

    Raises:
        DatasetError: If the audio directory does not exist.
    """
    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        raise DatasetError(f"Audio directory not found: {audio_path}")
    annotation_path = Path(annotation_dir) if annotation_dir else audio_path

    audio_files = sorted(p for p in audio_path.iterdir() if p.suffix.lower() == ".wav")
    annotation_files = {
        p.stem.lower(): p for p in annotation_path.iterdir() if p.suffix.lower() == ".txt"
    }

    paired_audio: list[Path] = []
    paired_annotations: list[Path] = []
    unpaired_audio: list[Path] = []
    for wav in audio_files:
        annotation = annotation_files.get(wav.stem.lower())
        if annotation is None:
            unpaired_audio.append(wav)
        else:
            paired_audio.append(wav)
            paired_annotations.append(annotation)

    paired_stems = {p.stem.lower() for p in paired_audio}
    unpaired_annotations = [p for stem, p in annotation_files.items() if stem not in paired_stems]
    return paired_audio, paired_annotations, unpaired_audio, unpaired_annotations


def build_manifest(
    audio_dir: str | Path,
    annotation_dir: str | Path | None = None,
    split_file: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the recording manifest.

    Returns:
        Tuple ``(manifest, issues)``. The manifest has the columns in
        :data:`MANIFEST_COLUMNS`; ``issues`` records unmatched files and
        recordings whose audio could not be opened.
    """
    paired_audio, paired_annotations, unpaired_audio, unpaired_annotations = discover_dataset_files(
        audio_dir, annotation_dir
    )

    splits = read_split_file(split_file) if split_file else {}

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for wav, annotation in zip(paired_audio, paired_annotations, strict=True):
        recording_id = wav.stem
        patient_id = infer_patient_id(recording_id)
        sample_rate = None
        channels = None
        duration = None
        try:
            info = sf_info(wav)
            sample_rate = info["sample_rate"]
            channels = info["channels"]
            duration = info["duration_seconds"]
        except (AudioLoadError, FileNotFoundError) as exc:
            issues.append(
                {
                    "severity": "error",
                    "code": "unreadable_audio",
                    "message": f"{recording_id}: {exc}",
                }
            )

        official_partition = splits.get(patient_id, "unassigned")
        rows.append(
            {
                "recording_id": recording_id,
                "patient_id": patient_id,
                "audio_path": str(wav),
                "annotation_path": str(annotation),
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_seconds": duration,
                "official_partition": official_partition,
            }
        )

    for wav in unpaired_audio:
        issues.append(
            {
                "severity": "warning",
                "code": "audio_without_annotations",
                "message": f"{wav.stem}: audio file has no matching annotation file",
            }
        )
    for annotation in unpaired_annotations:
        issues.append(
            {
                "severity": "warning",
                "code": "annotations_without_audio",
                "message": f"{annotation.stem}: annotation file has no matching audio file",
            }
        )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if not manifest.empty and splits:
        verify_patient_split_integrity(manifest)
    return manifest, issues


def sf_info(path: str | Path) -> dict[str, Any]:
    """Return cheap metadata (sample rate, channels, duration) for a WAV file."""
    import soundfile as _sf

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        info = _sf.info(file_path)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - surfaced via tests
        raise AudioLoadError(f"Cannot read audio metadata from {file_path}: {exc}") from exc
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "num_samples": int(info.frames),
        "duration_seconds": float(info.frames / info.samplerate) if info.samplerate else 0.0,
    }


def build_cycle_table(
    manifest: pd.DataFrame,
    min_cycle_seconds: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract annotated respiratory cycles for every manifest recording.

    Args:
        manifest: Manifest DataFrame as produced by :func:`build_manifest`.
        min_cycle_seconds: Skip cycles shorter than this duration.

    Returns:
        Tuple ``(cycles, issues)`` with cycle rows following
        :data:`CYCLE_COLUMNS`. Issues report skipped/invalid cycles.
    """
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        if row.get("sample_rate") is None:
            continue
        try:
            annotations = read_annotation_file(row["annotation_path"])
        except AnnotationError as exc:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_annotations",
                    "message": f"{row['recording_id']}: {exc}",
                }
            )
            continue
        except FileNotFoundError as exc:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_annotations",
                    "message": f"{row['recording_id']}: {exc}",
                }
            )
            continue

        for index, cycle in annotations.iterrows():
            start_time = float(cycle["start_time"])
            end_time = float(cycle["end_time"])
            if end_time > float(row["duration_seconds"]):
                issues.append(
                    {
                        "severity": "error",
                        "code": "cycle_beyond_duration",
                        "message": (
                            f"{row['recording_id']} cycle {index}: end_time {end_time:.3f}s "
                            f"exceeds duration {float(row['duration_seconds']):.3f}s"
                        ),
                    }
                )
                continue
            if (end_time - start_time) < min_cycle_seconds:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "cycle_too_short",
                        "message": (
                            f"{row['recording_id']} cycle {index}: duration "
                            f"{end_time - start_time:.3f}s below minimum "
                            f"{min_cycle_seconds:.3f}s"
                        ),
                    }
                )
                continue
            key = (int(cycle["crackle_label"]), int(cycle["wheeze_label"]))
            class_label = CRACKLE_WHEEZE_TO_CLASS[key]
            records.append(
                {
                    "recording_id": row["recording_id"],
                    "patient_id": row["patient_id"],
                    "cycle_id": f"{row['recording_id']}_{index:04d}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "crackle_label": key[0],
                    "wheeze_label": key[1],
                    "class_label": class_label,
                    "audio_path": row["audio_path"],
                    "sample_rate": row["sample_rate"],
                    "official_partition": row["official_partition"],
                }
            )

    cycles = pd.DataFrame(records, columns=list(CYCLE_COLUMNS))
    return cycles, issues
