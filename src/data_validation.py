"""Data and schema validation for metadata and EEG recordings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_GROUPS = {"A", "F", "C"}
GROUP_NAMES = {
    "A": "Alzheimer's Disease",
    "F": "Frontotemporal Dementia",
    "C": "Healthy Control",
}

REQUIRED_COLUMNS = [
    "subject_id",
    "group",
    "age",
    "gender",
    "mmse",
    "channels",
    "sampling_frequency",
    "samples",
    "duration_seconds",
    "duration_minutes",
    "status",
    "group_name",
]


@dataclass
class ValidationReport:
    """Container for validation results."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_subjects: int = 0
    n_valid: int = 0
    group_counts: dict[str, int] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Subjects: {self.n_subjects}",
            f"  Valid: {self.n_valid}",
            f"  Groups: {self.group_counts}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate_metadata_schema(df: pd.DataFrame) -> ValidationReport:
    """Validate metadata DataFrame schema and content.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame to validate.

    Returns
    -------
    ValidationReport
        Validation results.
    """
    report = ValidationReport()
    report.n_subjects = len(df)

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
        return report

    # Check for null values
    null_counts = df.isnull().sum()
    report.missing_values = {k: int(v) for k, v in null_counts.items() if v > 0}
    if report.missing_values:
        report.add_warning(f"Columns with missing values: {report.missing_values}")

    # Validate subject_id
    if df["subject_id"].duplicated().any():
        dups = df["subject_id"][df["subject_id"].duplicated()].tolist()
        report.add_error(f"Duplicate subject_ids: {dups}")

    # Validate group labels
    invalid_groups = set(df["group"].unique()) - VALID_GROUPS
    if invalid_groups:
        report.add_error(f"Invalid group labels: {invalid_groups}")

    # Validate group_name consistency
    for _, row in df.iterrows():
        expected_name = GROUP_NAMES.get(row["group"])
        if expected_name and row["group_name"] != expected_name:
            report.add_warning(
                f"Subject {row['subject_id']}: group='{row['group']}' "
                f"but group_name='{row['group_name']}' (expected '{expected_name}')"
            )

    # Validate numeric ranges
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        report.add_warning("Some age values are outside 0-120 range")

    if (df["mmse"] < 0).any() or (df["mmse"] > 30).any():
        report.add_warning("Some MMSE values are outside 0-30 range")

    if (df["sampling_frequency"] <= 0).any():
        report.add_error("Sampling frequency must be positive")

    if (df["channels"] <= 0).any():
        report.add_error("Channel count must be positive")

    # Count valid subjects
    report.n_valid = len(df[df["group"].isin(VALID_GROUPS)])
    report.group_counts = df["group"].value_counts().to_dict()

    return report


def validate_eeg_file(
    file_path: str | Path,
    expected_channels: int = 19,
    expected_sfreq: float = 500.0,
) -> dict[str, Any]:
    """Validate an EEG file without loading full data.

    Parameters
    ----------
    file_path : str or Path
        Path to EEG file.
    expected_channels : int
        Expected number of channels.
    expected_sfreq : float
        Expected sampling frequency.

    Returns
    -------
    dict
        Validation results with keys: valid, n_channels, sfreq, duration, error.
    """
    result: dict[str, Any] = {
        "valid": False,
        "n_channels": None,
        "sfreq": None,
        "duration": None,
        "error": None,
    }

    path = Path(file_path)
    if not path.exists():
        result["error"] = f"File not found: {path}"
        return result

    try:
        import mne

        raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
        result["n_channels"] = len(raw.ch_names)
        result["sfreq"] = raw.info["sfreq"]
        result["duration"] = raw.n_times / raw.info["sfreq"]

        if result["n_channels"] != expected_channels:
            result["error"] = (
                f"Channel count mismatch: got {result['n_channels']}, "
                f"expected {expected_channels}"
            )
            return result

        if result["sfreq"] != expected_sfreq:
            result["error"] = (
                f"Sampling rate mismatch: got {result['sfreq']}, "
                f"expected {expected_sfreq}"
            )
            return result

        result["valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def check_recording_availability(
    df: pd.DataFrame,
    data_root: str | Path,
) -> pd.DataFrame:
    """Check which recordings exist on disk.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame with subject_id column.
    data_root : str or Path
        Root directory of the EEG dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'recording_path' and 'recording_exists' columns.
    """
    root = Path(data_root)
    paths = []
    exists = []

    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        # BIDS-like structure: sub-001/eeg/sub-001_task-eyesclosed_eeg.set
        eeg_file = root / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"
        paths.append(str(eeg_file))
        exists.append(eeg_file.exists())

    result = df.copy()
    result["recording_path"] = paths
    result["recording_exists"] = exists

    n_found = sum(exists)
    logger.info("Recording availability: %d/%d found", n_found, len(df))

    return result
