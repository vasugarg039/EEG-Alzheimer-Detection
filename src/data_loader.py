"""Metadata and EEG data loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .data_validation import (
    check_recording_availability,
    validate_eeg_file,
    validate_metadata_schema,
)
from .utils import ensure_dir

logger = logging.getLogger(__name__)


class EEGDataset:
    """Main dataset class for EEG Alzheimer's detection.

    Handles metadata loading, validation, and EEG file access.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data_root: str | Path | None = None,
    ):
        """Initialize dataset.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        data_root : str or Path, optional
            Override for EEG data root path.
        """
        self.config = config
        self.paths = config["paths"]
        self.eeg_config = config["eeg"]

        self.data_root = Path(data_root or self.paths["data_root"])
        self.metadata_path = Path(self.paths["metadata"])

        self.metadata: pd.DataFrame | None = None
        self.validation_report = None
        self._eeg_cache: dict[str, mne.io.Raw] = {}

    def load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata CSV.

        Returns
        -------
        pd.DataFrame
            Validated metadata.

        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist.
        ValueError
            If metadata validation fails.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)
        logger.info("Loaded metadata: %d subjects", len(self.metadata))

        self.validation_report = validate_metadata_schema(self.metadata)
        if not self.validation_report.is_valid:
            raise ValueError(
                f"Metadata validation failed:\n{self.validation_report.summary()}"
            )

        # Check recording availability
        self.metadata = check_recording_availability(self.metadata, self.data_root)

        logger.info(self.validation_report.summary())
        return self.metadata

    def get_subjects_by_group(self, group: str) -> list[str]:
        """Get subject IDs for a specific group.

        Parameters
        ----------
        group : str
            Group code: 'A', 'F', or 'C'.

        Returns
        -------
        list of str
            Subject IDs.
        """
        if self.metadata is None:
            self.load_metadata()
        return self.metadata[self.metadata["group"] == group]["subject_id"].tolist()

    def get_recording_path(self, subject_id: str) -> Path:
        """Get the EEG recording path for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier.

        Returns
        -------
        Path
            Path to the .set file.
        """
        return (
            self.data_root
            / subject_id
            / "eeg"
            / f"{subject_id}_task-eyesclosed_eeg.set"
        )

    def load_eeg(
        self,
        subject_id: str,
        preload: bool = True,
    ) -> mne.io.Raw:
        """Load EEG data for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        preload : bool
            Whether to preload data into memory.

        Returns
        -------
        mne.io.Raw
            Loaded EEG data.
        """
        if subject_id in self._eeg_cache:
            return self._eeg_cache[subject_id]

        path = self.get_recording_path(subject_id)
        if not path.exists():
            raise FileNotFoundError(f"EEG file not found: {path}")

        logger.info("Loading EEG for %s from %s", subject_id, path)
        raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)

        # Validate
        validation = validate_eeg_file(
            path,
            expected_channels=self.eeg_config["expected_channels"],
            expected_sfreq=self.eeg_config["expected_sampling_rate"],
        )
        if not validation["valid"]:
            raise ValueError(
                f"EEG validation failed for {subject_id}: {validation['error']}"
            )

        self._eeg_cache[subject_id] = raw
        return raw

    def load_all_eeg(self) -> dict[str, mne.io.Raw]:
        """Load all available EEG recordings.

        Returns
        -------
        dict
            Mapping of subject_id to Raw objects.
        """
        if self.metadata is None:
            self.load_metadata()

        available = self.metadata[self.metadata["recording_exists"]]
        results = {}
        for _, row in available.iterrows():
            sid = row["subject_id"]
            try:
                results[sid] = self.load_eeg(sid)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", sid, exc)

        logger.info("Loaded %d/%d EEG recordings", len(results), len(available))
        return results

    def get_dataset_summary(self) -> dict[str, Any]:
        """Get a summary of the dataset.

        Returns
        -------
        dict
            Summary statistics.
        """
        if self.metadata is None:
            self.load_metadata()

        summary = {
            "n_subjects": len(self.metadata),
            "group_counts": self.metadata["group"].value_counts().to_dict(),
            "recording_available": int(self.metadata["recording_exists"].sum()),
            "age_range": [
                float(self.metadata["age"].min()),
                float(self.metadata["age"].max()),
            ],
            "mmse_range": [
                float(self.metadata["mmse"].min()),
                float(self.metadata["mmse"].max()),
            ],
            "duration_range_min": [
                float(self.metadata["duration_minutes"].min()),
                float(self.metadata["duration_minutes"].max()),
            ],
            "gender_counts": self.metadata["gender"].value_counts().to_dict(),
        }
        return summary

    def clear_cache(self) -> None:
        """Clear the EEG data cache."""
        self._eeg_cache.clear()
