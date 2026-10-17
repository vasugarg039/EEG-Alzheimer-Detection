"""Tests for data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import (
    ValidationReport,
    validate_metadata_schema,
    validate_eeg_file,
    check_recording_availability,
    VALID_GROUPS,
    GROUP_NAMES,
    REQUIRED_COLUMNS,
)


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_default_valid(self):
        report = ValidationReport()
        assert report.is_valid is True
        assert report.errors == []
        assert report.warnings == []

    def test_add_error(self):
        report = ValidationReport()
        report.add_error("Test error")
        assert report.is_valid is False
        assert "Test error" in report.errors

    def test_add_warning(self):
        report = ValidationReport()
        report.add_warning("Test warning")
        assert report.is_valid is True
        assert "Test warning" in report.warnings

    def test_summary(self):
        report = ValidationReport()
        report.n_subjects = 10
        report.n_valid = 8
        report.group_counts = {"A": 5, "C": 3}
        report.add_error("Test error")
        summary = report.summary()
        assert "FAILED" in summary
        assert "Test error" in summary


class TestValidateMetadataSchema:
    """Tests for metadata schema validation."""

    def test_valid_metadata(self, sample_metadata):
        report = validate_metadata_schema(sample_metadata)
        assert report.is_valid is True
        assert report.n_subjects == 3
        assert report.n_valid == 3

    def test_missing_columns(self):
        df = pd.DataFrame({"subject_id": ["sub-001"], "group": ["A"]})
        report = validate_metadata_schema(df)
        assert report.is_valid is False
        assert any("Missing required columns" in e for e in report.errors)

    def test_invalid_group(self, sample_metadata):
        sample_metadata.loc[0, "group"] = "X"
        report = validate_metadata_schema(sample_metadata)
        assert report.is_valid is False
        assert any("Invalid group" in e for e in report.errors)

    def test_duplicate_subject_id(self, sample_metadata):
        sample_metadata.loc[1, "subject_id"] = "sub-001"
        report = validate_metadata_schema(sample_metadata)
        assert report.is_valid is False
        assert any("Duplicate" in e for e in report.errors)

    def test_null_values(self, sample_metadata):
        sample_metadata.loc[0, "age"] = np.nan
        report = validate_metadata_schema(sample_metadata)
        assert report.is_valid is True  # Nulls are warnings, not errors
        assert "age" in report.missing_values

    def test_invalid_age_range(self, sample_metadata):
        sample_metadata.loc[0, "age"] = 150
        report = validate_metadata_schema(sample_metadata)
        assert any("age" in w.lower() for w in report.warnings)

    def test_invalid_mmse_range(self, sample_metadata):
        sample_metadata.loc[0, "mmse"] = 35
        report = validate_metadata_schema(sample_metadata)
        assert any("MMSE" in w for w in report.warnings)

    def test_group_name_mismatch(self, sample_metadata):
        sample_metadata.loc[0, "group_name"] = "Wrong Name"
        report = validate_metadata_schema(sample_metadata)
        assert any("group_name" in w for w in report.warnings)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        report = validate_metadata_schema(df)
        assert report.is_valid is True
        assert report.n_subjects == 0


class TestCheckRecordingAvailability:
    """Tests for recording availability check."""

    def test_all_exist(self, sample_metadata, tmp_path):
        # Create fake files
        for sid in sample_metadata["subject_id"]:
            eeg_dir = tmp_path / sid / "eeg"
            eeg_dir.mkdir(parents=True)
            (eeg_dir / f"{sid}_task-eyesclosed_eeg.set").touch()

        result = check_recording_availability(sample_metadata, tmp_path)
        assert result["recording_exists"].all()

    def test_none_exist(self, sample_metadata, tmp_path):
        result = check_recording_availability(sample_metadata, tmp_path)
        assert not result["recording_exists"].any()

    def test_partial_exist(self, sample_metadata, tmp_path):
        # Create only first subject's file
        sid = sample_metadata["subject_id"].iloc[0]
        eeg_dir = tmp_path / sid / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"{sid}_task-eyesclosed_eeg.set").touch()

        result = check_recording_availability(sample_metadata, tmp_path)
        assert result["recording_exists"].sum() == 1
