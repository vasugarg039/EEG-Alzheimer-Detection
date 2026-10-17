"""Tests for utilities module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.utils import (
    setup_logging,
    set_random_seed,
    ensure_dir,
    get_timestamp,
    format_duration,
    sanitize_filename,
    validate_file_path,
)


class TestSetupLogging:
    """Tests for logging setup."""

    def test_creates_log_file(self, tmp_path):
        logger = setup_logging(tmp_path, log_to_console=False)
        assert logger is not None
        # Check that a log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0


class TestSetRandomSeed:
    """Tests for random seed setting."""

    def test_reproducibility(self):
        set_random_seed(42)
        val1 = np.random.rand()
        set_random_seed(42)
        val2 = np.random.rand()
        assert val1 == val2


class TestEnsureDir:
    """Tests for directory creation."""

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory(self, tmp_path):
        result = ensure_dir(tmp_path)
        assert result.exists()


class TestGetTimestamp:
    """Tests for timestamp generation."""

    def test_returns_string(self):
        ts = get_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_format(self):
        ts = get_timestamp()
        # Should be like 20240101_120000
        assert "_" in ts


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds(self):
        assert "s" in format_duration(30.5)

    def test_minutes(self):
        result = format_duration(120)
        assert "min" in result

    def test_hours(self):
        result = format_duration(7200)
        assert "h" in result


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_removes_invalid_chars(self):
        result = sanitize_filename('file<name>:with/"|?*chars')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_strips_whitespace(self):
        result = sanitize_filename("  filename  ")
        assert result == "filename"


class TestValidateFilePath:
    """Tests for file path validation."""

    def test_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.touch()
        result = validate_file_path(test_file)
        assert result.exists()

    def test_missing_file(self, tmp_path):
        try:
            validate_file_path(tmp_path / "nonexistent.txt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
