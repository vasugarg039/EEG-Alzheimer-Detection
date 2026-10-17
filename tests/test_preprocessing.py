"""Tests for preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    EEGPreprocessor,
    validate_channel_consistency,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_defaults(self):
        config = PreprocessingConfig()
        assert config.highpass == 1.0
        assert config.lowpass == 45.0
        assert config.notch == 50.0
        assert config.ica_enabled is False

    def test_from_dict(self):
        config = PreprocessingConfig.from_dict({
            "highpass": 0.5,
            "lowpass": 40.0,
            "notch": 60.0,
        })
        assert config.highpass == 0.5
        assert config.lowpass == 40.0
        assert config.notch == 60.0


class TestEEGPreprocessor:
    """Tests for EEGPreprocessor."""

    def test_init_with_config(self):
        config = PreprocessingConfig(highpass=0.5, lowpass=40.0)
        preprocessor = EEGPreprocessor(config)
        assert preprocessor.config.highpass == 0.5

    def test_init_with_dict(self):
        preprocessor = EEGPreprocessor({"highpass": 2.0})
        assert preprocessor.config.highpass == 2.0

    def test_validate_channel_consistency(self):
        expected = ["Fp1", "Fp2", "F3"]
        actual = ["Fp1", "Fp2", "F3"]
        valid, issues = validate_channel_consistency(
            type("Raw", (), {"ch_names": actual})(), expected
        )
        assert valid is True
        assert issues == []

    def test_validate_missing_channels(self):
        expected = ["Fp1", "Fp2", "F3"]
        actual = ["Fp1", "Fp2"]
        valid, issues = validate_channel_consistency(
            type("Raw", (), {"ch_names": actual})(), expected
        )
        assert valid is False
        assert any("Missing" in i for i in issues)

    def test_validate_extra_channels(self):
        expected = ["Fp1", "Fp2"]
        actual = ["Fp1", "Fp2", "F3"]
        valid, issues = validate_channel_consistency(
            type("Raw", (), {"ch_names": actual})(), expected
        )
        assert valid is False
        assert any("Extra" in i for i in issues)
