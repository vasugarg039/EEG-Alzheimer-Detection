"""Tests for feature extraction module."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    FeatureConfig,
    FeatureExtractor,
    compute_band_power,
    compute_hjorth_parameters,
    compute_spectral_entropy,
    compute_peak_frequency,
)


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_defaults(self):
        config = FeatureConfig()
        assert "delta" in config.bands
        assert "theta" in config.bands
        assert config.n_fft == 512

    def test_from_dict(self):
        config = FeatureConfig.from_dict({"n_fft": 1024})
        assert config.n_fft == 1024


class TestComputeBandPower:
    """Tests for band power computation."""

    def test_known_frequency(self):
        # Create a 10 Hz sine wave at 500 Hz
        sfreq = 500.0
        t = np.arange(0, 10, 1 / sfreq)
        data = np.sin(2 * np.pi * 10 * t)
        power = compute_band_power(data, sfreq, [8, 12])
        assert power > 0

    def test_zero_outside_band(self):
        sfreq = 500.0
        t = np.arange(0, 10, 1 / sfreq)
        data = np.sin(2 * np.pi * 10 * t)
        power = compute_band_power(data, sfreq, [20, 30])
        # Spectral leakage means power is small but not exactly zero
        assert power < 1e-3

    def test_constant_signal(self):
        data = np.ones(5000)
        power = compute_band_power(data, 500.0, [0.5, 4.0])
        assert power >= 0


class TestComputeHjorthParameters:
    """Tests for Hjorth parameters."""

    def test_constant_signal(self):
        data = np.ones(5000)
        activity, mobility, complexity = compute_hjorth_parameters(data)
        assert activity == 0.0
        assert mobility == 0.0
        assert complexity == 0.0

    def test_random_signal(self):
        np.random.seed(42)
        data = np.random.randn(5000)
        activity, mobility, complexity = compute_hjorth_parameters(data)
        assert activity > 0
        assert mobility > 0
        assert complexity > 0


class TestComputeSpectralEntropy:
    """Tests for spectral entropy."""

    def test_pure_tone(self):
        sfreq = 500.0
        t = np.arange(0, 10, 1 / sfreq)
        data = np.sin(2 * np.pi * 10 * t)
        entropy = compute_spectral_entropy(data, sfreq)
        assert 0 <= entropy <= 1

    def test_white_noise(self):
        np.random.seed(42)
        data = np.random.randn(5000)
        entropy = compute_spectral_entropy(data, 500.0)
        assert 0 <= entropy <= 1


class TestComputePeakFrequency:
    """Tests for peak frequency."""

    def test_known_peak(self):
        sfreq = 500.0
        t = np.arange(0, 10, 1 / sfreq)
        data = np.sin(2 * np.pi * 10 * t)
        peak = compute_peak_frequency(data, sfreq)
        assert 8 <= peak <= 12


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_init_default(self):
        extractor = FeatureExtractor()
        assert extractor.config.n_fft == 512

    def test_extract_epoch_features(self, synthetic_eeg_data):
        extractor = FeatureExtractor()
        features = extractor.extract_epoch_features(synthetic_eeg_data, 500.0)
        assert isinstance(features, dict)
        assert len(features) > 0
        # Check for expected feature patterns
        assert any("mean" in k for k in features)
        assert any("std" in k for k in features)
        assert any("delta_power" in k for k in features)
        assert any("alpha_power" in k for k in features)

    def test_features_finite(self, synthetic_eeg_data):
        extractor = FeatureExtractor()
        features = extractor.extract_epoch_features(synthetic_eeg_data, 500.0)
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} is not finite"

    def test_no_nan_features(self, synthetic_eeg_data):
        extractor = FeatureExtractor()
        features = extractor.extract_epoch_features(synthetic_eeg_data, 500.0)
        for key, value in features.items():
            assert not np.isnan(value), f"Feature {key} is NaN"
