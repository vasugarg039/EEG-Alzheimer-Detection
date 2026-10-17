"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_metadata():
    """Create a sample metadata DataFrame."""
    return pd.DataFrame({
        "subject_id": ["sub-001", "sub-002", "sub-003"],
        "group": ["A", "F", "C"],
        "age": [65, 70, 60],
        "gender": ["F", "M", "F"],
        "mmse": [20, 22, 30],
        "channels": [19, 19, 19],
        "sampling_frequency": [500.0, 500.0, 500.0],
        "samples": [299900, 396550, 153550],
        "duration_seconds": [599.798, 793.098, 307.098],
        "duration_minutes": [9.99, 13.22, 5.12],
        "status": ["success", "success", "success"],
        "group_name": [
            "Alzheimer's Disease",
            "Frontotemporal Dementia",
            "Healthy Control",
        ],
    })


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "paths": {
            "data_root": "D:/EEG Dataset/ds004504",
            "metadata": "data/metadata/eeg_dataset_summary.csv",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "features_dir": "data/features",
            "models_dir": "models",
            "results_dir": "results",
            "reports_dir": "reports",
            "logs_dir": "logs",
        },
        "eeg": {
            "expected_channels": 19,
            "expected_sampling_rate": 500.0,
            "channel_names": ["Fp1", "Fp2", "F3", "F4", "C3", "C4"],
            "line_frequency": 50.0,
        },
        "preprocessing": {
            "highpass": 1.0,
            "lowpass": 45.0,
            "notch": 50.0,
            "resample_sfreq": None,
            "ica_enabled": False,
            "ica_n_components": 15,
            "reject_threshold_uv": 200.0,
        },
        "segmentation": {
            "epoch_length_sec": 2.0,
            "overlap": 0.5,
            "min_epoch_quality": 0.5,
        },
        "features": {
            "bands": {
                "delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 45.0],
            },
            "n_fft": 512,
            "n_per_seg": 256,
            "compute_hjorth": True,
            "compute_entropy": True,
            "compute_connectivity": False,
        },
        "models": {
            "random_seed": 42,
            "test_size": 0.2,
            "cv_folds": 5,
            "models": [
                {"name": "logistic_regression", "params": {"max_iter": 1000}},
                {"name": "svm", "params": {"kernel": "rbf"}},
                {"name": "random_forest", "params": {"n_estimators": 200}},
            ],
        },
        "training": {
            "primary_metric": "f1_macro",
            "secondary_metric": "balanced_accuracy",
        },
    }


@pytest.fixture
def synthetic_eeg_data():
    """Create synthetic EEG data for testing."""
    np.random.seed(42)
    n_channels = 19
    n_times = 5000  # 10 seconds at 500 Hz
    data = np.random.randn(n_channels, n_times) * 10  # microvolts
    return data


@pytest.fixture
def synthetic_features():
    """Create synthetic feature data for testing."""
    np.random.seed(42)
    n_samples = 30
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.array(["A"] * 10 + ["F"] * 10 + ["C"] * 10)
    groups = np.array([f"sub-{i:03d}" for i in range(n_samples)])
    return X, y, groups
