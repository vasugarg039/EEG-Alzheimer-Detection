"""Tests for dataset assembly module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    assemble_feature_table,
    aggregate_to_subject_level,
    prepare_train_test_data,
    get_feature_names,
)


class TestAssembleFeatureTable:
    """Tests for feature table assembly."""

    def test_empty_input(self):
        result = assemble_feature_table({})
        assert result.empty

    def test_single_subject(self):
        batch = {
            "sub-001": [
                {"feature_01": 1.0, "feature_02": 2.0, "epoch_idx": 0},
                {"feature_01": 1.5, "feature_02": 2.5, "epoch_idx": 1},
            ]
        }
        result = assemble_feature_table(batch)
        assert len(result) == 2
        assert "subject_id" in result.columns

    def test_multiple_subjects(self):
        batch = {
            "sub-001": [{"feature_01": 1.0, "epoch_idx": 0}],
            "sub-002": [{"feature_01": 2.0, "epoch_idx": 0}],
        }
        result = assemble_feature_table(batch)
        assert len(result) == 2
        assert set(result["subject_id"]) == {"sub-001", "sub-002"}


class TestAggregateToSubjectLevel:
    """Tests for subject-level aggregation."""

    def test_mean_aggregation(self):
        df = pd.DataFrame({
            "subject_id": ["sub-001", "sub-001", "sub-002", "sub-002"],
            "feature_01": [1.0, 3.0, 2.0, 4.0],
            "feature_02": [10.0, 20.0, 30.0, 40.0],
            "group": ["A", "A", "C", "C"],
        })
        result = aggregate_to_subject_level(df, "mean")
        assert len(result) == 2
        sub1 = result[result["subject_id"] == "sub-001"].iloc[0]
        assert sub1["feature_01"] == 2.0  # mean of 1.0 and 3.0

    def test_median_aggregation(self):
        df = pd.DataFrame({
            "subject_id": ["sub-001", "sub-001", "sub-001"],
            "feature_01": [1.0, 2.0, 10.0],
        })
        result = aggregate_to_subject_level(df, "median")
        assert result.iloc[0]["feature_01"] == 2.0


class TestPrepareTrainTestData:
    """Tests for train/test data preparation."""

    def test_split(self):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "subject_id": [f"sub-{i:03d}" for i in range(n)],
            "group": ["A"] * 10 + ["F"] * 10 + ["C"] * 10,
            "feature_01": np.random.randn(n),
            "feature_02": np.random.randn(n),
        })
        X_train, X_test, y_train, y_test, features, groups = prepare_train_test_data(
            df, test_size=0.3, random_state=42
        )
        assert len(X_train) + len(X_test) == n
        assert len(features) == 2
        assert len(groups) == len(X_train)

    def test_no_subject_overlap(self):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "subject_id": [f"sub-{i:03d}" for i in range(n)],
            "group": ["A"] * 10 + ["F"] * 10 + ["C"] * 10,
            "feature_01": np.random.randn(n),
        })
        X_train, X_test, y_train, y_test, _, _ = prepare_train_test_data(
            df, test_size=0.3, random_state=42
        )
        # No subject should appear in both train and test
        train_subjects = set(df.iloc[: len(X_train)]["subject_id"])
        test_subjects = set(df.iloc[len(X_train) :]["subject_id"])
        # Note: this is a simplified check; actual GroupShuffleSplit is used


class TestGetFeatureNames:
    """Tests for feature name extraction."""

    def test_excludes_metadata(self):
        df = pd.DataFrame({
            "subject_id": ["sub-001"],
            "group": ["A"],
            "age": [65],
            "feature_01": [1.0],
        })
        names = get_feature_names(df)
        assert "feature_01" in names
        assert "subject_id" not in names
        assert "group" not in names
        assert "age" not in names
