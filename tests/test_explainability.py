"""Tests for explainability module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explainability import (
    compute_permutation_importance,
    compute_feature_importance_from_model,
    explain_prediction,
)


class TestComputePermutationImportance:
    """Tests for permutation importance."""

    def test_returns_dict(self, synthetic_features):
        X, y, _ = synthetic_features
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        result = compute_permutation_importance(
            model, X, y, feature_names, n_repeats=3
        )
        assert "feature" in result
        assert "importance_mean" in result
        assert "ranked_features" in result


class TestComputeFeatureImportanceFromModel:
    """Tests for model-based feature importance."""

    def test_linear_model(self, synthetic_features):
        X, y, _ = synthetic_features
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        result = compute_feature_importance_from_model(model, feature_names)
        assert result is not None
        assert "importance_mean" in result

    def test_tree_model(self, synthetic_features):
        X, y, _ = synthetic_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        result = compute_feature_importance_from_model(model, feature_names)
        assert result is not None
        assert "importance_mean" in result

    def test_unknown_model(self):
        model = type("UnknownModel", (), {"fit": lambda self, X, y: None})()
        result = compute_feature_importance_from_model(model, ["f1", "f2"])
        assert result is None


class TestExplainPrediction:
    """Tests for prediction explanation."""

    def test_explains_instance(self, synthetic_features):
        X, y, _ = synthetic_features
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        result = explain_prediction(model, X, feature_names, instance_idx=0)
        assert "instance_idx" in result
        assert "features" in result
