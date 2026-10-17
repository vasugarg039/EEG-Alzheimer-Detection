"""Tests for training module."""

from __future__ import annotations

import numpy as np
import pytest

from src.train import (
    create_model,
    train_model,
    evaluate_model,
    select_best_model,
    save_model,
    load_model,
    ModelResult,
)


class TestCreateModel:
    """Tests for model creation."""

    def test_create_logistic_regression(self):
        model = create_model("logistic_regression")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_svm(self):
        model = create_model("svm")
        assert model is not None

    def test_create_random_forest(self):
        model = create_model("random_forest")
        assert model is not None

    def test_create_dummy(self):
        model = create_model("dummy")
        assert model is not None

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            create_model("unknown_model")


class TestTrainModel:
    """Tests for model training."""

    def test_train_logistic_regression(self, synthetic_features):
        X, y, groups = synthetic_features
        result = train_model(X, y, "logistic_regression", cv_folds=3)
        assert result.name == "logistic_regression"
        assert result.model is not None
        assert "accuracy" in result.cv_scores

    def test_train_with_groups(self, synthetic_features):
        X, y, groups = synthetic_features
        result = train_model(X, y, "logistic_regression", groups=groups, cv_folds=3)
        assert result.cv_scores is not None

    def test_train_svm(self, synthetic_features):
        X, y, _ = synthetic_features
        result = train_model(X, y, "svm", cv_folds=3)
        assert result.name == "svm"


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_evaluate(self, synthetic_features):
        X, y, _ = synthetic_features
        train_result = train_model(X, y, "logistic_regression", cv_folds=3)
        eval_result = evaluate_model(train_result, X, y)
        assert "accuracy" in eval_result.test_metrics
        assert "f1_macro" in eval_result.test_metrics
        assert eval_result.predictions is not None


class TestSelectBestModel:
    """Tests for best model selection."""

    def test_select_best(self, synthetic_features):
        X, y, _ = synthetic_features
        results = []
        for name in ["logistic_regression", "svm"]:
            result = train_model(X, y, name, cv_folds=3)
            result = evaluate_model(result, X, y)
            results.append(result)

        best = select_best_model(results, "f1_macro")
        assert best in results


class TestSaveLoadModel:
    """Tests for model persistence."""

    def test_save_and_load(self, synthetic_features, tmp_path):
        X, y, _ = synthetic_features
        result = train_model(X, y, "logistic_regression", cv_folds=3)
        result = evaluate_model(result, X, y)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        label_mapping = {"A": "Alzheimer's", "F": "FTD", "C": "Control"}

        model_dir = save_model(
            result, tmp_path, feature_names, label_mapping, {}
        )
        assert model_dir.exists()

        loaded_model, loaded_metadata = load_model(model_dir)
        assert loaded_model is not None
        assert loaded_metadata["model_name"] == "logistic_regression"
        assert "feature_names" in loaded_metadata
