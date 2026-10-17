"""Tests for evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    compute_metrics,
    generate_classification_report,
    save_metrics,
    cross_validation_summary,
)


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_perfect_predictions(self):
        y_true = np.array(["A", "A", "F", "F", "C", "C"])
        y_pred = np.array(["A", "A", "F", "F", "C", "C"])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_imperfect_predictions(self):
        y_true = np.array(["A", "A", "F", "F", "C", "C"])
        y_pred = np.array(["A", "F", "F", "C", "C", "A"])
        metrics = compute_metrics(y_true, y_pred)
        assert 0 < metrics["accuracy"] < 1

    def test_per_class_metrics(self):
        y_true = np.array(["A", "A", "F", "F", "C", "C"])
        y_pred = np.array(["A", "A", "F", "F", "C", "C"])
        metrics = compute_metrics(y_true, y_pred, labels=["A", "F", "C"])
        assert "per_class" in metrics
        assert "A" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["A"]

    def test_confusion_matrix(self):
        y_true = np.array(["A", "A", "F", "F", "C", "C"])
        y_pred = np.array(["A", "A", "F", "F", "C", "C"])
        metrics = compute_metrics(y_true, y_pred, labels=["A", "F", "C"])
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3


class TestGenerateClassificationReport:
    """Tests for classification report generation."""

    def test_report_generation(self):
        y_true = np.array(["A", "A", "F", "F", "C", "C"])
        y_pred = np.array(["A", "A", "F", "F", "C", "C"])
        report = generate_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report


class TestSaveMetrics:
    """Tests for saving metrics."""

    def test_save_json(self, tmp_path):
        metrics = {"accuracy": 0.95, "f1_macro": 0.93}
        path = save_metrics(metrics, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"


class TestCrossValidationSummary:
    """Tests for CV summary."""

    def test_summary(self):
        cv_scores = {
            "accuracy": [0.8, 0.85, 0.9],
            "f1_macro": [0.75, 0.8, 0.85],
        }
        summary = cross_validation_summary(cv_scores)
        assert "accuracy" in summary
        assert "mean" in summary["accuracy"]
        assert "std" in summary["accuracy"]
        assert summary["accuracy"]["n_folds"] == 3
