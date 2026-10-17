"""Model evaluation and metrics reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list of str, optional
        Label names.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    for i, label in enumerate(labels):
        metrics["per_class"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> str:
    """Generate a text classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list of str, optional
        Label names.

    Returns
    -------
    str
        Classification report.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def save_metrics(
    metrics: dict[str, Any],
    output_dir: str | Path,
    filename: str = "metrics.json",
) -> Path:
    """Save metrics to JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    output_dir : str or Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Path to saved file.
    """
    out_dir = ensure_dir(output_dir)
    filepath = out_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", filepath)
    return filepath


def save_classification_report(
    report: str,
    output_dir: str | Path,
    filename: str = "classification_report.txt",
) -> Path:
    """Save classification report to text file.

    Parameters
    ----------
    report : str
        Report text.
    output_dir : str or Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Path to saved file.
    """
    out_dir = ensure_dir(output_dir)
    filepath = out_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", filepath)
    return filepath


def evaluate_model_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate model from predictions and save results.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    output_dir : str or Path
        Output directory.
    labels : list of str, optional
        Label names.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    metrics = compute_metrics(y_true, y_pred, labels)
    save_metrics(metrics, output_dir)

    report = generate_classification_report(y_true, y_pred, labels)
    save_classification_report(report, output_dir)

    return metrics


def cross_validation_summary(cv_scores: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Summarize cross-validation scores.

    Parameters
    ----------
    cv_scores : dict
        Mapping of metric name to list of fold scores.

    Returns
    -------
    dict
        Summary statistics for each metric.
    """
    summary = {}
    for metric_name, scores in cv_scores.items():
        scores_arr = np.array(scores)
        summary[metric_name] = {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "n_folds": len(scores),
        }
    return summary
