"""Model training and selection with leakage-safe evaluation."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .utils import ensure_dir, get_timestamp

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result of training a single model."""

    name: str
    model: Pipeline
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    true_labels: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)


def create_model(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Create a scikit-learn pipeline with scaling.

    Parameters
    ----------
    model_name : str
        Model identifier.
    params : dict, optional
        Model hyperparameters.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline.
    """
    params = params or {}

    if model_name == "dummy":
        classifier = DummyClassifier(strategy="most_frequent", random_state=42)
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            random_state=params.get("random_state", 42),
            class_weight=params.get("class_weight", "balanced"),
        )
    elif model_name == "svm":
        classifier = SVC(
            kernel=params.get("kernel", "rbf"),
            random_state=params.get("random_state", 42),
            class_weight=params.get("class_weight", "balanced"),
            probability=params.get("probability", True),
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            random_state=params.get("random_state", 42),
            class_weight=params.get("class_weight", "balanced"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    return pipeline


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    params: dict[str, Any] | None = None,
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
) -> ModelResult:
    """Train a single model with cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : str
        Model identifier.
    params : dict, optional
        Model hyperparameters.
    groups : np.ndarray, optional
        Group labels for GroupKFold.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    ModelResult
        Training result.
    """
    logger.info("Training %s...", model_name)
    model = create_model(model_name, params)

    # Cross-validation
    if groups is not None and len(np.unique(groups)) >= cv_folds:
        cv = GroupKFold(n_splits=cv_folds)
        cv_results = cross_validate(
            model, X_train, y_train, groups=groups, cv=cv,
            scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            return_train_score=False,
        )
        cv_scores = {
            "accuracy": cv_results["test_accuracy"].tolist(),
            "balanced_accuracy": cv_results["test_balanced_accuracy"].tolist(),
            "f1_macro": cv_results["test_f1_macro"].tolist(),
        }
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            return_train_score=False,
        )
        cv_scores = {
            "accuracy": cv_results["test_accuracy"].tolist(),
            "balanced_accuracy": cv_results["test_balanced_accuracy"].tolist(),
            "f1_macro": cv_results["test_f1_macro"].tolist(),
        }

    # Fit on full training set
    model.fit(X_train, y_train)

    result = ModelResult(
        name=model_name,
        model=model,
        cv_scores=cv_scores,
    )

    logger.info(
        "%s CV: acc=%.3f, bal_acc=%.3f, f1_macro=%.3f",
        model_name,
        np.mean(cv_scores["accuracy"]),
        np.mean(cv_scores["balanced_accuracy"]),
        np.mean(cv_scores["f1_macro"]),
    )

    return result


def evaluate_model(
    model_result: ModelResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResult:
    """Evaluate a trained model on test data.

    Parameters
    ----------
    model_result : ModelResult
        Model to evaluate.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    ModelResult
        Updated result with test metrics.
    """
    model = model_result.model
    y_pred = model.predict(X_test)

    model_result.predictions = y_pred
    model_result.true_labels = y_test

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=["A", "F", "C"], zero_division=0
    )

    model_result.test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_A": float(precision[0]),
        "precision_F": float(precision[1]),
        "precision_C": float(precision[2]),
        "recall_A": float(recall[0]),
        "recall_F": float(recall[1]),
        "recall_C": float(recall[2]),
        "f1_A": float(f1[0]),
        "f1_F": float(f1[1]),
        "f1_C": float(f1[2]),
    }

    logger.info(
        "%s Test: acc=%.3f, bal_acc=%.3f, f1_macro=%.3f",
        model_result.name,
        model_result.test_metrics["accuracy"],
        model_result.test_metrics["balanced_accuracy"],
        model_result.test_metrics["f1_macro"],
    )

    return model_result


def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_configs: list[dict[str, Any]],
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
) -> list[ModelResult]:
    """Train and compare multiple models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    model_configs : list of dict
        List of model configurations.
    groups : np.ndarray, optional
        Group labels for GroupKFold.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    list of ModelResult
        Results for all models.
    """
    results = []
    for config in model_configs:
        name = config["name"]
        params = config.get("params", {})
        try:
            result = train_model(
                X_train, y_train, name, params, groups, cv_folds
            )
            result = evaluate_model(result, X_test, y_test)
            results.append(result)
        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)

    return results


def select_best_model(
    results: list[ModelResult],
    primary_metric: str = "f1_macro",
) -> ModelResult:
    """Select the best model based on a primary metric.

    Parameters
    ----------
    results : list of ModelResult
        Model results.
    primary_metric : str
        Metric to use for selection.

    Returns
    -------
    ModelResult
        Best model result.
    """
    if not results:
        raise ValueError("No model results to select from")

    best = max(results, key=lambda r: r.test_metrics.get(primary_metric, 0))
    logger.info(
        "Selected best model: %s (%s=%.3f)",
        best.name,
        primary_metric,
        best.test_metrics.get(primary_metric, 0),
    )
    return best


def save_model(
    model_result: ModelResult,
    output_dir: str | Path,
    feature_names: list[str],
    label_mapping: dict[str, str],
    config: dict[str, Any],
) -> Path:
    """Save a trained model with metadata.

    Parameters
    ----------
    model_result : ModelResult
        Model to save.
    output_dir : str or Path
        Output directory.
    feature_names : list of str
        Feature names.
    label_mapping : dict
        Label mapping.
    config : dict
        Configuration used.

    Returns
    -------
    Path
        Path to saved model.
    """
    out_dir = ensure_dir(output_dir)
    timestamp = get_timestamp()
    model_dir = out_dir / f"{model_result.name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(model_result.model, model_path)

    # Save metadata
    metadata = {
        "model_name": model_result.name,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "test_metrics": model_result.test_metrics,
        "cv_scores": model_result.cv_scores,
        "config": config,
        "timestamp": timestamp,
        "model_version": "1.0.0",
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Model saved to %s", model_dir)
    return model_dir


def load_model(model_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a saved model and its metadata.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the saved model.

    Returns
    -------
    tuple
        (Pipeline, metadata dict)
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Model loaded from %s", model_dir)
    return model, metadata
