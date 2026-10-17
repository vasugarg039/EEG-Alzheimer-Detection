"""Model explainability utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute permutation feature importance.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels.
    feature_names : list of str
        Feature names.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Importance results.
    """
    logger.info("Computing permutation importance...")
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    importance_df = {
        "feature": feature_names,
        "importance_mean": result.importances_mean.tolist(),
        "importance_std": result.importances_std.tolist(),
    }

    # Sort by importance
    sorted_idx = np.argsort(result.importances_mean)[::-1]
    importance_df["ranked_features"] = [feature_names[i] for i in sorted_idx]
    importance_df["ranked_importance"] = result.importances_mean[sorted_idx].tolist()

    logger.info("Permutation importance computed")
    return importance_df


def compute_feature_importance_from_model(
    model: Any,
    feature_names: list[str],
) -> dict[str, Any] | None:
    """Extract feature importance from model coefficients or feature_importances_.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    feature_names : list of str
        Feature names.

    Returns
    -------
    dict or None
        Importance results if available.
    """
    # Handle pipeline
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier", model)
    else:
        classifier = model

    if hasattr(classifier, "coef_"):
        # Linear models
        coef = classifier.coef_
        if coef.ndim > 1:
            # Multi-class: use mean absolute coefficient
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)

        return {
            "feature": feature_names,
            "importance_mean": importance.tolist(),
            "importance_std": [0.0] * len(feature_names),
            "ranked_features": [
                feature_names[i]
                for i in np.argsort(importance)[::-1]
            ],
            "ranked_importance": importance[np.argsort(importance)[::-1]].tolist(),
        }

    elif hasattr(classifier, "feature_importances_"):
        # Tree-based models
        importance = classifier.feature_importances_
        return {
            "feature": feature_names,
            "importance_mean": importance.tolist(),
            "importance_std": [0.0] * len(feature_names),
            "ranked_features": [
                feature_names[i]
                for i in np.argsort(importance)[::-1]
            ],
            "ranked_importance": importance[np.argsort(importance)[::-1]].tolist(),
        }

    return None


def explain_prediction(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    instance_idx: int = 0,
) -> dict[str, Any]:
    """Explain a single prediction using feature contributions.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X : np.ndarray
        Feature matrix.
    feature_names : list of str
        Feature names.
    instance_idx : int
        Index of instance to explain.

    Returns
    -------
    dict
        Explanation results.
    """
    instance = X[instance_idx]

    # Handle pipeline
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier", model)
    else:
        classifier = model

    explanation: dict[str, Any] = {
        "instance_idx": instance_idx,
        "features": {},
    }

    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
        if coef.ndim > 1:
            # Multi-class: explain predicted class
            pred = classifier.predict(instance.reshape(1, -1))[0]
            class_idx = list(classifier.classes_).index(pred)
            contributions = coef[class_idx] * instance
        else:
            contributions = coef * instance

        for i, fname in enumerate(feature_names):
            explanation["features"][fname] = {
                "value": float(instance[i]),
                "contribution": float(contributions[i]),
            }

    elif hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
        for i, fname in enumerate(feature_names):
            explanation["features"][fname] = {
                "value": float(instance[i]),
                "importance": float(importance[i]),
            }

    return explanation
