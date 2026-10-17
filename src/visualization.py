"""EEG and model visualization functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def plot_eeg_traces(
    raw: Any,
    duration: float = 10.0,
    start: float = 0.0,
    channels: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot EEG traces.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    duration : float
        Duration to plot in seconds.
    start : float
        Start time in seconds.
    channels : list of str, optional
        Channels to plot. Defaults to all.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    data, times = raw.copy().pick(channels or raw.ch_names).get_data(
        start=start, stop=start + duration, return_times=True
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(data) * 0.3)))
    for i, ch_data in enumerate(data):
        ax.plot(times, ch_data + i * np.std(data) * 3, linewidth=0.5, label=channels[i] if channels else raw.ch_names[i])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("EEG Traces")
    if channels:
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("EEG traces saved to %s", output_path)

    return fig


def plot_psd(
    raw: Any,
    fmin: float = 0.5,
    fmax: float = 45.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot power spectral density.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig = raw.plot_psd(fmin=fmin, fmax=fmax, show=False)
    fig.set_size_inches(10, 6)

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("PSD saved to %s", output_path)

    return fig


def plot_band_power(
    band_powers: dict[str, float],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot band power bar chart.

    Parameters
    ----------
    band_powers : dict
        Mapping of band name to power value.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    ax.bar(bands, powers, color=colors[: len(bands)])
    ax.set_ylabel("Power")
    ax.set_title("Frequency Band Power")
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Band power saved to %s", output_path)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list of str, optional
        Label names.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", output_path)

    return fig


def plot_feature_importance(
    importance_dict: dict[str, Any],
    top_n: int = 20,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot feature importance.

    Parameters
    ----------
    importance_dict : dict
        Importance dictionary with 'ranked_features' and 'ranked_importance'.
    top_n : int
        Number of top features to show.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    features = importance_dict["ranked_features"][:top_n]
    importance = importance_dict["ranked_importance"][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance saved to %s", output_path)

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot class distribution.

    Parameters
    ----------
    labels : np.ndarray
        Class labels.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique, counts, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Class distribution saved to %s", output_path)

    return fig


def plot_model_comparison(
    results: list[dict[str, Any]],
    metric: str = "f1_macro",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot model comparison.

    Parameters
    ----------
    results : list of dict
        List of model result dictionaries.
    metric : str
        Metric to compare.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    names = [r["name"] for r in results]
    values = [r["test_metrics"].get(metric, 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric}")
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom",
        )

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison saved to %s", output_path)

    return fig


def plot_feature_distribution(
    feature_values: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot feature distribution by class.

    Parameters
    ----------
    feature_values : np.ndarray
        Feature values.
    labels : np.ndarray
        Class labels.
    feature_name : str
        Feature name for title.
    output_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.hist(feature_values[mask], bins=30, alpha=0.5, label=label)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_name}")
    ax.legend()
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Feature distribution saved to %s", output_path)

    return fig
