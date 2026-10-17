"""Subject-level dataset assembly from extracted features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assemble_feature_table(
    batch_features: dict[str, list[dict[str, Any]]],
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble a flat feature table from batch feature extraction results.

    Parameters
    ----------
    batch_features : dict
        Mapping of subject_id to list of per-epoch feature dicts.
    metadata : pd.DataFrame, optional
        Metadata DataFrame to merge with features.

    Returns
    -------
    pd.DataFrame
        Feature table with one row per epoch.
    """
    rows = []
    for subject_id, epoch_list in batch_features.items():
        for epoch_features in epoch_list:
            row = dict(epoch_features)
            row["subject_id"] = subject_id
            rows.append(row)

    if not rows:
        logger.warning("No features to assemble")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Assembled feature table: %d rows, %d columns", len(df), len(df.columns))

    # Merge metadata if provided
    if metadata is not None and not metadata.empty:
        meta_cols = ["subject_id", "group", "age", "gender", "mmse"]
        available_meta_cols = [c for c in meta_cols if c in metadata.columns]
        df = df.merge(metadata[available_meta_cols], on="subject_id", how="left")

    return df


def aggregate_to_subject_level(
    feature_df: pd.DataFrame,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Aggregate epoch-level features to subject-level.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature table with one row per epoch.
    aggregation : str
        Aggregation method: 'mean', 'median', or 'std'.

    Returns
    -------
    pd.DataFrame
        Subject-level feature table.
    """
    if feature_df.empty:
        return feature_df

    # Identify feature columns (numeric, non-metadata)
    exclude_cols = {"subject_id", "group", "age", "gender", "mmse", "epoch_idx"}
    feature_cols = [
        c for c in feature_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(feature_df[c])
    ]

    if aggregation == "mean":
        agg_df = feature_df.groupby("subject_id")[feature_cols].mean()
    elif aggregation == "median":
        agg_df = feature_df.groupby("subject_id")[feature_cols].median()
    elif aggregation == "std":
        agg_df = feature_df.groupby("subject_id")[feature_cols].std()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Add back metadata
    meta_cols = ["group", "age", "gender", "mmse"]
    for col in meta_cols:
        if col in feature_df.columns:
            agg_df[col] = feature_df.groupby("subject_id")[col].first()

    agg_df = agg_df.reset_index()
    logger.info(
        "Aggregated to subject level: %d subjects, %d features",
        len(agg_df),
        len(feature_cols),
    )
    return agg_df


def prepare_train_test_data(
    subject_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Prepare train/test data with subject-level splitting.

    Parameters
    ----------
    subject_df : pd.DataFrame
        Subject-level feature DataFrame.
    test_size : float
        Fraction of subjects for test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, train_groups)
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    exclude_cols = {"subject_id", "group", "age", "gender", "mmse"}
    feature_cols = [
        c for c in subject_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(subject_df[c])
    ]

    X = subject_df[feature_cols].values
    y = subject_df["group"].values

    # Check for NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("NaN or Inf values found in features, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Subject-level split
    groups = subject_df["subject_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = np.asarray(y[train_idx]), np.asarray(y[test_idx])

    logger.info(
        "Train: %d subjects, Test: %d subjects, Features: %d",
        len(train_idx),
        len(test_idx),
        len(feature_cols),
    )

    train_groups = np.asarray(groups[train_idx])
    return X_train, X_test, y_train, y_test, feature_cols, train_groups


def get_feature_names(subject_df: pd.DataFrame) -> list[str]:
    """Get list of feature column names.

    Parameters
    ----------
    subject_df : pd.DataFrame
        Subject-level feature DataFrame.

    Returns
    -------
    list of str
        Feature column names.
    """
    exclude_cols = {"subject_id", "group", "age", "gender", "mmse", "epoch_idx"}
    return [
        c for c in subject_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(subject_df[c])
    ]
