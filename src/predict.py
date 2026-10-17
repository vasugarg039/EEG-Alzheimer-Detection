"""Inference and prediction interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FeatureExtractor, FeatureConfig
from .preprocessing import EEGPreprocessor, PreprocessingConfig
from .segmentation import EEGSegmenter, SegmentationConfig
from .train import load_model

logger = logging.getLogger(__name__)


class EEGPredictor:
    """End-to-end prediction from EEG file to class label."""

    def __init__(
        self,
        model_dir: str | Path,
        preprocessing_config: dict[str, Any] | None = None,
        segmentation_config: dict[str, Any] | None = None,
        feature_config: dict[str, Any] | None = None,
    ):
        """Initialize predictor with a saved model.

        Parameters
        ----------
        model_dir : str or Path
            Directory containing the saved model.
        preprocessing_config : dict, optional
            Preprocessing configuration.
        segmentation_config : dict, optional
            Segmentation configuration.
        feature_config : dict, optional
            Feature extraction configuration.
        """
        self.model_dir = Path(model_dir)
        self.model, self.metadata = load_model(model_dir)

        self.preprocessing_config = PreprocessingConfig.from_dict(
            preprocessing_config or {}
        )
        self.segmentation_config = SegmentationConfig.from_dict(
            segmentation_config or {}
        )
        self.feature_config = FeatureConfig.from_dict(feature_config or {})

        self.preprocessor = EEGPreprocessor(self.preprocessing_config)
        self.segmenter = EEGSegmenter(self.segmentation_config)
        self.extractor = FeatureExtractor(self.feature_config)

        self.feature_names = self.metadata.get("feature_names", [])
        self.label_mapping = self.metadata.get("label_mapping", {})

    def predict_file(self, eeg_file: str | Path) -> dict[str, Any]:
        """Predict class from an EEG file.

        Parameters
        ----------
        eeg_file : str or Path
            Path to EEG file.

        Returns
        -------
        dict
            Prediction results with class, probabilities, and metadata.
        """
        import mne

        # Load
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)

        # Preprocess
        preproc_result = self.preprocessor.preprocess(raw)
        if preproc_result.excluded:
            return {
                "success": False,
                "error": f"Preprocessing failed: {preproc_result.exclusion_reason}",
            }

        # Segment
        seg_result = self.segmenter.segment(preproc_result.raw_filtered)
        if seg_result.n_epochs_clean == 0:
            return {
                "success": False,
                "error": "No valid epochs after segmentation",
            }

        # Extract features
        epoch_features = self.extractor.extract_from_epochs(
            seg_result.epochs, subject_id="prediction"
        )

        if not epoch_features:
            return {
                "success": False,
                "error": "Feature extraction failed",
            }

        # Aggregate to subject level (mean across epochs)
        feature_vector = self._aggregate_epoch_features(epoch_features)

        # Predict
        return self._predict_features(feature_vector)

    def predict_features(self, features: np.ndarray) -> dict[str, Any]:
        """Predict class from a feature vector.

        Parameters
        ----------
        features : np.ndarray
            Feature vector matching the model's expected features.

        Returns
        -------
        dict
            Prediction results.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self._predict_features(features)

    def _predict_features(self, features: np.ndarray) -> dict[str, Any]:
        """Run prediction on feature matrix.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        dict
            Prediction results.
        """
        # Validate feature count
        expected = len(self.feature_names)
        if features.shape[1] != expected:
            return {
                "success": False,
                "error": f"Feature count mismatch: got {features.shape[1]}, expected {expected}",
            }

        # Predict
        predictions = self.model.predict(features)
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)

        results = []
        for i in range(len(predictions)):
            pred_class = predictions[i]
            result = {
                "success": True,
                "predicted_class": pred_class,
                "predicted_label": self.label_mapping.get(pred_class, pred_class),
            }
            if probabilities is not None:
                result["probabilities"] = {
                    cls: float(prob)
                    for cls, prob in zip(self.model.classes_, probabilities[i])
                }
                result["confidence"] = float(np.max(probabilities[i]))
            results.append(result)

        return results[0] if len(results) == 1 else results

    def _aggregate_epoch_features(
        self,
        epoch_features: list[dict[str, Any]],
    ) -> np.ndarray:
        """Aggregate epoch-level features to a single subject-level vector.

        Parameters
        ----------
        epoch_features : list of dict
            Per-epoch feature dictionaries.

        Returns
        -------
        np.ndarray
            Feature vector.
        """
        if not self.feature_names:
            # If no feature names stored, use all numeric keys
            all_keys = set()
            for ef in epoch_features:
                all_keys.update(ef.keys())
            feature_names = sorted(
                k for k in all_keys
                if k not in {"subject_id", "epoch_idx"}
            )
        else:
            feature_names = self.feature_names

        # Compute mean across epochs
        feature_vector = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            values = [ef.get(fname, 0.0) for ef in epoch_features]
            feature_vector[i] = np.mean(values)

        return feature_vector.reshape(1, -1)
