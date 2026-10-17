"""EEG segmentation into epochs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """Configuration for EEG segmentation."""

    epoch_length_sec: float = 2.0
    overlap: float = 0.5
    min_epoch_quality: float = 0.5

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SegmentationConfig":
        """Create from configuration dictionary."""
        return cls(
            epoch_length_sec=config.get("epoch_length_sec", 2.0),
            overlap=config.get("overlap", 0.5),
            min_epoch_quality=config.get("min_epoch_quality", 0.5),
        )


@dataclass
class SegmentationResult:
    """Result of segmenting a recording."""

    epochs: mne.Epochs
    n_epochs_total: int
    n_epochs_rejected: int
    n_epochs_clean: int
    epoch_length_sec: float
    overlap: float
    sfreq: float
    rejected_epoch_indices: list[int] | None = None


class EEGSegmenter:
    """Segment continuous EEG into fixed-length epochs."""

    def __init__(self, config: SegmentationConfig | dict[str, Any] | None = None):
        """Initialize segmenter.

        Parameters
        ----------
        config : SegmentationConfig or dict, optional
            Segmentation configuration.
        """
        if config is None:
            self.config = SegmentationConfig()
        elif isinstance(config, dict):
            self.config = SegmentationConfig.from_dict(config)
        else:
            self.config = config

    def segment(self, raw: mne.io.Raw) -> SegmentationResult:
        """Segment a Raw object into epochs.

        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed EEG data.

        Returns
        -------
        SegmentationResult
            Segmentation result with epochs and metadata.
        """
        sfreq = raw.info["sfreq"]
        epoch_samples = int(self.config.epoch_length_sec * sfreq)
        step_samples = int(epoch_samples * (1 - self.config.overlap))

        # Create events at regular intervals
        n_samples = raw.n_times
        n_epochs = max(0, (n_samples - epoch_samples) // step_samples + 1)

        if n_epochs == 0:
            logger.warning(
                "Recording too short for epoch length %.1fs at %.1f Hz",
                self.config.epoch_length_sec,
                sfreq,
            )
            # Create empty epochs
            epochs = mne.Epochs(
                raw,
                events=np.empty((0, 3), dtype=int),
                tmin=0,
                tmax=self.config.epoch_length_sec - 1 / sfreq,
                baseline=None,
                preload=True,
                verbose=False,
            )
            return SegmentationResult(
                epochs=epochs,
                n_epochs_total=0,
                n_epochs_rejected=0,
                n_epochs_clean=0,
                epoch_length_sec=self.config.epoch_length_sec,
                overlap=self.config.overlap,
                sfreq=sfreq,
            )

        events = np.array(
            [[i * step_samples, 0, 1] for i in range(n_epochs)], dtype=int
        )

        tmin = 0.0
        tmax = self.config.epoch_length_sec - 1 / sfreq

        epochs = mne.Epochs(
            raw,
            events,
            event_id=1,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        # Compute epoch quality (variance-based rejection)
        data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        epoch_vars = np.var(data, axis=(1, 2))
        var_threshold = np.percentile(epoch_vars, 100 * (1 - self.config.min_epoch_quality))
        good_mask = epoch_vars >= var_threshold

        n_total = len(epochs)
        n_clean = int(good_mask.sum())
        n_rejected = n_total - n_clean

        # Drop low-quality epochs
        if n_rejected > 0:
            bad_indices = np.where(~good_mask)[0].tolist()
            epochs.drop(bad_indices, reason="LOW_QUALITY")
        else:
            bad_indices = None

        logger.info(
            "Segmented into %d epochs (%d rejected)",
            n_clean,
            n_rejected,
        )

        return SegmentationResult(
            epochs=epochs,
            n_epochs_total=n_total,
            n_epochs_rejected=n_rejected,
            n_epochs_clean=n_clean,
            epoch_length_sec=self.config.epoch_length_sec,
            overlap=self.config.overlap,
            sfreq=sfreq,
            rejected_epoch_indices=bad_indices,
        )

    def segment_batch(
        self,
        raws: dict[str, mne.io.Raw],
        config: SegmentationConfig | dict[str, Any] | None = None,
    ) -> dict[str, SegmentationResult]:
        """Segment multiple recordings.

        Parameters
        ----------
        raws : dict
            Mapping of subject_id to Raw objects.
        config : SegmentationConfig or dict, optional
            Segmentation configuration.

        Returns
        -------
        dict
            Mapping of subject_id to SegmentationResult.
        """
        segmenter = EEGSegmenter(config)
        results = {}
        for subject_id, raw in raws.items():
            logger.info("Segmenting %s...", subject_id)
            try:
                results[subject_id] = segmenter.segment(raw)
            except Exception as exc:
                logger.warning("Segmentation failed for %s: %s", subject_id, exc)
        return results
