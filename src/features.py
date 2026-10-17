"""EEG feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
from scipy import signal
from scipy.stats import entropy, kurtosis, skew

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    bands: dict[str, list[float]] = field(
        default_factory=lambda: {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "gamma": [30.0, 45.0],
        }
    )
    n_fft: int = 512
    n_per_seg: int = 256
    compute_hjorth: bool = True
    compute_entropy: bool = True
    compute_connectivity: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FeatureConfig":
        """Create from configuration dictionary."""
        return cls(
            bands=config.get("bands", {
                "delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 45.0],
            }),
            n_fft=config.get("n_fft", 512),
            n_per_seg=config.get("n_per_seg", 256),
            compute_hjorth=config.get("compute_hjorth", True),
            compute_entropy=config.get("compute_entropy", True),
            compute_connectivity=config.get("compute_connectivity", False),
        )


def compute_band_power(
    data: np.ndarray,
    sfreq: float,
    band: list[float],
    n_fft: int = 512,
    n_per_seg: int = 256,
) -> float:
    """Compute average band power using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sfreq : float
        Sampling frequency.
    band : list of float
        [low, high] frequency band.
    n_fft : int
        FFT length.
    n_per_seg : int
        Length of each Welch segment.

    Returns
    -------
    float
        Band power.
    """
    freqs, psd = signal.welch(
        data, fs=sfreq, nfft=n_fft, nperseg=min(n_per_seg, len(data))
    )
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def compute_hjorth_parameters(data: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, and complexity.

    Parameters
    ----------
    data : np.ndarray
        1D signal.

    Returns
    -------
    tuple
        (activity, mobility, complexity)
    """
    activity = np.var(data)

    diff1 = np.diff(data)
    var_diff1 = np.var(diff1)
    mobility = np.sqrt(var_diff1 / activity) if activity > 0 else 0.0

    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)
    mobility_diff = np.sqrt(var_diff2 / var_diff1) if var_diff1 > 0 else 0.0
    complexity = mobility_diff / mobility if mobility > 0 else 0.0

    return float(activity), float(mobility), float(complexity)


def compute_spectral_entropy(
    data: np.ndarray,
    sfreq: float,
    n_fft: int = 512,
    n_per_seg: int = 256,
) -> float:
    """Compute spectral entropy.

    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sfreq : float
        Sampling frequency.
    n_fft : int
        FFT length.
    n_per_seg : int
        Length of each Welch segment.

    Returns
    -------
    float
        Spectral entropy (normalized 0-1).
    """
    freqs, psd = signal.welch(
        data, fs=sfreq, nfft=n_fft, nperseg=min(n_per_seg, len(data))
    )
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    psd_norm = psd / psd_sum
    # Filter zeros for entropy
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) <= 1:
        return 0.0
    # Normalize by log(n) to get value in [0, 1]
    return float(entropy(psd_norm) / np.log(len(psd_norm)))


def compute_peak_frequency(
    data: np.ndarray,
    sfreq: float,
    n_fft: int = 512,
    n_per_seg: int = 256,
) -> float:
    """Compute peak frequency.

    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sfreq : float
        Sampling frequency.
    n_fft : int
        FFT length.
    n_per_seg : int
        Length of each Welch segment.

    Returns
    -------
    float
        Peak frequency in Hz.
    """
    freqs, psd = signal.welch(
        data, fs=sfreq, nfft=n_fft, nperseg=min(n_per_seg, len(data))
    )
    if len(psd) == 0 or np.max(psd) == 0:
        return 0.0
    return float(freqs[np.argmax(psd)])


class FeatureExtractor:
    """Extract features from epoched EEG data."""

    def __init__(self, config: FeatureConfig | dict[str, Any] | None = None):
        """Initialize feature extractor.

        Parameters
        ----------
        config : FeatureConfig or dict, optional
            Feature extraction configuration.
        """
        if config is None:
            self.config = FeatureConfig()
        elif isinstance(config, dict):
            self.config = FeatureConfig.from_dict(config)
        else:
            self.config = config

    def extract_epoch_features(
        self,
        epoch_data: np.ndarray,
        sfreq: float,
    ) -> dict[str, float]:
        """Extract features from a single epoch.

        Parameters
        ----------
        epoch_data : np.ndarray
            2D array (n_channels, n_times).
        sfreq : float
            Sampling frequency.

        Returns
        -------
        dict
            Feature dictionary.
        """
        features: dict[str, float] = {}
        n_channels = epoch_data.shape[0]

        for ch_idx in range(n_channels):
            ch_data = epoch_data[ch_idx, :]
            ch_prefix = f"ch{ch_idx:02d}"

            # Time-domain features
            features[f"{ch_prefix}_mean"] = float(np.mean(ch_data))
            features[f"{ch_prefix}_std"] = float(np.std(ch_data))
            features[f"{ch_prefix}_var"] = float(np.var(ch_data))
            features[f"{ch_prefix}_rms"] = float(np.sqrt(np.mean(ch_data**2)))
            features[f"{ch_prefix}_ptp"] = float(np.ptp(ch_data))
            features[f"{ch_prefix}_skew"] = float(skew(ch_data))
            features[f"{ch_prefix}_kurtosis"] = float(kurtosis(ch_data))

            # Hjorth parameters
            if self.config.compute_hjorth:
                activity, mobility, complexity = compute_hjorth_parameters(ch_data)
                features[f"{ch_prefix}_hjorth_activity"] = activity
                features[f"{ch_prefix}_hjorth_mobility"] = mobility
                features[f"{ch_prefix}_hjorth_complexity"] = complexity

            # Frequency-domain features
            total_power = 0.0
            band_powers = {}
            for band_name, band_range in self.config.bands.items():
                bp = compute_band_power(
                    ch_data, sfreq, band_range, self.config.n_fft, self.config.n_per_seg
                )
                band_powers[band_name] = bp
                total_power += bp
                features[f"{ch_prefix}_{band_name}_power"] = bp

            # Relative band power
            if total_power > 0:
                for band_name, bp in band_powers.items():
                    features[f"{ch_prefix}_{band_name}_rel"] = bp / total_power
            else:
                for band_name in band_powers:
                    features[f"{ch_prefix}_{band_name}_rel"] = 0.0

            # Spectral entropy
            if self.config.compute_entropy:
                features[f"{ch_prefix}_spectral_entropy"] = compute_spectral_entropy(
                    ch_data, sfreq, self.config.n_fft, self.config.n_per_seg
                )

            # Peak frequency
            features[f"{ch_prefix}_peak_freq"] = compute_peak_frequency(
                ch_data, sfreq, self.config.n_fft, self.config.n_per_seg
            )

        return features

    def extract_from_epochs(
        self,
        epochs: mne.Epochs,
        subject_id: str = "",
    ) -> list[dict[str, Any]]:
        """Extract features from all epochs in an Epochs object.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched EEG data.
        subject_id : str
            Subject identifier.

        Returns
        -------
        list of dict
            List of feature dictionaries, one per epoch.
        """
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        sfreq = epochs.info["sfreq"]
        results = []

        for i in range(data.shape[0]):
            epoch_features = self.extract_epoch_features(data[i], sfreq)
            epoch_features["subject_id"] = subject_id
            epoch_features["epoch_idx"] = i
            results.append(epoch_features)

        logger.info(
            "Extracted features from %d epochs for %s", len(results), subject_id
        )
        return results

    def extract_batch(
        self,
        segmentation_results: dict[str, Any],
        config: FeatureConfig | dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract features from multiple subjects' epochs.

        Parameters
        ----------
        segmentation_results : dict
            Mapping of subject_id to SegmentationResult.
        config : FeatureConfig or dict, optional
            Feature extraction configuration.

        Returns
        -------
        dict
            Mapping of subject_id to list of feature dicts.
        """
        extractor = FeatureExtractor(config)
        results = {}
        for subject_id, seg_result in segmentation_results.items():
            if seg_result.excluded or seg_result.epochs is None:
                continue
            try:
                results[subject_id] = extractor.extract_from_epochs(
                    seg_result.epochs, subject_id
                )
            except Exception as exc:
                logger.warning(
                    "Feature extraction failed for %s: %s", subject_id, exc
                )
        return results
