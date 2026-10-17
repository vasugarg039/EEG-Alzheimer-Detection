"""EEG preprocessing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing."""

    highpass: float = 1.0
    lowpass: float = 45.0
    notch: float | None = 50.0
    resample_sfreq: float | None = None
    ica_enabled: bool = False
    ica_n_components: int = 15
    reject_threshold_uv: float = 200.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PreprocessingConfig":
        """Create from configuration dictionary."""
        return cls(
            highpass=config.get("highpass", 1.0),
            lowpass=config.get("lowpass", 45.0),
            notch=config.get("notch", 50.0),
            resample_sfreq=config.get("resample_sfreq"),
            ica_enabled=config.get("ica_enabled", False),
            ica_n_components=config.get("ica_n_components", 15),
            reject_threshold_uv=config.get("reject_threshold_uv", 200.0),
        )


@dataclass
class PreprocessingResult:
    """Result of preprocessing a single recording."""

    raw_filtered: mne.io.Raw
    n_channels_original: int
    n_channels_final: int
    sfreq_original: float
    sfreq_final: float
    duration_original: float
    duration_final: float
    ica_applied: bool = False
    n_ica_components: int = 0
    reject_threshold_uv: float = 0.0
    excluded: bool = False
    exclusion_reason: str = ""
    steps_applied: list[str] = field(default_factory=list)


class EEGPreprocessor:
    """EEG preprocessing pipeline.

    Applies filtering, optional ICA, and artifact rejection.
    """

    def __init__(self, config: PreprocessingConfig | dict[str, Any] | None = None):
        """Initialize preprocessor.

        Parameters
        ----------
        config : PreprocessingConfig or dict, optional
            Preprocessing configuration.
        """
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            self.config = PreprocessingConfig.from_dict(config)
        else:
            self.config = config

    def preprocess(self, raw: mne.io.Raw) -> PreprocessingResult:
        """Apply full preprocessing pipeline to a Raw object.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data.

        Returns
        -------
        PreprocessingResult
            Preprocessing result with filtered data and metadata.
        """
        result = PreprocessingResult(
            raw_filtered=raw.copy(),
            n_channels_original=len(raw.ch_names),
            n_channels_final=len(raw.ch_names),
            sfreq_original=raw.info["sfreq"],
            sfreq_final=raw.info["sfreq"],
            duration_original=raw.n_times / raw.info["sfreq"],
            duration_final=raw.n_times / raw.info["sfreq"],
            reject_threshold_uv=self.config.reject_threshold_uv,
        )

        raw_proc = result.raw_filtered

        # Step 1: Band-pass filter
        logger.info(
            "Applying band-pass filter: %.1f - %.1f Hz",
            self.config.highpass,
            self.config.lowpass,
        )
        raw_proc.filter(
            l_freq=self.config.highpass,
            h_freq=self.config.lowpass,
            picks="eeg",
            verbose=False,
        )
        result.steps_applied.append(
            f"bandpass_{self.config.highpass}_{self.config.lowpass}Hz"
        )

        # Step 2: Notch filter for line frequency
        if self.config.notch is not None:
            logger.info("Applying notch filter at %.1f Hz", self.config.notch)
            raw_proc.notch_filter(
                freqs=self.config.notch,
                picks="eeg",
                verbose=False,
            )
            result.steps_applied.append(f"notch_{self.config.notch}Hz")

        # Step 3: Resampling (if configured)
        if self.config.resample_sfreq is not None:
            if raw_proc.info["sfreq"] != self.config.resample_sfreq:
                logger.info(
                    "Resampling from %.1f to %.1f Hz",
                    raw_proc.info["sfreq"],
                    self.config.resample_sfreq,
                )
                raw_proc.resample(self.config.resample_sfreq, verbose=False)
                result.sfreq_final = raw_proc.info["sfreq"]
                result.steps_applied.append(
                    f"resample_{self.config.resample_sfreq}Hz"
                )

        # Step 4: ICA (if enabled)
        if self.config.ica_enabled:
            ica = mne.preprocessing.ICA(
                n_components=min(
                    self.config.ica_n_components, len(raw_proc.ch_names)
                ),
                random_state=42,
                verbose=False,
            )
            ica.fit(raw_proc, verbose=False)
            # Remove components correlated with EOG
            eog_indices, _ = ica.find_bads_eog(raw_proc, verbose=False)
            ica.exclude = eog_indices
            ica.apply(raw_proc, verbose=False)
            result.ica_applied = True
            result.n_ica_components = len(eog_indices)
            result.steps_applied.append(f"ica_removed_{len(eog_indices)}")

        # Update duration after potential resampling
        result.duration_final = raw_proc.n_times / raw_proc.info["sfreq"]
        result.raw_filtered = raw_proc

        return result

    def preprocess_batch(
        raws: dict[str, mne.io.Raw],
        config: PreprocessingConfig | dict[str, Any] | None = None,
    ) -> dict[str, PreprocessingResult]:
        """Preprocess multiple recordings.

        Parameters
        ----------
        raws : dict
            Mapping of subject_id to Raw objects.
        config : PreprocessingConfig or dict, optional
            Preprocessing configuration.

        Returns
        -------
        dict
            Mapping of subject_id to PreprocessingResult.
        """
        preprocessor = EEGPreprocessor(config)
        results = {}
        for subject_id, raw in raws.items():
            logger.info("Preprocessing %s...", subject_id)
            try:
                results[subject_id] = preprocessor.preprocess(raw)
            except Exception as exc:
                logger.warning("Preprocessing failed for %s: %s", subject_id, exc)
                results[subject_id] = PreprocessingResult(
                    raw_filtered=raw,
                    n_channels_original=len(raw.ch_names),
                    n_channels_final=0,
                    sfreq_original=raw.info["sfreq"],
                    sfreq_final=0,
                    duration_original=raw.n_times / raw.info["sfreq"],
                    duration_final=0,
                    excluded=True,
                    exclusion_reason=str(exc),
                )
        return results


def validate_channel_consistency(
    raw: mne.io.Raw,
    expected_channels: list[str],
) -> tuple[bool, list[str]]:
    """Validate channel names match expected set.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    expected_channels : list of str
        Expected channel names.

    Returns
    -------
    tuple
        (is_valid, list_of_issues)
    """
    issues = []
    actual = set(raw.ch_names)
    expected = set(expected_channels)

    missing = expected - actual
    extra = actual - expected

    if missing:
        issues.append(f"Missing channels: {sorted(missing)}")
    if extra:
        issues.append(f"Extra channels: {sorted(extra)}")

    return len(issues) == 0, issues
