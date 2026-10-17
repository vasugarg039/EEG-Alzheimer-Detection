"""Tests for segmentation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation import (
    SegmentationConfig,
    SegmentationResult,
    EEGSegmenter,
)


class TestSegmentationConfig:
    """Tests for SegmentationConfig."""

    def test_defaults(self):
        config = SegmentationConfig()
        assert config.epoch_length_sec == 2.0
        assert config.overlap == 0.5

    def test_from_dict(self):
        config = SegmentationConfig.from_dict({
            "epoch_length_sec": 3.0,
            "overlap": 0.25,
        })
        assert config.epoch_length_sec == 3.0
        assert config.overlap == 0.25


class TestEEGSegmenter:
    """Tests for EEGSegmenter."""

    def test_init_default(self):
        segmenter = EEGSegmenter()
        assert segmenter.config.epoch_length_sec == 2.0

    def test_init_with_config(self):
        config = SegmentationConfig(epoch_length_sec=4.0)
        segmenter = EEGSegmenter(config)
        assert segmenter.config.epoch_length_sec == 4.0
