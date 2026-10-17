"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config, get_project_root, resolve_path


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, dict)
        assert "paths" in config
        assert "eeg" in config
        assert "preprocessing" in config

    def test_load_custom_config(self, tmp_path):
        import yaml

        custom = {"test": {"value": 42}}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(custom, f)

        config = load_config(config_file)
        assert config["test"]["value"] == 42

    def test_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestGetProjectRoot:
    """Tests for project root detection."""

    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_has_expected_structure(self):
        root = get_project_root()
        assert (root / "src").exists()
        assert (root / "config").exists()


class TestResolvePath:
    """Tests for path resolution."""

    def test_absolute_path_unchanged(self):
        path = Path("/absolute/path")
        assert resolve_path(path) == path

    def test_relative_path_resolved(self):
        path = Path("data/metadata")
        resolved = resolve_path(path)
        assert resolved.is_absolute()
        # On Windows the separator is \, on Unix it's /
        assert str(resolved).endswith("data") and "metadata" in str(resolved)
