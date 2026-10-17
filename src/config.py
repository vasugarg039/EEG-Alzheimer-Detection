"""Configuration loading and path management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config YAML. Defaults to ``config/config.yaml``.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", path)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def resolve_path(path_str: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to project root if not absolute.

    Parameters
    ----------
    path_str : str or Path
        Path to resolve.
    root : Path, optional
        Root directory. Defaults to project root.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    p = Path(path_str)
    # On Windows, Path("/foo") is relative — check for root or drive
    if p.is_absolute() or p.drive or p.root:
        return p
    base = root or PROJECT_ROOT
    return base / p
