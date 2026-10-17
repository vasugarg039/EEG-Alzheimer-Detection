"""Shared utilities for logging, reproducibility, and common helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(
    log_dir: str | Path = "logs",
    log_level: int = logging.INFO,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configure logging with file and console handlers.

    Parameters
    ----------
    log_dir : str or Path
        Directory for log files.
    log_level : int
        Logging level.
    log_to_console : bool
        Whether to also log to stdout.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"eeg_classification_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # Console handler
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    return root_logger


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy and python.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return it as Path.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        Created directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_timestamp() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like "12.3s" or "2.1min".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Parameters
    ----------
    name : str
        Input string.

    Returns
    -------
    str
        Sanitized filename-safe string.
    """
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, "_")
    return name.strip()


def validate_file_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate and return a Path object.

    Parameters
    ----------
    path : str or Path
        Path to validate.
    must_exist : bool
        Whether the path must exist.

    Returns
    -------
    Path
        Validated path.

    Raises
    ------
    FileNotFoundError
        If must_exist is True and path doesn't exist.
    """
    p = Path(path)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p
