"""Report generation for experiments."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import ensure_dir, get_timestamp

logger = logging.getLogger(__name__)


def generate_text_report(
    dataset_summary: dict[str, Any],
    preprocessing_config: dict[str, Any],
    feature_config: dict[str, Any],
    model_results: list[dict[str, Any]],
    best_model_name: str,
    output_dir: str | Path,
) -> Path:
    """Generate a human-readable text report.

    Parameters
    ----------
    dataset_summary : dict
        Dataset summary.
    preprocessing_config : dict
        Preprocessing configuration.
    feature_config : dict
        Feature extraction configuration.
    model_results : list of dict
        Model results.
    best_model_name : str
        Name of the best model.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    Path
        Path to saved report.
    """
    out_dir = ensure_dir(output_dir)
    timestamp = get_timestamp()
    report_path = out_dir / f"report_{timestamp}.txt"

    lines = [
        "=" * 70,
        "EEG ALZHEIMER'S DETECTION — EXPERIMENT REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        "-" * 70,
        "DATASET SUMMARY",
        "-" * 70,
        f"Total subjects: {dataset_summary.get('n_subjects', 'N/A')}",
        f"Group counts: {dataset_summary.get('group_counts', {})}",
        f"Recordings available: {dataset_summary.get('recording_available', 'N/A')}",
        f"Age range: {dataset_summary.get('age_range', 'N/A')}",
        f"MMSE range: {dataset_summary.get('mmse_range', 'N/A')}",
        f"Duration range (min): {dataset_summary.get('duration_range_min', 'N/A')}",
        "",
        "-" * 70,
        "PREPROCESSING CONFIGURATION",
        "-" * 70,
    ]

    for key, value in preprocessing_config.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "-" * 70,
        "FEATURE EXTRACTION CONFIGURATION",
        "-" * 70,
    ])

    for key, value in feature_config.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "-" * 70,
        "MODEL RESULTS",
        "-" * 70,
        f"Best model: {best_model_name}",
        "",
    ])

    for result in model_results:
        lines.append(f"  Model: {result['name']}")
        metrics = result.get("test_metrics", {})
        for metric_name, value in metrics.items():
            lines.append(f"    {metric_name}: {value:.4f}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "LIMITATIONS",
        "-" * 70,
        "- This is a research/educational tool, not a clinical diagnostic device.",
        "- Results do not constitute medical advice.",
        "- Model performance depends on data quality and representativeness.",
        "- Small sample sizes may lead to high variance in metrics.",
        "",
        "=" * 70,
    ])

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Report saved to %s", report_path)
    return report_path


def generate_json_report(
    dataset_summary: dict[str, Any],
    preprocessing_config: dict[str, Any],
    feature_config: dict[str, Any],
    model_results: list[dict[str, Any]],
    best_model_name: str,
    output_dir: str | Path,
) -> Path:
    """Generate a machine-readable JSON report.

    Parameters
    ----------
    dataset_summary : dict
        Dataset summary.
    preprocessing_config : dict
        Preprocessing configuration.
    feature_config : dict
        Feature extraction configuration.
    model_results : list of dict
        Model results.
    best_model_name : str
        Name of the best model.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    Path
        Path to saved report.
    """
    out_dir = ensure_dir(output_dir)
    timestamp = get_timestamp()
    report_path = out_dir / f"report_{timestamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_summary": dataset_summary,
        "preprocessing_config": preprocessing_config,
        "feature_config": feature_config,
        "model_results": model_results,
        "best_model": best_model_name,
        "limitations": [
            "Research/educational tool only",
            "Not a clinical diagnostic device",
            "Results do not constitute medical advice",
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("JSON report saved to %s", report_path)
    return report_path


def generate_csv_export(
    model_results: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Generate CSV export of model results.

    Parameters
    ----------
    model_results : list of dict
        Model results.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    Path
        Path to saved CSV.
    """
    import pandas as pd

    out_dir = ensure_dir(output_dir)
    timestamp = get_timestamp()
    csv_path = out_dir / f"model_results_{timestamp}.csv"

    rows = []
    for result in model_results:
        row = {"model": result["name"]}
        row.update(result.get("test_metrics", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    logger.info("CSV export saved to %s", csv_path)
    return csv_path
