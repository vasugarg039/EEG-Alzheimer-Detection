"""FastAPI backend for the EEG Alzheimer's Detection web app.

Serves the static HTML/CSS/JS frontend and exposes JSON API endpoints that
wrap the existing ``src/`` pipeline modules.

Research and educational use only — not a clinical diagnostic tool.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable so `src.*` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data_loader import EEGDataset  # noqa: E402
from src.data_validation import GROUP_NAMES  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.explainability import compute_permutation_importance  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.preprocessing import EEGPreprocessor  # noqa: E402
from src.segmentation import EEGSegmenter  # noqa: E402
from src.train import load_model  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EEG Alzheimer's Detection", version="1.0.0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the cached project configuration with absolute paths.

    Relative paths from the YAML are resolved against the project root so the
    server behaves identically regardless of the launch directory.
    """
    global _config
    if _config is None:
        _config = load_config()
        paths = _config.get("paths", {})
        for key, value in list(paths.items()):
            if value and not Path(str(value)).is_absolute():
                paths[key] = str((PROJECT_ROOT / str(value)).resolve())
    return _config


def get_dataset() -> EEGDataset:
    """Create a dataset object with metadata loaded."""
    ds = EEGDataset(get_config())
    ds.load_metadata()
    return ds


SUPPORTED_EEG_SUFFIXES = {".set", ".edf", ".fif", ".vhdr", ".csv", ".npy"}


def read_eeg_file(path: Path):
    """Read an EEG file using MNE or a generic reader.

    Parameters
    ----------
    path : Path
        Path to the EEG file.

    Returns
    -------
    mne.io.Raw
        Loaded raw EEG data.

    Raises
    ------
    ValueError
        If the format is unsupported or cannot be read.
    """
    import mne

    suffix = path.suffix.lower()
    try:
        if suffix == ".set":
            return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        if suffix == ".edf":
            return mne.io.read_raw_edf(path, preload=True, verbose=False)
        if suffix == ".fif":
            return mne.io.read_raw_fif(path, preload=True, verbose=False)
        if suffix == ".vhdr":
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        if suffix == ".csv":
            data = np.loadtxt(path, delimiter=",")
            return _array_to_raw(data)
        if suffix == ".npy":
            data = np.load(path)
            return _array_to_raw(data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read EEG file '{path.name}': {exc}") from exc

    raise ValueError(f"Unsupported EEG format: {suffix}")


def _array_to_raw(data: np.ndarray):
    """Convert a 2D array to an MNE Raw object (assumes microvolts)."""
    import mne

    if data.ndim != 2:
        raise ValueError("Array must be 2D (channels x samples) or (samples x channels)")

    # Heuristic: assume the larger axis is time
    if data.shape[0] > data.shape[1]:
        data = data.T  # channels x samples

    sfreq = 500.0
    ch_names = [f"ch{i:02d}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # uV -> V
    return raw


def decimate_signal(data: np.ndarray, max_points: int) -> tuple[np.ndarray, int]:
    """Downsample a signal along the last axis so it has at most max_points.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., n_samples).
    max_points : int
        Maximum number of points to keep.

    Returns
    -------
    tuple
        (decimated array, stride used)
    """
    n_samples = data.shape[-1]
    if n_samples <= max_points:
        return data, 1
    stride = int(np.ceil(n_samples / max_points))
    return data[..., ::stride], stride


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Return configuration and availability status."""
    try:
        config = get_config()
    except Exception as exc:  # noqa: BLE001
        return {"config_loaded": False, "error": str(exc)}

    data_root = Path(config["paths"]["data_root"])
    models_dir = Path(config["paths"]["models_dir"])
    features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"
    metrics_path = Path(config["paths"]["results_dir"]) / "metrics.json"

    return _jsonify(
        {
            "config_loaded": True,
            "data_root": str(data_root),
            "data_available": data_root.exists(),
            "features_available": features_path.exists(),
            "metrics_available": metrics_path.exists(),
            "models": sorted(p.parent.name for p in models_dir.glob("*/model.joblib"))
            if models_dir.exists()
            else [],
            "disclaimer": (
                "Research and educational tool only. "
                "Not a clinical diagnostic device."
            ),
        }
    )


@app.get("/api/dataset/summary")
def api_dataset_summary() -> dict[str, Any]:
    """Return a summary of the dataset."""
    try:
        ds = get_dataset()
        summary = ds.get_dataset_summary()
        summary["group_names"] = GROUP_NAMES
        return _jsonify(summary)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dataset/metadata")
def api_dataset_metadata(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    """Return metadata records."""
    try:
        ds = get_dataset()
        df = ds.metadata.head(limit)
        columns = [c for c in df.columns if c != "recording_path"]
        return {"columns": columns, "rows": df[columns].to_dict(orient="records")}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/subjects")
def api_subjects() -> dict[str, Any]:
    """Return the list of subjects with available recordings."""
    try:
        ds = get_dataset()
        df = ds.metadata
        available = df[df["recording_exists"]]
        return _jsonify(
            {
                "subjects": available[
                    ["subject_id", "group", "age", "gender", "mmse"]
                ].to_dict(orient="records"),
                "total": int(len(available)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eeg/{subject_id}")
def api_eeg(
    subject_id: str,
    channels: str | None = Query(None, description="Comma-separated channel names"),
    start: float = Query(0.0, ge=0.0),
    duration: float = Query(10.0, gt=0.0, le=300.0),
    max_points: int = Query(3000, ge=100, le=20000),
) -> dict[str, Any]:
    """Return a window of EEG data for plotting."""
    try:
        ds = get_dataset()
        raw = ds.load_eeg(subject_id)

        sfreq = raw.info["sfreq"]
        start_sample = int(start * sfreq)
        stop_sample = int((start + duration) * sfreq)
        stop_sample = min(stop_sample, raw.n_times)

        if start_sample >= raw.n_times:
            raise HTTPException(status_code=400, detail="Start time beyond recording")

        ch_names = raw.ch_names
        if channels:
            selected = [c.strip() for c in channels.split(",") if c.strip()]
            picks = [ch_names.index(c) for c in selected if c in ch_names]
        else:
            picks = list(range(len(ch_names)))

        data = raw.get_data(picks=picks, start=start_sample, stop=stop_sample)
        data_uv = data * 1e6  # V -> uV

        decimated, stride = decimate_signal(data_uv, max_points)
        times = (np.arange(decimated.shape[1]) * stride) / sfreq + start

        return _jsonify(
            {
                "subject_id": subject_id,
                "sfreq": sfreq,
                "duration": float(raw.n_times / sfreq),
                "start": start,
                "window_duration": duration,
                "channels": [ch_names[p] for p in picks],
                "times": times,
                "data": decimated,
                "stride": stride,
                "n_channels": len(ch_names),
                "all_channels": ch_names,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/features")
def api_features(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Return the extracted subject-level feature table."""
    config = get_config()
    path = Path(config["paths"]["features_dir"]) / "subject_features.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No features found. Run 'python main.py features' first.",
        )
    import pandas as pd

    df = pd.read_csv(path)
    meta_cols = [c for c in ["subject_id", "group", "age", "gender", "mmse"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return {
        "n_subjects": int(len(df)),
        "n_features": len(feature_cols),
        "columns": feature_cols,
        "meta_columns": meta_cols,
        "rows": df.head(limit).to_dict(orient="records"),
    }


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    """Return saved models and their metadata."""
    import json

    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    if not models_dir.exists():
        return {"models": []}

    models = []
    for model_file in sorted(models_dir.glob("*/model.joblib")):
        meta_file = model_file.parent / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        models.append(
            {
                "name": model_file.parent.name,
                "model_name": meta.get("model_name", "unknown"),
                "timestamp": meta.get("timestamp", ""),
                "test_metrics": meta.get("test_metrics", {}),
                "cv_scores": meta.get("cv_scores", {}),
            }
        )
    return _jsonify({"models": models})


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    """Return saved evaluation metrics."""
    import json

    config = get_config()
    path = Path(config["paths"]["results_dir"]) / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train a model first.")
    with open(path, "r", encoding="utf-8") as f:
        return _jsonify(json.load(f))


@app.get("/api/explainability")
def api_explainability(
    model: str = Query(..., description="Model directory name"),
    n_repeats: int = Query(5, ge=1, le=50),
    top_n: int = Query(15, ge=1, le=100),
) -> dict[str, Any]:
    """Compute permutation feature importance for a saved model."""
    import pandas as pd

    config = get_config()
    model_dir = Path(config["paths"]["models_dir"]) / model
    if not (model_dir / "model.joblib").exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")

    features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"
    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Features not found.")

    clf, meta = load_model(model_dir)
    feature_names = meta.get("feature_names", [])
    df = pd.read_csv(features_path)
    X = np.nan_to_num(df[feature_names].values)
    y = np.asarray(df["group"].values)

    importance = compute_permutation_importance(
        clf, X, y, feature_names, n_repeats=n_repeats
    )
    importance["ranked_features"] = importance["ranked_features"][:top_n]
    importance["ranked_importance"] = importance["ranked_importance"][:top_n]
    return _jsonify(importance)


@app.post("/api/upload/eeg")
async def api_upload_eeg(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload an EEG file and return basic validation info."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EEG_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_EEG_SUFFIXES)}",
        )

    dest = UPLOAD_DIR / (file.filename or "upload")
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    info: dict[str, Any] = {"filename": dest.name, "size_bytes": len(contents)}
    try:
        raw = read_eeg_file(dest)
        info.update(
            {
                "valid": True,
                "n_channels": len(raw.ch_names),
                "channels": raw.ch_names,
                "sfreq": float(raw.info["sfreq"]),
                "duration": float(raw.n_times / raw.info["sfreq"]),
            }
        )
    except Exception as exc:  # noqa: BLE001
        info.update({"valid": False, "error": str(exc)})

    return _jsonify(info)


@app.post("/api/upload/metadata")
async def api_upload_metadata(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a metadata CSV and validate its schema."""
    import pandas as pd

    from src.data_validation import validate_metadata_schema

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Metadata must be a .csv file")

    dest = UPLOAD_DIR / (file.filename or "metadata.csv")
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    try:
        df = pd.read_csv(dest)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    report = validate_metadata_schema(df)
    return _jsonify(
        {
            "filename": dest.name,
            "n_rows": int(len(df)),
            "columns": df.columns.tolist(),
            "is_valid": report.is_valid,
            "errors": report.errors,
            "warnings": report.warnings,
            "group_counts": report.group_counts,
            "summary": report.summary(),
        }
    )


@app.post("/api/preprocess")
def api_preprocess(
    subject_id: str = Query(...),
) -> dict[str, Any]:
    """Preprocess a subject's recording and return summary statistics."""
    try:
        config = get_config()
        ds = get_dataset()
        raw = ds.load_eeg(subject_id)
        preprocessor = EEGPreprocessor(config["preprocessing"])
        result = preprocessor.preprocess(raw)
        return _jsonify(
            {
                "subject_id": subject_id,
                "steps_applied": result.steps_applied,
                "n_channels_original": result.n_channels_original,
                "n_channels_final": result.n_channels_final,
                "sfreq_original": result.sfreq_original,
                "sfreq_final": result.sfreq_final,
                "duration_original": result.duration_original,
                "duration_final": result.duration_final,
                "excluded": result.excluded,
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/segment")
def api_segment(subject_id: str = Query(...)) -> dict[str, Any]:
    """Preprocess and segment a subject, returning epoch counts."""
    try:
        config = get_config()
        ds = get_dataset()
        raw = ds.load_eeg(subject_id)
        preproc = EEGPreprocessor(config["preprocessing"]).preprocess(raw)
        seg = EEGSegmenter(config["segmentation"]).segment(preproc.raw_filtered)
        return _jsonify(
            {
                "subject_id": subject_id,
                "n_epochs_total": seg.n_epochs_total,
                "n_epochs_rejected": seg.n_epochs_rejected,
                "n_epochs_clean": seg.n_epochs_clean,
                "epoch_length_sec": seg.epoch_length_sec,
                "overlap": seg.overlap,
                "sfreq": seg.sfreq,
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/extract")
def api_extract(subject_id: str = Query(...)) -> dict[str, Any]:
    """Extract features for a subject and return band-power style summary."""
    try:
        config = get_config()
        ds = get_dataset()
        raw = ds.load_eeg(subject_id)
        preproc = EEGPreprocessor(config["preprocessing"]).preprocess(raw)
        seg = EEGSegmenter(config["segmentation"]).segment(preproc.raw_filtered)
        extractor = FeatureExtractor(config["features"])
        epoch_features = extractor.extract_from_epochs(seg.epochs, subject_id)

        # Aggregate band powers across channels and epochs for display
        bands = list(config["features"]["bands"].keys())
        band_means: dict[str, float] = {}
        for band in bands:
            vals = [
                ef[f"ch{ci:02d}_{band}_power"]
                for ef in epoch_features
                for ci in range(len(seg.epochs.ch_names))
                if f"ch{ci:02d}_{band}_power" in ef
            ]
            band_means[band] = float(np.mean(vals)) if vals else 0.0

        return _jsonify(
            {
                "subject_id": subject_id,
                "n_epochs": len(epoch_features),
                "n_features_per_epoch": len(epoch_features[0]) if epoch_features else 0,
                "band_power": band_means,
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def api_predict(
    model: str = Query(...),
    subject_id: str | None = Query(None),
    filename: str | None = Query(None),
) -> dict[str, Any]:
    """Predict the class for a subject or an uploaded file."""
    try:
        config = get_config()
        model_dir = Path(config["paths"]["models_dir"]) / model
        if not (model_dir / "model.joblib").exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model}")

        clf, meta = load_model(model_dir)
        feature_names = meta.get("feature_names", [])

        # Resolve EEG source
        if filename:
            eeg_path = UPLOAD_DIR / filename
            if not eeg_path.exists():
                raise HTTPException(status_code=404, detail=f"Upload not found: {filename}")
            raw = read_eeg_file(eeg_path)
            source = filename
        elif subject_id:
            ds = get_dataset()
            raw = ds.load_eeg(subject_id)
            source = subject_id
        else:
            raise HTTPException(status_code=400, detail="Provide subject_id or filename")

        preproc = EEGPreprocessor(config["preprocessing"]).preprocess(raw)
        if preproc.excluded:
            raise HTTPException(status_code=400, detail="Preprocessing failed for input")

        seg = EEGSegmenter(config["segmentation"]).segment(preproc.raw_filtered)
        if seg.n_epochs_clean == 0:
            raise HTTPException(status_code=400, detail="No valid epochs produced")

        extractor = FeatureExtractor(config["features"])
        epoch_features = extractor.extract_from_epochs(seg.epochs, source or "input")

        # Aggregate to subject level (mean)
        vector = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            vector[i] = np.mean([ef.get(fname, 0.0) for ef in epoch_features])

        X = vector.reshape(1, -1)
        pred = clf.predict(X)[0]

        result: dict[str, Any] = {
            "success": True,
            "source": source,
            "predicted_class": str(pred),
            "predicted_label": meta.get("label_mapping", {}).get(str(pred), str(pred)),
            "n_epochs": len(epoch_features),
        }
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)
            result["probabilities"] = {
                str(c): float(p) for c, p in zip(classes, proba)
            }
            result["confidence"] = float(np.max(proba))
        return _jsonify(result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/uploaded")
def api_uploaded() -> dict[str, Any]:
    """List uploaded EEG files."""
    files = []
    for p in sorted(UPLOAD_DIR.glob("*")):
        if p.is_file():
            files.append({"filename": p.name, "size_bytes": p.stat().st_size})
    return {"files": files}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the web server."""
    import uvicorn

    setup_logging(get_config()["paths"]["logs_dir"])
    logger.info("Starting EEG web app on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
