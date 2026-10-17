# AGENTS.md

## Project

EEG-based 3-class classification: Alzheimer's Disease (`A`), Frontotemporal Dementia (`F`), Healthy Controls (`C`). Research and educational use only — not a clinical diagnostic tool.

## Dataset

Metadata for 88 subjects lives in `data/metadata/eeg_dataset_summary.csv` (36 AD, 29 HC, 23 FTD). Recordings are 19-channel EEG at 500 Hz, ~5-21 minutes each. Raw data is NOT in the repo — set `paths.data_root` in `config/config.yaml` to the EEG dataset location.

## Code Status

All `src/` modules are fully implemented. `main.py` provides CLI commands: `validate`, `preprocess`, `features`, `train`, `evaluate`, `predict`, `app`. Tests cover 92 cases across all modules.

## Key Commands

```powershell
python main.py validate      # Check dataset metadata and recordings
python main.py features      # Extract features from EEG
python main.py train         # Train and compare models
python main.py app           # Launch Streamlit interface
pytest tests/ -v             # Run test suite
```

## Architecture

- `src/config.py` — Configuration loading
- `src/data_validation.py` — Schema validation, label mapping
- `src/data_loader.py` — Metadata + EEG file loading
- `src/preprocessing.py` — Band-pass, notch, ICA, artifact rejection
- `src/segmentation.py` — Fixed-length epoching
- `src/features.py` — Time-domain, frequency-domain, Hjorth features
- `src/dataset.py` — Feature assembly, subject-level aggregation, train/test split
- `src/train.py` — Model training, comparison, persistence
- `src/evaluate.py` — Metrics computation and reporting
- `src/predict.py` — End-to-end inference
- `src/explainability.py` — Permutation importance, model-based importance
- `src/visualization.py` — EEG traces, PSD, confusion matrix, feature importance
- `src/reporting.py` — Text, JSON, CSV report generation
