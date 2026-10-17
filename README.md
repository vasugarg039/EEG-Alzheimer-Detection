# EEG Alzheimer's Detection

**Research and Educational Tool — Not a Clinical Diagnostic Device**

This project implements a machine learning pipeline for three-class EEG classification:
- **A** — Alzheimer's Disease (AD)
- **F** — Frontotemporal Dementia (FTD)
- **C** — Healthy Controls (HC)

## Disclaimer

This software is intended for **research and educational purposes only**. It is not a clinically validated diagnostic tool and should not be used for medical diagnosis. Results do not constitute medical advice.

## Repository Structure

```
.
├── main.py                  # CLI entry point
├── app.py                   # Streamlit application
├── requirements.txt         # Python dependencies
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration loading
│   ├── utils.py             # Shared utilities
│   ├── data_validation.py   # Data/schema validation
│   ├── data_loader.py       # Metadata and EEG loading
│   ├── preprocessing.py     # EEG preprocessing
│   ├── segmentation.py      # EEG windowing
│   ├── features.py          # Feature extraction
│   ├── dataset.py           # Dataset assembly
│   ├── train.py             # Model training
│   ├── evaluate.py          # Evaluation metrics
│   ├── predict.py           # Inference
│   ├── explainability.py     # Model interpretation
│   ├── visualization.py     # Plotting functions
│   └── reporting.py         # Report generation
├── tests/                   # Test suite
├── data/
│   ├── metadata/            # Dataset metadata
│   ├── raw/                 # Raw EEG (gitignored)
│   ├── processed/           # Processed EEG (gitignored)
│   └── features/            # Extracted features (gitignored)
├── models/                  # Trained models (gitignored)
├── results/                 # Evaluation results
├── reports/                 # Generated reports
└── logs/                    # Log files (gitignored)
```

## Setup

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd EEG-Alzheimer-Detection
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```powershell
   python main.py validate
   ```

## Dataset

### Metadata

The dataset metadata is stored in `data/metadata/eeg_dataset_summary.csv` with the following schema:

| Column | Type | Description |
|---|---|---|
| `subject_id` | string | Unique subject identifier (e.g., `sub-001`) |
| `group` | string | Class label: `A`, `F`, or `C` |
| `age` | int | Subject age |
| `gender` | string | `M` or `F` |
| `mmse` | int | MMSE score (0–30) |
| `channels` | int | Number of EEG channels |
| `sampling_frequency` | float | Sampling rate in Hz |
| `samples` | int | Total sample count |
| `duration_seconds` | float | Recording duration |
| `duration_minutes` | float | Recording duration in minutes |
| `status` | string | Processing status |
| `group_name` | string | Full group name |

### EEG Data

- **Format**: EEGLAB `.set` files
- **Channels**: 19 (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz)
- **Sampling rate**: 500 Hz
- **Task**: Eyes-closed resting state
- **Structure**: BIDS-like (`sub-001/eeg/sub-001_task-eyesclosed_eeg.set`)

### Configuring Data Path

Edit `config/config.yaml` to set the `paths.data_root` to your EEG dataset location:

```yaml
paths:
  data_root: "D:/EEG Dataset/ds004504"
```

## Usage

### Command-Line Interface

All commands support `--config` to specify a custom config file.

#### Validate Dataset
```powershell
python main.py validate
```
Checks metadata schema, class distribution, and recording availability.

#### Preprocess EEG
```powershell
python main.py preprocess
```
Applies band-pass filter, notch filter, and optional ICA.

#### Extract Features
```powershell
python main.py features
```
Segments EEG into epochs and extracts time-domain, frequency-domain, and complexity features.

#### Train Models
```powershell
python main.py train
```
Trains Logistic Regression, SVM, and Random Forest with subject-level cross-validation.

#### Evaluate Model
```powershell
python main.py evaluate --model models/logistic_regression_20240101_120000
```
Evaluates a saved model on the full dataset.

#### Predict
```powershell
python main.py predict --model models/logistic_regression_20240101_120000 --input path/to/eeg.set
```
Classifies a new EEG recording.

#### Launch App
```powershell
python main.py app
```
Starts the Streamlit web interface.

### Streamlit Application

```powershell
python main.py app
```

The app provides pages for:
- Dashboard
- Dataset Overview
- EEG Viewer
- Preprocessing
- Feature Analysis
- Model Training
- Evaluation
- Explainability
- Prediction
- Report Export

## Testing

Run the test suite:

```powershell
pytest tests/ -v
```

Run with coverage:

```powershell
pytest tests/ --cov=src --cov-report=html
```

## Configuration

All configuration is in `config/config.yaml`:

- **paths**: Data, model, and output directories
- **eeg**: Expected channels, sampling rate, channel names
- **preprocessing**: Filter parameters, ICA settings
- **segmentation**: Epoch length, overlap
- **features**: Frequency bands, FFT parameters
- **models**: Model hyperparameters, CV settings
- **training**: Primary/secondary metrics
- **explainability**: Method, repeats
- **visualization**: DPI, figure size
- **report**: Export settings

## Methodology

### Preprocessing
1. Band-pass filter: 1–45 Hz
2. Notch filter: 50 Hz (power-line noise)
3. Optional ICA for artifact removal

### Segmentation
- 2-second epochs with 50% overlap
- Quality-based rejection (variance threshold)

### Features
- **Time-domain**: mean, std, variance, RMS, peak-to-peak, skewness, kurtosis
- **Hjorth parameters**: activity, mobility, complexity
- **Frequency-domain**: band power (delta, theta, alpha, beta, gamma), relative band power
- **Spectral**: entropy, peak frequency

### Training
- Subject-level train/test split (no window leakage)
- GroupKFold cross-validation
- StandardScaler + classifier pipelines
- Models: Logistic Regression, SVM, Random Forest

### Evaluation
- Accuracy, balanced accuracy, macro F1
- Per-class precision, recall, F1
- Confusion matrix
- Cross-validation summaries

## Reproducibility

- Fixed random seeds (configurable)
- All parameters in `config/config.yaml`
- Model metadata includes feature names, label mapping, config, and timestamp
- Logs saved with timestamps

## Limitations

- Small sample size (88 subjects) may lead to high variance
- Class imbalance (36/29/23) requires careful evaluation
- Results are not clinically validated
- EEG data quality affects model performance
- Feature importance reflects model behavior, not biological causation

## License

This project is for research and educational use only.
