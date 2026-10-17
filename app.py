"""Streamlit application for EEG Alzheimer's Detection.

Research and educational use only — not a clinical diagnostic tool.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data_loader import EEGDataset
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="EEG Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "config" not in st.session_state:
        try:
            st.session_state.config = load_config()
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            st.session_state.config = None

    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    if "model" not in st.session_state:
        st.session_state.model = None

    if "model_metadata" not in st.session_state:
        st.session_state.model_metadata = None


def sidebar_navigation() -> str:
    """Render sidebar navigation and return selected page."""
    st.sidebar.title("EEG Alzheimer's Detection")
    st.sidebar.markdown("---")

    pages = [
        "Dashboard",
        "Dataset Overview",
        "EEG Viewer",
        "Preprocessing",
        "Feature Analysis",
        "Model Training",
        "Evaluation",
        "Explainability",
        "Prediction",
        "Report Export",
    ]

    selected = st.sidebar.radio("Navigation", pages)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Disclaimer**: This is a research and educational tool.
    It is not a clinical diagnostic device and should not be used
    for medical diagnosis.
    """)

    return selected


def page_dashboard() -> None:
    """Dashboard page."""
    st.markdown('<p class="main-header">EEG Alzheimer\'s Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Research & Educational Tool</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    <strong>Disclaimer</strong>: This application is for research and educational purposes only.
    It is not a clinically validated diagnostic tool and should not be used for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Subjects", "88")
    with col2:
        st.metric("EEG Channels", "19")
    with col3:
        st.metric("Sampling Rate", "500 Hz")

    st.markdown("---")
    st.subheader("Quick Start")

    st.markdown("""
    1. **Dataset Overview** — Explore the dataset metadata and class distribution
    2. **EEG Viewer** — Visualize raw EEG recordings
    3. **Preprocessing** — Apply filtering and artifact removal
    4. **Feature Analysis** — Extract and explore EEG features
    5. **Model Training** — Train and compare ML models
    6. **Evaluation** — View model performance metrics
    7. **Prediction** — Classify new EEG recordings
    """)

    # Status
    st.markdown("---")
    st.subheader("System Status")

    config = st.session_state.get("config")
    if config:
        st.success("Configuration loaded")
        data_root = config.get("paths", {}).get("data_root", "N/A")
        st.info(f"Data root: `{data_root}`")

        # Check data availability
        if Path(data_root).exists():
            st.success("EEG data directory found")
        else:
            st.warning("EEG data directory not found — some features will be unavailable")
    else:
        st.error("Configuration not loaded")


def page_dataset_overview() -> None:
    """Dataset overview page."""
    st.title("Dataset Overview")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    try:
        dataset = EEGDataset(config)
        dataset.load_metadata()
        summary = dataset.get_dataset_summary()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Subjects", summary["n_subjects"])
        col2.metric("Recordings Available", summary["recording_available"])
        col3.metric("Age Range", f"{summary['age_range'][0]:.0f}–{summary['age_range'][1]:.0f}")
        col4.metric("MMSE Range", f"{summary['mmse_range'][0]:.0f}–{summary['mmse_range'][1]:.0f}")

        # Class distribution
        st.subheader("Class Distribution")
        group_counts = summary["group_counts"]
        st.bar_chart(group_counts)

        # Metadata table
        st.subheader("Metadata")
        st.dataframe(dataset.metadata, use_container_width=True)

        # Validation report
        if dataset.validation_report:
            st.subheader("Validation Report")
            st.text(dataset.validation_report.summary())

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")


def page_eeg_viewer() -> None:
    """EEG viewer page."""
    st.title("EEG Viewer")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    try:
        dataset = EEGDataset(config)
        dataset.load_metadata()

        available = dataset.metadata[dataset.metadata["recording_exists"]]
        if available.empty:
            st.warning("No EEG recordings available")
            return

        # Subject selection
        subject_ids = available["subject_id"].tolist()
        selected_subject = st.selectbox("Select Subject", subject_ids)

        if selected_subject:
            raw = dataset.load_eeg(selected_subject)

            # Display info
            st.subheader(f"Subject: {selected_subject}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Channels", len(raw.ch_names))
            col2.metric("Sampling Rate", f"{raw.info['sfreq']:.0f} Hz")
            col3.metric("Duration", f"{raw.n_times / raw.info['sfreq']:.1f} s")

            # Channel selection
            channels = st.multiselect(
                "Channels",
                raw.ch_names,
                default=raw.ch_names[:5],
            )

            # Time range
            max_time = raw.n_times / raw.info["sfreq"]
            time_range = st.slider(
                "Time Range (s)",
                0.0,
                max_time,
                (0.0, min(10.0, max_time)),
            )

            # Plot
            if channels:
                from src.visualization import plot_eeg_traces
                fig = plot_eeg_traces(
                    raw,
                    duration=time_range[1] - time_range[0],
                    start=time_range[0],
                    channels=channels,
                )
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load EEG: {e}")


def page_preprocessing() -> None:
    """Preprocessing page."""
    st.title("Preprocessing")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    st.subheader("Preprocessing Configuration")
    prep_config = config.get("preprocessing", {})

    col1, col2, col3 = st.columns(3)
    highpass = col1.number_input("High-pass (Hz)", 0.0, 10.0, prep_config.get("highpass", 1.0))
    lowpass = col2.number_input("Low-pass (Hz)", 10.0, 100.0, prep_config.get("lowpass", 45.0))
    notch = col3.number_input("Notch (Hz)", 0.0, 100.0, prep_config.get("notch", 50.0))

    st.markdown("---")
    st.info("Preprocessing is applied during feature extraction. See the Feature Analysis page to run the full pipeline.")


def page_feature_analysis() -> None:
    """Feature analysis page."""
    st.title("Feature Analysis")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
        st.subheader("Extracted Features")
        st.dataframe(df, use_container_width=True)

        st.subheader("Feature Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("No features extracted yet. Run feature extraction first.")
        if st.button("Extract Features"):
            st.info("Run `python main.py features` from the command line to extract features.")


def page_model_training() -> None:
    """Model training page."""
    st.title("Model Training")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"

    if not features_path.exists():
        st.warning("No features available. Extract features first.")
        return

    st.subheader("Available Models")
    model_configs = config.get("models", {}).get("models", [])
    for model in model_configs:
        st.write(f"- **{model['name']}**")

    st.markdown("---")
    st.info("Run `python main.py train` from the command line to train models.")

    # Check for saved models
    models_dir = Path(config["paths"]["models_dir"])
    if models_dir.exists():
        saved_models = list(models_dir.glob("*/model.joblib"))
        if saved_models:
            st.subheader("Saved Models")
            for model_path in saved_models:
                st.write(f"- {model_path.parent.name}")


def page_evaluation() -> None:
    """Evaluation page."""
    st.title("Evaluation")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    results_path = Path(config["paths"]["results_dir"]) / "metrics.json"

    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)

        st.subheader("Model Results")
        st.json(results)
    else:
        st.warning("No evaluation results available. Train models first.")


def page_explainability() -> None:
    """Explainability page."""
    st.title("Explainability")

    st.info("Model explainability will be available after training.")

    st.markdown("""
    ### About Explainability

    Feature importance shows which EEG features the model uses for classification.
    This reflects model behavior, not biological causation.

    **Important**: Feature importance does not prove Alzheimer's pathology or
    any other biological mechanism.
    """)


def page_prediction() -> None:
    """Prediction page."""
    st.title("Prediction")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    models_dir = Path(config["paths"]["models_dir"])
    if not models_dir.exists() or not list(models_dir.glob("*/model.joblib")):
        st.warning("No trained model available. Train a model first.")
        return

    # Model selection
    saved_models = [p.parent for p in models_dir.glob("*/model.joblib")]
    model_names = [p.name for p in saved_models]
    selected_model = st.selectbox("Select Model", model_names)

    if selected_model:
        st.info(f"Selected model: {selected_model}")
        st.markdown("---")
        st.info("Run `python main.py predict --model <model_dir> --input <eeg_file>` from the command line to make predictions.")


def page_report_export() -> None:
    """Report export page."""
    st.title("Report Export")

    config = st.session_state.get("config")
    if not config:
        st.error("Configuration not loaded")
        return

    st.subheader("Generate Report")

    st.info("Run `python main.py train` to generate experiment reports automatically.")

    reports_dir = Path(config["paths"]["reports_dir"])
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.txt")) + list(reports_dir.glob("*.json"))
        if reports:
            st.subheader("Available Reports")
            for report in reports:
                st.write(f"- {report.name}")
                with open(report) as f:
                    with st.expander("Preview"):
                        st.text(f.read()[:2000])


def main() -> None:
    """Main application entry point."""
    init_session_state()
    page = sidebar_navigation()

    pages = {
        "Dashboard": page_dashboard,
        "Dataset Overview": page_dataset_overview,
        "EEG Viewer": page_eeg_viewer,
        "Preprocessing": page_preprocessing,
        "Feature Analysis": page_feature_analysis,
        "Model Training": page_model_training,
        "Evaluation": page_evaluation,
        "Explainability": page_explainability,
        "Prediction": page_prediction,
        "Report Export": page_report_export,
    }

    pages[page]()


if __name__ == "__main__":
    main()
