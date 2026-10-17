"""EEG Alzheimer's Detection — CLI entry point.

Usage:
    python main.py validate [--config CONFIG]
    python main.py preprocess [--config CONFIG]
    python main.py features [--config CONFIG]
    python main.py train [--config CONFIG]
    python main.py evaluate [--config CONFIG] [--model MODEL_DIR]
    python main.py predict [--config CONFIG] [--model MODEL_DIR] --input FILE
    python main.py app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.utils import setup_logging, set_random_seed

logger = logging.getLogger(__name__)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate dataset metadata and recordings."""
    from src.data_loader import EEGDataset

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])

    try:
        dataset = EEGDataset(config)
        dataset.load_metadata()
        summary = dataset.get_dataset_summary()

        print("\n" + "=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)
        print(f"Subjects: {summary['n_subjects']}")
        print(f"Groups: {summary['group_counts']}")
        print(f"Recordings available: {summary['recording_available']}")
        print(f"Age range: {summary['age_range']}")
        print(f"MMSE range: {summary['mmse_range']}")
        print(f"Duration range (min): {summary['duration_range_min']}")
        print("=" * 60)

        if dataset.validation_report:
            print(dataset.validation_report.summary())

        return 0
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Preprocess EEG recordings."""
    from src.data_loader import EEGDataset
    from src.preprocessing import EEGPreprocessor

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])

    try:
        dataset = EEGDataset(config)
        dataset.load_metadata()

        available = dataset.metadata[dataset.metadata["recording_exists"]]
        logger.info("Preprocessing %d recordings...", len(available))

        preprocessor = EEGPreprocessor(config["preprocessing"])
        results = {}
        for _, row in available.iterrows():
            sid = row["subject_id"]
            try:
                raw = dataset.load_eeg(sid)
                result = preprocessor.preprocess(raw)
                results[sid] = result
                logger.info(
                    "%s: %d channels, %.1fs, steps=%s",
                    sid,
                    result.n_channels_final,
                    result.duration_final,
                    result.steps_applied,
                )
            except Exception as exc:
                logger.warning("Failed to preprocess %s: %s", sid, exc)

        print(f"\nPreprocessed {len(results)}/{len(available)} recordings")
        return 0
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_features(args: argparse.Namespace) -> int:
    """Extract features from EEG recordings."""
    from src.data_loader import EEGDataset
    from src.preprocessing import EEGPreprocessor
    from src.segmentation import EEGSegmenter
    from src.features import FeatureExtractor
    from src.dataset import assemble_feature_table, aggregate_to_subject_level

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])

    try:
        dataset = EEGDataset(config)
        dataset.load_metadata()

        available = dataset.metadata[dataset.metadata["recording_exists"]]
        logger.info("Extracting features from %d recordings...", len(available))

        preprocessor = EEGPreprocessor(config["preprocessing"])
        segmenter = EEGSegmenter(config["segmentation"])
        extractor = FeatureExtractor(config["features"])

        batch_features = {}
        for _, row in available.iterrows():
            sid = row["subject_id"]
            try:
                raw = dataset.load_eeg(sid)
                preproc = preprocessor.preprocess(raw)
                if preproc.excluded:
                    continue
                seg = segmenter.segment(preproc.raw_filtered)
                if seg.n_epochs_clean == 0:
                    continue
                features = extractor.extract_from_epochs(seg.epochs, sid)
                batch_features[sid] = features
            except Exception as exc:
                logger.warning("Feature extraction failed for %s: %s", sid, exc)

        # Assemble and aggregate
        feature_df = assemble_feature_table(batch_features, dataset.metadata)
        subject_df = aggregate_to_subject_level(feature_df, "mean")

        # Save
        output_dir = Path(config["paths"]["features_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        subject_df.to_csv(output_dir / "subject_features.csv", index=False)
        feature_df.to_csv(output_dir / "epoch_features.csv", index=False)

        print(f"\nFeatures extracted for {len(subject_df)} subjects")
        print(f"Saved to {output_dir}")
        return 0
    except Exception as exc:
        logger.error("Feature extraction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train and compare models."""
    from src.dataset import prepare_train_test_data
    from src.train import train_and_compare, select_best_model, save_model
    from src.evaluate import save_metrics

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])
    set_random_seed(config["models"]["random_seed"])

    try:
        import pandas as pd

        features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"
        if not features_path.exists():
            print(f"ERROR: Features file not found: {features_path}", file=sys.stderr)
            print("Run 'python main.py features' first.", file=sys.stderr)
            return 1

        subject_df = pd.read_csv(features_path)
        X_train, X_test, y_train, y_test, feature_names, groups = prepare_train_test_data(
            subject_df,
            test_size=config["models"]["test_size"],
            random_state=config["models"]["random_seed"],
        )

        model_configs = config["models"]["models"]
        results = train_and_compare(
            X_train, y_train, X_test, y_test,
            model_configs,
            groups=groups,
            cv_folds=config["models"]["cv_folds"],
        )

        best = select_best_model(results, config["training"]["primary_metric"])

        # Save best model
        label_mapping = {"A": "Alzheimer's Disease", "F": "Frontotemporal Dementia", "C": "Healthy Control"}
        model_dir = save_model(
            best,
            config["paths"]["models_dir"],
            feature_names,
            label_mapping,
            config,
        )

        # Save metrics
        results_dict = []
        for r in results:
            results_dict.append({
                "name": r.name,
                "test_metrics": r.test_metrics,
                "cv_scores": r.cv_scores,
            })
        save_metrics(
            {"best_model": best.name, "results": results_dict},
            config["paths"]["results_dir"],
        )

        print(f"\nBest model: {best.name}")
        print(f"Test metrics: {best.test_metrics}")
        print(f"Model saved to: {model_dir}")
        return 0
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a saved model."""
    from src.train import load_model
    from src.evaluate import compute_metrics, save_metrics

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])

    try:
        import pandas as pd
        import numpy as np

        model, metadata = load_model(args.model)

        features_path = Path(config["paths"]["features_dir"]) / "subject_features.csv"
        if not features_path.exists():
            print(f"ERROR: Features file not found: {features_path}", file=sys.stderr)
            return 1

        subject_df = pd.read_csv(features_path)
        feature_names = metadata.get("feature_names", [])

        if not feature_names:
            exclude_cols = {"subject_id", "group", "age", "gender", "mmse"}
            feature_names = [
                c for c in subject_df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(subject_df[c])
            ]

        X = subject_df[feature_names].values
        y = subject_df["group"].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y_pred = model.predict(X)
        metrics = compute_metrics(y, y_pred)
        save_metrics(metrics, config["paths"]["results_dir"])

        print(f"\nEvaluation results for {args.model}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 macro: {metrics['f1_macro']:.4f}")
        return 0
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict from an EEG file."""
    from src.predict import EEGPredictor

    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"])

    try:
        predictor = EEGPredictor(
            args.model,
            config["preprocessing"],
            config["segmentation"],
            config["features"],
        )
        result = predictor.predict_file(args.input)
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_app(args: argparse.Namespace) -> int:
    """Launch the Streamlit app."""
    import subprocess

    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print(f"ERROR: App file not found: {app_path}", file=sys.stderr)
        return 1

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EEG Alzheimer's Detection — Research & Educational Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", default=None, help="Path to config YAML file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # validate
    subparsers.add_parser("validate", help="Validate dataset metadata and recordings")

    # preprocess
    subparsers.add_parser("preprocess", help="Preprocess EEG recordings")

    # features
    subparsers.add_parser("features", help="Extract features from EEG recordings")

    # train
    subparsers.add_parser("train", help="Train and compare models")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument("--model", "-m", required=True, help="Path to saved model directory")

    # predict
    pred_parser = subparsers.add_parser("predict", help="Predict from an EEG file")
    pred_parser.add_argument("--model", "-m", required=True, help="Path to saved model directory")
    pred_parser.add_argument("--input", "-i", required=True, help="Path to EEG file")

    # app
    subparsers.add_parser("app", help="Launch the Streamlit app")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "validate": cmd_validate,
        "preprocess": cmd_preprocess,
        "features": cmd_features,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "app": cmd_app,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
