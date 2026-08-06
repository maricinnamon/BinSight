#!/usr/bin/env python
"""Evaluates the trained Keras model on the untouched test split.

Reports macro F1 alongside accuracy, because on a dataset where `trash` is a
quarter the size of `paper`, accuracy alone can look healthy while the smallest
class is barely predicted.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from binsight_training import data as data_module
from binsight_training import metrics as metrics_module
from binsight_training.config import CLASS_NAMES, load_config
from binsight_training.utils import get_logger, read_json, set_seeds, write_json

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()
    set_seeds(config.data.seed)

    import tensorflow as tf

    keras_path = config.models_dir / "keras" / "binsight_trashnet.keras"
    if not keras_path.exists():
        LOGGER.error("No trained model at %s — run scripts/train.py first.", keras_path)
        return 1

    model = tf.keras.models.load_model(str(keras_path), compile=False)  # inference only: restoring the legacy M1 Adam state fails and is not needed
    test_frame = data_module.load_split(config, "test")
    LOGGER.info("Evaluating on %d held-out test images", len(test_frame))

    dataset = data_module.make_dataset(test_frame, config, shuffle=False, augment=False)
    probabilities = model.predict(dataset, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    y_true = test_frame["label_index"].to_numpy()

    assert len(y_pred) == len(y_true), "prediction/label count mismatch"

    results = metrics_module.compute_metrics(y_true, y_pred)
    LOGGER.info("accuracy    : %.4f", results["accuracy"])
    LOGGER.info("macro F1    : %.4f", results["macro_f1"])
    LOGGER.info("weighted F1 : %.4f", results["weighted_f1"])
    for name, values in results["per_class"].items():
        LOGGER.info("  %-10s P=%.3f R=%.3f F1=%.3f n=%d",
                    name, values["precision"], values["recall"], values["f1"], values["support"])

    write_json(config.reports_dir / "test_metrics.json", results)
    text = metrics_module.classification_report_text(y_true, y_pred)
    (config.reports_dir / "classification_report.txt").write_text(text, encoding="utf-8")
    metrics_module.report_frame(results).to_csv(
        config.reports_dir / "classification_report.csv", index=False)
    print("\n" + text)

    # --- Per-prediction records ----------------------------------------
    confidence = probabilities.max(axis=1)
    predictions = pd.DataFrame({
        "path": test_frame["path"].tolist(),
        "filename": test_frame["filename"].tolist(),
        "true_index": y_true,
        "true_label": [CLASS_NAMES[i] for i in y_true],
        "predicted_index": y_pred,
        "predicted_label": [CLASS_NAMES[i] for i in y_pred],
        "confidence": confidence,
        "correct": y_true == y_pred,
    })
    for index, name in enumerate(CLASS_NAMES):
        predictions[f"p_{name}"] = probabilities[:, index]
    predictions.to_csv(config.predictions_dir / "test_predictions.csv", index=False)

    misclassified = predictions[~predictions["correct"]].sort_values("confidence", ascending=False)
    misclassified.to_csv(config.predictions_dir / "misclassified_examples.csv", index=False)
    LOGGER.info("%d misclassified of %d", len(misclassified), len(predictions))

    # --- Threshold behaviour -------------------------------------------
    threshold = config.inference.confidence_threshold
    above = predictions["confidence"] >= threshold
    coverage = float(above.mean())
    precision_above = float(predictions[above]["correct"].mean()) if above.any() else 0.0
    threshold_report = {
        "threshold": threshold,
        "coverage": coverage,
        "accuracy_above_threshold": precision_above,
        "accuracy_below_threshold": float(predictions[~above]["correct"].mean()) if (~above).any() else None,
        "note": (
            "Coverage is the fraction of test images the app would name a class for. "
            "The rest would show 'Not sure'."
        ),
    }
    write_json(config.reports_dir / "threshold_report.json", threshold_report)
    LOGGER.info("At threshold %.2f: coverage %.1f%%, accuracy on covered %.1f%%",
                threshold, coverage * 100, precision_above * 100)

    # --- Figures --------------------------------------------------------
    metrics_module.plot_confusion_matrix(
        results, config.figures_dir / "confusion_matrix.png", normalise=True)
    metrics_module.plot_confusion_matrix(
        results, config.figures_dir / "confusion_matrix_counts.png", normalise=False)
    metrics_module.plot_per_class_f1(results, config.figures_dir / "per_class_f1.png")

    summary_path = config.reports_dir / "training_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        history = summary.get("history", {})
        boundary = summary.get("stage_a_epoch_count")
        metrics_module.plot_history(history, "accuracy",
                                    config.figures_dir / "training_accuracy.png", boundary)
        metrics_module.plot_history(history, "loss",
                                    config.figures_dir / "training_loss.png", boundary)

    correct_high = predictions[predictions["correct"]].sort_values("confidence", ascending=False)
    wrong_high = misclassified
    low_confidence = predictions.sort_values("confidence").head(8)

    metrics_module.plot_example_grid(
        correct_high, config.figures_dir / "examples_correct_high_confidence.png",
        "Correct, high confidence")
    metrics_module.plot_example_grid(
        wrong_high, config.figures_dir / "examples_incorrect_high_confidence.png",
        "Incorrect, high confidence — the ones worth studying")
    metrics_module.plot_example_grid(
        low_confidence, config.figures_dir / "examples_low_confidence.png",
        "Lowest confidence")
    per_class_examples = predictions.groupby("true_label", group_keys=False).head(1)
    metrics_module.plot_example_grid(
        per_class_examples, config.figures_dir / "examples_per_class.png",
        "One example per class", columns=6)

    LOGGER.info("Reports and figures written under %s", config.reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
