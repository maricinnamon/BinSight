#!/usr/bin/env python
"""Verifies every exported TFLite model on the real test split and selects one.

Conversion succeeding is not evidence a model still works: a silently wrong
quantisation produces a valid file that predicts nonsense. Each candidate is
therefore run over the whole test split and compared against Keras.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys

import _bootstrap  # noqa: F401

import numpy as np

from binsight_training import data as data_module
from binsight_training import export as export_module
from binsight_training import metrics as metrics_module
from binsight_training.config import CLASS_NAMES, load_config
from binsight_training.utils import get_logger, read_json, set_seeds, sha256_file, write_json

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
    model = tf.keras.models.load_model(str(keras_path), compile=False)  # inference only: restoring the legacy M1 Adam state fails and is not needed

    test_frame = data_module.load_split(config, "test")
    dataset = data_module.make_dataset(test_frame, config, shuffle=False, augment=False)
    images = np.concatenate([x.numpy() for x, _ in dataset])
    y_true = test_frame["label_index"].to_numpy()
    assert len(images) == len(y_true), "image/label count mismatch"
    LOGGER.info("Test split: %d images", len(images))

    keras_probabilities = model.predict(images, batch_size=config.data.batch_size, verbose=0)
    keras_pred = keras_probabilities.argmax(axis=1)
    keras_metrics = metrics_module.compute_metrics(y_true, keras_pred)
    LOGGER.info("Keras   accuracy=%.4f macro_f1=%.4f",
                keras_metrics["accuracy"], keras_metrics["macro_f1"])

    candidates: dict[str, dict] = {}
    for name in ("float32", "float16", "int8"):
        path = config.tflite_dir / f"binsight_trashnet_{name}.tflite"
        if not path.exists():
            LOGGER.info("%-8s not present, skipping", name)
            continue

        blob = path.read_bytes()
        contract = export_module.verify_contract(blob, config.data.image_size)
        if not contract["ok"]:
            LOGGER.error("%-8s contract problems: %s", name, contract["problems"])

        probabilities = export_module.run_tflite(blob, images)
        predictions = probabilities.argmax(axis=1)
        model_metrics = metrics_module.compute_metrics(y_true, predictions)

        agreement = float(np.mean(predictions == keras_pred))
        max_diff = float(np.max(np.abs(probabilities - keras_probabilities)))
        mean_diff = float(np.mean(np.abs(probabilities - keras_probabilities)))

        candidates[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / 1e6, 3),
            "sha256": sha256_file(path),
            "verified": contract["ok"],
            "contract_problems": contract["problems"],
            "input_name": contract["input_name"],
            "input_shape": contract["input_shape"],
            "input_dtype": contract["input_dtype"],
            "output_name": contract["output_name"],
            "output_shape": contract["output_shape"],
            "output_dtype": contract["output_dtype"],
            "input_quantization": contract["input_quantization"],
            "output_quantization": contract["output_quantization"],
            "accuracy": model_metrics["accuracy"],
            "macro_f1": model_metrics["macro_f1"],
            "weighted_f1": model_metrics["weighted_f1"],
            "per_class": model_metrics["per_class"],
            "top1_agreement_vs_keras": agreement,
            "max_abs_prob_diff_vs_keras": max_diff,
            "mean_abs_prob_diff_vs_keras": mean_diff,
        }
        LOGGER.info(
            "%-8s acc=%.4f macroF1=%.4f agree=%.4f maxdiff=%.5f %6.2f MB %s/%s",
            name, model_metrics["accuracy"], model_metrics["macro_f1"], agreement,
            max_diff, path.stat().st_size / 1e6,
            contract["input_dtype"], contract["output_dtype"],
        )

    if "float32" not in candidates:
        LOGGER.error("float32 candidate missing — run scripts/export_tflite.py first.")
        return 1

    selected, ranking = export_module.select_candidate(
        candidates, config.export.max_quantisation_degradation_pp)
    LOGGER.info("Selected: %s", selected)
    for entry in ranking:
        status = ("REJECTED: " + "; ".join(entry["rejected_because"])) if entry["rejected_because"] else "eligible"
        LOGGER.info("  %-8s %6.2f MB acc%+.2fpp f1%+.2fpp float-io=%s  %s%s",
                    entry["name"], entry["size_bytes"] / 1e6,
                    -entry["accuracy_drop_pp"], -entry["macro_f1_drop_pp"],
                    entry["float_io"], status,
                    "  <-- SELECTED" if entry["name"] == selected else "")

    # The canonical artefact the iOS app consumes.
    final_path = config.tflite_dir / "binsight_trashnet.tflite"
    shutil.copyfile(candidates[selected]["path"], final_path)
    LOGGER.info("Canonical model -> %s (%.2f MB)", final_path, final_path.stat().st_size / 1e6)

    details = export_module.verify_contract(final_path.read_bytes(), config.data.image_size)
    summary = read_json(config.reports_dir / "training_summary.json")

    contract = export_module.write_model_contract(
        config.root / "model_contract.json",
        config=config,
        model_path=final_path,
        selected=selected,
        details=details,
        keras_metrics=keras_metrics,
        tflite_metrics=candidates[selected],
        training_date=summary.get("training_date", datetime.datetime.now().astimezone().isoformat()),
        extra={
            "selection": {
                "selected": selected,
                "source_file": candidates[selected]["path"],
                "source_sha256": candidates[selected]["sha256"],
                "ranking": ranking,
                "rule": (
                    "reject on failed contract or >"
                    f"{config.export.max_quantisation_degradation_pp}pp drop in accuracy OR macro F1 "
                    "vs float32; then prefer float I/O; then smallest file"
                ),
            },
            "numerical_validation": {
                "top1_agreement_vs_keras": candidates[selected]["top1_agreement_vs_keras"],
                "max_abs_prob_diff_vs_keras": candidates[selected]["max_abs_prob_diff_vs_keras"],
                "mean_abs_prob_diff_vs_keras": candidates[selected]["mean_abs_prob_diff_vs_keras"],
            },
            "per_class_f1": {
                name: values["f1"] for name, values in candidates[selected]["per_class"].items()
            },
        },
    )

    write_json(config.reports_dir / "tflite_verification.json", {
        "keras": keras_metrics,
        "candidates": candidates,
        "selected": selected,
        "ranking": ranking,
    })
    LOGGER.info("model_contract.json written; sha256=%s", contract["model"]["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
