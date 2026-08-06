#!/usr/bin/env python
"""Two-stage transfer training for the BinSight waste classifier.

Stage A trains the head on a frozen backbone; stage B fine-tunes the backbone
tail at a much lower learning rate. Stage B is kept only if it actually improves
validation loss — measured, not assumed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import _bootstrap  # noqa: F401

import numpy as np

from binsight_training import data as data_module
from binsight_training import model as model_module
from binsight_training.config import CLASS_NAMES, as_dict, load_config
from binsight_training.utils import get_logger, package_versions, set_seeds, write_json

LOGGER = get_logger()


def _merge_history(a: dict, b: dict | None) -> dict:
    merged = {k: list(v) for k, v in a.items()}
    if b:
        for key, values in b.items():
            merged.setdefault(key, [])
            merged[key].extend(values)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="auto prefers CPU: Metal is unstable on this host")
    parser.add_argument("--smoke", action="store_true",
                        help="one short step per stage, to prove the pipeline runs")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()
    set_seeds(config.data.seed)

    import tensorflow as tf
    from binsight_training.utils import configure_device

    device = configure_device(args.device)
    LOGGER.info("TensorFlow %s | device=%s | visible=%s",
                tf.__version__, device["chosen"], device["visible_devices"])
    if device["chosen"] == "cpu" and device["gpus_present"]:
        LOGGER.warning(
            "Metal GPU is present but disabled: tensorflow-metal 1.1.0 crashed "
            "mid-training on this host (SIGABRT/SIGSEGV). Run with --device gpu to retry."
        )

    train_frame = data_module.load_split(config, "train")
    val_frame = data_module.load_split(config, "validation")
    LOGGER.info("train=%d validation=%d", len(train_frame), len(val_frame))

    if args.smoke:
        train_frame = train_frame.groupby("class_name", group_keys=False).head(8)
        val_frame = val_frame.groupby("class_name", group_keys=False).head(4)
        LOGGER.info("SMOKE: reduced to train=%d validation=%d", len(train_frame), len(val_frame))

    weights = data_module.class_weights(train_frame) if config.train.use_class_weights else None
    if weights:
        LOGGER.info("Class weights (imbalance is real: trash is ~1/4 of paper):")
        for index, name in enumerate(CLASS_NAMES):
            LOGGER.info("  %d %-10s %.3f", index, name, weights[index])

    # --- Batch-size ladder: step down on OOM rather than dying -----------
    last_error: Exception | None = None
    for batch_size in config.train.batch_size_ladder:
        try:
            LOGGER.info("Building pipelines with batch_size=%d", batch_size)
            train_ds = data_module.make_dataset(
                train_frame, config, shuffle=True, augment=True, batch_size=batch_size)
            val_ds = data_module.make_dataset(
                val_frame, config, shuffle=False, augment=False, batch_size=batch_size)
            images, labels = next(iter(train_ds))
            LOGGER.info("batch %s %s | pixel range [%.1f, %.1f]",
                        images.shape, images.dtype,
                        float(tf.reduce_min(images)), float(tf.reduce_max(images)))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("batch_size=%d failed (%s); stepping down", batch_size, exc)
    else:
        LOGGER.error("Every batch size failed: %s", last_error)
        return 1

    model, backbone = model_module.build_model(config)
    LOGGER.info("Model built. Trainable parameters (stage A): %d",
                model_module.trainable_parameter_count(model))

    checkpoints = config.models_dir / "keras" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ A
    stage_a = config.train.stage_a
    model_module.compile_model(model, stage_a.learning_rate)
    epochs_a = 1 if args.smoke else stage_a.epochs

    LOGGER.info("=== Stage A: frozen backbone, %d epochs ===", epochs_a)
    started = time.time()
    history_a = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_a,
        class_weight=weights,
        callbacks=model_module.make_callbacks(
            config, checkpoints / "stage_a.weights.h5", stage_a.patience,
            config.reports_dir / "logs" / "stage_a.csv"),
        verbose=2,
    )
    seconds_a = time.time() - started
    best_a = float(min(history_a.history["val_loss"]))
    LOGGER.info("Stage A done in %.1f min | best val_loss %.4f", seconds_a / 60, best_a)

    # ------------------------------------------------------------------ B
    stage_b = config.train.stage_b
    epochs_b = 1 if args.smoke else stage_b.epochs
    history_b = None
    seconds_b = 0.0
    used_stage_b = False
    best_b = None

    stage_a_weights = checkpoints / "stage_a_final.weights.h5"
    model.save_weights(str(stage_a_weights))

    if epochs_b > 0:
        info = model_module.unfreeze_backbone_tail(backbone, config.train.fine_tune_fraction)
        LOGGER.info("=== Stage B: fine-tuning ===")
        LOGGER.info("  %s", json.dumps(info))
        model_module.compile_model(model, stage_b.learning_rate)
        LOGGER.info("  trainable parameters: %d", model_module.trainable_parameter_count(model))

        started = time.time()
        history_b = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs_b,
            class_weight=weights,
            callbacks=model_module.make_callbacks(
                config, checkpoints / "stage_b.weights.h5", stage_b.patience,
                config.reports_dir / "logs" / "stage_b.csv"),
            verbose=2,
        )
        seconds_b = time.time() - started
        best_b = float(min(history_b.history["val_loss"]))
        LOGGER.info("Stage B done in %.1f min | best val_loss %.4f", seconds_b / 60, best_b)

        if best_b < best_a:
            used_stage_b = True
            LOGGER.info("-> Stage B improved validation loss. Keeping fine-tuned weights.")
        else:
            model.load_weights(str(stage_a_weights))
            LOGGER.info("-> Stage B did NOT improve validation loss. Reverting to stage A.")

    # ------------------------------------------------------------------ save
    keras_path = config.models_dir / "keras" / "binsight_trashnet.keras"
    model.save(str(keras_path))
    LOGGER.info("Saved Keras model -> %s", keras_path)

    saved_model_dir = config.models_dir / "saved_model" / "binsight_trashnet"
    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)
    model.export(str(saved_model_dir)) if hasattr(model, "export") else tf.saved_model.save(model, str(saved_model_dir))
    LOGGER.info("Saved SavedModel -> %s", saved_model_dir)

    history = _merge_history(history_a.history, history_b.history if history_b else None)
    summary = {
        "smoke": args.smoke,
        "training_date": datetime.datetime.now().astimezone().isoformat(),
        "config": as_dict(config),
        "versions": package_versions(),
        "batch_size_used": batch_size,
        "device": device,
        "class_weights": weights,
        "stage_a": {
            "epochs_run": len(history_a.history["loss"]),
            "best_val_loss": best_a,
            "best_val_accuracy": float(max(history_a.history["val_accuracy"])),
            "minutes": round(seconds_a / 60, 2),
        },
        "stage_b": {
            "epochs_run": len(history_b.history["loss"]) if history_b else 0,
            "best_val_loss": best_b,
            "best_val_accuracy": float(max(history_b.history["val_accuracy"])) if history_b else None,
            "minutes": round(seconds_b / 60, 2),
            "kept": used_stage_b,
        },
        "selected_stage": "stage_b_finetuned" if used_stage_b else "stage_a_frozen",
        "stage_a_epoch_count": len(history_a.history["loss"]),
        "history": history,
    }
    write_json(config.reports_dir / "training_summary.json", summary)
    LOGGER.info("Training summary -> %s", config.reports_dir / "training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
