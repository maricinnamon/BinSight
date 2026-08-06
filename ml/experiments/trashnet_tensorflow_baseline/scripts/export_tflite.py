#!/usr/bin/env python
"""Exports float32, float16 and (optionally) int8 TFLite models.

Verification and selection live in verify_tflite.py — this script only produces
candidates and records how each conversion went.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import numpy as np

from binsight_training import data as data_module
from binsight_training import export as export_module
from binsight_training.config import CLASS_NAMES, load_config
from binsight_training.utils import get_logger, set_seeds, sha256_file, write_json

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()
    set_seeds(config.data.seed)

    import tensorflow as tf

    keras_path = config.models_dir / "keras" / "binsight_trashnet.keras"
    saved_model_dir = config.models_dir / "saved_model" / "binsight_trashnet"
    if not keras_path.exists():
        LOGGER.error("No trained model at %s — run scripts/train.py first.", keras_path)
        return 1

    model = tf.keras.models.load_model(str(keras_path), compile=False)  # inference only: restoring the legacy M1 Adam state fails and is not needed
    if not saved_model_dir.exists():
        LOGGER.info("Re-exporting SavedModel")
        if hasattr(model, "export"):
            model.export(str(saved_model_dir))
        else:
            tf.saved_model.save(model, str(saved_model_dir))

    # Labels must match the model's output width, always.
    labels_path = config.root / "labels.txt"
    labels_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    assert model.output_shape[-1] == len(CLASS_NAMES), (
        f"model emits {model.output_shape[-1]} classes but there are {len(CLASS_NAMES)} labels"
    )
    LOGGER.info("labels.txt written in output-index order: %s", CLASS_NAMES)

    results: dict[str, dict] = {}

    def record(name: str, blob: bytes | None, error: str | None = None) -> None:
        if blob is None:
            results[name] = {"ok": False, "error": error}
            LOGGER.error("%-8s conversion FAILED: %s", name, error)
            return
        path = config.tflite_dir / f"binsight_trashnet_{name}.tflite"
        path.write_bytes(blob)
        results[name] = {
            "ok": True,
            "path": str(path),
            "size_bytes": len(blob),
            "size_mb": round(len(blob) / 1e6, 3),
            "sha256": sha256_file(path),
        }
        LOGGER.info("%-8s %6.2f MB -> %s", name, len(blob) / 1e6, path.name)

    # --- float32 baseline ------------------------------------------------
    try:
        record("float32", export_module.convert_float32(saved_model_dir, model))
    except Exception as exc:  # noqa: BLE001
        record("float32", None, f"{type(exc).__name__}: {exc}")
        LOGGER.error(
            "If this mentions SELECT_TF_OPS or Flex ops, treat it as a BLOCKER: the "
            "plain iOS LiteRT runtime cannot execute such a model."
        )
        return 1

    # --- float16 ---------------------------------------------------------
    try:
        record("float16", export_module.convert_float16(saved_model_dir, model))
    except Exception as exc:  # noqa: BLE001
        record("float16", None, f"{type(exc).__name__}: {exc}")

    # --- int8 (optional) -------------------------------------------------
    if not args.skip_int8:
        try:
            train_frame = data_module.load_split(config, "train")
            sample = train_frame.sample(
                n=min(config.export.representative_samples, len(train_frame)),
                random_state=config.data.seed,
            )
            calibration = data_module.make_dataset(
                sample, config, shuffle=False, augment=False, batch_size=1)

            def representative():
                for images, _ in calibration:
                    yield [tf.cast(images, tf.float32)]

            record("int8", export_module.convert_int8(saved_model_dir, representative, model))
        except Exception as exc:  # noqa: BLE001
            record("int8", None, f"{type(exc).__name__}: {exc}")
            LOGGER.warning("int8 is an optional candidate; continuing without it.")

    write_json(config.reports_dir / "export_results.json", results)
    LOGGER.info("Candidates: %s", [k for k, v in results.items() if v.get("ok")])
    return 0 if results.get("float32", {}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
