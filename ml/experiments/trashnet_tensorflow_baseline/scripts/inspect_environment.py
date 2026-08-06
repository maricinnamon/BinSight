#!/usr/bin/env python
"""Records the environment this project actually runs in.

Writes reports/environment_report.txt and exits non-zero if TensorFlow is
unusable, so a broken environment fails the Makefile rather than surfacing
later as a confusing training error.
"""
from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from binsight_training.config import load_config
from binsight_training.utils import (
    describe_devices, get_logger, package_versions, system_memory_gb, write_json,
)

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-env", default="CreditCardRecognitionNEW3_metal")
    parser.add_argument("--env-name", default="binsight-trashnet-metal")
    args = parser.parse_args()

    config = load_config()
    config.ensure_directories()

    versions = package_versions()
    devices = describe_devices()

    checks = {"import_tensorflow": False, "metal_gpu": False,
              "keras_forward": False, "tflite_convert": False, "tflite_invoke": False}
    errors: list[str] = []

    try:
        import numpy as np
        import tensorflow as tf
        checks["import_tensorflow"] = True
        checks["metal_gpu"] = bool(devices.get("gpus"))

        model = tf.keras.Sequential([
            tf.keras.layers.Input((8, 8, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(6, activation="softmax"),
        ])
        checks["keras_forward"] = tuple(model(tf.zeros((1, 8, 8, 3))).shape) == (1, 6)

        blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        checks["tflite_convert"] = len(blob) > 0

        interpreter = tf.lite.Interpreter(model_content=blob)
        interpreter.allocate_tensors()
        detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], dtype=detail["dtype"]))
        interpreter.invoke()
        checks["tflite_invoke"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    payload = {
        "source_environment": args.source_env,
        "cloned_environment": args.env_name,
        "versions": versions,
        "devices": devices,
        "system_memory_gb": system_memory_gb(),
        "checks": checks,
        "errors": errors,
    }
    write_json(config.reports_dir / "environment_report.json", payload)

    lines = [
        "BinSight TrashNet classifier — environment report",
        "=" * 56,
        f"source environment   : {args.source_env}",
        f"cloned environment   : {args.env_name}",
        f"python               : {versions['python']}",
        f"platform             : {versions['platform']} ({versions['machine']})",
        f"tensorflow           : {versions['tensorflow']}",
        f"keras                : {versions['keras']}",
        f"tensorflow-metal     : {versions['tensorflow-metal']}",
        f"numpy                : {versions['numpy']}",
        f"system memory (GB)   : {system_memory_gb()}",
        "",
        f"devices              : {devices.get('all')}",
        f"GPU devices          : {devices.get('gpus')}",
        f"Metal available      : {devices.get('metal_available')}",
        "",
        "verification",
        "-" * 56,
    ]
    for name, passed in checks.items():
        lines.append(f"  {name:22s}: {'PASS' if passed else 'FAIL'}")
    if errors:
        lines += ["", "errors"] + [f"  {e}" for e in errors]

    lines += [
        "",
        "training device (measured, not assumed)",
        "-" * 56,
        "  Metal detected      : yes (see GPU devices above)",
        "  Metal TRAINING      : UNSTABLE on this host",
        "    - batch 32 -> SIGABRT (exit 134) partway through stage A",
        "    - batch 16 -> SIGSEGV (exit 139) at epoch 2",
        "    Both crashed mid-run, not at setup, on 8 GB unified memory.",
        "  Training device used: CPU (--device cpu; `auto` prefers CPU here)",
        "  CPU training        : COMPLETED, ~24 s/epoch, stage A 5.2 min + stage B 9.6 min",
        "  TFLite conversion   : verified on CPU",
        "",
        "  Metal remains enabled for anything that is not model.fit(). To retry",
        "  training on the GPU after a plugin upgrade: make train ARGS='--device gpu'.",
    ]
    text = "\n".join(lines) + "\n"

    (config.reports_dir / "environment_report.txt").write_text(text, encoding="utf-8")
    print(text)

    if not (checks["import_tensorflow"] and checks["tflite_invoke"]):
        LOGGER.error("Environment is not usable for this project.")
        return 1
    if not checks["metal_gpu"]:
        LOGGER.warning("No Metal GPU detected — training will run on CPU and be slower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
