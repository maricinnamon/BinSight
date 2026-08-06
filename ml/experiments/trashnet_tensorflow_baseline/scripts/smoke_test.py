#!/usr/bin/env python
"""End-to-end smoke test: prove every moving part works before a real run.

Loads data, runs one batch, one training step, saves and reloads a checkpoint,
converts to TFLite and invokes it. Exits non-zero on any failure, so a broken
pipeline is caught in seconds rather than after an hour of training.
"""
from __future__ import annotations

import sys
import tempfile
import pathlib

import _bootstrap  # noqa: F401

import numpy as np

from binsight_training import data as data_module
from binsight_training import model as model_module
from binsight_training.config import CLASS_NAMES, load_config
from binsight_training.utils import get_logger, set_seeds

LOGGER = get_logger()


def main() -> int:
    config = load_config()
    set_seeds(config.data.seed)
    import tensorflow as tf

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        LOGGER.info("%-28s %s %s", name, "PASS" if ok else "FAIL", detail)

    try:
        train_frame = data_module.load_split(config, "train").groupby(
            "class_name", group_keys=False).head(6)
        record("load split", len(train_frame) > 0, f"{len(train_frame)} rows")

        dataset = data_module.make_dataset(
            train_frame, config, shuffle=True, augment=True, batch_size=4)
        images, labels = next(iter(dataset))
        ok_batch = (
            tuple(images.shape[1:]) == (config.data.image_size, config.data.image_size, 3)
            and images.dtype == tf.float32
        )
        record("one batch", ok_batch, f"{images.shape} {images.dtype}")

        in_range = float(tf.reduce_min(images)) >= -1.0 and float(tf.reduce_max(images)) <= 300.0
        record("pixel range 0..255", in_range,
               f"[{float(tf.reduce_min(images)):.1f}, {float(tf.reduce_max(images)):.1f}]")

        model, _ = model_module.build_model(config)
        outputs = model(images, training=False)
        record("forward pass", tuple(outputs.shape) == (images.shape[0], len(CLASS_NAMES)),
               str(outputs.shape))

        sums = np.sum(outputs.numpy(), axis=1)
        record("softmax sums to 1", bool(np.allclose(sums, 1.0, atol=1e-4)), f"{sums[:2]}")

        model_module.compile_model(model, 1e-3)
        history = model.fit(dataset, epochs=1, verbose=0)
        record("one training step", "loss" in history.history,
               f"loss={history.history['loss'][0]:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            weights = tmp_path / "smoke.weights.h5"
            model.save_weights(str(weights))
            before = model.predict(images, verbose=0)
            model.load_weights(str(weights))
            after = model.predict(images, verbose=0)
            record("checkpoint round-trip", bool(np.allclose(before, after, atol=1e-6)))

            saved = tmp_path / "saved_model"
            if hasattr(model, "export"):
                model.export(str(saved))
            else:
                tf.saved_model.save(model, str(saved))
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            blob = converter.convert()
            record("TFLite conversion", len(blob) > 0, f"{len(blob)/1e6:.2f} MB")

            interpreter = tf.lite.Interpreter(model_content=blob)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            interpreter.set_tensor(inp["index"],
                                   images.numpy()[:1].astype(inp["dtype"]))
            interpreter.invoke()
            values = interpreter.get_tensor(out["index"])
            record("TFLite invoke", values.shape[-1] == len(CLASS_NAMES), str(values.shape))
            record("TFLite input contract",
                   list(inp["shape"]) == [1, config.data.image_size, config.data.image_size, 3],
                   str(list(inp["shape"])))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Smoke test raised")
        record("unexpected exception", False, f"{type(exc).__name__}: {exc}")

    failed = [name for name, ok, _ in checks if not ok]
    print()
    if failed:
        LOGGER.error("SMOKE TEST FAILED: %s", failed)
        return 1
    LOGGER.info("SMOKE TEST PASSED (%d checks)", len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
