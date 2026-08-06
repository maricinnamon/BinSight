"""TFLite conversion, verification and candidate selection."""
from __future__ import annotations

import pathlib
import time
from typing import Any, Callable

import numpy as np

from .config import CLASS_NAMES, Config
from .utils import get_logger, sha256_file

LOGGER = get_logger()


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def _base_converter(saved_model_dir: pathlib.Path, model=None):
    """Converter pinned to builtin ops only.

    `SELECT_TF_OPS` is deliberately not enabled: a model needing Flex delegates
    cannot run on the plain iOS LiteRT runtime, so a failure here is a blocker
    to report rather than something to paper over with a bigger binary.
    """
    import tensorflow as tf
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("from_saved_model failed (%s); falling back to from_keras_model", exc)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter


def convert_float32(saved_model_dir: pathlib.Path, model=None) -> bytes:
    return _base_converter(saved_model_dir, model).convert()


def convert_float16(saved_model_dir: pathlib.Path, model=None) -> bytes:
    """Half the size, float input/output preserved — the iOS contract is unchanged."""
    import tensorflow as tf
    converter = _base_converter(saved_model_dir, model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(
    saved_model_dir: pathlib.Path,
    representative_data: Callable[[], Any],
    model=None,
) -> bytes:
    """Full int8 weights with a real calibration set, float I/O at the boundary.

    The boundary stays float on purpose: int8 I/O would push scale and
    zero-point arithmetic into Swift for a saving that does not matter at this
    model size.
    """
    import tensorflow as tf
    converter = _base_converter(saved_model_dir, model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def interpreter_details(blob: bytes) -> dict[str, Any]:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "input_name": inp["name"],
        "input_shape": [int(v) for v in inp["shape"]],
        "input_dtype": np.dtype(inp["dtype"]).name,
        "input_quantization": {
            "scale": float(inp["quantization"][0]),
            "zero_point": int(inp["quantization"][1]),
        },
        "output_name": out["name"],
        "output_shape": [int(v) for v in out["shape"]],
        "output_dtype": np.dtype(out["dtype"]).name,
        "output_quantization": {
            "scale": float(out["quantization"][0]),
            "zero_point": int(out["quantization"][1]),
        },
    }


def run_tflite(blob: bytes, images: np.ndarray) -> np.ndarray:
    """Runs a batch of [0,255] float32 images one at a time, returns probabilities."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    outputs = []
    for image in images:
        sample = np.expand_dims(image, 0)
        if inp["dtype"] in (np.int8, np.uint8):
            scale, zero_point = inp["quantization"]
            sample = np.round(sample / scale + zero_point).astype(inp["dtype"])
        else:
            sample = sample.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        values = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] in (np.int8, np.uint8):
            scale, zero_point = out["quantization"]
            values = (values - zero_point) * scale
        outputs.append(values)
    return np.stack(outputs)


def verify_contract(blob: bytes, image_size: int) -> dict[str, Any]:
    """Checks a converted model against the contract the iOS side relies on."""
    details = interpreter_details(blob)
    problems: list[str] = []

    expected_input = [1, image_size, image_size, 3]
    if details["input_shape"] != expected_input:
        problems.append(f"input shape {details['input_shape']} != {expected_input}")
    if details["output_shape"] != [1, len(CLASS_NAMES)]:
        problems.append(f"output shape {details['output_shape']} != [1, {len(CLASS_NAMES)}]")

    # Probabilities on a real-ish input.
    probe = np.random.default_rng(0).uniform(0, 255, size=(2, image_size, image_size, 3)).astype(np.float32)
    probabilities = run_tflite(blob, probe)
    if not np.all(np.isfinite(probabilities)):
        problems.append("non-finite values in output")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=0.02):
        problems.append(f"probabilities do not sum to 1 (got {sums.tolist()})")
    if probabilities.shape[1] != len(CLASS_NAMES):
        problems.append(f"{probabilities.shape[1]} outputs for {len(CLASS_NAMES)} labels")

    details["ok"] = not problems
    details["problems"] = problems
    return details


def benchmark(blob: bytes, image_size: int, runs: int = 100, warmup: int = 10) -> dict[str, Any]:
    """Relative development benchmark on this Mac. **Not** iPhone latency."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    rng = np.random.default_rng(0)
    sample = rng.uniform(0, 255, size=inp["shape"]).astype(np.float32)
    if inp["dtype"] in (np.int8, np.uint8):
        scale, zero_point = inp["quantization"]
        sample = np.round(sample / scale + zero_point).astype(inp["dtype"])
    else:
        sample = sample.astype(inp["dtype"])

    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(out["index"])
        timings.append((time.perf_counter() - start) * 1000.0)

    array = np.array(timings)
    return {
        "runs": runs,
        "mean_ms": round(float(array.mean()), 3),
        "median_ms": round(float(np.median(array)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "min_ms": round(float(array.min()), 3),
        "max_ms": round(float(array.max()), 3),
    }


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def select_candidate(
    candidates: dict[str, dict[str, Any]],
    max_degradation_pp: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Picks the model to ship. Not on size, and not on accuracy alone.

    Rules, in order:
      1. must have passed contract verification;
      2. must stay within `max_degradation_pp` of float32 on BOTH accuracy and
         macro F1 — a quantised model that only holds accuracy while dropping a
         minority class is exactly what macro F1 exists to catch;
      3. prefer a float input/output contract, so Swift needs no dequantisation;
      4. then take the smallest file.
    """
    baseline = candidates.get("float32")
    if baseline is None:
        raise ValueError("float32 candidate is required as the comparison baseline")

    ranked: list[dict[str, Any]] = []
    for name, info in candidates.items():
        reasons: list[str] = []
        if not info.get("verified", False):
            reasons.append("failed contract verification")

        acc_drop = (baseline["accuracy"] - info["accuracy"]) * 100.0
        f1_drop = (baseline["macro_f1"] - info["macro_f1"]) * 100.0
        if name != "float32":
            if acc_drop > max_degradation_pp:
                reasons.append(f"accuracy down {acc_drop:.2f}pp (> {max_degradation_pp})")
            if f1_drop > max_degradation_pp:
                reasons.append(f"macro F1 down {f1_drop:.2f}pp (> {max_degradation_pp})")

        float_io = info["input_dtype"].startswith("float") and info["output_dtype"].startswith("float")
        ranked.append({
            "name": name,
            "size_bytes": info["size_bytes"],
            "accuracy": info["accuracy"],
            "macro_f1": info["macro_f1"],
            "accuracy_drop_pp": round(acc_drop, 3),
            "macro_f1_drop_pp": round(f1_drop, 3),
            "float_io": float_io,
            "rejected_because": reasons,
        })

    eligible = [c for c in ranked if not c["rejected_because"]]
    if not eligible:
        raise ValueError(f"no candidate passed selection: {ranked}")

    eligible.sort(key=lambda c: (not c["float_io"], c["size_bytes"]))
    return eligible[0]["name"], ranked


def write_model_contract(
    path: pathlib.Path,
    *,
    config: Config,
    model_path: pathlib.Path,
    selected: str,
    details: dict[str, Any],
    keras_metrics: dict[str, Any],
    tflite_metrics: dict[str, Any],
    training_date: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "schema_version": 2,
        "verified_by_execution": True,
        "model": {
            "name": config.export.model_name,
            "version": config.export.model_version,
            "file": model_path.name,
            "backbone": config.model.backbone,
            "quantization": selected,
            "training_date": training_date,
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "labels": CLASS_NAMES,
        "input": {
            "name": details["input_name"],
            "shape": details["input_shape"],
            "dtype": details["input_dtype"],
            "color_order": "RGB",
            "layout": "NHWC",
            "value_range": [0.0, 255.0],
            "resize": "bilinear to 224x224",
            "crop": (
                "centre crop of the aspect-fill preview region, then the reticle "
                "square; see PreviewCropGeometry/FrameCropPlan in the iOS app"
            ),
            "preprocessing_note": (
                "MobileNetV3 preprocessing is baked into the graph. Pass raw "
                "0-255 RGB float32 pixels; do NOT normalise on the client."
            ),
            "quantization": details["input_quantization"],
        },
        "output": {
            "name": details["output_name"],
            "shape": details["output_shape"],
            "dtype": details["output_dtype"],
            "activation": "softmax",
            "is_logits": False,
            "quantization": details["output_quantization"],
            "labels_in_index_order": CLASS_NAMES,
        },
        "inference": {
            "recommended_confidence_threshold": config.inference.confidence_threshold,
            "below_threshold_behaviour": "show 'Not sure' rather than a class",
        },
        "metrics": {
            "keras_test_accuracy": keras_metrics["accuracy"],
            "keras_macro_f1": keras_metrics["macro_f1"],
            "tflite_test_accuracy": tflite_metrics["accuracy"],
            "tflite_macro_f1": tflite_metrics["macro_f1"],
        },
        "runtime": {"requires_select_tf_ops": False, "target_ops": "TFLITE_BUILTINS"},
    }
    if extra:
        contract.update(extra)
    from .utils import write_json
    write_json(path, contract)
    return contract
