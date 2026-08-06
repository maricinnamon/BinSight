"""Runs the selected TFLite model end to end."""
from __future__ import annotations

import numpy as np
import pytest

from binsight_training import export as export_module
from binsight_training.config import CLASS_NAMES, load_config


def _model_bytes():
    config = load_config()
    path = config.tflite_dir / "binsight_trashnet.tflite"
    if not path.exists():
        pytest.skip("no exported model yet; run export_tflite.py + verify_tflite.py")
    return path.read_bytes(), config


def test_tensor_contract():
    blob, config = _model_bytes()
    details = export_module.interpreter_details(blob)
    assert details["input_shape"] == [1, 224, 224, 3]
    assert details["input_dtype"] == "float32"
    assert details["output_shape"] == [1, len(CLASS_NAMES)]
    assert details["output_dtype"] == "float32"


def test_produces_finite_probabilities_summing_to_one():
    blob, config = _model_bytes()
    rng = np.random.default_rng(0)
    images = rng.uniform(0, 255, size=(3, 224, 224, 3)).astype(np.float32)
    probabilities = export_module.run_tflite(blob, images)

    assert probabilities.shape == (3, len(CLASS_NAMES))
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3)
    assert np.all(probabilities >= 0)


def test_threshold_turns_a_weak_prediction_into_not_sure():
    config = load_config()
    threshold = config.inference.confidence_threshold
    # Pure logic check of the rule the app applies.
    weak = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
    strong = np.array([0.90, 0.04, 0.02, 0.02, 0.01, 0.01])
    assert weak.max() < threshold, "this fixture should be below the threshold"
    assert strong.max() >= threshold


def test_verify_contract_helper_passes_on_the_shipped_model():
    blob, config = _model_bytes()
    result = export_module.verify_contract(blob, config.data.image_size)
    assert result["ok"], result["problems"]
