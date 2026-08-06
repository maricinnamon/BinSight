"""The contract the iOS app depends on."""
from __future__ import annotations

import json

import pytest

from binsight_training.config import CLASS_NAMES, DISPLAY_NAMES, load_config


def _contract():
    config = load_config()
    path = config.root / "model_contract.json"
    if not path.exists():
        pytest.skip("model_contract.json not generated yet; run verify_tflite.py")
    return json.loads(path.read_text())


def test_labels_file_matches_class_order():
    config = load_config()
    labels = (config.root / "labels.txt").read_text().split()
    assert labels == CLASS_NAMES


def test_every_class_has_a_display_name():
    assert set(DISPLAY_NAMES) == set(CLASS_NAMES)
    # trash is the residual class and must never be presented as recyclable.
    assert DISPLAY_NAMES["trash"] == "General waste"


def test_contract_input_is_the_documented_shape_and_range():
    contract = _contract()
    assert contract["input"]["shape"] == [1, 224, 224, 3]
    assert contract["input"]["dtype"] == "float32"
    assert contract["input"]["color_order"] == "RGB"
    assert contract["input"]["layout"] == "NHWC"
    assert contract["input"]["value_range"] == [0.0, 255.0]


def test_contract_output_is_six_softmax_probabilities():
    contract = _contract()
    assert contract["output"]["shape"] == [1, 6]
    assert contract["output"]["dtype"] == "float32"
    assert contract["output"]["activation"] == "softmax"
    assert contract["output"]["is_logits"] is False
    assert contract["output"]["labels_in_index_order"] == CLASS_NAMES


def test_contract_does_not_require_flex_ops():
    contract = _contract()
    # A Flex-dependent model cannot run on the plain iOS LiteRT runtime.
    assert contract["runtime"]["requires_select_tf_ops"] is False


def test_contract_records_a_real_checksum_and_size():
    contract = _contract()
    assert len(contract["model"]["sha256"]) == 64
    assert contract["model"]["size_bytes"] > 0
