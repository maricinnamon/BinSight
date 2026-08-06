"""Invariants of the bbox-only Detect pipeline.

The expensive failure this guards against is silent: a polygon-derived target, a
sixth value on a label line, or a segmentation head slipping in would all train
without error and produce a model the iOS app cannot consume.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys

import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from bbox_only import (  # noqa: E402
    ALLOWED_ANNOTATION_KEYS, FORBIDDEN_ANNOTATION_KEYS, assert_no_segmentation_keys,
    convert_bbox, parse_label_line, yolo_to_pixels,
)

DATA = HERE / "data"
MANIFESTS = HERE / "manifests"
REPORTS = HERE / "reports"
EXPECTED_CLASSES = ["bag_wrapper", "bottle", "bottle_cap", "can",
                    "carton", "cigarette", "cup", "straw"]


# ------------------------------------------------------------ the contract

def test_loader_whitelist_excludes_every_mask_key():
    assert not (ALLOWED_ANNOTATION_KEYS & FORBIDDEN_ANNOTATION_KEYS)
    assert "segmentation" in FORBIDDEN_ANNOTATION_KEYS
    assert "segmentation" not in ALLOWED_ANNOTATION_KEYS
    assert "bbox" in ALLOWED_ANNOTATION_KEYS


def test_assert_no_segmentation_keys_actually_fires():
    with pytest.raises(AssertionError):
        assert_no_segmentation_keys([{"id": 1, "segmentation": [[0, 0, 1, 1]]}])
    assert_no_segmentation_keys([{"id": 1, "bbox": [0, 0, 1, 1]}])   # must not raise


def test_label_line_is_always_five_values():
    image = {"width": 100, "height": 200}
    ann = {"id": 1, "image_id": 1, "bbox": [10, 20, 30, 40]}
    record, reason = convert_bbox(ann, image, 3, "can")
    assert reason is None
    assert len(record.to_label_line().split()) == 5


def test_parse_rejects_polygon_style_lines():
    parse_label_line("3 0.5 0.5 0.1 0.2")                      # valid Detect
    for bad in ("3 0.5 0.5 0.1 0.2 0.3",                        # six values
                "3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2",            # YOLO-seg polygon
                "3 0.5 0.5 0.1"):                               # four values
        with pytest.raises(ValueError):
            parse_label_line(bad)


def test_conversion_round_trips():
    image = {"width": 640, "height": 480}
    ann = {"id": 1, "image_id": 1, "bbox": [100.0, 50.0, 200.0, 120.0]}
    record, _ = convert_bbox(ann, image, 0, "bag_wrapper")
    x, y, w, h = yolo_to_pixels(record)
    for got, want in zip((x, y, w, h), (100.0, 50.0, 200.0, 120.0)):
        assert abs(got - want) < 1e-6


@pytest.mark.parametrize("bbox,reason", [
    (None, "bbox_missing"),
    ([1, 2, 3], "bbox_malformed"),
    (["a", 0, 1, 1], "bbox_non_numeric"),
    ([0, 0, 0, 10], "bbox_non_positive_size"),
    ([0, 0, 10, -5], "bbox_non_positive_size"),
    ([float("nan"), 0, 1, 1], "bbox_non_finite"),
])
def test_invalid_bboxes_are_excluded_not_repaired(bbox, reason):
    ann = {"id": 1, "image_id": 1, "bbox": bbox,
           "segmentation": [[0, 0, 50, 0, 50, 50]]}   # present but must be ignored
    record, got = convert_bbox(ann, {"width": 100, "height": 100}, 0, "bag_wrapper")
    assert record is None
    assert got == reason


# ------------------------------------------------------------- the dataset

@pytest.fixture(scope="module")
def label_files():
    files = [p for s in ("train", "val", "test") for p in (DATA / "labels" / s).glob("*.txt")]
    if not files:
        pytest.skip("dataset not built")
    return files


def test_every_label_line_has_exactly_five_fields(label_files):
    counts = collections.Counter()
    for path in label_files:
        for line in path.read_text().splitlines():
            if line.strip():
                counts[len(line.split())] += 1
    assert set(counts) == {5}, f"non-Detect label rows present: {dict(counts)}"


def test_class_ids_and_geometry_are_valid(label_files):
    for path in label_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            cid, xc, yc, w, h = parse_label_line(line)
            assert 0 <= cid < 8
            assert w > 0 and h > 0
            assert -1e-4 <= xc <= 1 + 1e-4 and -1e-4 <= yc <= 1 + 1e-4
            assert xc - w / 2 >= -1e-4 and xc + w / 2 <= 1 + 1e-4
            assert yc - h / 2 >= -1e-4 and yc + h / 2 <= 1 + 1e-4


def test_no_mask_or_polygon_artefacts_exist():
    strays = [p for pattern in ("*mask*", "*.png", "*seg*", "*polygon*")
              for p in DATA.rglob(pattern)]
    assert not strays, f"segmentation-style artefacts present: {strays[:5]}"


def test_every_image_has_a_label_and_resolves():
    for split in ("train", "val", "test"):
        images = {p.stem for p in (DATA / "images" / split).iterdir()}
        labels = {p.stem for p in (DATA / "labels" / split).glob("*.txt")}
        assert images == labels, f"{split}: image/label mismatch"
        dangling = [p for p in (DATA / "images" / split).iterdir()
                    if p.is_symlink() and not p.exists()]
        assert not dangling, f"{split}: {len(dangling)} dangling symlinks"


def test_no_split_leakage():
    rows = list(csv.DictReader(open(MANIFESTS / "bbox_annotations.csv")))
    by_id, by_path = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        by_id[r["image_id"]].add(r["split"])
        by_path[r["source_relative_path"]].add(r["split"])
    assert not [k for k, v in by_id.items() if len(v) > 1]
    assert not [k for k, v in by_path.items() if len(v) > 1]


def test_class_order_is_unchanged():
    config = yaml.safe_load((HERE / "configs" / "dataset.yaml").read_text())
    assert [config["names"][i] for i in range(8)] == EXPECTED_CLASSES


def test_training_config_is_detect_at_640_from_pretrained():
    config = yaml.safe_load((HERE / "configs" / "train_final.yaml").read_text())
    assert config["task"] == "detect"
    assert config["imgsz"] == 640
    assert config["model"] == "yolo26n.pt", "must start from pretrained, not a prior run"
    assert "seg" not in str(config["model"]).lower()


def test_detection_only_audit_passed():
    path = REPORTS / "detection_only_audit.json"
    if not path.exists():
        pytest.skip("audit not run")
    report = json.loads(path.read_text())
    assert report["result"] == "PASS", report["failures"]
    assert report["task"] == "detect"
    assert report["head_is_detect"] and not report["head_is_segment"]
    assert report["all_labels_have_5_fields"]
    assert not report["mask_labels_generated"]


def test_historical_experiments_untouched():
    """Run 001's frozen weights must still hash to their recorded value."""
    import hashlib
    best = HERE.parent / "taco_yolo26_detector" / "runs" / "yolo26n_run_001" / "weights" / "best.pt"
    if not best.exists():
        pytest.skip("archived run not present")
    h = hashlib.sha256()
    with open(best, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == "304aefbd0014b6e4e436e16d4c0b5d84ada6f46bb4d15c5d94a4494fc1039a69"
