"""Converted YOLO labels are valid and agree with the source COCO boxes."""
from __future__ import annotations
import json, pathlib
import pytest, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"
TOL = 1e-3


@pytest.fixture(scope="module")
def classes():
    path = ROOT / "configs" / "classes.txt"
    if not path.exists():
        pytest.skip("mapping not generated")
    return path.read_text().split()


def _rows():
    for split in ("train", "val", "test"):
        directory = PREPARED / "labels" / split
        if not directory.exists():
            continue
        for label in directory.glob("*.txt"):
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts:
                    yield label, parts


def test_every_row_has_five_values_and_valid_ranges(classes):
    seen = 0
    for label, parts in _rows():
        seen += 1
        assert len(parts) == 5, f"{label.name}: {len(parts)} values"
        cid = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:])
        assert 0 <= cid < len(classes)
        assert -TOL <= xc <= 1 + TOL and -TOL <= yc <= 1 + TOL
        assert 0 < w <= 1 + TOL and 0 < h <= 1 + TOL
        # The reconstructed corners must stay inside the frame.
        assert xc - w / 2 >= -TOL and xc + w / 2 <= 1 + TOL
        assert yc - h / 2 >= -TOL and yc + h / 2 <= 1 + TOL
    if seen == 0:
        pytest.skip("conversion not run")


def test_boxes_round_trip_against_the_source_coco_boxes(classes):
    """Denormalising a YOLO row must reproduce the original COCO bbox."""
    ann_path = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
    if not ann_path.exists() or not (PREPARED / "labels").exists():
        pytest.skip("source or conversion missing")
    coco = json.loads(ann_path.read_text())
    images = {i["id"]: i for i in coco["images"]}
    by_stem = {images[i]["file_name"].replace("/", "_").rsplit(".", 1)[0]: images[i]
               for i in images}

    mapping = yaml.safe_load((ROOT / "configs" / "class_mapping.yaml").read_text())
    included = {n for n, e in mapping["mapping"].items() if e.get("include")}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    coco_boxes = {}
    for ann in coco["annotations"]:
        if categories.get(ann["category_id"]) not in included:
            continue
        image = images.get(ann["image_id"])
        if image is None:
            continue
        stem = image["file_name"].replace("/", "_").rsplit(".", 1)[0]
        coco_boxes.setdefault(stem, []).append(ann["bbox"])

    checked = 0
    for split in ("train", "val", "test"):
        directory = PREPARED / "labels" / split
        if not directory.exists():
            continue
        for label in list(directory.glob("*.txt"))[:60]:
            image = by_stem.get(label.stem)
            if image is None:
                continue
            iw, ih = image["width"], image["height"]
            source = coco_boxes.get(label.stem, [])
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = (float(v) for v in parts)
                px_w, px_h = w * iw, h * ih
                x0, y0 = xc * iw - px_w / 2, yc * ih - px_h / 2
                # Some COCO box in this image must match within a pixel or two.
                assert any(
                    abs(x0 - bx) < 2 and abs(y0 - by) < 2
                    and abs(px_w - bw) < 2 and abs(px_h - bh) < 2
                    for bx, by, bw, bh in source
                ), f"{label.name}: no source box matches {x0:.1f},{y0:.1f},{px_w:.1f},{px_h:.1f}"
                checked += 1
    if checked == 0:
        pytest.skip("nothing to check")
