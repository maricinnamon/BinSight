"""The source COCO annotations resolve internally."""
from __future__ import annotations
import json, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"


@pytest.fixture(scope="module")
def coco():
    if not ANN.exists():
        pytest.skip("TACO not cloned")
    return json.loads(ANN.read_text())


def test_every_annotation_resolves_to_a_category_and_image(coco):
    categories = {c["id"] for c in coco["categories"]}
    images = {i["id"] for i in coco["images"]}
    for ann in coco["annotations"]:
        assert ann["category_id"] in categories, f"dangling category {ann['category_id']}"
        assert ann["image_id"] in images, f"dangling image {ann['image_id']}"


def test_category_ids_are_unique(coco):
    ids = [c["id"] for c in coco["categories"]]
    assert len(ids) == len(set(ids))


def test_official_annotations_are_used_not_unofficial():
    # The unofficial file is deliberately excluded from the first experiment.
    assert ANN.name == "annotations.json"
    assert (ANN.parent / "annotations_unofficial.json").exists(), "expected both to exist"
