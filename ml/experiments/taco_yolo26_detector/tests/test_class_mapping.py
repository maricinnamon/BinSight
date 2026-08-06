"""Every source category has an explicit decision; class IDs are contiguous."""
from __future__ import annotations
import json, pathlib
import pytest, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mapping():
    path = ROOT / "configs" / "class_mapping.yaml"
    if not path.exists():
        pytest.skip("mapping not generated")
    return yaml.safe_load(path.read_text())


def test_every_taco_category_is_mentioned(mapping):
    ann = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
    if not ann.exists():
        pytest.skip("TACO not cloned")
    names = {c["name"] for c in json.loads(ann.read_text())["categories"]}
    assert set(mapping["mapping"]) == names, "a category is silently unmapped"


def test_no_entry_is_left_undecided(mapping):
    for name, entry in mapping["mapping"].items():
        assert "include" in entry, f"{name} has no include flag"
        if entry["include"]:
            assert entry["target"] in mapping["target_classes"], \
                f"{name} targets an unknown class {entry['target']}"
        else:
            assert entry["target"] is None
            assert entry.get("reason"), f"{name} is excluded without a reason"


def test_class_ids_are_contiguous_from_zero(mapping):
    classes = mapping["target_classes"]
    assert classes == list(dict.fromkeys(classes)), "duplicate class name"
    txt = (ROOT / "configs" / "classes.txt").read_text().split()
    assert txt == classes, "classes.txt disagrees with class_mapping.yaml"
    dataset = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())
    assert [dataset["names"][i] for i in range(len(classes))] == classes, \
        "dataset.yaml name order disagrees — every prediction would be mislabelled"
