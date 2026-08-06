"""Invariants the final dataset must hold, or every final metric is worthless.

The expensive failure here is silent: a crop derived from a validation or test
photograph appearing in train would inflate the final numbers without producing
any visible error. These tests exist so that failure cannot happen quietly.

They also pin the thing the whole run depends on — that val and test are still
byte-identical to Run 002's untouched originals.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "prepared_final"
SOURCE = ROOT / "data" / "prepared_run_002"
REPORTS = ROOT / "reports" / "final_dataset"
CONFIGS = ROOT / "configs"

EXPECTED_CLASSES = ["bag_wrapper", "bottle", "bottle_cap", "can",
                    "carton", "cigarette", "cup", "straw"]
SIZE_BUCKETS = [("tiny", 0.0, 0.01), ("small", 0.01, 0.05),
                ("medium", 0.05, 0.20), ("large", 0.20, 1.01)]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stems(split: str, root: pathlib.Path = FINAL) -> set[str]:
    return {p.stem for p in (root / "images" / split).iterdir()}


@pytest.fixture(scope="module")
def manifest():
    path = FINAL / "source_crop_manifest.csv"
    if not path.exists():
        pytest.skip("final dataset not built")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------ leakage

def test_no_crop_derives_from_val_or_test(manifest):
    forbidden = set()
    for split in ("val", "test"):
        forbidden |= {p.stem for p in (SOURCE / "labels" / split).glob("*.txt")}
    offenders = [r["crop_filename"] for r in manifest
                 if r["source_image_id"] in forbidden]
    assert not offenders, f"{len(offenders)} crops derive from held-out images"


def test_every_crop_source_is_a_train_image(manifest):
    train_sources = {p.stem for p in (SOURCE / "labels" / "train").glob("*.txt")}
    offenders = [r["crop_filename"] for r in manifest
                 if r["source_image_id"] not in train_sources]
    assert not offenders, f"{len(offenders)} crops have a non-train source"


def test_manifest_records_a_source_for_every_generated_crop(manifest):
    on_disk = {p.name for p in (FINAL / "images" / "train").glob("crop_*.jpg")}
    recorded = {r["crop_filename"] for r in manifest}
    assert on_disk == recorded, "generated crops and manifest rows disagree"
    assert all(r["source_image_id"] for r in manifest)


def test_no_generated_crop_in_val_or_test():
    for split in ("val", "test"):
        assert not [s for s in stems(split) if s.startswith("crop_")], \
            f"a generated crop reached {split}"


def test_no_image_appears_in_two_splits():
    names = ["train", "val", "test"]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not stems(left) & stems(right), f"{left}/{right} share images"


def test_no_identical_image_content_across_splits():
    digests = {}
    for split in ("train", "val", "test"):
        digests[split] = {sha256(p) for p in (FINAL / "images" / split).iterdir()
                          if p.exists()}
    names = list(digests)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not digests[left] & digests[right], \
                f"identical image bytes in {left} and {right}"


# ------------------------------------------------- val/test must be untouched

def test_val_and_test_are_exactly_run_002_originals():
    for split in ("val", "test"):
        assert stems(split) == stems(split, SOURCE), \
            f"{split} differs from the untouched Run 002 split"


def test_val_and_test_labels_are_byte_identical_to_run_002():
    for split in ("val", "test"):
        for label in (FINAL / "labels" / split).glob("*.txt"):
            original = SOURCE / "labels" / split / label.name
            assert original.exists(), f"{label.name} has no Run 002 counterpart"
            assert label.read_bytes() == original.read_bytes(), \
                f"{split}/{label.name} was modified"


def test_split_manifest_guard_hashes_still_match():
    guard = REPORTS / "val_test_split_guard.json"
    if not guard.exists():
        pytest.skip("guard not recorded")
    for relative, expected in json.loads(guard.read_text())["hashes"].items():
        assert sha256(ROOT / relative) == expected, \
            f"{relative} changed after the guard was recorded"


# --------------------------------------------------------------- correctness

def test_class_ids_and_order_are_unchanged():
    config = yaml.safe_load((CONFIGS / "dataset_final.yaml").read_text())
    assert [config["names"][i] for i in range(8)] == EXPECTED_CLASSES


def test_cigarette_and_straw_were_not_dropped():
    config = yaml.safe_load((CONFIGS / "dataset_final.yaml").read_text())
    names = set(config["names"].values())
    assert {"cigarette", "straw"} <= names


def test_all_labels_are_well_formed():
    count = 0
    for split in ("train", "val", "test"):
        for label in (FINAL / "labels" / split).glob("*.txt"):
            body = label.read_text().strip()
            assert body, f"{split}/{label.name} is empty"
            for line in body.splitlines():
                parts = line.split()
                assert len(parts) == 5
                cid = int(parts[0])
                xc, yc, w, h = (float(v) for v in parts[1:])
                assert 0 <= cid < 8
                assert w > 0 and h > 0, "zero-area box"
                assert -1e-4 <= xc <= 1 + 1e-4 and -1e-4 <= yc <= 1 + 1e-4
                assert xc - w / 2 >= -1e-4 and xc + w / 2 <= 1 + 1e-4
                assert yc - h / 2 >= -1e-4 and yc + h / 2 <= 1 + 1e-4
                count += 1
    assert count > 0


def test_every_image_has_a_label():
    for split in ("train", "val", "test"):
        images = stems(split)
        labels = {p.stem for p in (FINAL / "labels" / split).glob("*.txt")}
        assert images == labels, f"{split}: image/label mismatch"


def test_every_crop_enlarges_its_target(manifest):
    """The one thing a crop must do. Measured against the source, per object."""
    shrunk = [r["crop_filename"] for r in manifest
              if float(r["new_rel_area"]) <= float(r["original_rel_area"])]
    assert not shrunk, f"{len(shrunk)} crops did not enlarge their target"


def test_crops_target_only_tiny_or_small_objects(manifest):
    assert all(r["original_bucket"] in ("tiny", "small") for r in manifest)


def test_crop_generation_respects_the_caps(manifest):
    per_source = {}
    per_source_class = {}
    for row in manifest:
        per_source[row["source_image_id"]] = per_source.get(row["source_image_id"], 0) + 1
        key = (row["source_image_id"], row["target_class"])
        per_source_class[key] = per_source_class.get(key, 0) + 1
    assert max(per_source.values()) <= 4, "more than 4 crops from one source image"
    assert max(per_source_class.values()) <= 2, "more than 2 crops per class per image"


def test_all_original_training_scenes_are_still_present():
    """Crops supplement the scenes; they must not have replaced any of them."""
    originals = {s for s in stems("train") if not s.startswith("crop_")}
    assert originals == stems("train", SOURCE), \
        "the final train split lost or gained an original scene"


def test_previous_runs_are_untouched():
    expected = "304aefbd0014b6e4e436e16d4c0b5d84ada6f46bb4d15c5d94a4494fc1039a69"
    best_001 = ROOT / "runs" / "yolo26n_run_001" / "weights" / "best.pt"
    if best_001.exists():
        assert sha256(best_001) == expected, "Run 001 weights changed"
    frozen = ROOT / "reports" / "runs_001_002_frozen_manifest.json"
    if frozen.exists():
        for relative, entry in json.loads(frozen.read_text())["file_hashes"].items():
            path = ROOT / relative
            if entry.get("missing") or not path.exists():
                continue
            assert path.stat().st_size == entry["bytes"], f"{relative} changed size"
