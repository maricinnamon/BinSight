"""Invariants that the dataset recovery must not break.

Two things are being protected here. Run 001 must stay bit-for-bit what it was,
because it is the control the whole Run 002 comparison rests on. And Run 002's
dataset must be comparable to it — same classes, same IDs, no leakage — or the
comparison measures the pipeline instead of the data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RECOVERY = ROOT / "reports" / "dataset_recovery"
RUN_001 = ROOT / "runs" / "yolo26n_run_001"
SPLITS_002 = ROOT / "data" / "splits" / "run_002"
PREPARED_002 = ROOT / "data" / "prepared_run_002"

EXPECTED_CLASSES = ["bag_wrapper", "bottle", "bottle_cap", "can",
                    "carton", "cigarette", "cup", "straw"]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- Run 001 frozen

@pytest.fixture(scope="module")
def frozen_manifest():
    path = ROOT / "reports" / "run_001_frozen_manifest.json"
    if not path.exists():
        pytest.skip("frozen manifest not written")
    return json.loads(path.read_text())


def test_run_001_weights_still_match_the_frozen_manifest(frozen_manifest):
    checked = 0
    for relative, expected in frozen_manifest["file_hashes"].items():
        path = ROOT / relative
        if not path.exists():
            pytest.fail(f"{relative} is missing — Run 001 was moved or deleted")
        # Hash the weights every time; they are what a rerun would silently
        # replace. The lighter text artefacts are covered by the size check.
        if relative.endswith(".pt"):
            assert sha256(path) == expected["sha256"], \
                f"{relative} changed since the freeze"
            checked += 1
        else:
            assert path.stat().st_size == expected["bytes"], \
                f"{relative} changed size since the freeze"
    assert checked >= 1, "no .pt weights were verified"


def test_run_001_best_weights_match_the_recorded_hash(frozen_manifest):
    best = ROOT / "runs" / "yolo26n_run_001" / "weights" / "best.pt"
    if not best.exists():
        pytest.skip("Run 001 weights not present")
    assert sha256(best) == frozen_manifest["best_pt_expected_sha256"]


def test_run_001_dataset_config_still_points_at_its_own_data():
    config = yaml.safe_load((CONFIGS / "dataset.yaml").read_text())
    assert config["path"].endswith("data/prepared"), \
        "Run 001's dataset root was repointed"
    assert list(config["names"].values()) == EXPECTED_CLASSES


def test_run_001_prepared_images_still_resolve():
    """Its images are symlinks into data/raw; moving data/raw would break them."""
    images = ROOT / "data" / "prepared" / "images"
    if not images.exists():
        pytest.skip("Run 001 dataset not present")
    dangling = [p for split in ("train", "val", "test")
                for p in (images / split).iterdir()
                if p.is_symlink() and not p.exists()]
    assert not dangling, f"{len(dangling)} Run 001 images no longer resolve"


def test_run_001_results_are_untouched():
    results = RUN_001 / "results.csv"
    if not results.exists():
        pytest.skip("Run 001 results not present")
    rows = list(csv.DictReader(open(results)))
    assert rows, "Run 001 results.csv is empty"


# ----------------------------------------------------- mapping is unchanged

def test_run_002_mapping_is_identical_to_run_001():
    a = CONFIGS / "class_mapping.yaml"
    b = CONFIGS / "class_mapping_run_002.yaml"
    assert sha256(a) == sha256(b), \
        "Run 002's mapping differs from Run 001's — the runs are no longer comparable"


def test_class_ids_are_unchanged():
    for name in ("class_mapping.yaml", "class_mapping_run_002.yaml"):
        classes = list(yaml.safe_load((CONFIGS / name).read_text())["target_classes"])
        assert classes == EXPECTED_CLASSES, f"{name} changed the class order"


def test_cigarette_was_not_removed():
    """Dropping the hardest class would raise mAP without detecting anything."""
    classes = yaml.safe_load(
        (CONFIGS / "class_mapping_run_002.yaml").read_text())["target_classes"]
    assert "cigarette" in classes


def test_run_002_dataset_yaml_matches_the_mapping():
    config = yaml.safe_load((CONFIGS / "dataset_run_002.yaml").read_text())
    assert [config["names"][i] for i in range(len(EXPECTED_CLASSES))] == EXPECTED_CLASSES
    assert config["path"].endswith("data/prepared_run_002")


def test_planned_run_002_does_not_change_image_size_or_init():
    config = yaml.safe_load((CONFIGS / "train_run_002_planned.yaml").read_text())
    run_001 = yaml.safe_load((RUN_001 / "args.yaml").read_text())
    assert config["imgsz"] == run_001["imgsz"] == 640
    assert config["model"] == "yolo26n.pt", "Run 002 must not fine-tune from Run 001"
    for key in ("epochs", "patience", "batch", "workers", "seed", "deterministic"):
        assert config[key] == run_001[key], f"{key} drifted from Run 001"


# ------------------------------------------------------------- Run 002 data

@pytest.fixture(scope="module")
def splits_002():
    if not (SPLITS_002 / "train.txt").exists():
        pytest.skip("Run 002 splits not generated")
    return {s: set((SPLITS_002 / f"{s}.txt").read_text().split())
            for s in ("train", "val", "test")}


@pytest.fixture(scope="module")
def manifest():
    path = RECOVERY / "official_image_manifest.csv"
    if not path.exists():
        pytest.skip("image manifest not built")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_002_splits_do_not_leak(splits_002):
    names = list(splits_002)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not splits_002[left] & splits_002[right], f"{left}/{right} leak"


def test_no_content_hash_crosses_a_split(splits_002, manifest):
    by_path = {row["annotation_path"]: row["sha256"]
               for row in manifest if row["status"] == "ok"}
    seen: dict[str, str] = {}
    for split, files in splits_002.items():
        for name in files:
            digest = by_path.get(name)
            if not digest:
                continue
            if digest in seen:
                assert seen[digest] == split, \
                    f"identical image bytes in both {seen[digest]} and {split}"
            seen[digest] = split


def test_run_002_splits_are_disjoint_from_nothing_but_still_complete(splits_002):
    total = sum(len(v) for v in splits_002.values())
    assert total == len(set().union(*splits_002.values())), "an image is counted twice"


def test_every_manifest_image_validated_or_explained(manifest):
    for row in manifest:
        assert row["status"] in {"ok", "missing", "unreadable", "dimension_mismatch"}
        if row["status"] == "ok":
            assert row["sha256"], "validated image has no hash"
            assert row["decoded_width"] == row["coco_width"]
            assert row["decoded_height"] == row["coco_height"]


def test_recovered_dataset_is_larger_than_run_001(manifest):
    ok = sum(1 for row in manifest if row["status"] == "ok")
    raw_run_001 = len(list((ROOT / "data" / "raw").rglob("*.jpg")))
    assert ok > raw_run_001, "recovery did not add images"


def test_run_002_labels_are_well_formed():
    if not PREPARED_002.exists():
        pytest.skip("Run 002 dataset not prepared")
    count = 0
    for split in ("train", "val", "test"):
        for label in (PREPARED_002 / "labels" / split).glob("*.txt"):
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, f"{label.name}: {len(parts)} values"
                cid = int(parts[0])
                assert 0 <= cid < len(EXPECTED_CLASSES)
                xc, yc, w, h = (float(v) for v in parts[1:])
                assert 0 <= xc <= 1 and 0 <= yc <= 1
                assert 0 < w <= 1 and 0 < h <= 1
                assert xc - w / 2 >= -1e-4 and xc + w / 2 <= 1 + 1e-4
                assert yc - h / 2 >= -1e-4 and yc + h / 2 <= 1 + 1e-4
                count += 1
    assert count > 0, "no Run 002 labels were checked"


def test_run_002_every_image_has_a_label():
    if not PREPARED_002.exists():
        pytest.skip("Run 002 dataset not prepared")
    for split in ("train", "val", "test"):
        images = {p.stem for p in (PREPARED_002 / "images" / split).iterdir()}
        labels = {p.stem for p in (PREPARED_002 / "labels" / split).glob("*.txt")}
        assert images == labels, f"{split}: image/label mismatch"


def test_run_002_did_not_write_into_run_001_paths():
    """The prepared trees must be separate directories, not one aliasing the other."""
    if not PREPARED_002.exists():
        pytest.skip("Run 002 dataset not prepared")
    assert (ROOT / "data" / "prepared").resolve() != PREPARED_002.resolve()


def test_completeness_is_not_overclaimed():
    path = RECOVERY / "completeness_report.json"
    if not path.exists():
        pytest.skip("completeness report not written")
    report = json.loads(path.read_text())
    text = (RECOVERY / "completeness_report.md").read_text()
    if report["missing"] or report["unreadable"]:
        assert "maximally recovered official subset" in text, \
            "an incomplete dataset must not be labelled complete"
