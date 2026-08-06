"""Splits are image-level and leak-free."""
from __future__ import annotations
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


@pytest.fixture(scope="module")
def splits():
    if not (SPLITS / "train.txt").exists():
        pytest.skip("splits not generated")
    return {s: set((SPLITS / f"{s}.txt").read_text().split())
            for s in ("train", "val", "test")}


def test_no_image_appears_in_two_splits(splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not splits[left] & splits[right], f"{left}/{right} leak"


def test_splits_are_non_empty(splits):
    for name, files in splits.items():
        assert files, f"{name} is empty"


def test_prepared_images_match_the_manifests(splits):
    prepared = ROOT / "data" / "prepared" / "images"
    if not prepared.exists():
        pytest.skip("conversion not run")
    for split, files in splits.items():
        on_disk = {p.stem for p in (prepared / split).iterdir()}
        expected = {f.replace("/", "_").rsplit(".", 1)[0] for f in files}
        assert on_disk <= expected, f"{split} has images not in its manifest"
