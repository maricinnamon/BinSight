"""Dataset discovery, validation and split integrity."""
from __future__ import annotations

import pandas as pd
import pytest

from binsight_training.config import CLASS_NAMES, load_config
from binsight_training import data as data_module


def test_class_names_are_the_six_trashnet_classes():
    assert CLASS_NAMES == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert len(set(CLASS_NAMES)) == 6


def test_splits_exist_and_are_disjoint():
    config = load_config()
    frames = {}
    for name in ("train", "validation", "test"):
        path = config.splits_dir / f"{name}.csv"
        if not path.exists():
            pytest.skip("splits not generated yet; run scripts/prepare_dataset.py")
        frames[name] = pd.read_csv(path)

    names = list(frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            # By path...
            assert not set(frames[left]["path"]) & set(frames[right]["path"]), \
                f"{left}/{right} share files"
            # ...and by content hash, so duplicates cannot leak either.
            assert not set(frames[left]["sha256"]) & set(frames[right]["sha256"]), \
                f"{left}/{right} share duplicate content"


def test_every_split_contains_every_class():
    config = load_config()
    for name in ("train", "validation", "test"):
        path = config.splits_dir / f"{name}.csv"
        if not path.exists():
            pytest.skip("splits not generated yet")
        frame = pd.read_csv(path)
        assert set(frame["class_name"]) == set(CLASS_NAMES), f"{name} is missing classes"


def test_label_index_matches_class_order():
    config = load_config()
    path = config.splits_dir / "train.csv"
    if not path.exists():
        pytest.skip("splits not generated yet")
    frame = pd.read_csv(path)
    for _, row in frame.sample(min(50, len(frame)), random_state=0).iterrows():
        assert CLASS_NAMES[int(row["label_index"])] == row["class_name"]


def test_class_weights_favour_the_smallest_class():
    frame = pd.DataFrame({"label_index": [0] * 100 + [5] * 10})
    weights = data_module.class_weights(frame)
    # trash (index 5) is rarer, so it must carry more weight than cardboard.
    assert weights[5] > weights[0]
