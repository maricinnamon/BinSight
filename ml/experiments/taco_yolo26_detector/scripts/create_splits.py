#!/usr/bin/env python
"""Deterministic image-level train/val/test splits, seed 42.

Splitting is at **image** level, never annotation level: an image's objects must
all land in the same split, or the same background leaks across the boundary and
the validation score becomes meaningless.

Images carry multiple classes, so exact stratification is impossible. This uses
iterative assignment — repeatedly place the image whose classes are furthest
behind their target — which keeps every class close to its 70/15/15 share
without ever splitting an image.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_mapping() -> tuple[list[str], dict[str, str]]:
    config = yaml.safe_load((CONFIGS / "class_mapping.yaml").read_text())
    classes = list(config["target_classes"])
    mapping = {name: entry["target"]
               for name, entry in config["mapping"].items()
               if entry.get("include")}
    return classes, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-images", action="store_true", default=True,
                        help="only include images present on disk")
    args = parser.parse_args()

    classes, mapping = load_mapping()
    class_index = {name: i for i, name in enumerate(classes)}
    data = json.loads(ANNOTATIONS.read_text())
    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    # Which images have at least one mapped annotation, and what classes?
    per_image: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for ann in data["annotations"]:
        name = categories.get(ann["category_id"])
        target = mapping.get(name)
        if target is None:
            continue
        image = images.get(ann["image_id"])
        if image is None:
            continue
        if args.require_images and not (RAW / image["file_name"]).exists():
            continue
        per_image[ann["image_id"]][class_index[target]] += 1

    usable = sorted(per_image)
    if not usable:
        print("ERROR: no usable images — download the dataset first.")
        return 1

    total_per_class = collections.Counter()
    for counts in per_image.values():
        total_per_class.update(counts)

    targets = {
        split: {cid: total_per_class[cid] * frac for cid in total_per_class}
        for split, frac in FRACTIONS.items()
    }

    # Deterministic order, then a seeded shuffle: stable regardless of filesystem.
    rng = random.Random(SEED)
    order = list(usable)
    rng.shuffle(order)
    # Rarest-class-first: images carrying scarce classes are placed while there
    # is still room to balance them.
    rarity = {cid: total_per_class[cid] for cid in total_per_class}
    order.sort(key=lambda iid: min(rarity[c] for c in per_image[iid]))

    assigned: dict[int, str] = {}
    current = {split: collections.Counter() for split in FRACTIONS}
    split_images = {split: 0 for split in FRACTIONS}
    target_images = {split: len(order) * frac for split, frac in FRACTIONS.items()}

    for image_id in order:
        counts = per_image[image_id]
        best, best_score = None, None
        for split in FRACTIONS:
            # Give the image to whichever split is furthest BEHIND its target for
            # the classes this image carries. Scoring the deficit *after* the
            # hypothetical add and minimising it sends everything to the smallest
            # split, which is what the first version did.
            deficit = sum(
                max(targets[split][cid] - current[split][cid], 0) / max(targets[split][cid], 1e-9)
                for cid, n in counts.items()
            )
            # Tie-break on overall image-count balance, same normalisation.
            image_deficit = (max(target_images[split] - split_images[split], 0)
                             / max(target_images[split], 1e-9))
            score = (deficit, image_deficit)
            if best_score is None or score > best_score:
                best, best_score = split, score
        assigned[image_id] = best
        current[best].update(counts)
        split_images[best] += 1

    SPLITS.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {s: [] for s in FRACTIONS}
    for image_id, split in assigned.items():
        by_split[split].append(images[image_id]["file_name"])
    for split, files in by_split.items():
        (SPLITS / f"{split}.txt").write_text("\n".join(sorted(files)) + "\n")

    # ------------------------------------------------------------- leakage
    sets = {s: set(f) for s, f in by_split.items()}
    problems = []
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                problems.append(f"{len(overlap)} images shared between {left} and {right}")
    if sum(len(v) for v in sets.values()) != len(usable):
        problems.append("split sizes do not sum to the usable image count")
    if problems:
        for p in problems:
            print("ERROR:", p)
        return 1

    # ------------------------------------------------------------- report
    rows = []
    for cid, name in enumerate(classes):
        row = {"class_id": cid, "class": name, "total_annotations": total_per_class[cid]}
        for split in ("train", "val", "test"):
            row[f"{split}_annotations"] = current[split][cid]
            row[f"{split}_images"] = sum(1 for iid, s in assigned.items()
                                         if s == split and cid in per_image[iid])
        total = max(total_per_class[cid], 1)
        for split in ("train", "val", "test"):
            row[f"{split}_pct"] = round(100 * current[split][cid] / total, 1)
        rows.append(row)

    with open(REPORTS / "split_statistics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Split statistics",
        "",
        f"Deterministic, seed **{SEED}**, split at **image** level so no image's",
        "annotations straddle a boundary. Target 70/15/15.",
        "",
        "| Split | Images | Annotations |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {split_images[split]} | {sum(current[split].values())} |")
    lines += [
        f"| **total** | **{len(usable)}** | **{sum(total_per_class.values())}** |",
        "",
        "## Per class",
        "",
        "| ID | Class | Total | Train | Val | Test | Train % | Val % | Test % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_id']} | {row['class']} | {row['total_annotations']} "
            f"| {row['train_annotations']} | {row['val_annotations']} | {row['test_annotations']} "
            f"| {row['train_pct']}% | {row['val_pct']}% | {row['test_pct']}% |"
        )
    worst = max(rows, key=lambda r: abs(r["train_pct"] - 70))
    lines += [
        "",
        "## Stratification quality",
        "",
        "Images carry multiple classes, so exact stratification is impossible without",
        "splitting an image — which would leak. The assignment is greedy over class",
        "deficits, rarest class first.",
        "",
        f"Largest deviation from the 70% train target: **`{worst['class']}` at "
        f"{worst['train_pct']}%**.",
        "",
        "Leakage check: no image appears in more than one split (verified by",
        "`create_splits.py`, which exits non-zero otherwise).",
        "",
    ]
    (REPORTS / "split_statistics.md").write_text("\n".join(lines) + "\n")

    print(f"train={split_images['train']} val={split_images['val']} test={split_images['test']} "
          f"(usable images {len(usable)})")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} annotations={sum(current[split].values())}")
    print("no leakage; manifests ->", SPLITS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
