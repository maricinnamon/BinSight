#!/usr/bin/env python
"""Deterministic Run 002 train/val/test splits over the recovered dataset.

Same algorithm and same seed as Run 001's `create_splits.py` — image-level,
rarest-class-first, greedy on per-class deficit — so any difference between the
two runs comes from the data, not from a changed splitting rule.

Two things differ, both forced by recovery:

* the usable set comes from `reports/dataset_recovery/official_image_manifest.csv`
  rather than a directory listing, because images now live in two source trees;
* images are grouped by **content hash** before assignment, so byte-identical
  images cannot be split across a boundary even if they carry different ids.

Run 001's manifests in `data/splits/` are never touched; output goes to
`data/splits/run_002/`.
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
MANIFEST = ROOT / "reports" / "dataset_recovery" / "official_image_manifest.csv"
SPLITS = ROOT / "data" / "splits" / "run_002"
REPORTS = ROOT / "reports" / "dataset_recovery"
CONFIGS = ROOT / "configs"

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    config = yaml.safe_load((CONFIGS / "class_mapping_run_002.yaml").read_text())
    classes = list(config["target_classes"])
    class_index = {name: i for i, name in enumerate(classes)}
    mapping = {name: entry["target"] for name, entry in config["mapping"].items()
               if entry.get("include")}

    with open(MANIFEST, newline="") as handle:
        manifest = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    available = {int(row["image_id"]): row for row in manifest}
    if not available:
        print("ERROR: no validated images — run scripts/index_official_images.py first.")
        return 1

    data = json.loads(ANNOTATIONS.read_text())
    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    per_image: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for ann in data["annotations"]:
        target = mapping.get(categories.get(ann["category_id"]))
        if target is None or ann["image_id"] not in available:
            continue
        per_image[ann["image_id"]][class_index[target]] += 1

    usable = sorted(per_image)
    if not usable:
        print("ERROR: no image carries a mapped class.")
        return 1

    # Group by content hash. Distinct image records with identical bytes must
    # travel together or the same picture appears on both sides of the split.
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for image_id in usable:
        groups[available[image_id]["sha256"]].append(image_id)
    group_counts = {
        h: sum((per_image[i] for i in ids), collections.Counter())
        for h, ids in groups.items()
    }

    total_per_class = collections.Counter()
    for counts in per_image.values():
        total_per_class.update(counts)
    targets = {split: {cid: total_per_class[cid] * frac for cid in total_per_class}
               for split, frac in FRACTIONS.items()}

    rng = random.Random(SEED)
    order = sorted(groups)
    rng.shuffle(order)
    rarity = dict(total_per_class)
    order.sort(key=lambda h: min(rarity[c] for c in group_counts[h]))

    assigned: dict[str, str] = {}
    current = {split: collections.Counter() for split in FRACTIONS}
    split_groups = {split: 0 for split in FRACTIONS}
    target_groups = {split: len(order) * frac for split, frac in FRACTIONS.items()}

    for key in order:
        counts = group_counts[key]
        best, best_score = None, None
        for split in FRACTIONS:
            deficit = sum(
                max(targets[split][cid] - current[split][cid], 0) / max(targets[split][cid], 1e-9)
                for cid in counts
            )
            group_deficit = (max(target_groups[split] - split_groups[split], 0)
                             / max(target_groups[split], 1e-9))
            score = (deficit, group_deficit)
            if best_score is None or score > best_score:
                best, best_score = split, score
        assigned[key] = best
        current[best].update(counts)
        split_groups[best] += 1

    split_of_image = {image_id: assigned[available[image_id]["sha256"]] for image_id in usable}

    SPLITS.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {s: [] for s in FRACTIONS}
    for image_id, split in split_of_image.items():
        by_split[split].append(images[image_id]["file_name"])
    for split, files in by_split.items():
        (SPLITS / f"{split}.txt").write_text("\n".join(sorted(files)) + "\n")

    # --------------------------------------------------------------- leakage
    problems = []
    sets = {s: set(f) for s, f in by_split.items()}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if overlap := sets[left] & sets[right]:
                problems.append(f"{len(overlap)} images shared between {left} and {right}")
    if sum(len(v) for v in sets.values()) != len(usable):
        problems.append("split sizes do not sum to the usable image count")
    hash_splits = collections.defaultdict(set)
    for image_id, split in split_of_image.items():
        hash_splits[available[image_id]["sha256"]].add(split)
    crossing = [h for h, s in hash_splits.items() if len(s) > 1]
    if crossing:
        problems.append(f"{len(crossing)} content hashes appear in more than one split")
    if problems:
        for p in problems:
            print("ERROR:", p)
        return 1

    # ---------------------------------------------------------------- report
    split_images = {s: len(v) for s, v in by_split.items()}
    rows = []
    for cid, name in enumerate(classes):
        row = {"class_id": cid, "class": name, "total_annotations": total_per_class[cid]}
        for split in ("train", "val", "test"):
            row[f"{split}_annotations"] = current[split][cid]
            row[f"{split}_images"] = sum(1 for iid, s in split_of_image.items()
                                         if s == split and cid in per_image[iid])
        total = max(total_per_class[cid], 1)
        for split in ("train", "val", "test"):
            row[f"{split}_pct"] = round(100 * current[split][cid] / total, 1)
        rows.append(row)

    with open(REPORTS / "run_002_split_statistics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    duplicate_groups = sum(1 for ids in groups.values() if len(ids) > 1)
    lines = [
        "# Run 002 split statistics", "",
        f"Deterministic, seed **{SEED}**, target 70/15/15, split at **image** level — "
        "the same algorithm as Run 001, so differences between the runs come from the "
        "data rather than the rule.", "",
        "| Split | Images | Annotations |", "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {split_images[split]} | {sum(current[split].values())} |")
    lines += [
        f"| **total** | **{len(usable)}** | **{sum(total_per_class.values())}** |", "",
        "## Per class", "",
        "| ID | Class | Total | Train | Val | Test | Train % | Val % | Test % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_id']} | {row['class']} | {row['total_annotations']} "
            f"| {row['train_annotations']} | {row['val_annotations']} | {row['test_annotations']} "
            f"| {row['train_pct']}% | {row['val_pct']}% | {row['test_pct']}% |")
    worst = max(rows, key=lambda r: abs(r["train_pct"] - 70))
    lines += [
        "", "## Integrity", "",
        f"- Content-hash groups: **{len(groups)}** over {len(usable)} images "
        f"({duplicate_groups} groups hold more than one image record).",
        "- No image appears in two splits.",
        "- No content hash appears in two splits — byte-identical pictures travel "
        "together, so a duplicate cannot straddle the train/val boundary.",
        f"- Largest deviation from the 70% train target: **`{worst['class']}` at "
        f"{worst['train_pct']}%**.",
        "",
        "`create_splits_run_002.py` exits non-zero if any of the first three fail.", "",
    ]
    (REPORTS / "run_002_split_statistics.md").write_text("\n".join(lines) + "\n")

    print(f"train={split_images['train']} val={split_images['val']} test={split_images['test']} "
          f"(usable images {len(usable)}, hash groups {len(groups)})")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} annotations={sum(current[split].values())}")
    print("no leakage; manifests ->", SPLITS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
