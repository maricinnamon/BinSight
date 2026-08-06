#!/usr/bin/env python
"""Builds the final training set: full scenes plus object-centric crops of small targets.

Run 002's diagnosis was unambiguous — 86.5% of false negatives were tiny or small
objects, and tiny recall was 0.095. Raising input resolution helps the pixels; it
does nothing about the fact that the *training signal* for those classes is a
handful of pixels per example. This script attacks the second half: for every
tiny/small ground-truth object in TRAIN, it emits a crop in which that object is
a large fraction of the frame while keeping enough surroundings that the model
still learns litter-in-context rather than litter-on-a-white-background.

Three rules protect the experiment:

**Crops come only from TRAIN.** Val and test are copied through untouched. A crop
inherits its source image's split, and since only TRAIN images are cropped, no
crop can reach val or test. The manifest records every source id so this is
auditable rather than merely asserted.

**Crops supplement, never replace.** Every original training scene stays in the
set. The caps below stop a single litter-strewn photograph from contributing
dozens of near-identical samples.

**Geometry is never stretched.** The crop window is a rectangle taken from the
original pixels and saved as-is. Boxes are clipped to it and re-normalised
against the crop's own dimensions; anything left with a negligible visible
fraction is dropped rather than kept as a sliver the model cannot learn from.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import shutil
import sys

from PIL import Image, ImageOps

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_PREPARED = ROOT / "data" / "prepared_run_002"
FINAL = ROOT / "data" / "prepared_final"
REPORTS = ROOT / "reports" / "final_dataset"
CONFIGS = ROOT / "configs"

SEED = 42

# Identical to Run 001 and Run 002. Relative area of the frame.
SIZE_BUCKETS = [("tiny", 0.0, 0.01), ("small", 0.01, 0.05),
                ("medium", 0.05, 0.20), ("large", 0.20, 1.01)]
TARGET_BUCKETS = {"tiny", "small"}

CONTEXT_FACTOR = 2.0      # initial window is 2.0x the box in each dimension...
MIN_CROP_PX = 192         # ...but never smaller than this, or context vanishes
MAX_CROP_FRACTION = 0.85  # ...and never so large that the crop is the whole image
KEEP_VISIBLE_FRACTION = 0.35   # a clipped box must retain this much of its area
MIN_BOX_PX = 4            # below this a box is not a learnable target

MAX_CROPS_PER_IMAGE_PER_CLASS = 2
MAX_CROPS_PER_IMAGE = 4


def bucket_for(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return "large"


def read_labels(path: pathlib.Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return rows


def crop_window(cx: float, cy: float, bw: float, bh: float,
                width: int, height: int) -> tuple[int, int, int, int]:
    """A square-ish window centred on the box, clamped inside the image.

    Square keeps the object's aspect ratio intact once the crop is letterboxed to
    768 by the dataloader; a window matching the box's own aspect would distort
    thin objects like straws after resize.
    """
    side = max(bw, bh) * CONTEXT_FACTOR
    side = max(side, MIN_CROP_PX)
    side = min(side, width * MAX_CROP_FRACTION, height * MAX_CROP_FRACTION)
    side = min(side, float(min(width, height)))

    half = side / 2
    x0, y0 = cx - half, cy - half
    # Shift rather than shrink when the window runs off an edge — shrinking would
    # silently reduce the context for objects near the frame border.
    x0 = min(max(x0, 0.0), width - side)
    y0 = min(max(y0, 0.0), height - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    classes = list(yaml.safe_load(
        (CONFIGS / "class_mapping_run_002.yaml").read_text())["target_classes"])
    rng = random.Random(SEED)
    REPORTS.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = FINAL / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    stats = {
        "originals": collections.Counter(),
        "crops": collections.Counter(),
        "class_before": collections.Counter(),
        "class_after": collections.Counter(),
        "bucket_before": collections.Counter(),
        "bucket_after": collections.Counter(),
        "skipped": collections.Counter(),
    }
    manifest_rows: list[dict] = []
    source_of: dict[str, str] = {}     # final filename -> source image stem
    split_of_source: dict[str, str] = {}

    # ------------------------------------------------- pass 1: copy originals
    for split in ("train", "val", "test"):
        for label in sorted((SOURCE_PREPARED / "labels" / split).glob("*.txt")):
            matches = list((SOURCE_PREPARED / "images" / split).glob(label.stem + ".*"))
            if not matches:
                stats["skipped"]["original_without_image"] += 1
                continue
            source_image = matches[0]
            destination = FINAL / "images" / split / f"{label.stem}.jpg"
            # Link to the fully resolved real file, not to prepared_run_002's own
            # symlink, so this dataset does not depend on that tree's internals.
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source_image.resolve())
            shutil.copyfile(label, FINAL / "labels" / split / label.name)
            stats["originals"][split] += 1
            source_of[f"{label.stem}.jpg"] = label.stem
            split_of_source[label.stem] = split

            # before/after both describe TRAIN only — val and test are untouched,
            # so including them would dilute the very effect being measured.
            if split == "train":
                for cid, _, _, w, h in read_labels(label):
                    stats["class_before"][classes[cid]] += 1
                    stats["bucket_before"][bucket_for(w * h)] += 1
                    stats["class_after"][classes[cid]] += 1
                    stats["bucket_after"][bucket_for(w * h)] += 1

    # -------------------------------------- pass 2: object-centric TRAIN crops
    train_labels = sorted((SOURCE_PREPARED / "labels" / "train").glob("*.txt"))
    for label in train_labels:
        matches = list((SOURCE_PREPARED / "images" / "train").glob(label.stem + ".*"))
        if not matches:
            continue
        source_image = matches[0]
        rows = read_labels(label)
        candidates = [(i, r) for i, r in enumerate(rows)
                      if bucket_for(r[3] * r[4]) in TARGET_BUCKETS]
        if not candidates:
            continue

        # Deterministic selection: shuffle with the seeded RNG, then apply caps.
        # Rarest class first so a photograph full of cigarettes still yields a
        # straw crop if it has one.
        rng.shuffle(candidates)
        class_totals = collections.Counter(classes[r[0]] for _, r in candidates)
        candidates.sort(key=lambda pair: class_totals[classes[pair[1][0]]])

        per_class = collections.Counter()
        made = 0
        with Image.open(source_image) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size

            for index, (cid, xc, yc, nw, nh) in candidates:
                if made >= MAX_CROPS_PER_IMAGE:
                    break
                name = classes[cid]
                if per_class[name] >= MAX_CROPS_PER_IMAGE_PER_CLASS:
                    continue

                cx, cy = xc * width, yc * height
                bw, bh = nw * width, nh * height
                if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                    stats["skipped"]["target_below_min_px"] += 1
                    continue

                x0, y0, x1, y1 = crop_window(cx, cy, bw, bh, width, height)
                cw, ch = x1 - x0, y1 - y0
                if cw < MIN_BOX_PX * 2 or ch < MIN_BOX_PX * 2:
                    stats["skipped"]["crop_window_too_small"] += 1
                    continue

                # Every mapped object that survives the window, not just the target.
                kept = []
                for row_index, (ocid, oxc, oyc, onw, onh) in enumerate(rows):
                    ax0 = (oxc - onw / 2) * width
                    ay0 = (oyc - onh / 2) * height
                    ax1 = (oxc + onw / 2) * width
                    ay1 = (oyc + onh / 2) * height
                    original_area = max((ax1 - ax0) * (ay1 - ay0), 1e-9)

                    ix0, iy0 = max(ax0, x0), max(ay0, y0)
                    ix1, iy1 = min(ax1, x1), min(ay1, y1)
                    iw, ih = ix1 - ix0, iy1 - iy0
                    if iw <= 0 or ih <= 0:
                        continue
                    if (iw * ih) / original_area < KEEP_VISIBLE_FRACTION:
                        continue          # a sliver: drop rather than mislabel
                    if iw < MIN_BOX_PX or ih < MIN_BOX_PX:
                        continue

                    kept.append((row_index, ocid,
                                 ((ix0 + ix1) / 2 - x0) / cw,
                                 ((iy0 + iy1) / 2 - y0) / ch,
                                 iw / cw, ih / ch))

                # Identity by row index, never by class: the target must be *this*
                # object, not another instance of the same class in the frame.
                if not any(k[0] == index for k in kept):
                    stats["skipped"]["target_lost_after_clip"] += 1
                    continue

                stem = f"crop_{label.stem}_{index:03d}_{name}"
                crop = im.crop((x0, y0, x1, y1))
                crop.save(FINAL / "images" / "train" / f"{stem}.jpg", "JPEG",
                          quality=args.jpeg_quality)
                (FINAL / "labels" / "train" / f"{stem}.txt").write_text(
                    "\n".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}"
                              for _, c, a, b, w, h in kept) + "\n")

                target = next(k for k in kept if k[0] == index)
                manifest_rows.append({
                    "crop_filename": f"{stem}.jpg",
                    "source_image_id": label.stem,
                    "source_image_path": str(source_image.resolve().relative_to(ROOT)),
                    "source_split": "train",
                    "target_class_id": cid,
                    "target_class": name,
                    "original_bbox_norm": f"{xc:.6f},{yc:.6f},{nw:.6f},{nh:.6f}",
                    "original_bbox_px": f"{cx - bw/2:.1f},{cy - bh/2:.1f},{bw:.1f},{bh:.1f}",
                    "crop_window_px": f"{x0},{y0},{x1},{y1}",
                    "crop_size_px": f"{cw}x{ch}",
                    "source_size_px": f"{width}x{height}",
                    "resulting_bbox_norm": f"{target[2]:.6f},{target[3]:.6f},"
                                           f"{target[4]:.6f},{target[5]:.6f}",
                    "original_rel_area": round(nw * nh, 8),
                    "new_rel_area": round(target[4] * target[5], 8),
                    "area_gain": round((target[4] * target[5]) / max(nw * nh, 1e-12), 1),
                    "original_bucket": bucket_for(nw * nh),
                    "new_bucket": bucket_for(target[4] * target[5]),
                    "boxes_in_crop": len(kept),
                })
                for _, c, _, _, w, h in kept:
                    stats["class_after"][classes[c]] += 1
                    stats["bucket_after"][bucket_for(w * h)] += 1
                stats["crops"][name] += 1
                source_of[f"{stem}.jpg"] = label.stem
                per_class[name] += 1
                made += 1

    # ----------------------------------------------------------- manifest
    with open(FINAL / "source_crop_manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    # A second copy under data/final_training/ as the brief names that path.
    alt = ROOT / "data" / "final_training"
    alt.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL / "source_crop_manifest.csv", alt / "source_crop_manifest.csv")

    summary = {
        "seed": SEED,
        "geometry": {
            "context_factor": CONTEXT_FACTOR, "min_crop_px": MIN_CROP_PX,
            "max_crop_fraction_of_image": MAX_CROP_FRACTION,
            "keep_visible_fraction": KEEP_VISIBLE_FRACTION, "min_box_px": MIN_BOX_PX,
            "window": "square, centred on the target, shifted (not shrunk) at edges",
        },
        "caps": {"per_image_per_class": MAX_CROPS_PER_IMAGE_PER_CLASS,
                 "per_image_total": MAX_CROPS_PER_IMAGE},
        "images": {
            "original_train": stats["originals"]["train"],
            "generated_crops": len(manifest_rows),
            "final_train": stats["originals"]["train"] + len(manifest_rows),
            "val": stats["originals"]["val"], "test": stats["originals"]["test"],
        },
        "crops_by_target_class": dict(stats["crops"]),
        "train_annotations_before": dict(stats["class_before"]),
        "train_annotations_after": dict(stats["class_after"]),
        "bucket_before": dict(stats["bucket_before"]),
        "bucket_after": dict(stats["bucket_after"]),
        "skipped": dict(stats["skipped"]),
        "median_area_gain": (sorted(r["area_gain"] for r in manifest_rows)[len(manifest_rows)//2]
                             if manifest_rows else None),
    }
    (REPORTS / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({k: summary[k] for k in
                      ("images", "crops_by_target_class", "skipped", "median_area_gain")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
