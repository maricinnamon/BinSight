#!/usr/bin/env python
"""Converts the mapped COCO annotations into YOLO detection labels.

YOLO format, one row per object, normalised:

    class_id x_center y_center width height

Source images are never modified. Clipping to image bounds is deterministic and
every clipped box is logged with its original and corrected coordinates — a
silent repair would hide a real annotation problem.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import shutil
import sys

from PIL import Image, ImageOps

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
RAW = ROOT / "data" / "raw"
PREPARED = ROOT / "data" / "prepared"
SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

# Boxes may exceed image bounds by this many pixels before clipping is logged.
CLIP_TOLERANCE = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link", choices=["symlink", "copy"], default="symlink",
                        help="how images enter data/prepared/images")
    args = parser.parse_args()

    config = yaml.safe_load((CONFIGS / "class_mapping.yaml").read_text())
    classes = list(config["target_classes"])
    class_index = {name: i for i, name in enumerate(classes)}
    mapping = {name: entry["target"] for name, entry in config["mapping"].items()
               if entry.get("include")}

    data = json.loads(ANNOTATIONS.read_text())
    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = SPLITS / f"{split}.txt"
        if not path.exists():
            print(f"ERROR: {path} not found — run scripts/create_splits.py first.")
            return 1
        for line in path.read_text().split():
            split_of[line] = split

    by_image: dict[int, list[str]] = collections.defaultdict(list)
    clipped: list[dict] = []
    skipped = collections.Counter()
    written = collections.Counter()

    for ann in data["annotations"]:
        name = categories.get(ann["category_id"])
        target = mapping.get(name)
        if target is None:
            skipped["unmapped_category"] += 1
            continue

        image = images.get(ann["image_id"])
        if image is None:
            skipped["missing_image_record"] += 1
            continue
        if image["file_name"] not in split_of:
            skipped["image_not_in_split"] += 1
            continue

        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            skipped["malformed_bbox"] += 1
            continue
        x, y, w, h = (float(v) for v in bbox)
        iw, ih = float(image["width"]), float(image["height"])

        if not all(v == v and abs(v) != float("inf") for v in (x, y, w, h)):
            skipped["non_finite"] += 1
            continue
        if w <= 0 or h <= 0:
            skipped["non_positive_size"] += 1
            continue

        x0, y0, x1, y1 = x, y, x + w, y + h
        needs_clip = (x0 < -CLIP_TOLERANCE or y0 < -CLIP_TOLERANCE
                      or x1 > iw + CLIP_TOLERANCE or y1 > ih + CLIP_TOLERANCE)
        cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
        cx1, cy1 = min(x1, iw), min(y1, ih)
        if cx1 - cx0 <= 0 or cy1 - cy0 <= 0:
            skipped["empty_after_clip"] += 1
            continue
        if needs_clip:
            clipped.append({
                "annotation_id": ann["id"], "image_id": ann["image_id"],
                "category": name, "target": target,
                "original_bbox": f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}",
                "corrected_bbox": f"{cx0:.2f},{cy0:.2f},{cx1-cx0:.2f},{cy1-cy0:.2f}",
                "image_width": iw, "image_height": ih,
            })

        xc = ((cx0 + cx1) / 2) / iw
        yc = ((cy0 + cy1) / 2) / ih
        nw = (cx1 - cx0) / iw
        nh = (cy1 - cy0) / ih
        # Guard against float drift pushing a value fractionally out of range.
        xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)

        by_image[ann["image_id"]].append(
            f"{class_index[target]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        written[target] += 1

    # ------------------------------------------------------------- write out
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            directory = PREPARED / kind / split
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    linked = collections.Counter()
    normalised = collections.Counter()
    for image_id, rows in by_image.items():
        image = images[image_id]
        split = split_of[image["file_name"]]
        # TACO nests images in batch_N/; flatten so YOLO's image/label pairing
        # by stem works and names stay unique.
        stem = image["file_name"].replace("/", "_")
        # This image's own dimensions — not the loop variable from the
        # annotation pass above, which belongs to a different image.
        target_w, target_h = int(image["width"]), int(image["height"])
        source = RAW / image["file_name"]
        if not source.exists():
            skipped["image_file_missing"] += 1
            continue

        destination = PREPARED / "images" / split / stem
        if destination.exists() or destination.is_symlink():
            destination.unlink()

        # TACO's annotations use the EXIF-corrected orientation, but the JPEGs on
        # disk store raw sensor pixels plus an Orientation tag. For ~25% of the
        # dataset that swaps width and height, and every box for those images
        # lands in the wrong place. Neither PIL nor OpenCV applies EXIF by
        # default, so the fix is to bake the rotation into a normalised copy —
        # a symlink cannot carry it.
        with Image.open(source) as probe:
            raw_size = probe.size
            orientation = probe.getexif().get(274)
        needs_normalising = (raw_size != (target_w, target_h)) or (orientation not in (None, 1))

        if needs_normalising:
            with Image.open(source) as probe:
                fixed = ImageOps.exif_transpose(probe)
                if fixed.size != (target_w, target_h):
                    skipped["size_mismatch_after_exif"] += 1
                    continue
                fixed.convert("RGB").save(destination, "JPEG", quality=95)
            normalised[split] += 1
        elif args.link == "symlink":
            destination.symlink_to(source.resolve())
        else:
            shutil.copyfile(source, destination)
        linked[split] += 1

        label = PREPARED / "labels" / split / (pathlib.Path(stem).stem + ".txt")
        label.write_text("\n".join(rows) + "\n")

    # ------------------------------------------------------------- reports
    if clipped:
        with open(REPORTS / "clipped_annotations.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(clipped[0].keys()))
            writer.writeheader()
            writer.writerows(clipped)

    summary = {
        "image_strategy": args.link,
        "classes": classes,
        "annotations_written": sum(written.values()),
        "per_class": dict(written),
        "images_per_split": dict(linked),
        "images_exif_normalised": dict(normalised),
        "skipped": dict(skipped),
        "clipped": len(clipped),
        "clip_tolerance_px": CLIP_TOLERANCE,
        "note": ("Source images and annotations are never modified. Clipped boxes are "
                 "listed with original and corrected coordinates in "
                 "reports/clipped_annotations.csv. Images whose EXIF orientation "
                 "disagrees with annotations.json are written as rotation-baked "
                 "copies; the rest are symlinked."),
    }
    (REPORTS / "conversion_report.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"annotations written: {sum(written.values())}")
    print(f"images linked ({args.link}): {dict(linked)}")
    print(f"EXIF-normalised copies: {dict(normalised)}")
    print(f"skipped: {dict(skipped)}")
    print(f"clipped: {len(clipped)}")

    if sum(written.values()) == 0:
        print("ERROR: nothing was converted.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
