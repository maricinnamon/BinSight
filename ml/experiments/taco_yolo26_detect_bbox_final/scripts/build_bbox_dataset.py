#!/usr/bin/env python
"""Builds the final bbox-only YOLO Detect dataset from official TACO.

Reuses three things that were already verified in the archived experiment rather
than rebuilding them, because re-deriving them would change the numbers without
improving them:

* the recovered image index (which of TACO's 1500 official records resolve to a
  file on disk, and in which of the two source trees);
* the 8-class TACO → BinSight category mapping;
* the deterministic train/val/test split at source-image level.

What is new is the conversion itself: `annotation["bbox"]` only, five-value YOLO
Detect labels, and an explicit record of every annotation that was excluded and
why. Nothing is derived from segmentation polygons.

Source images are never modified. Images whose EXIF orientation disagrees with
the COCO record are written as rotation-baked copies — that defect put a quarter
of Run 001's boxes in empty background and is corrected here the same way.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import shutil
import sys

from PIL import Image, ImageOps

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from bbox_only import (  # noqa: E402
    SIZE_BUCKETS, assert_no_segmentation_keys, bucket_for, convert_bbox,
    load_annotations_bbox_only,
)

ARCHIVE = HERE.parent / "taco_yolo26_detector"
ANNOTATIONS = ARCHIVE / "data" / "source" / "TACO" / "data" / "annotations.json"
IMAGE_MANIFEST = ARCHIVE / "reports" / "dataset_recovery" / "official_image_manifest.csv"
MAPPING = ARCHIVE / "configs" / "class_mapping_run_002.yaml"
SPLITS = ARCHIVE / "data" / "splits" / "run_002"

DATA = HERE / "data"
MANIFESTS = HERE / "manifests"
REPORTS = HERE / "reports"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    for d in (MANIFESTS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            path = DATA / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- inputs
    config = yaml.safe_load(MAPPING.read_text())
    classes = list(config["target_classes"])
    class_index = {name: i for i, name in enumerate(classes)}
    category_to_target = {name: entry["target"]
                          for name, entry in config["mapping"].items()
                          if entry.get("include")}

    images, annotations, categories = load_annotations_bbox_only(ANNOTATIONS)
    assert_no_segmentation_keys(annotations)

    with open(IMAGE_MANIFEST, newline="") as handle:
        resolved = {int(r["image_id"]): ARCHIVE / r["resolved_path"]
                    for r in csv.DictReader(handle) if r["status"] == "ok"}

    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for line in (SPLITS / f"{split}.txt").read_text().split():
            split_of[line] = split

    # ------------------------------------------------------------ conversion
    per_image: dict[int, list] = collections.defaultdict(list)
    excluded: list[dict] = []
    exclusion_counts = collections.Counter()
    clipped = 0

    for ann in annotations:
        category = categories.get(ann.get("category_id"))
        target = category_to_target.get(category)
        if target is None:
            exclusion_counts["category_not_mapped"] += 1
            continue
        image = images.get(ann.get("image_id"))
        if image is None:
            exclusion_counts["image_record_missing"] += 1
            continue
        if ann["image_id"] not in resolved:
            exclusion_counts["image_file_unavailable"] += 1
            continue
        if image["file_name"] not in split_of:
            exclusion_counts["image_not_in_split"] += 1
            continue

        record, reason = convert_bbox(ann, image, class_index[target], target)
        if record is None:
            exclusion_counts[reason] += 1
            excluded.append({"annotation_id": ann.get("id"),
                             "image_id": ann.get("image_id"),
                             "category": category, "target": target,
                             "reason": reason, "raw_bbox": str(ann.get("bbox"))})
            continue
        if record.clipped:
            clipped += 1
        per_image[record.image_id].append(record)

    # -------------------------------------------------------------- write out
    counts = {s: {"images": 0, "boxes": 0} for s in ("train", "val", "test")}
    per_class = {s: collections.Counter() for s in ("train", "val", "test")}
    buckets = {s: collections.Counter() for s in ("train", "val", "test")}
    normalised_copies = collections.Counter()
    manifest_rows: list[dict] = []

    for image_id, records in sorted(per_image.items()):
        image = images[image_id]
        split = split_of[image["file_name"]]
        stem = image["file_name"].replace("/", "_")
        target_w, target_h = int(image["width"]), int(image["height"])
        source = resolved[image_id]
        destination = DATA / "images" / split / stem

        with Image.open(source) as probe:
            raw_size = probe.size
            orientation = probe.getexif().get(274)
        # TACO's annotations describe the EXIF-corrected orientation while the
        # JPEG stores raw sensor pixels plus a rotation tag. A symlink cannot
        # carry that rotation, so those images become baked copies.
        needs_rotation = raw_size != (target_w, target_h) or orientation not in (None, 1)
        if needs_rotation:
            with Image.open(source) as probe:
                fixed = ImageOps.exif_transpose(probe)
                if fixed.size != (target_w, target_h):
                    exclusion_counts["image_size_mismatch_after_exif"] += len(records)
                    continue
                fixed.convert("RGB").save(destination, "JPEG", quality=args.jpeg_quality)
            normalised_copies[split] += 1
        else:
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source.resolve())

        label = DATA / "labels" / split / f"{pathlib.Path(stem).stem}.txt"
        label.write_text("\n".join(r.to_label_line() for r in records) + "\n")

        counts[split]["images"] += 1
        counts[split]["boxes"] += len(records)
        for r in records:
            per_class[split][r.class_name] += 1
            buckets[split][r.size_bucket] += 1
            manifest_rows.append({
                "split": split, "image_id": image_id,
                "source_relative_path": image["file_name"],
                "prepared_name": stem, "annotation_id": r.annotation_id,
                "class_id": r.class_id, "class": r.class_name,
                "coco_bbox_xywh": ",".join(f"{v:.2f}" for v in r.coco_bbox),
                "yolo_xc_yc_w_h": f"{r.x_center:.6f},{r.y_center:.6f},"
                                  f"{r.width:.6f},{r.height:.6f}",
                "image_width": r.image_size[0], "image_height": r.image_size[1],
                "relative_area": round(r.relative_area, 8),
                "size_bucket": r.size_bucket, "clipped": r.clipped,
            })

    # -------------------------------------------------------------- manifests
    with open(MANIFESTS / "bbox_annotations.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    if excluded:
        with open(MANIFESTS / "excluded_annotations.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded[0].keys()))
            writer.writeheader()
            writer.writerows(excluded)
    for split in ("train", "val", "test"):
        shutil.copyfile(SPLITS / f"{split}.txt", MANIFESTS / f"{split}.txt")

    total_boxes = sum(c["boxes"] for c in counts.values())
    summary = {
        "source": {
            "annotations_json": str(ANNOTATIONS.relative_to(HERE.parent.parent.parent)),
            "official_image_records": len(images),
            "official_annotations": len(annotations),
            "images_resolved_on_disk": len(resolved),
            "split_manifests": "reused from the archived experiment, unchanged",
        },
        "contract": {
            "coco_field_used": "bbox",
            "coco_segmentation_used": False,
            "label_format": "class_id x_center y_center width height",
            "values_per_line": 5,
        },
        "classes": classes,
        "counts": counts,
        "total_images": sum(c["images"] for c in counts.values()),
        "total_boxes": total_boxes,
        "per_class": {s: dict(c) for s, c in per_class.items()},
        "per_class_total": dict(sum((per_class[s] for s in per_class), collections.Counter())),
        "size_buckets": {s: dict(b) for s, b in buckets.items()},
        "size_buckets_total": dict(sum((buckets[s] for s in buckets), collections.Counter())),
        "excluded": dict(exclusion_counts),
        "excluded_total": sum(exclusion_counts.values()),
        "excluded_mapped_only": len(excluded),
        "clipped_to_frame": clipped,
        "exif_normalised_copies": dict(normalised_copies),
    }
    (REPORTS / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({k: summary[k] for k in
                      ("counts", "total_images", "total_boxes", "per_class_total",
                       "size_buckets_total", "excluded", "clipped_to_frame")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
