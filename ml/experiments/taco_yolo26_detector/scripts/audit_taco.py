#!/usr/bin/env python
"""Audits the official TACO COCO annotations.

Everything here is read from `annotations.json` — no category list is taken from
memory or from a tutorial. Produces the taxonomy tables, the bounding-box
validity report and the object-size analysis that the class mapping is built on.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
REPORTS = ROOT / "reports"
TAXONOMY = REPORTS / "taxonomy"
FIGURES = REPORTS / "figures"

# Relative bbox area buckets. Thresholds chosen for a phone-camera use case:
# below 1% of frame an object is a few dozen pixels at 640px input and is
# effectively invisible to a small detector; above 20% it fills the viewfinder.
SIZE_BUCKETS = [
    ("tiny", 0.0, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.20),
    ("large", 0.20, 1.01),
]

# Boxes may exceed image bounds by this fraction of a pixel before being called
# invalid; COCO coordinates are floats and rounding is normal.
BOUNDS_TOLERANCE = 1.0


def bucket_for(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(ANNOTATIONS))
    args = parser.parse_args()

    path = pathlib.Path(args.annotations)
    if not path.exists():
        print(f"ERROR: {path} not found — clone the TACO repository first.")
        return 1

    data = json.loads(path.read_text())
    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c for c in data["categories"]}
    annotations = data["annotations"]

    for directory in (TAXONOMY, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- bbox validation
    invalid: list[dict] = []
    per_category: dict[int, dict] = {
        cid: {"annotations": 0, "images": set(), "rel_areas": []} for cid in categories
    }
    per_image_counts: collections.Counter = collections.Counter()
    rel_areas: list[float] = []
    buckets: collections.Counter = collections.Counter()

    for ann in annotations:
        problems: list[str] = []
        image = images.get(ann["image_id"])
        category = categories.get(ann["category_id"])

        if image is None:
            problems.append("image_id not found")
        if category is None:
            problems.append("category_id not found")

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            problems.append("bbox missing or malformed")
        else:
            x, y, w, h = (float(v) for v in bbox)
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                problems.append("non-finite bbox value")
            else:
                if w <= 0:
                    problems.append(f"width <= 0 ({w})")
                if h <= 0:
                    problems.append(f"height <= 0 ({h})")
                if image is not None:
                    iw, ih = float(image["width"]), float(image["height"])
                    if x < -BOUNDS_TOLERANCE:
                        problems.append(f"x < 0 ({x})")
                    if y < -BOUNDS_TOLERANCE:
                        problems.append(f"y < 0 ({y})")
                    if x + w > iw + BOUNDS_TOLERANCE:
                        problems.append(f"x+w exceeds width ({x + w:.1f} > {iw})")
                    if y + h > ih + BOUNDS_TOLERANCE:
                        problems.append(f"y+h exceeds height ({y + h:.1f} > {ih})")

        if problems:
            invalid.append({
                "annotation_id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "category_name": category["name"] if category else "?",
                "bbox": ann.get("bbox"),
                "image_width": image["width"] if image else None,
                "image_height": image["height"] if image else None,
                "problems": "; ".join(problems),
            })
            continue

        iw, ih = float(image["width"]), float(image["height"])
        relative = (w * h) / (iw * ih)
        rel_areas.append(relative)
        buckets[bucket_for(relative)] += 1
        per_image_counts[ann["image_id"]] += 1
        entry = per_category[ann["category_id"]]
        entry["annotations"] += 1
        entry["images"].add(ann["image_id"])
        entry["rel_areas"].append(relative)

    valid_count = len(annotations) - len(invalid)

    # ---------------------------------------------------------------- taxonomy tables
    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    rows = []
    for cid, category in categories.items():
        entry = per_category[cid]
        areas = entry["rel_areas"]
        rows.append({
            "category_id": cid,
            "name": category["name"],
            "supercategory": category.get("supercategory", ""),
            "annotations": entry["annotations"],
            "unique_images": len(entry["images"]),
            "pct_of_annotations": round(100 * entry["annotations"] / max(valid_count, 1), 3),
            "median_rel_area": round(median(areas), 6),
            "min_rel_area": round(min(areas), 6) if areas else 0.0,
            "max_rel_area": round(max(areas), 6) if areas else 0.0,
        })
    rows.sort(key=lambda r: (-r["annotations"], r["name"]))

    with open(TAXONOMY / "taco_categories.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (TAXONOMY / "taco_categories.json").write_text(json.dumps(rows, indent=2) + "\n")

    # ---------------------------------------------------------------- dataset level
    widths = [img["width"] for img in data["images"]]
    heights = [img["height"] for img in data["images"]]
    aspects = [w / h for w, h in zip(widths, heights)]
    annotated_images = len(per_image_counts)
    per_image = list(per_image_counts.values())

    summary = {
        "source": "https://github.com/pedropro/TACO",
        "annotation_file": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
        "images": len(data["images"]),
        "annotations_total": len(annotations),
        "annotations_valid": valid_count,
        "annotations_invalid": len(invalid),
        "categories": len(categories),
        "supercategories": len({c.get("supercategory") for c in categories.values()}),
        "images_with_annotations": annotated_images,
        "images_without_annotations": len(data["images"]) - annotated_images,
        "annotations_per_image": {
            "min": min(per_image) if per_image else 0,
            "max": max(per_image) if per_image else 0,
            "mean": round(sum(per_image) / max(len(per_image), 1), 3),
            "median": median([float(v) for v in per_image]),
        },
        "image_dimensions": {
            "width": {"min": min(widths), "max": max(widths), "median": median([float(w) for w in widths])},
            "height": {"min": min(heights), "max": max(heights), "median": median([float(h) for h in heights])},
            "aspect_ratio": {"min": round(min(aspects), 4), "max": round(max(aspects), 4),
                             "median": round(median(aspects), 4)},
            "portrait_images": sum(1 for a in aspects if a < 1),
            "landscape_images": sum(1 for a in aspects if a > 1),
        },
        "relative_bbox_area": {
            "min": round(min(rel_areas), 8) if rel_areas else 0,
            "max": round(max(rel_areas), 6) if rel_areas else 0,
            "median": round(median(rel_areas), 6),
            "mean": round(sum(rel_areas) / max(len(rel_areas), 1), 6),
        },
        "size_buckets": {
            name: {
                "count": buckets[name],
                "pct": round(100 * buckets[name] / max(valid_count, 1), 2),
                "range": f"[{low:.0%}, {high:.0%})" if high <= 1 else f">= {low:.0%}",
            }
            for name, low, high in SIZE_BUCKETS
        },
        "bucket_thresholds_note": (
            "tiny <1% of frame area, small 1-5%, medium 5-20%, large >20%. "
            "Chosen for a phone-camera use case: below 1% an object is a few dozen "
            "pixels at a 640 px detector input."
        ),
    }
    (REPORTS / "dataset_audit.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---------------------------------------------------------------- validation report
    validation = {
        "annotations_checked": len(annotations),
        "valid": valid_count,
        "invalid": len(invalid),
        "bounds_tolerance_px": BOUNDS_TOLERANCE,
        "checks": [
            "width > 0", "height > 0", "finite coordinates",
            "bbox within image bounds (±1 px)", "category_id resolves",
            "image_id resolves",
        ],
        "problem_breakdown": dict(collections.Counter(
            p for row in invalid for p in row["problems"].split("; ")
        )),
        "policy": (
            "Source annotations are never modified. Invalid rows are recorded here "
            "and excluded from conversion; any deterministic clipping applied later "
            "is logged with original and corrected coordinates."
        ),
    }
    (REPORTS / "annotation_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    if invalid:
        with open(REPORTS / "invalid_annotations.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(invalid[0].keys()))
            writer.writeheader()
            writer.writerows(invalid)
    else:
        (REPORTS / "invalid_annotations.csv").write_text(
            "annotation_id,image_id,category_id,category_name,bbox,"
            "image_width,image_height,problems\n")

    # ---------------------------------------------------------------- markdown
    lines = [
        "# TACO taxonomy audit",
        "",
        f"Source: <https://github.com/pedropro/TACO> · `data/annotations.json` "
        "(official, reviewed — `annotations_unofficial.json` is deliberately not used).",
        "",
        "## Dataset",
        "",
        "| | |",
        "|---|---:|",
        f"| Images | {summary['images']} |",
        f"| Annotations | {summary['annotations_total']} |",
        f"| Valid annotations | {summary['annotations_valid']} |",
        f"| Invalid annotations | {summary['annotations_invalid']} |",
        f"| Categories | {summary['categories']} |",
        f"| Supercategories | {summary['supercategories']} |",
        f"| Images with ≥1 annotation | {summary['images_with_annotations']} |",
        f"| Images with no annotation | {summary['images_without_annotations']} |",
        f"| Annotations per image (mean / median / max) | "
        f"{summary['annotations_per_image']['mean']} / "
        f"{summary['annotations_per_image']['median']} / "
        f"{summary['annotations_per_image']['max']} |",
        "",
        "## Object size",
        "",
        "`relative_bbox_area = (bbox_w × bbox_h) / (image_w × image_h)`",
        "",
        "| Bucket | Range | Annotations | Share |",
        "|---|---|---:|---:|",
    ]
    for name, _, _ in SIZE_BUCKETS:
        info = summary["size_buckets"][name]
        lines.append(f"| {name} | {info['range']} | {info['count']} | {info['pct']}% |")
    lines += [
        "",
        f"Median object occupies **{summary['relative_bbox_area']['median']:.2%}** of the frame.",
        "",
        summary["bucket_thresholds_note"],
        "",
        "## Categories",
        "",
        f"All {len(rows)} original categories, sorted by annotation count.",
        "",
        "| # | ID | Category | Supercategory | Anns | Images | % | Median rel. area |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['category_id']} | {row['name']} | {row['supercategory']} "
            f"| {row['annotations']} | {row['unique_images']} "
            f"| {row['pct_of_annotations']}% | {row['median_rel_area']:.4%} |"
        )
    tail = [r for r in rows if r["annotations"] < 20]
    lines += [
        "",
        f"**{len(tail)} of {len(rows)} categories have fewer than 20 annotations.** "
        "That long tail is the central problem for the class mapping: most original "
        "categories cannot support a detector class on their own.",
        "",
    ]
    (TAXONOMY / "taco_categories.md").write_text("\n".join(lines))

    print(f"images={summary['images']} annotations={summary['annotations_total']} "
          f"valid={valid_count} invalid={len(invalid)} categories={len(categories)}")
    print("size buckets:", {k: v["pct"] for k, v in summary["size_buckets"].items()})
    print(f"reports -> {TAXONOMY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
