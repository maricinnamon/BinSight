#!/usr/bin/env python
"""Hard validation of the converted YOLO Detect labels. Exits non-zero on failure.

Two kinds of check, and the second is the one that matters.

**Well-formedness** — five fields per line, class id in range, finite values,
positive extent, coordinates inside the frame, image/label pairing, and no mask
directories anywhere. A YOLO-seg polygon line has a variable number of trailing
floats, so it cannot pass the five-field rule; that is the mechanism by which
segmentation labels are made impossible rather than merely discouraged.

**Round trip** — every normalised box is converted back to COCO pixels and
compared against the original annotation. Well-formed numbers prove nothing
about correctness: a swapped axis or a wrong divisor produces perfectly valid
coordinates in the wrong place, which is exactly the defect that survived
programmatic validation in Run 001 and was only caught by eye.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from bbox_only import parse_label_line  # noqa: E402

DATA = HERE / "data"
MANIFESTS = HERE / "manifests"
REPORTS = HERE / "reports"

TOLERANCE = 1e-4          # normalised coordinate slack
ROUNDTRIP_TOLERANCE_PX = 1.5   # pixels; clipping legitimately moves an edge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0,
                        help="round-trip only N boxes (0 = all)")
    args = parser.parse_args()

    classes = list(yaml.safe_load(
        (HERE.parent / "taco_yolo26_detector" / "configs" /
         "class_mapping_run_002.yaml").read_text())["target_classes"])

    problems: list[dict] = []
    stats = collections.Counter()
    per_split: dict[str, dict] = {}
    field_counts = collections.Counter()

    # ------------------------------------------------------ well-formedness
    for split in ("train", "val", "test"):
        image_dir, label_dir = DATA / "images" / split, DATA / "labels" / split
        if not label_dir.exists():
            problems.append({"where": split, "problem": "label directory missing"})
            continue
        image_stems = {p.stem for p in image_dir.iterdir() if p.is_file() or p.is_symlink()}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}

        for orphan in sorted(label_stems - image_stems):
            problems.append({"where": f"{split}/{orphan}", "problem": "orphan label"})
        for missing in sorted(image_stems - label_stems):
            problems.append({"where": f"{split}/{missing}", "problem": "image without label"})
        for stem in sorted(image_stems):
            path = image_dir / f"{stem}.jpg"
            if path.is_symlink() and not path.exists():
                problems.append({"where": f"{split}/{stem}", "problem": "dangling symlink"})

        boxes = 0
        for label in sorted(label_dir.glob("*.txt")):
            text = label.read_text()
            if not text.strip():
                problems.append({"where": f"{split}/{label.name}", "problem": "empty label file"})
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                where = f"{split}/{label.name}:{n}"
                field_counts[len(line.split())] += 1
                try:
                    cid, xc, yc, w, h = parse_label_line(line)
                except ValueError as exc:
                    problems.append({"where": where, "problem": str(exc)})
                    continue
                if not 0 <= cid < len(classes):
                    problems.append({"where": where,
                                     "problem": f"class_id {cid} outside 0..{len(classes)-1}"})
                if not all(math.isfinite(v) for v in (xc, yc, w, h)):
                    problems.append({"where": where, "problem": "NaN or Inf coordinate"})
                    continue
                if w <= 0:
                    problems.append({"where": where, "problem": f"width {w} <= 0"})
                if h <= 0:
                    problems.append({"where": where, "problem": f"height {h} <= 0"})
                if not (-TOLERANCE <= xc <= 1 + TOLERANCE) or not (-TOLERANCE <= yc <= 1 + TOLERANCE):
                    problems.append({"where": where, "problem": "centre outside [0,1]"})
                if w > 1 + TOLERANCE or h > 1 + TOLERANCE:
                    problems.append({"where": where, "problem": "extent larger than frame"})
                if (xc - w / 2 < -TOLERANCE or xc + w / 2 > 1 + TOLERANCE
                        or yc - h / 2 < -TOLERANCE or yc + h / 2 > 1 + TOLERANCE):
                    problems.append({"where": where, "problem": "box leaves the frame"})
                boxes += 1
                stats["boxes"] += 1
        per_split[split] = {"images": len(image_stems), "labels": len(label_stems),
                            "boxes": boxes}

    # A YOLO-seg dataset would show field counts other than 5 here.
    if set(field_counts) - {5}:
        problems.append({"where": "labels",
                         "problem": f"non-5-field lines present: {dict(field_counts)}"})

    # ----------------------------------------------- no mask artefacts at all
    for pattern in ("*mask*", "*.png", "*seg*", "*polygon*"):
        for stray in DATA.rglob(pattern):
            problems.append({"where": str(stray.relative_to(HERE)),
                             "problem": f"unexpected artefact matching {pattern}"})

    # ------------------------------------------------------------ round trip
    rows = list(csv.DictReader(open(MANIFESTS / "bbox_annotations.csv")))
    checked = rows if args.sample <= 0 else rows[:args.sample]
    worst = 0.0
    roundtrip_failures = 0
    for row in checked:
        cx, cy, w, h = (float(v) for v in row["yolo_xc_yc_w_h"].split(","))
        iw, ih = int(row["image_width"]), int(row["image_height"])
        px_w, px_h = w * iw, h * ih
        px_x, px_y = cx * iw - px_w / 2, cy * ih - px_h / 2
        ox, oy, ow, oh = (float(v) for v in row["coco_bbox_xywh"].split(","))
        if row["clipped"] == "True":
            continue          # clipping legitimately moves an edge
        deltas = [abs(px_x - ox), abs(px_y - oy), abs(px_w - ow), abs(px_h - oh)]
        worst = max(worst, max(deltas))
        if max(deltas) > ROUNDTRIP_TOLERANCE_PX:
            roundtrip_failures += 1
            if roundtrip_failures <= 20:
                problems.append({
                    "where": f"annotation {row['annotation_id']}",
                    "problem": f"round trip off by {max(deltas):.3f}px "
                               f"(coco {ox:.1f},{oy:.1f},{ow:.1f},{oh:.1f})"})

    ok = not problems
    report = {
        "result": "PASS" if ok else "FAIL",
        "checks": [
            "exactly 5 fields per label line", "class_id within 0..7",
            "no NaN/Inf", "width > 0", "height > 0", "centre within [0,1]",
            "box inside frame", "image/label pairing", "no orphan labels",
            "no empty label files", "no dangling symlinks",
            "no polygon/segmentation-style lines", "no mask artefacts on disk",
            "COCO -> YOLO -> COCO round trip within tolerance",
        ],
        "per_split": per_split,
        "total_boxes": stats["boxes"],
        "field_count_histogram": {str(k): v for k, v in sorted(field_counts.items())},
        "roundtrip": {
            "checked": len(checked),
            "skipped_clipped": sum(1 for r in checked if r["clipped"] == "True"),
            "tolerance_px": ROUNDTRIP_TOLERANCE_PX,
            "max_error_px": round(worst, 6),
            "failures": roundtrip_failures,
        },
        "problem_count": len(problems),
        "problems": problems[:100],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dataset_validation.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [f"# Label validation — **{report['result']}**", "",
          f"{stats['boxes']} boxes across {sum(v['labels'] for v in per_split.values())} "
          f"label files, {len(problems)} problems.", "",
          "| Split | Images | Labels | Boxes |", "|---|---:|---:|---:|"]
    for s, v in per_split.items():
        md.append(f"| {s} | {v['images']} | {v['labels']} | {v['boxes']} |")
    md += ["", "## Field-count histogram", "",
           "A YOLO **Detect** label has exactly five values. A YOLO-seg polygon label has "
           "a variable number, so anything other than 5 below would mean segmentation "
           "data reached the dataset.", "",
           "| Values per line | Lines |", "|---:|---:|"]
    md += [f"| {k} | {v} |" for k, v in sorted(field_counts.items())]
    md += ["", "## Round trip: COCO → YOLO → COCO", "",
           f"Every normalised box was converted back to pixels and compared with the "
           f"original `annotation[\"bbox\"]`.", "",
           f"- checked: **{report['roundtrip']['checked']}**",
           f"- skipped (legitimately clipped to the frame): "
           f"{report['roundtrip']['skipped_clipped']}",
           f"- tolerance: {ROUNDTRIP_TOLERANCE_PX} px",
           f"- worst error: **{report['roundtrip']['max_error_px']} px**",
           f"- failures: **{report['roundtrip']['failures']}**", "",
           "## Checks applied", ""]
    md += [f"- {c}" for c in report["checks"]]
    if problems:
        md += ["", "## Problems", "", "| Where | Problem |", "|---|---|"]
        md += [f"| `{p['where']}` | {p['problem']} |" for p in problems[:40]]
    else:
        md += ["", "No problems found. Numbers being well-formed is necessary but not "
               "sufficient — see `bbox_visual_validation.jpg`, which renders these "
               "labels back onto the images.", ""]
    (REPORTS / "dataset_validation.md").write_text("\n".join(md) + "\n")

    print(f"{report['result']}: {stats['boxes']} boxes, {len(problems)} problems, "
          f"field counts {dict(field_counts)}, worst round trip "
          f"{report['roundtrip']['max_error_px']} px")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
