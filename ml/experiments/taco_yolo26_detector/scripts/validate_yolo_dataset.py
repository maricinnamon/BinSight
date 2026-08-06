#!/usr/bin/env python
"""Validates the converted YOLO dataset. Exits non-zero on real corruption.

Programmatic validation only proves the numbers are well-formed. It cannot prove
the conversion put the box in the right place — `visualise_annotations.py` does
that by rendering the YOLO labels back onto the images.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

TOLERANCE = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults reproduce the Run 001 invocation exactly; Run 002 passes its own
    # tree and report prefix so neither run can overwrite the other's report.
    parser.add_argument("--prepared", default="data/prepared",
                        help="dataset root holding images/ and labels/")
    parser.add_argument("--report-prefix", default="yolo_validation",
                        help="report basename under reports/")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    prepared = ROOT / args.prepared
    reports = ROOT / args.report_dir

    classes = (CONFIGS / "classes.txt").read_text().split()
    problems: list[dict] = []
    stats = {"label_files": 0, "boxes": 0, "images": 0}
    per_class: collections.Counter = collections.Counter()
    per_split: dict[str, dict] = {}
    stems_by_split: dict[str, set] = {}

    for split in ("train", "val", "test"):
        image_dir = prepared / "images" / split
        label_dir = prepared / "labels" / split
        if not label_dir.exists():
            problems.append({"split": split, "file": "-", "problem": "label directory missing"})
            continue

        image_stems = {p.stem for p in image_dir.iterdir() if p.is_file() or p.is_symlink()}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        stems_by_split[split] = image_stems

        for stem in sorted(image_stems):
            path = image_dir / f"{stem}.jpg"
            if path.is_symlink() and not path.exists():
                problems.append({"split": split, "file": f"{stem}.jpg",
                                 "problem": "dangling symlink: source image is gone"})

        for orphan in sorted(label_stems - image_stems):
            problems.append({"split": split, "file": f"{orphan}.txt",
                             "problem": "orphan label: no matching image"})
        for missing in sorted(image_stems - label_stems):
            problems.append({"split": split, "file": missing,
                             "problem": "image has no label file"})

        split_boxes = 0
        for label in sorted(label_dir.glob("*.txt")):
            stats["label_files"] += 1
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                where = f"{label.name}:{line_no}"

                if len(parts) != 5:
                    problems.append({"split": split, "file": where,
                                     "problem": f"expected 5 values, got {len(parts)}"})
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    problems.append({"split": split, "file": where,
                                     "problem": "non-numeric value"})
                    continue

                if not all(math.isfinite(v) for v in (xc, yc, w, h)):
                    problems.append({"split": split, "file": where,
                                     "problem": "non-finite coordinate"})
                    continue
                if not 0 <= cid < len(classes):
                    problems.append({"split": split, "file": where,
                                     "problem": f"class id {cid} outside 0..{len(classes)-1}"})
                    continue
                for label_name, value in (("x_center", xc), ("y_center", yc)):
                    if not -TOLERANCE <= value <= 1 + TOLERANCE:
                        problems.append({"split": split, "file": where,
                                         "problem": f"{label_name}={value} outside [0,1]"})
                for label_name, value in (("width", w), ("height", h)):
                    if not 0 < value <= 1 + TOLERANCE:
                        problems.append({"split": split, "file": where,
                                         "problem": f"{label_name}={value} outside (0,1]"})
                # Reconstructed corners must sit inside the frame.
                if xc - w / 2 < -TOLERANCE or xc + w / 2 > 1 + TOLERANCE \
                        or yc - h / 2 < -TOLERANCE or yc + h / 2 > 1 + TOLERANCE:
                    problems.append({"split": split, "file": where,
                                     "problem": "reconstructed box leaves the image"})

                per_class[cid] += 1
                split_boxes += 1
                stats["boxes"] += 1

        per_split[split] = {"images": len(image_stems), "labels": len(label_stems),
                            "boxes": split_boxes}
        stats["images"] += len(image_stems)

    # An image reaching two splits is silent, catastrophic leakage: the score
    # would be measured on pictures the model trained on.
    names = list(stems_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for stem in sorted(stems_by_split[left] & stems_by_split[right]):
                problems.append({"split": f"{left}+{right}", "file": stem,
                                 "problem": "image present in two splits (leakage)"})

    report = {
        "dataset": args.prepared,
        "classes": classes,
        "checks": [
            "5 values per row", "class id within range", "finite coordinates",
            "x_center/y_center in [0,1]", "0 < width,height <= 1",
            "reconstructed box inside image", "no orphan labels",
            "no image without a label", "no dangling image symlink",
            "no image shared between splits",
        ],
        "totals": stats,
        "per_split": per_split,
        "per_class": {classes[i]: per_class[i] for i in range(len(classes))},
        "problems": problems[:500],
        "problem_count": len(problems),
        "ok": not problems,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.report_prefix}.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# YOLO label validation",
        "",
        f"Dataset: `{args.prepared}`",
        "",
        f"**Result: {'PASS' if report['ok'] else 'FAIL'}** — "
        f"{stats['boxes']} boxes across {stats['label_files']} label files, "
        f"{len(problems)} problems.",
        "",
        "| Split | Images | Label files | Boxes |",
        "|---|---:|---:|---:|",
    ]
    for split, info in per_split.items():
        lines.append(f"| {split} | {info['images']} | {info['labels']} | {info['boxes']} |")
    lines += [
        "",
        "## Boxes per class",
        "",
        "| ID | Class | Boxes |",
        "|---:|---|---:|",
    ]
    for index, name in enumerate(classes):
        lines.append(f"| {index} | {name} | {per_class[index]} |")
    lines += ["", "## Checks applied", ""]
    lines += [f"- {check}" for check in report["checks"]]
    if problems:
        lines += ["", "## Problems", "", "| Split | File | Problem |", "|---|---|---|"]
        lines += [f"| {p['split']} | `{p['file']}` | {p['problem']} |" for p in problems[:50]]
        if len(problems) > 50:
            lines.append(f"| … | | {len(problems) - 50} more in the JSON report |")
    else:
        lines += ["", "No problems found.", ""]
    lines += [
        "",
        "Programmatic validation proves the numbers are well-formed. It does not prove",
        "the boxes are in the right *place* — see `reports/samples/`, which renders",
        "these YOLO labels back onto the images.",
        "",
    ]
    (reports / f"{args.report_prefix}.md").write_text("\n".join(lines))

    print(f"{'PASS' if report['ok'] else 'FAIL'}: {stats['boxes']} boxes, "
          f"{stats['label_files']} label files, {len(problems)} problems")
    for split, info in per_split.items():
        print(f"  {split:<6} images={info['images']} labels={info['labels']} boxes={info['boxes']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
