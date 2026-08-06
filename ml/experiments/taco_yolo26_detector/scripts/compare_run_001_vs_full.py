#!/usr/bin/env python
"""Quantifies what recovery added, per class and per object size.

Two comparisons, both read straight off the two prepared label trees rather than
off the annotations — what the model sees is what gets counted:

* per-class annotation and image counts, Run 001 vs Run 002;
* object size distribution in COCO terms (small <32², medium <96², large),
  measured at the 640 px letterboxed scale the model actually trains at.

The size half matters because Run 001's weakest classes — cigarette, straw,
bottle_cap — are also its smallest objects, and more images only helps if the
new objects are not all in the same unresolvable size bucket.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_001 = ROOT / "data" / "prepared" / "labels"
RUN_002 = ROOT / "data" / "prepared_run_002" / "labels"
REPORTS = ROOT / "reports" / "dataset_recovery"
FIGURES = ROOT / "reports" / "figures"
CONFIGS = ROOT / "configs"

IMGSZ = 640  # unchanged from Run 001; sizes are quoted at the training scale
SMALL, MEDIUM = 32 ** 2, 96 ** 2


def read_labels(root: pathlib.Path) -> tuple[collections.Counter, collections.Counter, list]:
    """Returns (annotations per class, images per class, [(cid, w, h) normalised])."""
    annotations = collections.Counter()
    images = collections.Counter()
    boxes = []
    for split in ("train", "val", "test"):
        for path in sorted((root / split).glob("*.txt")):
            present = set()
            for line in path.read_text().split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                annotations[cid] += 1
                present.add(cid)
                boxes.append((cid, float(parts[3]), float(parts[4])))
            images.update(present)
    return annotations, images, boxes


def bucket(area_px: float) -> str:
    if area_px < SMALL:
        return "small"
    if area_px < MEDIUM:
        return "medium"
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    classes = list(yaml.safe_load(
        (CONFIGS / "class_mapping_run_002.yaml").read_text())["target_classes"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    a1, i1, b1 = read_labels(RUN_001)
    a2, i2, b2 = read_labels(RUN_002)

    # ------------------------------------------------------ per-class growth
    rows = []
    for cid, name in enumerate(classes):
        old, new = a1[cid], a2[cid]
        rows.append({
            "class_id": cid, "class": name,
            "run_001_annotations": old, "run_002_annotations": new,
            "annotation_delta": new - old,
            "annotation_multiplier": round(new / old, 2) if old else "",
            "run_001_images": i1[cid], "run_002_images": i2[cid],
            "image_delta": i2[cid] - i1[cid],
            "image_multiplier": round(i2[cid] / i1[cid], 2) if i1[cid] else "",
        })
    totals = {
        "class_id": "", "class": "TOTAL",
        "run_001_annotations": sum(a1.values()), "run_002_annotations": sum(a2.values()),
        "annotation_delta": sum(a2.values()) - sum(a1.values()),
        "annotation_multiplier": round(sum(a2.values()) / max(sum(a1.values()), 1), 2),
        "run_001_images": "", "run_002_images": "", "image_delta": "", "image_multiplier": "",
    }
    with open(REPORTS / "run_001_vs_full_dataset.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows + [totals])

    scarcest = min(rows, key=lambda r: r["run_002_annotations"])
    biggest = max(rows, key=lambda r: r["annotation_delta"])
    lines = [
        "# What recovery added, per class", "",
        "Counted from the two prepared label trees — `data/prepared/labels/` for Run 001 "
        "and `data/prepared_run_002/labels/` — so these are the boxes the model is "
        "actually handed, not the raw COCO totals.", "",
        "| ID | Class | Ann. 001 | Ann. 002 | Δ | × | Img 001 | Img 002 | Δ img |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_id']} | `{row['class']}` | {row['run_001_annotations']} "
            f"| {row['run_002_annotations']} | +{row['annotation_delta']} "
            f"| {row['annotation_multiplier']}× | {row['run_001_images']} "
            f"| {row['run_002_images']} | +{row['image_delta']} |")
    lines += [
        f"| | **TOTAL** | **{totals['run_001_annotations']}** "
        f"| **{totals['run_002_annotations']}** | **+{totals['annotation_delta']}** "
        f"| **{totals['annotation_multiplier']}×** | | | |", "",
        "## Reading it", "",
        f"- Largest absolute gain: **`{biggest['class']}`**, "
        f"+{biggest['annotation_delta']} annotations "
        f"({biggest['run_001_annotations']} → {biggest['run_002_annotations']}).",
        f"- Still the scarcest class: **`{scarcest['class']}`** at "
        f"{scarcest['run_002_annotations']} annotations — recovery narrowed the "
        "imbalance but did not remove it.",
        "", "Class IDs are unchanged from Run 001, so Run 002's numbers are directly "
        "comparable to Run 001's per-class metrics.", "",
    ]
    (REPORTS / "run_001_vs_full_dataset.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------ size distribution
    def distribution(boxes):
        per_class = collections.defaultdict(collections.Counter)
        areas = collections.defaultdict(list)
        for cid, w, h in boxes:
            area = (w * IMGSZ) * (h * IMGSZ)
            per_class[cid][bucket(area)] += 1
            areas[cid].append(area)
        return per_class, areas

    d1, _ = distribution(b1)
    d2, areas2 = distribution(b2)

    size_rows = []
    for cid, name in enumerate(classes):
        values = sorted(areas2.get(cid, []))
        total = max(sum(d2[cid].values()), 1)
        size_rows.append({
            "class_id": cid, "class": name,
            "annotations": sum(d2[cid].values()),
            "small": d2[cid]["small"], "medium": d2[cid]["medium"], "large": d2[cid]["large"],
            "small_pct": round(100 * d2[cid]["small"] / total, 1),
            "median_area_px2": round(values[len(values) // 2], 1) if values else "",
            "p10_area_px2": round(values[int(0.10 * (len(values) - 1))], 1) if values else "",
            "p90_area_px2": round(values[int(0.90 * (len(values) - 1))], 1) if values else "",
        })
    with open(REPORTS / "full_object_size_distribution.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(size_rows[0].keys()))
        writer.writeheader()
        writer.writerows(size_rows)

    all_small = sum(r["small"] for r in size_rows)
    all_total = max(sum(r["annotations"] for r in size_rows), 1)
    tiniest = max(size_rows, key=lambda r: r["small_pct"])
    size_md = [
        "# Object size distribution, recovered dataset", "",
        f"Areas are measured at **{IMGSZ} px**, the training size — unchanged from Run 001. "
        "Buckets follow the COCO convention: small < 32² px, medium < 96² px, large above.", "",
        "| ID | Class | Ann. | Small | Medium | Large | Small % | p10 | Median | p90 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_rows:
        size_md.append(
            f"| {row['class_id']} | `{row['class']}` | {row['annotations']} | {row['small']} "
            f"| {row['medium']} | {row['large']} | {row['small_pct']}% "
            f"| {row['p10_area_px2']} | {row['median_area_px2']} | {row['p90_area_px2']} |")
    size_md += [
        "", f"Overall **{round(100 * all_small / all_total, 1)}%** of objects fall in the "
        f"small bucket. The extreme is **`{tiniest['class']}` at {tiniest['small_pct']}%** "
        f"small, median area {tiniest['median_area_px2']} px² — a few pixels across at "
        f"{IMGSZ}. That is a resolution problem, not a data-volume problem, and no amount "
        "of extra images removes it at this input size.", "",
    ]
    (REPORTS / "full_object_size_distribution.md").write_text("\n".join(size_md) + "\n")

    # ------------------------------------------- 001 vs full, size comparison
    comp = [
        "# Object sizes: Run 001 vs recovered dataset", "",
        "The question this answers: did recovery add *different* objects, or more of "
        "the same size profile? A shifted profile changes what Run 002 can be expected "
        "to learn; an unchanged one means the gain is purely volume.", "",
        "| Class | Small 001 | Small 002 | Small % 001 | Small % 002 | Δ pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cid, name in enumerate(classes):
        t1 = max(sum(d1[cid].values()), 1)
        t2 = max(sum(d2[cid].values()), 1)
        p1 = 100 * d1[cid]["small"] / t1
        p2 = 100 * d2[cid]["small"] / t2
        comp.append(f"| `{name}` | {d1[cid]['small']} | {d2[cid]['small']} "
                    f"| {p1:.1f}% | {p2:.1f}% | {p2 - p1:+.1f} |")
    t1all = max(sum(sum(d1[c].values()) for c in range(len(classes))), 1)
    p1all = 100 * sum(d1[c]["small"] for c in range(len(classes))) / t1all
    p2all = 100 * all_small / all_total
    shifted = max(
        ((name, 100 * d2[c]["small"] / max(sum(d2[c].values()), 1)
          - 100 * d1[c]["small"] / max(sum(d1[c].values()), 1))
         for c, name in enumerate(classes)),
        key=lambda pair: abs(pair[1]))
    comp += [
        "", f"Overall small-object share moved from **{p1all:.1f}%** to "
        f"**{p2all:.1f}%** — a shift of **{p2all - p1all:+.1f} pp**. The largest "
        f"per-class move is **`{shifted[0]}` at {shifted[1]:+.1f} pp**.", "",
    ]
    if abs(p2all - p1all) < 2.0:
        comp += [
            "That is close to flat: recovery scaled the dataset without changing its "
            "difficulty profile, so any Run 002 improvement is attributable to volume "
            "rather than to an easier test set.", ""]
    else:
        comp += [
            "**This is not a flat rescale — the recovered dataset is harder.** The "
            "images Flickr refused were not a random sample of TACO: they carried "
            "proportionally more tiny objects, so recovery raised the small-object "
            "share rather than leaving it alone.", "",
            "Two consequences for reading Run 002:", "",
            "1. Run 002's mAP is **not directly comparable** to Run 001's as a measure "
            "of the model. The test set changed and got harder, so a flat or slightly "
            "lower mAP would still be consistent with a better detector.",
            "2. The honest comparison is per-class and size-bucketed, not a single "
            "headline number. `evaluate_run.py` already reports per class; the size "
            "buckets in `full_object_size_distribution.md` are what to read it "
            "against.", "",
            "Run 001 remains frozen precisely so this comparison stays available — its "
            "metrics were measured on its own splits and are not retroactively "
            "invalidated, only differently scoped.", ""]
    (REPORTS / "run_001_vs_full_object_sizes.md").write_text("\n".join(comp) + "\n")

    # ------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import math

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for cid, name in enumerate(classes):
            values = [math.sqrt(a) for a in areas2.get(cid, []) if a > 0]
            if values:
                ax.hist(values, bins=40, range=(0, 320), histtype="step",
                        linewidth=1.6, label=name)
        ax.axvline(32, color="crimson", linestyle="--", linewidth=1)
        ax.axvline(96, color="darkorange", linestyle="--", linewidth=1)
        ax.set_xlabel(f"box side, sqrt(area) in px at {IMGSZ}")
        ax.set_ylabel("annotations")
        ax.set_title("Box size by class (dashed: COCO small / medium cut-offs)")
        ax.legend(fontsize=8)

        ax = axes[1]
        width = 0.6
        bottoms = [0.0] * len(classes)
        for key, colour in (("small", "#d1495b"), ("medium", "#edae49"), ("large", "#00798c")):
            values = [100 * d2[c][key] / max(sum(d2[c].values()), 1) for c in range(len(classes))]
            ax.bar(range(len(classes)), values, width, bottom=bottoms, label=key, color=colour)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("% of annotations")
        ax.set_title("Size bucket composition, recovered dataset")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "full_bbox_area_distribution.png", dpi=150)
        plt.close(fig)
        figure_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: figure not written ({type(exc).__name__}: {exc})")
        figure_ok = False

    print(json.dumps({
        "run_001_annotations": sum(a1.values()),
        "run_002_annotations": sum(a2.values()),
        "small_pct_run_001": round(p1all, 1),
        "small_pct_run_002": round(100 * all_small / all_total, 1),
        "figure_written": figure_ok,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
