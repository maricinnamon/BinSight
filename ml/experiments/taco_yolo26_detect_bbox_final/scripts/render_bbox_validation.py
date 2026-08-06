#!/usr/bin/env python
"""Renders the converted YOLO labels back onto the images, as plain rectangles.

This is the check that programmatic validation cannot perform. Well-formed
coordinates prove nothing about placement: a swapped axis or a wrong divisor
produces perfectly valid numbers pointing at empty background. In Run 001 exactly
that happened to a quarter of the dataset — every coordinate passed validation,
and the defect was found only by looking at rendered samples.

Rectangles only. No polygon overlay, no mask, no filled region — the output is
what the detector is actually being asked to predict.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from bbox_only import parse_label_line  # noqa: E402

DATA = HERE / "data"
REPORTS = HERE / "reports"
PALETTE = ["#FF5C8A", "#5DE2B8", "#8B5CF6", "#D7FF5F",
           "#F5A524", "#4CC9F0", "#FF8A5C", "#B8A8FF"]


def draw(ax, image_path, boxes, classes, caption):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        width, height = im.size
        ax.imshow(im)
    for cid, xc, yc, w, h in boxes:
        bw, bh = w * width, h * height
        # Plain rectangle — this is a Detect dataset, there is nothing else to draw.
        ax.add_patch(patches.Rectangle((xc * width - bw / 2, yc * height - bh / 2),
                                       bw, bh, linewidth=2.0,
                                       edgecolor=PALETTE[cid % len(PALETTE)],
                                       facecolor="none"))
        ax.text(xc * width - bw / 2, max(yc * height - bh / 2 - 6, 12), classes[cid],
                fontsize=7, color="black",
                bbox=dict(facecolor=PALETTE[cid % len(PALETTE)], edgecolor="none",
                          pad=1.2, alpha=0.95))
    ax.set_title(caption, fontsize=8)
    ax.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    classes = list(yaml.safe_load(
        (HERE.parent / "taco_yolo26_detector" / "configs" /
         "class_mapping_run_002.yaml").read_text())["target_classes"])
    rng = random.Random(42)

    # Index every label file by the classes it contains, so the sheet can show
    # each of the eight classes rather than whatever the shuffle happens to pick.
    by_class = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for label in sorted((DATA / "labels" / split).glob("*.txt")):
            image = next((DATA / "images" / split).glob(label.stem + ".*"), None)
            if image is None or not image.exists():
                continue
            boxes = [parse_label_line(l) for l in label.read_text().splitlines() if l.strip()]
            for cid in {b[0] for b in boxes}:
                by_class[cid].append((image, boxes, split))

    tiles = []
    missing = []
    for cid, name in enumerate(classes):
        entries = by_class.get(cid, [])
        if not entries:
            missing.append(name)
            continue
        # Prefer frames where this class is legible: largest instance first,
        # then a random one, so the sheet shows both easy and typical cases.
        entries = sorted(entries, key=lambda e: -max(
            (b[3] * b[4] for b in e[1] if b[0] == cid), default=0))
        picked = [entries[0]]
        rest = entries[1:]
        rng.shuffle(rest)
        picked += rest[: max(args.per_class - 1, 0)]
        for image, boxes, split in picked:
            area = max((b[3] * b[4] for b in boxes if b[0] == cid), default=0)
            tiles.append((image, boxes,
                          f"{name} · {split} · {len(boxes)} box(es) · "
                          f"largest {100*area:.2f}% of frame"))

    rows = (len(tiles) + args.columns - 1) // args.columns
    fig, axes = plt.subplots(rows, args.columns,
                             figsize=(4.4 * args.columns, 4.6 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (image, boxes, caption) in zip(axes, tiles):
        draw(ax, image, boxes, classes, caption)
    fig.suptitle("BinSight final dataset — YOLO Detect labels rendered as rectangles\n"
                 "(bbox-only; no segmentation polygons anywhere in this pipeline)",
                 fontsize=13)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "bbox_visual_validation.jpg"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)

    print(f"{out.relative_to(HERE)}: {len(tiles)} tiles covering "
          f"{len(classes) - len(missing)}/{len(classes)} classes")
    if missing:
        print(f"  WARNING: no example found for {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
