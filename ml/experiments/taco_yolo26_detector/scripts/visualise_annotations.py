#!/usr/bin/env python
"""Renders the CONVERTED YOLO labels back onto real images.

This is the check programmatic validation cannot do. Well-formed numbers prove
nothing about whether a box landed on the object — an axis swap or a
normalisation mistake produces perfectly valid coordinates in the wrong place.
The boxes drawn here come from `data/prepared/labels/`, never from the source
COCO polygons.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# Classes whose objects are mostly a few pixels across at 640. A full-frame
# render of these is unreadable, so they also get zoomed crops.
TINY_CLASSES = ("cigarette", "straw", "bottle_cap")

PALETTE = ["#FF5C8A", "#5DE2B8", "#8B5CF6", "#D7FF5F",
           "#F5A524", "#4CC9F0", "#FF8A5C", "#B8A8FF"]


def read_labels(path: pathlib.Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return rows


def draw(ax, image_path: pathlib.Path, labels, classes) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        ax.imshow(image)
    for cid, xc, yc, w, h in labels:
        # Denormalise using the *image's own* size — this is the step that
        # would expose a wrong normalisation.
        bw, bh = w * width, h * height
        x0, y0 = (xc * width) - bw / 2, (yc * height) - bh / 2
        colour = PALETTE[cid % len(PALETTE)]
        ax.add_patch(patches.Rectangle((x0, y0), bw, bh, linewidth=2.2,
                                       edgecolor=colour, facecolor="none"))
        ax.text(x0, max(y0 - 6, 10), classes[cid], fontsize=8, color="black",
                bbox=dict(facecolor=colour, edgecolor="none", pad=1.5, alpha=0.95))
    ax.axis("off")


def draw_crop(ax, image_path: pathlib.Path, box, classes, margin: float = 3.0) -> None:
    """Renders one box zoomed in, so a 78 px² object is actually inspectable."""
    cid, xc, yc, w, h = box
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        bw, bh = w * width, h * height
        cx, cy = xc * width, yc * height
        half = max(bw, bh) * (1 + margin) / 2
        half = max(half, 40.0)
        left, top = max(cx - half, 0), max(cy - half, 0)
        right, bottom = min(cx + half, width), min(cy + half, height)
        crop = image.crop((int(left), int(top), int(right), int(bottom)))
        ax.imshow(crop)
    colour = PALETTE[cid % len(PALETTE)]
    ax.add_patch(patches.Rectangle((cx - bw / 2 - left, cy - bh / 2 - top), bw, bh,
                                   linewidth=1.8, edgecolor=colour, facecolor="none"))
    ax.set_title(f"{classes[cid]} — {bw * bh * (640 * 640) / (width * height):.0f} px² @640",
                 fontsize=8)
    ax.axis("off")


def crop_grid(items, classes, path: pathlib.Path, title: str, columns: int = 4) -> None:
    if not items:
        return
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (image_path, box) in zip(axes, items):
        draw_crop(ax, image_path, box, classes)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}  ({len(items)} crops)")


def grid(pairs, classes, path: pathlib.Path, title: str, columns: int = 3) -> None:
    if not pairs:
        return
    rows = (len(pairs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.2 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (image_path, label_path) in zip(axes, pairs):
        draw(ax, image_path, read_labels(label_path), classes)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}  ({len(pairs)} images)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-split", type=int, default=9)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--per-tiny-class", type=int, default=16,
                        help="zoomed crops per hard-to-see class")
    # Defaults reproduce the Run 001 invocation; Run 002 passes its own paths.
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--out", default="reports/samples")
    args = parser.parse_args()

    prepared = ROOT / args.prepared
    samples = ROOT / args.out

    classes = (CONFIGS / "classes.txt").read_text().split()
    rng = random.Random(42)
    samples.mkdir(parents=True, exist_ok=True)

    by_class: dict[int, list] = collections.defaultdict(list)
    tiny_boxes: dict[int, list] = collections.defaultdict(list)
    multi_class: list = []

    for split in ("train", "val", "test"):
        label_dir = prepared / "labels" / split
        image_dir = prepared / "images" / split
        if not label_dir.exists():
            continue
        pairs = []
        for label in sorted(label_dir.glob("*.txt")):
            matches = list(image_dir.glob(label.stem + ".*"))
            if not matches:
                continue
            entry = (matches[0], label)
            pairs.append(entry)
            rows = read_labels(label)
            present = {cid for cid, *_ in rows}
            for cid in present:
                by_class[cid].append(entry)
            for box in rows:
                if classes[box[0]] in TINY_CLASSES:
                    tiny_boxes[box[0]].append((matches[0], box))
            if len(present) > 1:
                multi_class.append(entry)

        rng.shuffle(pairs)
        grid(pairs[: args.per_split], classes, samples / f"{split}_samples.jpg",
             f"{split} — boxes rendered from the converted YOLO labels")

    per_class_dir = samples / "per_class"
    for cid, name in enumerate(classes):
        entries = by_class.get(cid, [])
        rng.shuffle(entries)
        grid(entries[: args.per_class], classes, per_class_dir / f"{name}.jpg",
             f"class {cid}: {name}")

    # Zoomed crops for the classes a full-frame render cannot show. Sorted
    # smallest-first so the grid opens on the genuinely marginal cases rather
    # than on a comfortable example.
    tiny_dir = samples / "tiny_objects"
    for cid, items in sorted(tiny_boxes.items()):
        items.sort(key=lambda pair: pair[1][3] * pair[1][4])
        crop_grid(items[: args.per_tiny_class], classes,
                  tiny_dir / f"{classes[cid]}_smallest.jpg",
                  f"{classes[cid]}: the {args.per_tiny_class} smallest boxes in the dataset")
        rng.shuffle(items)
        crop_grid(items[: args.per_tiny_class], classes,
                  tiny_dir / f"{classes[cid]}_random.jpg",
                  f"{classes[cid]}: random boxes, zoomed")

    rng.shuffle(multi_class)
    grid(multi_class[: args.per_split], classes, samples / "multi_class_samples.jpg",
         "images containing more than one class")

    missing = [classes[c] for c in range(len(classes)) if not by_class.get(c)]
    if missing:
        print(f"WARNING: no rendered examples for: {missing}")
    print(f"samples -> {samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
