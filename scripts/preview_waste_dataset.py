#!/usr/bin/env python3
"""Renders final labels back onto images, so placement can be checked by eye.

Programmatic validation proves the numbers are well-formed. It cannot prove a box
is on the object: a swapped axis or a wrong divisor yields perfectly valid
coordinates pointing at background. This is the check that catches that.

Plain rectangles only — this is a detection dataset, there is nothing else to
draw. One image is opened, drawn, saved and closed at a time; nothing is cached.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

COLOURS = {0: (255, 92, 138), 1: (93, 226, 184), 2: (139, 92, 246)}   # paper/plastic/metal
NAMES = {0: "paper", 1: "plastic", 2: "metal"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/binsight_waste_3class")
    parser.add_argument("--out", default="artifacts/dataset_preview")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = pathlib.Path(args.dataset).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()

    from PIL import Image, ImageDraw

    pool = []
    for split in ("train", "val", "test"):
        for label in sorted((root / "labels" / split).glob("*.txt")):
            image = next((root / "images" / split).glob(label.stem + ".*"), None)
            if image is not None:
                pool.append((split, image, label))
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    # Bias the sample toward multi-class frames — they exercise the mapping harder
    # than a single-object photo, where a wrong class id is easy to miss.
    def rank(entry):
        classes = {int(l.split()[0]) for l in entry[2].read_text().splitlines() if l.strip()}
        return -len(classes)
    pool.sort(key=rank)
    chosen = pool[: args.count]

    written = 0
    for split, image_path, label_path in chosen:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            width, height = im.size
            draw = ImageDraw.Draw(im)
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cid, x, y, w, h = line.split()
                cid = int(cid)
                x, y, w, h = float(x), float(y), float(w), float(h)
                bw, bh = w * width, h * height
                x0, y0 = x * width - bw / 2, y * height - bh / 2
                colour = COLOURS.get(cid, (255, 255, 255))
                draw.rectangle([x0, y0, x0 + bw, y0 + bh], outline=colour, width=3)
                tag = NAMES.get(cid, str(cid))
                tw = 7 * len(tag) + 6
                draw.rectangle([x0, max(y0 - 16, 0), x0 + tw, max(y0, 16)], fill=colour)
                draw.text((x0 + 3, max(y0 - 14, 2)), tag, fill=(0, 0, 0))
            im.save(out / f"{split}_{image_path.stem[:40]}.jpg", "JPEG", quality=88)
            del draw
        written += 1
    print(f"wrote {written} previews to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
