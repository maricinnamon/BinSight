#!/usr/bin/env python3
"""Independent hard validation of the final 3-class YOLO detection dataset.

Deliberately re-reads everything from disk rather than trusting the preparation
script's own summary — a bug shared between builder and checker would otherwise
be invisible. Exits non-zero if any gate fails.

Memory: one file at a time, images opened only to verify and read size, hashing
in chunks, no caching, no multiprocessing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

EXPECTED = {0: "paper", 1: "plastic", 2: "metal"}
FORBIDDEN_NAMES = {"biodegradable", "cardboard", "glass"}
TOL = 1e-6


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/binsight_waste_3class")
    args = parser.parse_args()
    root = pathlib.Path(args.dataset).expanduser().resolve()

    failures: list[str] = []
    problems: list[dict] = []
    counts = {s: {"images": 0, "boxes": 0} for s in ("train", "val", "test")}
    per_class = {s: collections.Counter() for s in ("train", "val", "test")}
    field_counts = collections.Counter()
    class_ids_seen: set[int] = set()
    hashes: dict[str, list[str]] = collections.defaultdict(list)
    stems_by_split: dict[str, set] = {}

    from PIL import Image

    for split in ("train", "val", "test"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            failures.append(f"{split}: images/ or labels/ directory missing")
            continue

        image_stems = {p.stem for p in image_dir.iterdir() if p.is_file()}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        stems_by_split[split] = image_stems

        for orphan in sorted(label_stems - image_stems):
            problems.append({"where": f"{split}/{orphan}.txt", "problem": "orphan label"})
        for missing in sorted(image_stems - label_stems):
            problems.append({"where": f"{split}/{missing}", "problem": "image without label"})

        for image in sorted(image_dir.iterdir()):
            if not image.is_file():
                continue
            try:
                with Image.open(image) as im:
                    im.verify()
                with Image.open(image) as im:
                    width, height = im.size
                del im
            except Exception as exc:  # noqa: BLE001
                problems.append({"where": f"{split}/{image.name}",
                                 "problem": f"corrupt/unreadable: {type(exc).__name__}"})
                continue
            if width <= 0 or height <= 0:
                problems.append({"where": f"{split}/{image.name}",
                                 "problem": f"zero-size {width}x{height}"})
                continue
            hashes[sha256_file(image)].append(f"{split}/{image.name}")
            counts[split]["images"] += 1

        for label in sorted(label_dir.glob("*.txt")):
            text = label.read_text()
            if not text.strip():
                problems.append({"where": f"{split}/{label.name}", "problem": "empty label"})
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                where = f"{split}/{label.name}:{n}"
                parts = line.split()
                field_counts[len(parts)] += 1
                if len(parts) != 5:
                    problems.append({"where": where,
                                     "problem": f"{len(parts)} fields, expected 5"})
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    problems.append({"where": where, "problem": "non-numeric value"})
                    continue
                class_ids_seen.add(cid)
                if cid not in EXPECTED:
                    problems.append({"where": where, "problem": f"class_id {cid} not in 0..2"})
                    continue
                if any(v != v or v in (float("inf"), float("-inf")) for v in (x, y, w, h)):
                    problems.append({"where": where, "problem": "NaN or Inf"})
                    continue
                if not (0 - TOL <= x <= 1 + TOL) or not (0 - TOL <= y <= 1 + TOL):
                    problems.append({"where": where, "problem": f"centre outside [0,1]: {x},{y}"})
                if not (0 < w <= 1 + TOL) or not (0 < h <= 1 + TOL):
                    problems.append({"where": where, "problem": f"extent invalid: {w}x{h}"})
                if (x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3
                        or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3):
                    problems.append({"where": where, "problem": "box leaves the frame"})
                counts[split]["boxes"] += 1
                per_class[split][EXPECTED[cid]] += 1

    # ------------------------------------------------------------- gates
    names = list(stems_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = stems_by_split[left] & stems_by_split[right]
            if shared:
                failures.append(f"{len(shared)} filenames shared between {left} and {right}")
    cross_hash = [(h, v) for h, v in hashes.items()
                  if len({p.split("/")[0] for p in v}) > 1]
    if cross_hash:
        failures.append(f"{len(cross_hash)} identical images across splits (leakage)")
    duplicate_within = sum(1 for v in hashes.values() if len(v) > 1)

    if set(field_counts) - {5}:
        failures.append(f"non-5-field label rows present: {dict(field_counts)}")
    stray = class_ids_seen - set(EXPECTED)
    if stray:
        failures.append(f"class ids outside 0..2: {sorted(stray)}")
    if class_ids_seen != set(EXPECTED):
        failures.append(f"expected classes {sorted(EXPECTED)}, found {sorted(class_ids_seen)}")
    if problems:
        failures.append(f"{len(problems)} per-file problems")

    # ------------------------------------------------------------ data.yaml
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        failures.append("data.yaml missing")
    else:
        import yaml as pyyaml
        cfg = pyyaml.safe_load(yaml_path.read_text())
        declared = {int(k): str(v).lower() for k, v in cfg["names"].items()}
        if declared != EXPECTED:
            failures.append(f"data.yaml names {declared} != {EXPECTED}")
        for name in declared.values():
            if name in FORBIDDEN_NAMES:
                failures.append(f"data.yaml still declares excluded class {name!r}")
        base = yaml_path.parent / str(cfg.get("path", "."))
        for key in ("train", "val", "test"):
            resolved = (base / cfg[key]).resolve()
            if not resolved.is_dir():
                failures.append(f"data.yaml {key} does not resolve: {resolved}")

    total = sum(c["boxes"] for c in counts.values())
    tally = sum((per_class[s] for s in per_class), collections.Counter())
    report = {
        "result": "PASS" if not failures else "FAIL",
        "failures": failures,
        "counts": counts,
        "per_class": {s: dict(per_class[s]) for s in per_class},
        "per_class_total": dict(tally),
        "class_distribution_pct": {k: round(100 * v / max(total, 1), 2) for k, v in tally.items()},
        "class_ids_seen": sorted(class_ids_seen),
        "field_count_histogram": {str(k): v for k, v in sorted(field_counts.items())},
        "duplicate_content_within_split": duplicate_within,
        "identical_images_across_splits": len(cross_hash),
        "problem_count": len(problems),
        "problems": problems[:100],
    }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"VALIDATION: {report['result']}")
    print(f"  field counts   : {report['field_count_histogram']}")
    print(f"  class ids seen : {report['class_ids_seen']}")
    print(f"  per class      : {report['per_class_total']}")
    print(f"  distribution % : {report['class_distribution_pct']}")
    print(f"  dup within split: {duplicate_within}   across splits: {len(cross_hash)}")
    print(f"  problems       : {len(problems)}")
    for f in failures:
        print(f"  FAILURE: {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
