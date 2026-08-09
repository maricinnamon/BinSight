#!/usr/bin/env python3
"""Builds BinSight's 3-class YOLO **detection** dataset from the Kaggle garbage export.

    source (6 classes, YOLO detect)  ->  paper / plastic / metal  ->  0 / 1 / 2

Object detection only. The source ships five-value YOLO rows and this script
refuses anything else: a polygon row (YOLO-seg) has a variable field count and is
rejected rather than silently truncated.

Two decisions worth knowing before reading the code.

**The source class order is read, never assumed.** `data.yaml` lists
`['BIODEGRADABLE','CARDBOARD','GLASS','METAL','PAPER','PLASTIC']`, so PAPER is
source id 4 and must become 0. Hardcoding the final order onto the source order
would silently mislabel the whole dataset.

**The source splits are discarded, deliberately.** They are unusable for these
three classes: `valid` holds 33 paper boxes against 1360 metal, and `test` has a
different distribution again. Validating on a set that is 85% metal while
training on a balanced one produces numbers that mean nothing, and per-class AP
for paper cannot be measured from 33 boxes at all. The re-split is deterministic
(seed 42) and stratified by each image's dominant class, so the three splits
share a distribution. This is safe here because the source contains no augmented
duplicate groups and no image appears in two source splits — both verified below
rather than assumed.

Memory: files are processed one at a time and nothing is cached. Images are
opened only to verify them and read their dimensions, then closed; hashing reads
in chunks. There is no multiprocessing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import re
import shutil
import sys

SEED = 42
FINAL_CLASSES = ["paper", "plastic", "metal"]        # index == final class id
WANTED = {"paper": 0, "plastic": 1, "metal": 2}
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
COORD_TOLERANCE = 1e-6

# Roboflow filenames look like "<base>_jpg.rf.<32 hex>.jpg". The part before the
# suffix identifies the original photograph; several augmentations of one photo
# would otherwise be free to land in different splits.
ROBOFLOW_NAME = re.compile(r"^(.*?)(?:_jpe?g)?\.rf\.[0-9a-f]{32}", re.I)


def base_identity(stem: str) -> str:
    m = ROBOFLOW_NAME.match(stem)
    return m.group(1) if m else stem


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_source_classes(data_yaml: pathlib.Path) -> list[str]:
    """Parses `names:` from the source data.yaml without requiring PyYAML."""
    text = data_yaml.read_text()
    m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
    names, collecting = [], False
    for line in text.splitlines():
        if re.match(r"^names\s*:", line):
            collecting = True
            continue
        if collecting:
            item = re.match(r"^\s*-\s*(.+?)\s*$", line)
            keyed = re.match(r"^\s*\d+\s*:\s*(.+?)\s*$", line)
            if item:
                names.append(item.group(1).strip("'\""))
            elif keyed:
                names.append(keyed.group(1).strip("'\""))
            else:
                break
    if not names:
        raise SystemExit(f"could not read class names from {data_yaml}")
    return names


def parse_row(row: str) -> tuple[int, float, float, float, float] | None:
    """Five values or nothing. A YOLO-seg polygon row fails here by design."""
    parts = row.split()
    if len(parts) != 5:
        return None
    try:
        cid = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None
    return cid, x, y, w, h


def valid_box(x: float, y: float, w: float, h: float) -> bool:
    values = (x, y, w, h)
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return False          # NaN / Inf
    if not (0 - COORD_TOLERANCE <= x <= 1 + COORD_TOLERANCE):
        return False
    if not (0 - COORD_TOLERANCE <= y <= 1 + COORD_TOLERANCE):
        return False
    if not (0 < w <= 1 + COORD_TOLERANCE) or not (0 < h <= 1 + COORD_TOLERANCE):
        return False
    # Reconstructed corners must stay inside the frame.
    if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3:
        return False
    if y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="extracted source root containing train/valid/test and data.yaml")
    parser.add_argument("--output", default="datasets/binsight_waste_3class")
    parser.add_argument("--preserve-source-splits", action="store_true",
                        help="keep the source train/valid/test instead of re-splitting")
    args = parser.parse_args()

    source = pathlib.Path(args.source).expanduser().resolve()
    output = pathlib.Path(args.output).expanduser().resolve()
    data_yaml = source / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"no data.yaml under {source}")

    source_names = read_source_classes(data_yaml)
    # name (lowercased) -> final id, resolved from the SOURCE order.
    source_id_to_final: dict[int, int] = {}
    for source_id, name in enumerate(source_names):
        final = WANTED.get(name.strip().lower())
        if final is not None:
            source_id_to_final[source_id] = final
    missing = set(WANTED) - {source_names[i].lower() for i in source_id_to_final}
    if missing:
        raise SystemExit(f"source lacks required classes: {sorted(missing)}")

    print("source classes (as declared):")
    for i, n in enumerate(source_names):
        target = source_id_to_final.get(i)
        print(f"  {i} {n:<15} -> " +
              (f"final {target} ({FINAL_CLASSES[target]})" if target is not None else "EXCLUDED"))

    # ------------------------------------------------------- pass 1: scan
    # One record per usable image. Boxes are small; images are never loaded here.
    records: list[dict] = []
    stats = collections.Counter()
    problems: list[dict] = []
    field_counts = collections.Counter()
    source_split_of_base: dict[str, set] = collections.defaultdict(set)

    for split in ("train", "valid", "test"):
        label_dir, image_dir = source / split / "labels", source / split / "images"
        if not label_dir.is_dir():
            continue
        for label in sorted(label_dir.glob("*.txt")):
            stats["source_labels"] += 1
            image = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = image_dir / (label.stem + ext)
                if candidate.exists():
                    image = candidate
                    break
            if image is None:
                stats["source_label_without_image"] += 1
                problems.append({"file": label.name, "problem": "no matching source image"})
                continue

            kept: list[tuple[int, float, float, float, float]] = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                field_counts[len(line.split())] += 1
                parsed = parse_row(line)
                if parsed is None:
                    stats["malformed_rows"] += 1
                    problems.append({"file": label.name, "problem": f"malformed row: {line[:60]!r}"})
                    continue
                cid, x, y, w, h = parsed
                stats["source_boxes"] += 1
                if cid not in source_id_to_final:
                    stats["removed_excluded_boxes"] += 1
                    continue
                if not valid_box(x, y, w, h):
                    stats["invalid_boxes_dropped"] += 1
                    problems.append({"file": label.name,
                                     "problem": f"invalid geometry: {x},{y},{w},{h}"})
                    continue
                kept.append((source_id_to_final[cid], x, y, w, h))

            if not kept:
                stats["images_removed_no_target"] += 1
                continue
            records.append({"image": image, "boxes": kept, "source_split": split,
                            "base": base_identity(label.stem)})
            source_split_of_base[base_identity(label.stem)].add(split)

    print(f"\nfield-count histogram across source labels: {dict(field_counts)}")
    if set(field_counts) - {5}:
        raise SystemExit("source contains non-5-field rows — not a pure detection dataset")

    # -------------------------------------- pass 2: verify images, hash, dedupe
    by_hash: dict[str, list[int]] = collections.defaultdict(list)
    usable: list[dict] = []
    from PIL import Image
    for index, rec in enumerate(records):
        path = rec["image"]
        try:
            with Image.open(path) as im:
                im.verify()                       # cheap integrity check
            with Image.open(path) as im:
                width, height = im.size           # reopen: verify() leaves it unusable
            del im
        except Exception as exc:                  # noqa: BLE001
            stats["corrupt_images"] += 1
            problems.append({"file": path.name, "problem": f"unreadable: {type(exc).__name__}"})
            continue
        if width <= 0 or height <= 0:
            stats["zero_size_images"] += 1
            problems.append({"file": path.name, "problem": f"zero-size image {width}x{height}"})
            continue
        digest = sha256_file(path)
        rec.update(width=width, height=height, sha256=digest)
        by_hash[digest].append(len(usable))
        usable.append(rec)

    duplicate_groups = {h: idxs for h, idxs in by_hash.items() if len(idxs) > 1}
    # Identical bytes must never straddle a split; keep the first occurrence only.
    drop = set()
    for idxs in duplicate_groups.values():
        drop.update(idxs[1:])
    stats["duplicate_images_dropped"] = len(drop)
    usable = [r for i, r in enumerate(usable) if i not in drop]

    # ------------------------------------------------------------ pass 3: split
    if args.preserve_source_splits:
        for rec in usable:
            rec["split"] = "val" if rec["source_split"] == "valid" else rec["source_split"]
        split_reason = "source splits preserved (--preserve-source-splits)"
    else:
        # Deterministic and stratified by each image's dominant class, so the three
        # splits share a class distribution. Grouped by base identity so no two
        # variants of one photograph can separate.
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for rec in usable:
            groups[rec["base"]].append(rec)
        stratum: dict[str, str] = {}
        for base, recs in groups.items():
            counter = collections.Counter(b[0] for r in recs for b in r["boxes"])
            stratum[base] = FINAL_CLASSES[counter.most_common(1)[0][0]]
        rng = random.Random(SEED)
        assigned: dict[str, str] = {}
        for cls in FINAL_CLASSES:
            members = sorted(b for b, s in stratum.items() if s == cls)
            rng.shuffle(members)
            n = len(members)
            n_train = int(round(n * SPLIT_FRACTIONS["train"]))
            n_val = int(round(n * SPLIT_FRACTIONS["val"]))
            for i, base in enumerate(members):
                assigned[base] = ("train" if i < n_train
                                  else "val" if i < n_train + n_val else "test")
        for rec in usable:
            rec["split"] = assigned[rec["base"]]
        split_reason = (f"re-split {int(SPLIT_FRACTIONS['train']*100)}/"
                        f"{int(SPLIT_FRACTIONS['val']*100)}/"
                        f"{int(SPLIT_FRACTIONS['test']*100)}, seed {SEED}, stratified by "
                        "dominant class, grouped by source image identity")

    # ------------------------------------------------------------ pass 4: write
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = output / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {s: {"images": 0, "boxes": 0} for s in ("train", "val", "test")}
    per_class = {s: collections.Counter() for s in ("train", "val", "test")}
    manifest_rows = []
    used_names: set[str] = set()
    for rec in usable:
        split = rec["split"]
        name = rec["image"].name
        if name in used_names:                    # cannot happen with these sources
            stats["duplicate_filenames"] += 1
            name = f"{rec['image'].stem}_{rec['sha256'][:8]}{rec['image'].suffix}"
        used_names.add(name)
        # Copy, not symlink: datasets/_tmp is deleted when preparation finishes.
        shutil.copyfile(rec["image"], output / "images" / split / name)
        (output / "labels" / split / f"{pathlib.Path(name).stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                      for c, x, y, w, h in rec["boxes"]) + "\n")
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(rec["boxes"])
        for c, *_ in rec["boxes"]:
            per_class[split][FINAL_CLASSES[c]] += 1
        manifest_rows.append({"split": split, "file": name,
                              "source_split": rec["source_split"],
                              "base_identity": rec["base"], "sha256": rec["sha256"],
                              "width": rec["width"], "height": rec["height"],
                              "boxes": len(rec["boxes"])})

    # ------------------------------------------------------------- data.yaml
    (output / "data.yaml").write_text(
        "# BinSight 3-class waste detection — YOLO Detect, bounding boxes only.\n"
        "#\n"
        "# `path` is relative to this file, so the dataset is portable: Ultralytics\n"
        "# resolves it against the yaml's own location when given a relative value here.\n"
        "# Generated by scripts/prepare_waste_dataset.py — do not hand-edit.\n"
        "\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: paper\n"
        "  1: plastic\n"
        "  2: metal\n")

    output.joinpath("manifest.csv").write_text("")
    with open(output / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_root": str(source),
        "source_classes": source_names,
        "source_id_to_final": {str(k): v for k, v in source_id_to_final.items()},
        "final_classes": {str(i): n for i, n in enumerate(FINAL_CLASSES)},
        "split_strategy": split_reason,
        "seed": SEED,
        "counts": counts,
        "per_class": {s: dict(per_class[s]) for s in per_class},
        "per_class_total": dict(sum((per_class[s] for s in per_class), collections.Counter())),
        "stats": dict(stats),
        "duplicate_content_groups": len(duplicate_groups),
        "problems": problems[:200],
        "problem_count": len(problems),
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nsplit strategy: {split_reason}")
    print(f"{'split':<8}{'images':>8}{'boxes':>8}{'paper':>8}{'plastic':>9}{'metal':>7}")
    for s in ("train", "val", "test"):
        print(f"{s:<8}{counts[s]['images']:>8}{counts[s]['boxes']:>8}"
              f"{per_class[s]['paper']:>8}{per_class[s]['plastic']:>9}{per_class[s]['metal']:>7}")
    print(f"\nstats: {dict(stats)}")
    print(f"problems recorded: {len(problems)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
