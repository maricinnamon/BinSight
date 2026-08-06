#!/usr/bin/env python
"""Split-leakage audit, class distribution and object-size distribution.

Leakage is checked three independent ways, because a single check tests only the
mechanism you happened to think of: by source image id, by relative source path,
and by SHA-256 of the image bytes. The third is the one that catches genuine
duplicates — the same photograph resolving through two different records.

Size buckets are the historical BinSight definitions, unchanged since Run 001, so
the numbers here sit alongside the earlier reports rather than replacing them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from bbox_only import SIZE_BUCKETS  # noqa: E402

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
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = list(csv.DictReader(open(MANIFESTS / "bbox_annotations.csv")))
    build = json.loads((REPORTS / "build_summary.json").read_text())
    splits = ("train", "val", "test")

    # ------------------------------------------------------------- leakage
    by_id, by_path, by_hash = (collections.defaultdict(set) for _ in range(3))
    for r in rows:
        by_id[r["image_id"]].add(r["split"])
        by_path[r["source_relative_path"]].add(r["split"])
    hashes: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for split in splits:
        for image in sorted((DATA / "images" / split).iterdir()):
            if image.exists():
                hashes[sha256(image)].append((split, image.name))
    for digest, entries in hashes.items():
        for split, _ in entries:
            by_hash[digest].add(split)

    leaks = {
        "by_image_id": {k: sorted(v) for k, v in by_id.items() if len(v) > 1},
        "by_source_path": {k: sorted(v) for k, v in by_path.items() if len(v) > 1},
        "by_content_hash": {k[:16]: sorted(v) for k, v in by_hash.items() if len(v) > 1},
    }
    duplicate_content = {k[:16]: v for k, v in hashes.items() if len(v) > 1}
    leakage_ok = not any(leaks.values())

    # -------------------------------------------------------- distributions
    classes = build["classes"]
    per_class = {s: collections.Counter() for s in splits}
    per_bucket = {s: collections.Counter() for s in splits}
    bucket_by_class = collections.defaultdict(collections.Counter)
    areas = collections.defaultdict(list)
    for r in rows:
        per_class[r["split"]][r["class"]] += 1
        per_bucket[r["split"]][r["size_bucket"]] += 1
        bucket_by_class[r["class"]][r["size_bucket"]] += 1
        areas[r["class"]].append(float(r["relative_area"]))

    totals = sum((per_class[s] for s in splits), collections.Counter())
    bucket_totals = sum((per_bucket[s] for s in splits), collections.Counter())

    report = {
        "leakage": {"result": "PASS" if leakage_ok else "FAIL",
                    "checks": ["source image id", "relative source path",
                               "image content SHA-256"],
                    "violations": leaks,
                    "duplicate_content_groups": duplicate_content,
                    "unique_image_hashes": len(hashes)},
        "counts": build["counts"],
        "per_class": {s: dict(per_class[s]) for s in splits},
        "per_class_total": dict(totals),
        "size_buckets": {s: dict(per_bucket[s]) for s in splits},
        "size_buckets_total": dict(bucket_totals),
        "excluded": build["excluded"],
    }
    (REPORTS / "split_leakage_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    # ------------------------------------------------------------- markdown
    n_boxes = sum(totals.values())
    dist = ["# Dataset distribution", "",
            "Official reviewed TACO, converted with `annotation[\"bbox\"]` only. "
            "Splits are reused unchanged from the earlier verified experiment so the "
            "numbers stay comparable.", "",
            "| Split | Images | Boxes |", "|---|---:|---:|"]
    for s in splits:
        dist.append(f"| {s} | {build['counts'][s]['images']} | {build['counts'][s]['boxes']} |")
    dist += [f"| **total** | **{build['total_images']}** | **{build['total_boxes']}** |", "",
             "## Boxes per class", "",
             "| ID | Class | Train | Val | Test | Total | Share |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for i, name in enumerate(classes):
        t = totals[name]
        dist.append(f"| {i} | `{name}` | {per_class['train'][name]} | {per_class['val'][name]} "
                    f"| {per_class['test'][name]} | {t} | {100*t/max(n_boxes,1):.1f}% |")
    imbalance = max(totals.values()) / max(min(totals.values()), 1)
    dist += [f"| | **total** | **{sum(per_class['train'].values())}** "
             f"| **{sum(per_class['val'].values())}** | **{sum(per_class['test'].values())}** "
             f"| **{n_boxes}** | |", "",
             f"Imbalance: **{imbalance:.2f}:1** "
             f"(`{max(totals, key=totals.get)}` vs `{min(totals, key=totals.get)}`). "
             "No reweighting or resampling is applied — this run measures the dataset as "
             "it is.", ""]
    (REPORTS / "dataset_distribution.md").write_text("\n".join(dist) + "\n")

    ranges = {n: (lo, hi) for n, lo, hi in SIZE_BUCKETS}
    audit = ["# Bounding-box annotation audit", "",
             "Every target came from `annotation[\"bbox\"]`. Segmentation polygons were "
             "stripped at load time and never consulted.", "",
             "## Source", "", "| | |", "|---|---:|",
             f"| Official image records | {build['source']['official_image_records']} |",
             f"| Official annotations | {build['source']['official_annotations']} |",
             f"| Images resolved on disk | {build['source']['images_resolved_on_disk']} |",
             f"| **Images used** | **{build['total_images']}** |",
             f"| **Valid mapped boxes** | **{build['total_boxes']}** |",
             f"| Excluded annotations | {build['excluded_total']} |",
             f"| Boxes clipped to frame | {build['clipped_to_frame']} |", "",
             "## Why annotations were excluded", "",
             "| Reason | Count |", "|---|---:|"]
    audit += [f"| `{k}` | {v} |" for k, v in sorted(build["excluded"].items(),
                                                    key=lambda kv: -kv[1])]
    invalid = sum(v for k, v in build["excluded"].items()
                  if k.startswith("bbox_") or k.startswith("normalised"))
    audit += ["",
              f"**{invalid} annotations were excluded for an invalid or missing `bbox`.** "
              "None were reconstructed from a segmentation polygon — a mask-derived box "
              "is a different annotation, and substituting one would make the dataset a "
              "silent mixture of two sources.", "",
              "`category_not_mapped` is the deliberate taxonomy decision carried over "
              "from the earlier experiment: TACO's 60 categories collapse to 8 BinSight "
              "classes with no catch-all, and the remainder becomes background.", "",
              "## Object size distribution", "",
              "Historical BinSight buckets, by fraction of frame area — unchanged since "
              "Run 001.", "",
              "| Bucket | Range | Train | Val | Test | Total | Share |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, lo, hi in SIZE_BUCKETS:
        t = bucket_totals[name]
        audit.append(f"| {name} | {lo}–{hi} | {per_bucket['train'][name]} "
                     f"| {per_bucket['val'][name]} | {per_bucket['test'][name]} | {t} "
                     f"| {100*t/max(n_boxes,1):.1f}% |")
    small_share = 100 * (bucket_totals["tiny"] + bucket_totals["small"]) / max(n_boxes, 1)
    audit += ["", f"**{small_share:.1f}% of boxes are tiny or small.** That is the "
              "defining difficulty of this dataset and the reason earlier BinSight runs "
              "reported low recall: TACO is street litter photographed from standing "
              "height.", "",
              "## Size composition per class", "",
              "| Class | tiny | small | medium | large | tiny+small |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in classes:
        c = bucket_by_class[name]
        tot = max(sum(c.values()), 1)
        audit.append(f"| `{name}` | {c['tiny']} | {c['small']} | {c['medium']} | {c['large']} "
                     f"| {100*(c['tiny']+c['small'])/tot:.0f}% |")
    audit += ["", "## Split leakage", "",
              f"**{report['leakage']['result']}** — checked three independent ways:", "",
              f"- by source image id: {len(leaks['by_image_id'])} violations",
              f"- by relative source path: {len(leaks['by_source_path'])} violations",
              f"- by image content SHA-256: {len(leaks['by_content_hash'])} violations",
              "",
              f"{report['leakage']['unique_image_hashes']} unique image hashes across "
              f"{build['total_images']} images "
              f"({len(duplicate_content)} duplicate-content groups).", ""]
    (REPORTS / "bbox_annotation_audit.md").write_text("\n".join(audit) + "\n")

    print(f"leakage: {report['leakage']['result']}")
    print(f"  by id={len(leaks['by_image_id'])} path={len(leaks['by_source_path'])} "
          f"hash={len(leaks['by_content_hash'])}")
    print(f"  unique hashes {len(hashes)} / {build['total_images']} images")
    print(f"tiny+small share: {small_share:.1f}%")
    print(f"imbalance: {imbalance:.2f}:1")
    return 0 if leakage_ok else 1


if __name__ == "__main__":
    sys.exit(main())
