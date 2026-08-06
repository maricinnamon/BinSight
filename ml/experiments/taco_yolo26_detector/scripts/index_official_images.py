#!/usr/bin/env python
"""Builds a verified index of every official COCO image record to a local file.

Two sources are searched, in order: the archived Zenodo release
(`data/official_complete/`) and the partial Flickr download kept from Run 001
(`data/raw/`). Matching is by the annotation's own relative path, never by
basename — TACO nests images in `batch_N/` and basenames repeat across batches.

Every resolved file is opened and decoded, and its dimensions are checked against
the COCO record. That check matters more than it looks: 25% of TACO JPEGs carry
an EXIF Orientation tag that swaps width and height relative to the annotations,
which is what put a quarter of Run 001's boxes in empty background.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
SOURCES = [
    ("zenodo_archive", ROOT / "data" / "official_complete" / "TACO" / "data"),
    ("flickr_partial", ROOT / "data" / "raw"),
]
REPORTS = ROOT / "reports" / "dataset_recovery"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    data = json.loads(ANNOTATIONS.read_text())
    images = data["images"]
    per_image = collections.Counter(a["image_id"] for a in data["annotations"])
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    by_source = collections.Counter()
    status_counts = collections.Counter()

    for record in images:
        relative = record["file_name"]
        resolved, source = None, None
        for name, base in SOURCES:
            candidate = base / relative
            if candidate.exists() and candidate.stat().st_size > 0:
                resolved, source = candidate, name
                break

        row = {
            "image_id": record["id"], "annotation_path": relative,
            "resolved_path": str(resolved.relative_to(ROOT)) if resolved else "",
            "source": source or "",
            "coco_width": record["width"], "coco_height": record["height"],
            "decoded_width": "", "decoded_height": "",
            "file_bytes": "", "sha256": "",
            "annotations": per_image.get(record["id"], 0),
            "status": "",
        }

        if resolved is None:
            row["status"] = "missing"
            status_counts["missing"] += 1
            failures.append({**row, "problem": "no local file in any source"})
            rows.append(row)
            continue

        try:
            with Image.open(resolved) as image:
                image.load()
                raw_size = image.size
                # Compare against the EXIF-corrected size, since that is what the
                # annotations describe.
                corrected = ImageOps.exif_transpose(image).size
        except Exception as exc:  # noqa: BLE001
            row["status"] = "unreadable"
            status_counts["unreadable"] += 1
            failures.append({**row, "problem": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            continue

        row["decoded_width"], row["decoded_height"] = corrected
        row["file_bytes"] = resolved.stat().st_size
        row["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()

        if corrected != (record["width"], record["height"]):
            row["status"] = "dimension_mismatch"
            status_counts["dimension_mismatch"] += 1
            failures.append({**row,
                             "problem": f"decoded {corrected} (raw {raw_size}) != "
                                        f"coco ({record['width']}, {record['height']})"})
        else:
            row["status"] = "ok"
            status_counts["ok"] += 1
            by_source[source] += 1
        rows.append(row)

    with open(REPORTS / "official_image_manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        with open(REPORTS / "image_validation_failures.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
    else:
        (REPORTS / "image_validation_failures.csv").write_text("image_id,problem\n")

    # Duplicate content across different image records would be a leakage risk.
    hashes = collections.Counter(r["sha256"] for r in rows if r["sha256"])
    duplicate_hashes = {h: c for h, c in hashes.items() if c > 1}

    usable_ids = {r["image_id"] for r in rows if r["status"] == "ok"}
    annotations_available = sum(1 for a in data["annotations"] if a["image_id"] in usable_ids)

    summary = {
        "official_image_records": len(images),
        "official_annotations": len(data["annotations"]),
        "resolved_ok": status_counts["ok"],
        "missing": status_counts["missing"],
        "unreadable": status_counts["unreadable"],
        "dimension_mismatch": status_counts["dimension_mismatch"],
        "by_source": dict(by_source),
        "annotations_on_available_images": annotations_available,
        "annotations_lost_to_missing_images": len(data["annotations"]) - annotations_available,
        "duplicate_content_hashes": len(duplicate_hashes),
        "coverage_pct": round(100 * status_counts["ok"] / len(images), 2),
    }
    (REPORTS / "completeness_report.json").write_text(json.dumps(summary, indent=2) + "\n")

    complete = summary["missing"] == 0 and summary["unreadable"] == 0
    label = ("complete official reviewed dataset" if complete
             else "maximally recovered official subset")
    md = [
        "# Dataset completeness after recovery", "",
        f"**Status: {label}**", "",
        "| | |", "|---|---:|",
        f"| Official image records | {summary['official_image_records']} |",
        f"| Official annotations | {summary['official_annotations']} |",
        f"| **Resolved and validated** | **{summary['resolved_ok']}** ({summary['coverage_pct']}%) |",
        f"| Missing | {summary['missing']} |",
        f"| Unreadable | {summary['unreadable']} |",
        f"| Dimension mismatches | {summary['dimension_mismatch']} |",
        f"| Annotations on available images | {summary['annotations_on_available_images']} |",
        f"| Annotations lost to missing images | {summary['annotations_lost_to_missing_images']} |",
        f"| Duplicate content hashes | {summary['duplicate_content_hashes']} |",
        "", "## By source", "", "| Source | Images |", "|---|---:|",
    ]
    md += [f"| `{k}` | {v} |" for k, v in by_source.most_common()]
    md += ["",
           "Every file was opened and fully decoded, and its **EXIF-corrected** size "
           "compared against the COCO record — the raw JPEG size disagrees for about a "
           "quarter of TACO, which is what misplaced a quarter of Run 001's boxes.", ""]
    if not complete:
        md += [
            "## Why this is not the complete dataset", "",
            f"{summary['missing']} image records could not be resolved from either the "
            "archived Zenodo release or the Flickr download. The archive predates the "
            "current annotation revision (it holds 715 images against the annotations' "
            "1500), and the remaining Flickr URLs are rate-limited or gone.", "",
            "It is therefore labelled a **maximally recovered official subset**, not the "
            "complete TACO dataset.", ""]
    (REPORTS / "completeness_report.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
