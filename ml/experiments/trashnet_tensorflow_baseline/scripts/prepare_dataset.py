#!/usr/bin/env python
"""Discovers, validates and splits the TrashNet dataset.

Writes the dataset reports and the three split manifests. Exits non-zero if the
data cannot be trusted — a silent bad split is worse than a failed build.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import pandas as pd

from binsight_training.config import CLASS_NAMES, load_config
from binsight_training import data as data_module
from binsight_training.utils import get_logger, write_json

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()

    LOGGER.info("Looking for the dataset in %s", config.raw_dir)
    data_module.extract_archives(config.raw_dir)

    class_dirs = data_module.find_class_directories(config.raw_dir)
    if len(class_dirs) < len(CLASS_NAMES):
        missing = sorted(set(CLASS_NAMES) - set(class_dirs))
        LOGGER.error("Missing class directories: %s", missing)
        LOGGER.error(
            "Place the TrashNet archive (or its extracted folders) in %s", config.raw_dir
        )
        return 1

    LOGGER.info("Found all %d class directories", len(class_dirs))
    for name in CLASS_NAMES:
        LOGGER.info("  %-10s -> %s", name, class_dirs[name])

    LOGGER.info("Validating images (full decode, not just headers)...")
    manifest, corrupt = data_module.build_manifest(class_dirs)
    duplicates = data_module.find_duplicates(manifest)

    if manifest.empty:
        LOGGER.error("No readable images found.")
        return 1

    summary = data_module.summarise(manifest, corrupt, duplicates)

    manifest.to_csv(config.reports_dir / "dataset_manifest.csv", index=False)
    corrupt.to_csv(config.reports_dir / "corrupt_images.csv", index=False)
    duplicates.to_csv(config.reports_dir / "duplicate_images.csv", index=False)
    write_json(config.reports_dir / "dataset_summary.json", summary)

    LOGGER.info("Images: %d readable, %d corrupt, %d duplicate groups",
                summary["total_images"], summary["corrupt_count"],
                summary["duplicate_group_count"])
    for name, count in summary["class_counts"].items():
        LOGGER.info("  %-10s %5d", name, count)
    LOGGER.info("Imbalance ratio: %.2f", summary["imbalance_ratio"] or 0)

    # --- Splits ---------------------------------------------------------
    splits = data_module.make_splits(manifest, config)
    verification = data_module.verify_splits(splits)

    if not verification["ok"]:
        for problem in verification["problems"]:
            LOGGER.error("Split problem: %s", problem)
        return 1
    LOGGER.info("Split verification passed: disjoint by path AND by content hash.")

    for split_name in ("train", "validation", "test"):
        subset = splits[splits["split"] == split_name]
        subset.to_csv(config.splits_dir / f"{split_name}.csv", index=False)
        LOGGER.info("  %-11s %5d images", split_name, len(subset))

    write_json(config.reports_dir / "split_verification.json", verification)

    # --- Markdown summary ----------------------------------------------
    per_split = verification["per_class"]
    lines = [
        "# TrashNet dataset summary",
        "",
        f"- **Source:** <https://github.com/garythung/trashnet> (`dataset-resized.zip`)",
        f"- **Licence:** MIT; the repository asks to be cited (see `THIRD_PARTY_NOTICES.md`)",
        f"- **Readable images:** {summary['total_images']}",
        f"- **Corrupt / unreadable:** {summary['corrupt_count']}",
        f"- **Duplicate groups:** {summary['duplicate_group_count']} "
        f"({summary['duplicate_file_count']} redundant files)",
        f"- **Imbalance ratio:** {summary['imbalance_ratio']:.2f} "
        f"({summary['largest_class']} vs {summary['smallest_class']})",
        "",
        "## Class distribution and splits",
        "",
        "| class | total | train | validation | test |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        lines.append(
            f"| {name} | {summary['class_counts'][name]} "
            f"| {per_split.get('train', {}).get(name, 0)} "
            f"| {per_split.get('validation', {}).get(name, 0)} "
            f"| {per_split.get('test', {}).get(name, 0)} |"
        )
    counts = verification["counts"]
    lines += [
        f"| **total** | **{summary['total_images']}** "
        f"| **{counts.get('train', 0)}** | **{counts.get('validation', 0)}** "
        f"| **{counts.get('test', 0)}** |",
        "",
        "## Image geometry",
        "",
        f"- width: {summary['width']}",
        f"- height: {summary['height']}",
        f"- aspect ratio: {summary['aspect_ratio']}",
        "",
        "## Split method",
        "",
        "Stratified 70/15/15, `seed=42`, split over **content hash groups** rather",
        "than individual files so byte-identical duplicates cannot straddle a",
        "boundary and inflate the test score. Verified disjoint by path and by hash.",
        "",
    ]
    (config.reports_dir / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("Reports written to %s", config.reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
