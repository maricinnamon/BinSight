#!/usr/bin/env python
"""Copies the selected model and labels into the iOS app bundle resources.

Records where the artefact came from and its checksum, so the file in the app
can always be traced back to a specific verified export.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys

import _bootstrap  # noqa: F401

from binsight_training.config import CLASS_NAMES, DISPLAY_NAMES, IOS_LABEL_ALIASES, load_config
from binsight_training.utils import get_logger, read_json, sha256_file, write_json

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    source_model = config.tflite_dir / "binsight_trashnet.tflite"
    contract_path = config.root / "model_contract.json"
    if not source_model.exists() or not contract_path.exists():
        LOGGER.error("Run export_tflite.py and verify_tflite.py first.")
        return 1

    destination_dir = config.repo_root / "BinSight" / "Resources" / "Models"
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_destination = destination_dir / "binsight_trashnet.tflite"
    labels_destination = destination_dir / "labels.txt"
    contract_destination = destination_dir / "model_contract.json"

    shutil.copyfile(source_model, model_destination)
    (labels_destination).write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    shutil.copyfile(contract_path, contract_destination)

    contract = read_json(contract_path)
    digest = sha256_file(model_destination)
    if digest != contract["model"]["sha256"]:
        LOGGER.error("Checksum mismatch after copy: %s != %s", digest, contract["model"]["sha256"])
        return 1

    provenance = {
        "copied_at": datetime.datetime.now().astimezone().isoformat(),
        "source_file": str(source_model.relative_to(config.repo_root)),
        "source_candidate": contract.get("selection", {}).get("selected"),
        "sha256": digest,
        "size_bytes": model_destination.stat().st_size,
        "labels": CLASS_NAMES,
        "display_names": DISPLAY_NAMES,
        "ios_label_aliases": IOS_LABEL_ALIASES,
        "note": (
            "The model label `trash` is TrashNet's residual class. iOS maps it to "
            "WasteCategory.generalWaste and displays 'General waste'. It is never "
            "presented as a recyclable stream."
        ),
    }
    write_json(destination_dir / "model_provenance.json", provenance)

    LOGGER.info("Model    -> %s (%.2f MB)", model_destination,
                model_destination.stat().st_size / 1e6)
    LOGGER.info("Labels   -> %s", labels_destination)
    LOGGER.info("Contract -> %s", contract_destination)
    LOGGER.info("sha256    = %s", digest)
    LOGGER.info(
        "Resources live in a file-system synchronized group, so Xcode picks them up "
        "automatically — no Copy Bundle Resources edit needed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
