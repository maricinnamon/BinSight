#!/usr/bin/env python
"""Records exactly what a training run is about to consume, before it starts.

Written before training rather than after, so the inputs cannot be adjusted to
fit the result. Hashes cover the dataset definition, the class mapping and all
three split manifests — change any of them and the manifest stops matching, which
is the point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed(path: pathlib.Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="e.g. run_002")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--mapping", default="configs/class_mapping_run_002.yaml")
    parser.add_argument("--splits", default="data/splits/run_002")
    parser.add_argument("--prepared", default="data/prepared_run_002")
    args = parser.parse_args()

    train_config = yaml.safe_load((ROOT / args.train_config).read_text())
    dataset_config = yaml.safe_load((ROOT / args.dataset_config).read_text())
    splits_dir = ROOT / args.splits
    prepared = ROOT / args.prepared

    import torch
    import ultralytics

    counts = {}
    for split in ("train", "val", "test"):
        labels = sorted((prepared / "labels" / split).glob("*.txt"))
        boxes = sum(len([r for r in p.read_text().splitlines() if r.strip()])
                    for p in labels)
        images = list((prepared / "images" / split).iterdir())
        counts[split] = {"images": len(images), "label_files": len(labels), "boxes": boxes}

    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120).stdout
        packages = {line.split("==")[0]: line.split("==")[-1]
                    for line in pip_freeze.splitlines() if "==" in line}
        relevant = {k: v for k, v in packages.items()
                    if k.lower() in {"ultralytics", "torch", "torchvision", "numpy",
                                     "opencv-python", "pillow", "pyyaml", "matplotlib",
                                     "pandas", "scipy"}}
    except Exception:  # noqa: BLE001
        relevant = {}

    manifest = {
        "run": args.run,
        "written_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "pre-training — no metrics exist yet",
        "model_source": train_config.get("model"),
        "seed": train_config.get("seed"),
        "imgsz": train_config.get("imgsz"),
        "batch": train_config.get("batch"),
        "device": train_config.get("device"),
        "epochs_max": train_config.get("epochs"),
        "patience": train_config.get("patience"),
        "class_order": [dataset_config["names"][i] for i in sorted(dataset_config["names"])],
        "dataset_root": dataset_config["path"],
        "dataset_counts": counts,
        "hashes": {
            "train_config": hashed(ROOT / args.train_config),
            "dataset_config": hashed(ROOT / args.dataset_config),
            "class_mapping": hashed(ROOT / args.mapping),
            "split_train": hashed(splits_dir / "train.txt"),
            "split_val": hashed(splits_dir / "val.txt"),
            "split_test": hashed(splits_dir / "test.txt"),
        },
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "packages": relevant,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{args.run}_pretraining_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"written": str(out.relative_to(ROOT)),
                      "counts": counts,
                      "classes": manifest["class_order"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
