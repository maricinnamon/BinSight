#!/usr/bin/env python
"""Copies the selected checkpoint into models/final/ and writes its manifest.

A copy, never a move: the source run keeps its own weights so the experimental
progression stays intact and reproducible. The manifest gathers every number the
next step needs from persisted reports rather than from arguments, so the frozen
model cannot claim metrics it did not earn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import shutil
import sys
from datetime import datetime, timezone

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FINAL = ROOT / "models" / "final"

READINESS = {"strong_deployment_candidate", "acceptable_prototype_candidate",
             "research_only", "not_suitable"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True, help="e.g. run_002")
    parser.add_argument("--run-dir", required=True, help="e.g. yolo26n_run_002_full_taco_640")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--head", required=True,
                        choices=["one_to_one_end2end", "one_to_many_nms"])
    parser.add_argument("--conf-balanced", type=float, required=True)
    parser.add_argument("--conf-high-precision", type=float, required=True)
    parser.add_argument("--readiness", required=True, choices=sorted(READINESS))
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--name", default="binsight_yolo26n_best.pt")
    args = parser.parse_args()

    source = ROOT / "runs" / args.run_dir / "weights" / "best.pt"
    if not source.exists():
        print(f"ERROR: {source} not found")
        return 1

    FINAL.mkdir(parents=True, exist_ok=True)
    destination = FINAL / args.name
    source_hash = sha256(source)
    shutil.copy2(source, destination)
    copied_hash = sha256(destination)
    if copied_hash != source_hash:
        print("ERROR: copy does not match source hash")
        return 1
    # The source run must be exactly what it was before this script ran.
    if sha256(source) != source_hash:
        print("ERROR: source checkpoint changed during copy")
        return 1

    metrics = read_json(REPORTS / args.run_key / "test_metrics.json") or {}
    errors = read_json(REPORTS / args.run_key / "error_analysis.json") or {}
    bench = read_json(REPORTS / f"pytorch_mps_benchmark_{args.run_key}.json") or {}
    training = read_json(REPORTS / f"training_{args.run_key}_summary.json") or {}
    pretrain = read_json(REPORTS / f"{args.run_key}_pretraining_manifest.json") or {}
    heads = read_json(REPORTS / "comparisons" / "yolo26_head_comparison.json") or []

    dataset_config = ROOT / (pretrain.get("dataset_config_path")
                             or f"configs/dataset_{args.run_key}.yaml")
    classes = []
    if dataset_config.exists():
        names = yaml.safe_load(dataset_config.read_text())["names"]
        classes = [names[i] for i in sorted(names)]

    head_row = next((r for r in heads
                     if r.get("head") == args.head and r.get("run") == args.run_dir), None)

    import torch
    import ultralytics

    manifest = {
        "model_file": args.name,
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "sha256": copied_hash,
        "bytes": destination.stat().st_size,
        "megabytes": round(destination.stat().st_size / 1e6, 3),
        "source": {
            "run_key": args.run_key,
            "run_name": args.run_dir,
            "checkpoint": str(source.relative_to(ROOT)),
            "source_sha256": source_hash,
            "note": "copied, not moved — the source run keeps its own weights",
        },
        "model": {
            "family": "YOLO26",
            "variant": "yolo26n",
            "task": "detect",
            "initialised_from": "yolo26n.pt (COCO pretrained)",
            "imgsz": args.imgsz,
            "preferred_inference_head": args.head,
            "requires_nms": args.head == "one_to_many_nms",
        },
        "classes": {
            "order": classes,
            "count": len(classes),
            "class_mapping_sha256": (pretrain.get("hashes", {})
                                     .get("class_mapping", {}).get("sha256")),
        },
        "confidence_thresholds": {
            "balanced": args.conf_balanced,
            "high_precision": args.conf_high_precision,
            "selected_on": "validation split only — never the test split",
        },
        "metrics": {
            "validation": metrics.get("val", {}).get("overall"),
            "test": metrics.get("test", {}).get("overall"),
            "per_class_test": metrics.get("test", {}).get("per_class"),
            "size_recall_test": errors.get("size_buckets"),
            "error_counts_test": errors.get("counts"),
            "head_comparison": ({"head": args.head,
                                 "val": head_row.get("val"), "test": head_row.get("test")}
                                if head_row else None),
        },
        "latency_mps": {
            "note": ("Mac MPS, batch 1. NOT iPhone latency and NOT a Neural Engine "
                     "measurement."),
            **{k: bench.get(k) for k in ("mean_ms", "median_ms", "p95_ms", "approx_fps")},
        },
        "training": {
            "epochs_completed": training.get("epochs_completed"),
            "best_epoch": training.get("best_epoch"),
            "training_hours": training.get("training_hours"),
            "batch": training.get("batch"),
            "device": training.get("device"),
            "seed": training.get("seed"),
        },
        "dataset_snapshot": {
            "counts": pretrain.get("dataset_counts"),
            "hashes": pretrain.get("hashes"),
            "provenance": ("maximally recovered official subset of TACO — 1300 of 1500 "
                           "official image records resolved; official reviewed "
                           "annotations.json only, annotations_unofficial.json never used"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "readiness": args.readiness,
        "rationale": args.rationale,
        "exports": {
            "coreml": "not exported",
            "tflite": "not exported",
            "onnx": "not exported",
            "note": "Export and parity validation are the next phase, deliberately not done here.",
        },
    }
    (FINAL / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frozen": str(destination.relative_to(ROOT)),
                      "sha256": copied_hash,
                      "megabytes": manifest["megabytes"],
                      "readiness": args.readiness}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
