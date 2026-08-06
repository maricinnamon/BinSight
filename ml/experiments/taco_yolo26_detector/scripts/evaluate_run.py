#!/usr/bin/env python
"""Evaluates a trained run on the validation and test splits.

Detection metrics only — precision, recall, mAP50, mAP50-95, per-class AP. This
model localises and classifies, so a single "accuracy" figure would not describe
it and is deliberately not produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"


def collect(result, classes: list[str]) -> dict:
    """Pulls the metrics out of an Ultralytics DetMetrics object."""
    box = result.box
    overall = {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map50_95": float(box.map),
    }
    overall["f1"] = (2 * overall["precision"] * overall["recall"] /
                     (overall["precision"] + overall["recall"])) \
        if (overall["precision"] + overall["recall"]) else 0.0
    try:
        overall["map75"] = float(box.map75)
    except Exception:
        overall["map75"] = None

    per_class = []
    # `ap_class_index` lists which class ids actually had instances.
    for position, class_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(position)
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class.append({
            "class_id": int(class_id), "class": classes[int(class_id)],
            "precision": round(float(p), 6), "recall": round(float(r), 6),
            "f1": round(float(f1), 6), "ap50": round(float(ap50), 6),
            "ap50_95": round(float(ap), 6),
        })
    per_class.sort(key=lambda row: row["class_id"])
    return {"overall": {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in overall.items()},
            "per_class": per_class}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="yolo26n_run_001")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    # Defaults reproduce the Run 001 invocation, so its reports keep their paths.
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--prefix", default="test_metrics_run_001")
    parser.add_argument("--label", default="Run 001")
    parser.add_argument("--per-class-csv", default=None,
                        help="optional extra per-class-only CSV")
    args = parser.parse_args()

    reports = ROOT / args.out_dir

    from ultralytics import YOLO

    run_dir = ROOT / "runs" / args.run
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: {best} not found.")
        return 1

    classes = (CONFIGS / "classes.txt").read_text().split()
    data = str(ROOT / args.data)
    results: dict = {"run": args.run, "weights": str(best), "imgsz": args.imgsz}

    for split in ("val", "test"):
        model = YOLO(str(best))       # reload: validation mutates model state
        metrics = model.val(data=data, split=split, imgsz=args.imgsz,
                            device=args.device, plots=True, verbose=False,
                            project=str(ROOT / "runs"),
                            name=f"{args.run}_{split}", exist_ok=True)
        results[split] = collect(metrics, classes)
        overall = results[split]["overall"]
        print(f"{split:<5} P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"mAP50={overall['map50']:.4f} mAP50-95={overall['map50_95']:.4f}")

    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.prefix}.json").write_text(json.dumps(results, indent=2) + "\n")

    rows = []
    for split in ("val", "test"):
        for row in results[split]["per_class"]:
            rows.append({"split": split, **row})
    with open(reports / f"{args.prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [f"# {args.label} — detection metrics", "",
          "Object-detection metrics. There is no single 'accuracy' figure here on",
          "purpose: this model localises *and* classifies, and a classification-style",
          "accuracy would describe neither job.", "",
          "## Overall", "",
          "| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("val", "test"):
        o = results[split]["overall"]
        m75 = f"{o['map75']:.4f}" if o.get("map75") is not None else "—"
        md.append(f"| {split} | {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} "
                  f"| {o['map50']:.4f} | {m75} | {o['map50_95']:.4f} |")
    for split in ("val", "test"):
        md += ["", f"## Per class — {split}", "",
               "| ID | Class | Precision | Recall | F1 | AP50 | AP50-95 |",
               "|---:|---|---:|---:|---:|---:|---:|"]
        for row in results[split]["per_class"]:
            md.append(f"| {row['class_id']} | {row['class']} | {row['precision']:.4f} "
                      f"| {row['recall']:.4f} | {row['f1']:.4f} | {row['ap50']:.4f} "
                      f"| {row['ap50_95']:.4f} |")
    (reports / f"{args.prefix}.md").write_text("\n".join(md) + "\n")

    if args.per_class_csv:
        path = ROOT / args.per_class_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        # Instance counts per class come from the labels, not from the metrics
        # object — Ultralytics reports them per split and they belong here.
        import collections
        prepared = pathlib.Path(yaml.safe_load(open(data))["path"])
        counts = {}
        for split in ("val", "test"):
            counter = collections.Counter()
            for label in (prepared / "labels" / split).glob("*.txt"):
                for line in label.read_text().splitlines():
                    if line.strip():
                        counter[int(line.split()[0])] += 1
            counts[split] = counter
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "split", "class_id", "class", "instances", "precision", "recall",
                "f1", "ap50", "ap50_95"])
            writer.writeheader()
            for split in ("val", "test"):
                for row in results[split]["per_class"]:
                    writer.writerow({"split": split,
                                     "instances": counts[split][row["class_id"]],
                                     **row})
        print("per-class ->", path)

    print("reports ->", reports / f"{args.prefix}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
