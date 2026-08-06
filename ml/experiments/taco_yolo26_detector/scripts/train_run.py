#!/usr/bin/env python
"""Trains one YOLO run from a committed config, and records what it did.

Reads a `configs/train_run_*.yaml`, passes only the keys that file names, and
lets every other Ultralytics default stand — the same discipline Run 001 used, so
the runs stay comparable. `project` is forced to an absolute path because a
relative one resolves against Ultralytics' own settings directory and lands the
run somewhere unexpected.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Keys that are ours rather than Ultralytics', or that we resolve ourselves.
LOCAL_KEYS = {"project", "data", "model"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="e.g. run_002")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text())
    kwargs = {k: v for k, v in config.items() if k not in LOCAL_KEYS}
    kwargs["data"] = str(ROOT / config["data"])
    kwargs["project"] = str(ROOT / config.get("project", "runs"))
    kwargs.pop("task", None)

    import torch
    from ultralytics import YOLO

    if kwargs.get("device") == "mps" and not torch.backends.mps.is_available():
        print("ERROR: device=mps requested but MPS is unavailable. "
              "Not falling back to CPU.")
        return 1

    print(f"model={config['model']}  data={kwargs['data']}")
    print(f"imgsz={kwargs.get('imgsz')} batch={kwargs.get('batch')} "
          f"device={kwargs.get('device')} epochs={kwargs.get('epochs')} "
          f"patience={kwargs.get('patience')}")

    started = time.time()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    model = YOLO(config["model"])
    model.train(**kwargs)
    hours = (time.time() - started) / 3600

    run_dir = pathlib.Path(model.trainer.save_dir)
    results_csv = run_dir / "results.csv"
    epochs_completed, best_epoch = None, None
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        epochs_completed = len(rows)
        # Ultralytics selects best on its own fitness; recompute the standard
        # 0.1*mAP50 + 0.9*mAP50-95 so the number is explainable rather than
        # implicit.
        def fitness(row: dict) -> float:
            m50 = float(row.get("metrics/mAP50(B)", 0) or 0)
            m5095 = float(row.get("metrics/mAP50-95(B)", 0) or 0)
            return 0.1 * m50 + 0.9 * m5095
        best_epoch = max(range(len(rows)), key=lambda i: fitness(rows[i])) + 1

    summary = {
        "run": config["name"],
        "key": args.run,
        "config": args.config,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "training_hours": round(hours, 3),
        "epochs_configured": kwargs.get("epochs"),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "early_stopped": (epochs_completed or 0) < (kwargs.get("epochs") or 0),
        "patience": kwargs.get("patience"),
        "imgsz": kwargs.get("imgsz"),
        "batch": kwargs.get("batch"),
        "device": kwargs.get("device"),
        "seed": kwargs.get("seed"),
        "save_dir": str(run_dir.relative_to(ROOT)),
        "weights_best": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"training_{args.run}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
