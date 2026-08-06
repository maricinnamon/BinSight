#!/usr/bin/env python
"""Resumes an interrupted Ultralytics run from its own last.pt.

Ultralytics' resume contract is deliberately minimal: hand it the checkpoint and
`resume=True`, and it restores weights, EMA, optimiser moments, the AMP scaler,
the LR schedule position, the epoch counter and the original `train_args` from
inside the file. Passing hyperparameters here would *override* what the run was
launched with, which is exactly what a resume must not do — so nothing is passed.

Wall-clock accounting needs care. Ultralytics' `time` column restarts at zero in
a resumed process, so total duration is the sum of segments, recorded separately
rather than silently added into one figure that looks like an uninterrupted run.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_results(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def fitness(row: dict) -> float:
    """Ultralytics' standard detection fitness: 0.1*mAP50 + 0.9*mAP50-95."""
    m50 = float(row.get("metrics/mAP50(B)", 0) or 0)
    m5095 = float(row.get("metrics/mAP50-95(B)", 0) or 0)
    return 0.1 * m50 + 0.9 * m5095


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run", required=True, help="report key, e.g. run_002")
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"ERROR: {checkpoint} not found")
        return 1

    import torch
    from ultralytics import YOLO

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed_before = int(ckpt["epoch"]) + 1
    stored = ckpt["train_args"]
    if stored.get("device") == "mps" and not torch.backends.mps.is_available():
        print("ERROR: the run was trained on MPS but MPS is unavailable. "
              "Not silently resuming on CPU.")
        return 1

    run_dir = checkpoint.parent.parent
    results_csv = run_dir / "results.csv"
    prior = read_results(results_csv)
    prior_seconds = float(prior[-1]["time"]) if prior else 0.0

    print(f"resuming {stored.get('name')} from epoch {completed_before}/{stored.get('epochs')}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  restored settings come from the checkpoint, not from this script")

    started = time.time()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    # `resume=True` and nothing else. Any other kwarg would override the very
    # settings the resume is supposed to preserve.
    model = YOLO(str(checkpoint))
    model.train(resume=True)
    segment_hours = (time.time() - started) / 3600

    rows = read_results(results_csv)
    epochs_completed = len(rows)
    best_epoch = (max(range(len(rows)), key=lambda i: fitness(rows[i])) + 1) if rows else None

    summary_path = REPORTS / f"training_{args.run}_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    segments = previous.get("segments", [])
    if not segments and prior_seconds:
        segments.append({
            "segment": 1,
            "epochs": f"1-{completed_before}",
            "hours": round(prior_seconds / 3600, 3),
            "note": "initial process, interrupted",
        })
    segments.append({
        "segment": len(segments) + 1,
        "epochs": f"{completed_before + 1}-{epochs_completed}",
        "hours": round(segment_hours, 3),
        "started_at": started_at,
        "note": "resumed from last.pt",
    })

    summary = {
        "run": stored.get("name"),
        "key": args.run,
        "config": "configs/train_run_002.yaml",
        "resumed": True,
        "resumed_from": str(checkpoint.relative_to(ROOT)),
        "epochs_completed_before_resume": completed_before,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "segments": segments,
        "training_hours": round(sum(s["hours"] for s in segments), 3),
        "training_hours_note": (
            "Sum of training segments. The run was interrupted after epoch "
            f"{completed_before} and resumed, so this is compute time, not uninterrupted "
            "wall clock. Run 001's 1.59 h was a single process; compare with that in mind."),
        "epochs_configured": stored.get("epochs"),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "early_stopped": epochs_completed < (stored.get("epochs") or 0),
        "patience": stored.get("patience"),
        "imgsz": stored.get("imgsz"),
        "batch": stored.get("batch"),
        "device": stored.get("device"),
        "seed": stored.get("seed"),
        "workers_effective": stored.get("workers"),
        "save_dir": str(run_dir.relative_to(ROOT)),
        "weights_best": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
