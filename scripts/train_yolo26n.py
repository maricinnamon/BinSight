#!/usr/bin/env python3
"""Fine-tunes the official pretrained YOLO26n **detection** checkpoint on BinSight's 3 classes.

This is transfer learning, not training from scratch. The model is constructed
from `yolo26n.pt` — real COCO-pretrained weights — never from `yolo26n.yaml`,
which would build the architecture with random initialisation. The script asserts
this before a single batch runs: it checks the checkpoint dict is present, the
head is `Detect`, and the BatchNorm running statistics are non-trivial, because
an untrained model has exactly zero means and unit variances.

Ultralytics adapts the 80-class COCO head to our `nc=3` itself. The backbone is
deliberately not frozen — generic visual features still benefit from adapting to
close-up waste photography.

Memory discipline for a ~8.6 GB machine: `workers=0`, `cache=False`, batch 2,
nothing accumulated in Python-side lists, and one `gc.collect()` +
`torch.mps.empty_cache()` per epoch — per epoch, not per batch, so the sweep
never becomes the bottleneck. Memory is sampled once per epoch to CSV.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIG = dict(
    data="datasets/binsight_waste_3class/data.yaml",
    device="mps",
    imgsz=640,
    batch=2,
    epochs=60,
    patience=12,
    workers=0,
    cache=False,
    seed=42,
    mosaic=0.3,
    close_mosaic=10,
    save=True,
    save_period=1,
    plots=True,
    verbose=True,
    project="runs/detect",
    name="binsight_yolo26n_pretrained_3class",
    exist_ok=True,
)
PRETRAINED = "yolo26n.pt"
EXPECTED_NAMES = {0: "paper", 1: "plastic", 2: "metal"}


def memory_sample() -> dict:
    """Cheap snapshot. psutil for the process, sysctl for swap."""
    import psutil
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    proc = psutil.Process()
    return {
        "process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "ram_used_gb": round((vm.total - vm.available) / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "swap_used_gb": round(swap.used / 1e9, 2),
    }


def assert_pretrained(model) -> dict:
    """Refuses to train unless these really are trained detection weights."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect, Segment

    core = model.model
    head = core.model[-1]
    failures = []

    if model.ckpt is None:
        failures.append("no checkpoint dict — model was built from YAML, not .pt weights")
    if not str(getattr(model, "ckpt_path", "")).endswith(".pt"):
        failures.append(f"ckpt_path is not a .pt file: {getattr(model, 'ckpt_path', None)!r}")
    if not isinstance(core, DetectionModel):
        failures.append(f"model is {type(core).__name__}, not DetectionModel")
    if isinstance(head, Segment) or not isinstance(head, Detect):
        failures.append(f"head is {type(head).__name__}, not Detect")
    if model.task != "detect":
        failures.append(f"task is {model.task!r}, not 'detect'")

    # Trained BatchNorm statistics are the hard evidence. A freshly initialised
    # model has running_mean exactly 0 and running_var exactly 1 everywhere.
    bns = [m for m in core.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    trained_bn = any(
        float(b.running_mean.abs().max()) > 0 and not bool((b.running_var == 1).all())
        for b in bns
    )
    if not trained_bn:
        failures.append("BatchNorm statistics are untrained — these are random weights")

    info = {
        "checkpoint": str(getattr(model, "ckpt_path", PRETRAINED)),
        "model_class": type(core).__name__,
        "head_class": type(head).__name__,
        "task": model.task,
        "checkpoint_dict_present": model.ckpt is not None,
        "pretrained_class_count": core.yaml.get("nc"),
        "pretrained_sample_names": [model.names[i] for i in list(model.names)[:5]],
        "parameters": sum(p.numel() for p in core.parameters()),
        "batchnorm_layers_trained": trained_bn,
        "from_scratch": False,
    }
    if failures:
        for f in failures:
            print(f"  REFUSING: {f}")
        raise SystemExit(2)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=CONFIG["batch"],
                        help="only lower this if memory conclusively forces it")
    parser.add_argument("--resume", default=None,
                        help="path to last.pt, to continue an interrupted run")
    parser.add_argument("--name", default=CONFIG["name"],
                        help="run directory name; encode the batch actually used")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--close-mosaic", type=int, default=CONFIG["close_mosaic"])
    parser.add_argument("--weights", default=PRETRAINED,
                        help="starting weights. Defaults to the COCO-pretrained "
                             "yolo26n.pt; pass an already-fine-tuned BinSight "
                             "checkpoint to continue training it further.")
    parser.add_argument("--lr0", type=float, default=None,
                        help="initial learning rate. Leave unset for Ultralytics' "
                             "auto choice; lower it when continuing from an "
                             "already-converged checkpoint so the first epochs do "
                             "not undo what the model has already learned.")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.backends.mps.is_available():
        print("REFUSING: device=mps requested but MPS is unavailable. No CPU fallback.")
        return 2

    run_dir = ROOT / CONFIG["project"] / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mem_csv = run_dir / "memory_log.csv"
    fields = ["timestamp", "epoch", "process_rss_mb", "ram_used_gb",
              "ram_available_gb", "swap_used_gb", "note"]
    if not mem_csv.exists():
        with open(mem_csv, "w", newline="") as h:
            csv.DictWriter(h, fieldnames=fields).writeheader()

    def log_memory(epoch, note=""):
        row = {"timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
               "epoch": epoch, "note": note, **memory_sample()}
        with open(mem_csv, "a", newline="") as h:
            csv.DictWriter(h, fieldnames=fields).writerow(row)
        return row

    baseline = log_memory(0, "before training")
    print(f"baseline memory: {baseline}")

    if args.resume:
        model = YOLO(args.resume)
        kwargs = {"resume": True}
        print(f"resuming from {args.resume}")
        pretrained_info = {"resumed_from": args.resume, "from_scratch": False}
    else:
        # THE critical line: .pt weights, never .yaml.
        model = YOLO(args.weights)
        pretrained_info = assert_pretrained(model)
        print("pretrained verification passed:")
        for k, v in pretrained_info.items():
            print(f"  {k}: {v}")
        kwargs = {k: v for k, v in CONFIG.items() if k not in ("data", "project")}
        kwargs["batch"] = args.batch
        kwargs["name"] = args.name
        kwargs["epochs"] = args.epochs
        kwargs["patience"] = args.patience
        kwargs["close_mosaic"] = args.close_mosaic
        if args.lr0 is not None:
            # `optimizer="auto"` overrides lr0 outright — Ultralytics logs
            # "ignoring 'lr0=...'" and picks its own. Pinning the optimizer is
            # what makes the flag actually take effect.
            kwargs["lr0"] = args.lr0
            kwargs["optimizer"] = "AdamW"
        kwargs["data"] = str(ROOT / CONFIG["data"])
        kwargs["project"] = str(ROOT / CONFIG["project"])

    peak = {"rss": baseline["process_rss_mb"], "swap": baseline["swap_used_gb"]}

    def on_epoch_end(trainer):
        epoch = int(trainer.epoch) + 1
        # Once per epoch, not per batch: enough to stop drift, cheap enough to ignore.
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        row = log_memory(epoch, "after epoch + cleanup")
        peak["rss"] = max(peak["rss"], row["process_rss_mb"])
        peak["swap"] = max(peak["swap"], row["swap_used_gb"])

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    started = time.time()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    model.train(**kwargs)
    hours = (time.time() - started) / 3600

    results_csv = run_dir / "results.csv"
    epochs_done, best_epoch = 0, None
    if results_csv.exists():
        rows = list(csv.DictReader(open(results_csv)))
        epochs_done = len(rows)
        def fitness(r):
            return (0.1 * float(r.get("metrics/mAP50(B)", 0) or 0)
                    + 0.9 * float(r.get("metrics/mAP50-95(B)", 0) or 0))
        if rows:
            best_epoch = max(range(len(rows)), key=lambda i: fitness(rows[i])) + 1

    summary = {
        "training_type": "transfer learning / fine-tuning from official pretrained "
                         "yolo26n.pt detection weights — NOT trained from scratch",
        "starting_weights": args.weights,
        "pretrained": pretrained_info,
        "config": {**{k: v for k, v in CONFIG.items()}, "batch": args.batch,
                   "name": args.name, "epochs": args.epochs,
                   "patience": args.patience, "close_mosaic": args.close_mosaic},
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "training_hours": round(hours, 3),
        "epochs_configured": args.epochs,
        "epochs_completed": epochs_done,
        "best_epoch": best_epoch,
        "early_stopped": epochs_done < args.epochs,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "best_pt": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
        "last_pt": str((run_dir / "weights" / "last.pt").relative_to(ROOT)),
        "memory": {"baseline": baseline, "peak_process_rss_mb": peak["rss"],
                   "peak_swap_gb": peak["swap"], "log": str(mem_csv.relative_to(ROOT))},
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
