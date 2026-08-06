#!/usr/bin/env python
"""The disposable training child. Trains a bounded number of epochs, then exits.

This process is deliberately short-lived. The supervisor recycles it every few
epochs so that whatever accumulates in a long-lived MPS process is released back
to the OS on exit — observed behaviour on this host, not a diagnosed leak.

Two modes:
  --start   fresh training from yolo26n.pt (first child only)
  --resume  continue the same logical run from last.pt

`--stop-after-epoch N` is the recycling mechanism: an Ultralytics callback fires
at the end of each epoch, and once the target is reached the child raises to
unwind training cleanly *after* the checkpoint has been written. Killing mid-epoch
would discard that epoch's work; this does not.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]

# Keys that belong to us, not to Ultralytics.
LOCAL_KEYS = {"project", "data", "model", "task"}


class StopAfterEpoch(Exception):
    """Raised to unwind training cleanly once the epoch budget is spent."""


def memory_snapshot() -> dict:
    import subprocess
    out = {}
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                            text=True, timeout=10).stdout
        for part in sw.split():
            if part.startswith("used"):
                pass
        used = [p for p in sw.replace("=", " ").split() if p.endswith("M")]
        out["swap_used_mb"] = float(used[1].rstrip("M")) if len(used) > 1 else None
    except Exception:  # noqa: BLE001
        out["swap_used_mb"] = None
    try:
        import torch
        out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
    except Exception:  # noqa: BLE001
        out["mps_allocated_mb"] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_final.yaml")
    parser.add_argument("--mode", choices=["start", "resume"], required=True)
    parser.add_argument("--checkpoint", default=None, help="last.pt, for --resume")
    parser.add_argument("--stop-after-epoch", type=int, required=True,
                        help="absolute epoch index (1-based) after which to exit")
    parser.add_argument("--status", default=None, help="JSON status file to write")
    args = parser.parse_args()

    config = yaml.safe_load((HERE / args.config).read_text())
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect, Segment

    if config.get("task", "detect") != "detect":
        print(f"REFUSING: task is {config.get('task')!r}, must be 'detect'")
        return 2
    if config.get("device") == "mps" and not torch.backends.mps.is_available():
        print("REFUSING: device=mps requested but MPS unavailable; no CPU fallback")
        return 2

    if args.mode == "start":
        model = YOLO(config["model"])
        kwargs = {k: v for k, v in config.items() if k not in LOCAL_KEYS}
        kwargs["data"] = str(HERE / config["data"])
        kwargs["project"] = str(HERE / config.get("project", "runs"))
        kwargs["exist_ok"] = True
    else:
        checkpoint = pathlib.Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"REFUSING: checkpoint {checkpoint} not found")
            return 2
        model = YOLO(str(checkpoint))
        # resume=True and nothing else — any other kwarg would override the
        # settings the resume exists to restore.
        kwargs = {"resume": True}

    # --- detection-only gate, on the actual loaded model -------------------
    core = model.model
    head = core.model[-1]
    if not isinstance(core, DetectionModel):
        print(f"REFUSING: model is {type(core).__name__}, not DetectionModel")
        return 2
    if isinstance(head, Segment) or not isinstance(head, Detect):
        print(f"REFUSING: head is {type(head).__name__}, not Detect")
        return 2
    if getattr(model, "task", None) not in (None, "detect"):
        print(f"REFUSING: model task is {model.task!r}")
        return 2
    print(f"detect-only gate passed: {type(core).__name__} / {type(head).__name__}")

    # --- stop-after-epoch callback ----------------------------------------
    state = {"last_epoch": None, "stopped_by_budget": False}

    def on_epoch_end(trainer):
        # trainer.epoch is 0-based; +1 gives the count of completed epochs.
        completed = int(trainer.epoch) + 1
        state["last_epoch"] = completed
        if completed >= args.stop_after_epoch:
            state["stopped_by_budget"] = True
            print(f"\nepoch budget reached ({completed}); checkpoint written, exiting cleanly")
            raise StopAfterEpoch(completed)

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    started = time.time()
    before = memory_snapshot()
    outcome = "completed"
    try:
        model.train(**kwargs)
    except StopAfterEpoch:
        outcome = "recycled"
    except KeyboardInterrupt:
        outcome = "interrupted"
    finally:
        # Release what we can before the process exits. On MPS the meaningful
        # reclamation happens at process exit; this simply makes the before/after
        # numbers in the log honest.
        try:
            del model
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    after = memory_snapshot()
    status = {
        "outcome": outcome,
        "mode": args.mode,
        "last_completed_epoch": state["last_epoch"],
        "stop_after_epoch": args.stop_after_epoch,
        "elapsed_seconds": round(time.time() - started, 1),
        "memory_before_cleanup": before,
        "memory_after_cleanup": after,
    }
    if args.status:
        pathlib.Path(args.status).write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
