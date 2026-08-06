#!/usr/bin/env python
"""Lightweight supervisor that recycles the training child on a schedule.

Four training processes died to SIGKILL in the archived experiments, each time
mid-batch with no traceback — the signature of macOS reclaiming memory from a
long-lived process on an ~8.6 GB machine. This is stated as observed host
behaviour; no PyTorch or MPS leak was ever diagnosed, and none is claimed.

The design turns that unpredictable kill into a scheduled, cheap event:

    child trains N epochs -> checkpoint written -> child exits cleanly
      -> supervisor verifies last.pt loads and the epoch advanced
      -> brief pause for the OS to reclaim
      -> new child resumes from the SAME last.pt with resume=True

It remains ONE logical run in ONE run directory. No `run2`, no `_resume` suffix,
no restart from pretrained weights. `results.csv` continues across recycles.

The supervisor itself imports no torch and holds no model — it stays small so it
is never the process the kernel decides to reclaim.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parents[1]
REPORTS = HERE / "reports"
PYTHON = sys.executable

# Swap thresholds, in MB. Host-operational safeguards for this machine — not
# universal ML advice.
SWAP_WARNING_MB = 2048
SWAP_EARLY_RECYCLE_MB = 3584
SWAP_CRITICAL_MB = 5120

MAX_UNEXPECTED_KILLS = 3
RECLAIM_PAUSE_SECONDS = 20


def swap_used_mb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", out)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def memory_pressure() -> str | None:
    """macOS memory_pressure, read-only, no sudo."""
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
        return f"{m.group(1)}% free" if m else out.strip().splitlines()[-1][:60]
    except Exception:  # noqa: BLE001
        return None


def free_memory_gb() -> float | None:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        vals = {}
        for line in out.splitlines():
            m = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
            if m:
                vals[m.group(1)] = int(m.group(2))
        return round(sum(vals.values()) * 16384 / 1e9, 2) if vals else None
    except Exception:  # noqa: BLE001
        return None


def rss_mb(pid: int) -> float | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return round(int(out) / 1024, 1) if out else None
    except Exception:  # noqa: BLE001
        return None


class MemoryLog:
    FIELDS = ["timestamp", "event", "epoch", "child_pid", "child_rss_mb",
              "swap_used_mb", "free_plus_inactive_gb", "memory_pressure",
              "checkpoint_state"]

    def __init__(self, path: pathlib.Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as h:
                csv.DictWriter(h, fieldnames=self.FIELDS).writeheader()

    def write(self, event: str, epoch=None, pid=None, checkpoint="") -> dict:
        row = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "event": event, "epoch": epoch, "child_pid": pid,
            "child_rss_mb": rss_mb(pid) if pid else None,
            "swap_used_mb": swap_used_mb(),
            "free_plus_inactive_gb": free_memory_gb(),
            "memory_pressure": memory_pressure(),
            "checkpoint_state": checkpoint,
        }
        with open(self.path, "a", newline="") as h:
            csv.DictWriter(h, fieldnames=self.FIELDS).writerow(row)
        return row


def completed_epochs(results_csv: pathlib.Path) -> int:
    if not results_csv.exists():
        return 0
    with open(results_csv) as h:
        return sum(1 for _ in csv.DictReader(h))


def verify_checkpoint(path: pathlib.Path) -> tuple[bool, dict]:
    """Loads last.pt in a throwaway process — the supervisor never imports torch."""
    probe = (
        "import json,sys,torch;"
        "ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
        "print(json.dumps({'epoch':int(ck.get('epoch',-1)),"
        "'has_optimizer':ck.get('optimizer') is not None,"
        "'has_ema':ck.get('ema') is not None,"
        "'has_scaler':ck.get('scaler') is not None,"
        "'has_train_args':bool(ck.get('train_args')),"
        "'best_fitness':float(ck.get('best_fitness') or 0),"
        "'nc':int(getattr(ck.get('ema') or ck.get('model'),'nc',-1))}))"
    )
    try:
        out = subprocess.run([PYTHON, "-c", probe, str(path)],
                             capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            return False, {"error": out.stderr.strip()[-300:]}
        return True, json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return False, {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_final.yaml")
    parser.add_argument("--total-epochs", type=int, default=60)
    parser.add_argument("--epochs-per-child", type=int, default=5)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--memory-log", default="reports/memory_log.csv")
    parser.add_argument("--state", default="reports/supervisor_state.json")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load((HERE / args.config).read_text())
    run_name = args.run_name or config["name"]
    run_dir = HERE / config.get("project", "runs") / run_name
    results_csv = run_dir / "results.csv"
    last_pt = run_dir / "weights" / "last.pt"

    log = MemoryLog(HERE / args.memory_log)
    state = {
        "run": run_name, "run_dir": str(run_dir.relative_to(HERE)),
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "total_epochs": args.total_epochs, "epochs_per_child": args.epochs_per_child,
        "children": [], "planned_recycles": 0, "memory_recycles": 0,
        "unexpected_kills": 0, "max_swap_mb": 0.0, "outcome": None,
    }

    def save_state():
        (HERE / args.state).write_text(json.dumps(state, indent=2) + "\n")

    log.write("supervisor_start")
    print(f"supervisor: run={run_name} total={args.total_epochs} "
          f"per_child={args.epochs_per_child}")

    while True:
        done = completed_epochs(results_csv)
        if done >= args.total_epochs:
            state["outcome"] = "epoch_limit_reached"
            break

        fresh = done == 0 and not last_pt.exists()
        target = min(done + args.epochs_per_child, args.total_epochs)
        status_file = HERE / "reports" / ".child_status.json"
        cmd = [PYTHON, str(HERE / "scripts" / "train_child.py"),
               "--config", args.config,
               "--stop-after-epoch", str(target),
               "--status", str(status_file)]
        cmd += ["--mode", "start"] if fresh else ["--mode", "resume",
                                                  "--checkpoint", str(last_pt)]

        child_log = HERE / "reports" / f"child_{len(state['children']) + 1:03d}.log"
        print(f"\n--- child {len(state['children'])+1}: epochs {done+1}..{target} "
              f"({'start' if fresh else 'resume'}) ---")
        with open(child_log, "w") as sink:
            child = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT,
                                     cwd=str(HERE))
        entry = {"index": len(state["children"]) + 1, "pid": child.pid,
                 "mode": "start" if fresh else "resume",
                 "epochs_from": done + 1, "epochs_to": target,
                 "log": str(child_log.relative_to(HERE))}
        log.write("child_start", epoch=done, pid=child.pid,
                  checkpoint="fresh" if fresh else "resume")

        early_recycle = False
        while child.poll() is None:
            time.sleep(args.poll_seconds)
            row = log.write("poll", epoch=completed_epochs(results_csv), pid=child.pid,
                            checkpoint="training")
            swap = row["swap_used_mb"] or 0.0
            state["max_swap_mb"] = max(state["max_swap_mb"], swap)
            if swap >= SWAP_CRITICAL_MB:
                # Do not wait to be SIGKILLed mid-epoch. Stop the child ourselves;
                # the last COMPLETED epoch's checkpoint stays canonical.
                print(f"  swap {swap:.0f} MB >= critical {SWAP_CRITICAL_MB} — "
                      "terminating child now")
                log.write("critical_memory_terminate", pid=child.pid, checkpoint="terminating")
                child.terminate()
                early_recycle = True
                break
            if swap >= SWAP_EARLY_RECYCLE_MB and not early_recycle:
                print(f"  swap {swap:.0f} MB >= early-recycle {SWAP_EARLY_RECYCLE_MB} — "
                      "will recycle after this epoch")
                log.write("early_recycle_planned", pid=child.pid, checkpoint="training")
                early_recycle = True
            elif swap >= SWAP_WARNING_MB:
                log.write("swap_warning", pid=child.pid, checkpoint="training")

        try:
            child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=60)
        rc = child.returncode
        entry["returncode"] = rc
        status = {}
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text())
                status_file.unlink()
            except Exception:  # noqa: BLE001
                pass
        entry["child_status"] = status

        after = completed_epochs(results_csv)
        entry["epochs_completed_after"] = after
        log.write("child_exit", epoch=after, checkpoint=f"rc={rc}")

        # rc<0 means killed by a signal. Our own terminate() is expected; anything
        # else on this host is the SIGKILL pattern the design exists to survive.
        if rc is not None and rc < 0 and not early_recycle:
            state["unexpected_kills"] += 1
            entry["classification"] = f"unexpected_signal_{-rc}"
            print(f"  UNEXPECTED: child died on signal {-rc} "
                  f"({state['unexpected_kills']}/{MAX_UNEXPECTED_KILLS})")
            log.write("unexpected_kill", epoch=after, checkpoint=f"signal {-rc}")
        elif early_recycle:
            state["memory_recycles"] += 1
            entry["classification"] = "memory_triggered_recycle"
        elif rc == 0 and after < args.total_epochs:
            state["planned_recycles"] += 1
            entry["classification"] = "planned_recycle"
        elif rc == 0:
            entry["classification"] = "final_child"
        else:
            entry["classification"] = f"error_rc_{rc}"

        state["children"].append(entry)
        save_state()

        if rc not in (0, None) and rc > 0 and entry["classification"].startswith("error"):
            print(f"  child exited {rc} — see {child_log}")
            state["outcome"] = f"child_error_rc_{rc}"
            break
        if state["unexpected_kills"] >= MAX_UNEXPECTED_KILLS:
            state["outcome"] = "aborted_repeated_unexpected_kills"
            print("  three unexpected kills — the memory-safe design is not holding. "
                  "Stopping rather than looping.")
            break
        if after == 0:
            state["outcome"] = "no_epoch_completed"
            print("  child produced no completed epoch — stopping")
            break

        # Early stopping inside Ultralytics: a child that exited 0 without
        # reaching its epoch budget and without advancing means patience fired.
        if rc == 0 and status.get("outcome") == "completed" and after < args.total_epochs:
            state["outcome"] = "early_stopping"
            print(f"  Ultralytics stopped at epoch {after} (patience)")
            break

        ok, info = verify_checkpoint(last_pt)
        entry["checkpoint_verified"] = ok
        entry["checkpoint_info"] = info
        save_state()
        if not ok:
            state["outcome"] = "checkpoint_unverifiable"
            print(f"  last.pt did not load: {info}")
            break
        print(f"  checkpoint ok: epoch={info.get('epoch')} "
              f"optimizer={info.get('has_optimizer')} ema={info.get('has_ema')} "
              f"nc={info.get('nc')}")

        log.write("reclaim_pause", epoch=after, checkpoint="between_children")
        time.sleep(RECLAIM_PAUSE_SECONDS)
        log.write("reclaim_done", epoch=after, checkpoint="between_children")

    state["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    state["epochs_completed"] = completed_epochs(results_csv)
    save_state()
    log.write("supervisor_end", epoch=state["epochs_completed"],
              checkpoint=state["outcome"] or "")
    print(f"\nsupervisor finished: {state['outcome']}, "
          f"{state['epochs_completed']} epochs, "
          f"{state['planned_recycles']} planned / {state['memory_recycles']} memory "
          f"recycles, {state['unexpected_kills']} unexpected kills, "
          f"max swap {state['max_swap_mb']:.0f} MB")
    return 0 if state["outcome"] in ("epoch_limit_reached", "early_stopping") else 1


if __name__ == "__main__":
    sys.exit(main())
