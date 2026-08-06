#!/usr/bin/env python
"""Development benchmark of best.pt on this Mac via MPS.

**Mac/MPS development benchmark only.** It says nothing about iPhone latency and
nothing about the Neural Engine — different silicon, different runtime, different
thermal envelope.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, platform, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="yolo26n_run_001")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    # Defaults reproduce the Run 001 invocation.
    ap.add_argument("--prepared", default="data/prepared")
    ap.add_argument("--prefix", default="pytorch_mps_benchmark_run_001")
    ap.add_argument("--label", default="Run 001")
    a = ap.parse_args()

    from ultralytics import YOLO
    best = ROOT / "runs" / a.run / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: {best} missing"); return 1

    model = YOLO(str(best))
    sample = next((ROOT / a.prepared / "images" / "test").iterdir())

    for _ in range(a.warmup):
        model.predict(str(sample), imgsz=a.imgsz, device=a.device, verbose=False)

    timings = []
    for _ in range(a.runs):
        start = time.perf_counter()
        model.predict(str(sample), imgsz=a.imgsz, device=a.device, verbose=False)
        timings.append((time.perf_counter() - start) * 1000)
    arr = np.array(timings)

    report = {
        "note": ("Mac/MPS development benchmark. NOT iPhone latency and NOT a Neural "
                 "Engine measurement — no on-device benchmark has been taken."),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "weights": str(best.relative_to(ROOT)),
        "weights_bytes": best.stat().st_size,
        "weights_mb": round(best.stat().st_size / 1e6, 3),
        "sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "device": a.device, "imgsz": a.imgsz, "batch": 1,
        "warmup_runs": a.warmup, "measured_runs": a.runs,
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "approx_fps": round(1000.0 / float(np.median(arr)), 2),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{a.prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORTS / f"{a.prefix}.md").write_text("\n".join([
        f"# PyTorch MPS benchmark — {a.label}", "",
        f"> {report['note']}", "",
        "| | |", "|---|---:|",
        f"| Device | {a.device} |", f"| Image size | {a.imgsz} |", "| Batch | 1 |",
        f"| Warm-up / measured runs | {a.warmup} / {a.runs} |",
        f"| Mean | {report['mean_ms']} ms |",
        f"| Median | {report['median_ms']} ms |",
        f"| p95 | {report['p95_ms']} ms |",
        f"| Approx FPS | {report['approx_fps']} |",
        f"| Weights | {report['weights_mb']} MB |",
        f"| SHA-256 | `{report['sha256']}` |", ""]) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("mean_ms", "median_ms", "p95_ms", "approx_fps", "weights_mb")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
