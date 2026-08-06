#!/usr/bin/env python
"""Measures YOLO26's two inference heads on an already-trained checkpoint.

YOLO26 trains both branches at once and keeps both in the checkpoint, so this is
an evaluation-time switch, not a retrain:

* **one-to-one (end2end, default)** — the head emits final boxes directly. No NMS,
  which makes it markedly simpler to export and to run on a phone.
* **one-to-many (`end2end=False`)** — the classic dense head, which needs NMS
  afterwards. Often slightly stronger on metrics, at the cost of post-processing
  that has to be reimplemented on the deployment target.

The point is to choose from measurements rather than from the usual claim that
one-to-many "should" score higher. Nothing is retrained here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "comparisons"

HEADS = [("one_to_one_end2end", True), ("one_to_many_nms", False)]


def measure_latency(model, sample, imgsz, device, head_end2end, warmup=15, runs=100):
    for _ in range(warmup):
        model.predict(str(sample), imgsz=imgsz, device=device, verbose=False,
                      end2end=head_end2end)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(str(sample), imgsz=imgsz, device=device, verbose=False,
                      end2end=head_end2end)
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return {
        "mean_ms": round(statistics.fmean(timings), 3),
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(timings[int(0.95 * (len(timings) - 1))], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                        help="comma-separated run_key:run_dir:imgsz:dataset_yaml entries")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--latency-runs", type=int, default=100)
    args = parser.parse_args()

    from ultralytics import YOLO

    records = []
    for spec in args.candidates.split(","):
        key, run_dir, imgsz, dataset = spec.split(":")
        imgsz = int(imgsz)
        best = ROOT / "runs" / run_dir / "weights" / "best.pt"
        if not best.exists():
            print(f"skipping {key}: {best} missing")
            continue
        prepared = pathlib.Path(
            __import__("yaml").safe_load((ROOT / dataset).read_text())["path"])
        sample = next((prepared / "images" / "test").iterdir())

        for head_name, end2end in HEADS:
            row = {"candidate": key, "run": run_dir, "imgsz": imgsz, "head": head_name,
                   "end2end": end2end}
            for split in ("val", "test"):
                try:
                    model = YOLO(str(best))   # reload: val mutates model state
                    m = model.val(data=str(ROOT / dataset), split=split, imgsz=imgsz,
                                  device=args.device, plots=False, verbose=False,
                                  end2end=end2end,
                                  project=str(ROOT / "runs" / "head_comparison"),
                                  name=f"{key}_{head_name}_{split}", exist_ok=True)
                    p, r = float(m.box.mp), float(m.box.mr)
                    row[split] = {
                        "precision": round(p, 6), "recall": round(r, 6),
                        "f1": round(2 * p * r / (p + r), 6) if (p + r) else 0.0,
                        "map50": round(float(m.box.map50), 6),
                        "map50_95": round(float(m.box.map), 6),
                    }
                except Exception as exc:  # noqa: BLE001
                    row[split] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"  {key}/{head_name}/{split} FAILED: {exc}")
            try:
                model = YOLO(str(best))
                row["latency"] = measure_latency(model, sample, imgsz, args.device,
                                                 end2end, runs=args.latency_runs)
            except Exception as exc:  # noqa: BLE001
                row["latency"] = {"error": f"{type(exc).__name__}: {exc}"}
            records.append(row)
            summary = row.get("test", {})
            print(f"{key:<10} {head_name:<20} test mAP50-95="
                  f"{summary.get('map50_95', 'ERR')} "
                  f"p50={row.get('latency', {}).get('median_ms', 'ERR')} ms")

    if not records:
        print("ERROR: no candidates evaluated")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yolo26_head_comparison.json").write_text(json.dumps(records, indent=2) + "\n")

    md = [
        "# YOLO26 inference heads — measured, not assumed", "",
        "YOLO26 trains a one-to-one branch and a one-to-many branch together and keeps "
        "both in the checkpoint, so this comparison is an evaluation-time switch on the "
        "**same weights**. Nothing was retrained.", "",
        "| Head | What it costs | What it buys |", "|---|---|---|",
        "| `one_to_one_end2end` (default) | nothing extra | no NMS — a much simpler "
        "CoreML/TFLite export and no post-processing to reimplement on-device |",
        "| `one_to_many_nms` (`end2end=False`) | NMS after every inference | usually a "
        "small metric gain |", "",
        "## Results", "",
        "| Candidate | imgsz | Head | Split | P | R | F1 | mAP50 | mAP50-95 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        for split in ("val", "test"):
            s = row.get(split, {})
            if "error" in s:
                md.append(f"| {row['candidate']} | {row['imgsz']} | `{row['head']}` "
                          f"| {split} | — | — | — | — | failed: {s['error']} |")
                continue
            md.append(
                f"| {row['candidate']} | {row['imgsz']} | `{row['head']}` | {split} "
                f"| {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} "
                f"| {s['map50']:.4f} | {s['map50_95']:.4f} |")
    md += ["", "## Latency — Mac MPS, batch 1", "",
           "Not iPhone latency. Useful only for comparing the two heads on identical "
           "hardware.", "",
           "| Candidate | Head | Mean | Median | p95 |", "|---|---|---:|---:|---:|"]
    for row in records:
        lat = row.get("latency", {})
        if "error" in lat:
            md.append(f"| {row['candidate']} | `{row['head']}` | — | — | {lat['error']} |")
        else:
            md.append(f"| {row['candidate']} | `{row['head']}` | {lat['mean_ms']} ms "
                      f"| {lat['median_ms']} ms | {lat['p95_ms']} ms |")

    # A recommendation, stated from the measured gap rather than from theory.
    best_rows = [r for r in records if "error" not in r.get("test", {})]
    if best_rows:
        by_candidate: dict[str, dict] = {}
        for row in best_rows:
            by_candidate.setdefault(row["candidate"], {})[row["head"]] = row
        md += ["", "## Reading it", ""]
        for candidate, heads in by_candidate.items():
            one = heads.get("one_to_one_end2end")
            many = heads.get("one_to_many_nms")
            if not (one and many):
                continue
            gap = many["test"]["map50_95"] - one["test"]["map50_95"]
            recall_gap = many["test"]["recall"] - one["test"]["recall"]
            verdict = ("the one-to-many head is not worth the NMS: it does not lead"
                       if gap <= 0.005 else
                       "the one-to-many head leads by enough to be worth considering")
            md += [f"**{candidate}** — one-to-many minus one-to-one on test: "
                   f"mAP50-95 {gap:+.4f}, recall {recall_gap:+.4f}. On this evidence "
                   f"{verdict}.", ""]
        md += ["For a phone target the NMS-free head is preferred unless the "
               "one-to-many gap is large, because NMS has to be reimplemented and "
               "matched exactly on the deployment runtime — a real source of "
               "train/deploy divergence, not just extra milliseconds.", ""]
    (OUT / "yolo26_head_comparison.md").write_text("\n".join(md) + "\n")
    print("wrote", (OUT / "yolo26_head_comparison.md").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
