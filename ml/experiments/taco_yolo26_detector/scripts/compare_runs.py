#!/usr/bin/env python
"""Compares two or more trained runs on identical metrics, and builds the ablation table.

Everything here is read from persisted reports, never recomputed from memory, so
a number in the comparison can always be traced back to the file that produced
it. Missing inputs are reported as missing rather than filled in.

A caveat carried through the output: Run 001 and Run 002 were scored on
*different* test splits — recovery enlarged the pool and raised the small-object
share from 29.1% to 38.4%. The deltas are still worth having, but they measure
"model + data together", not the model alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "comparisons"

OVERALL_KEYS = ["precision", "recall", "f1", "map50", "map75", "map50_95"]
BUCKETS = ["tiny", "small", "medium", "large"]


def load(run_key: str) -> dict | None:
    """Gathers one run's persisted evidence. Returns None if it was never trained."""
    if run_key == "run_001":
        metrics = REPORTS / "test_metrics_run_001.json"
        summary = REPORTS / "training_run_001_summary.json"
        sizes = REPORTS / "object_size_performance_run_001.csv"
        manifest = None
    else:
        metrics = REPORTS / run_key / "test_metrics.json"
        summary = REPORTS / f"training_{run_key}_summary.json"
        sizes = REPORTS / run_key / "object_size_performance.csv"
        manifest = REPORTS / f"{run_key}_pretraining_manifest.json"
    if not metrics.exists():
        return None

    data = json.loads(metrics.read_text())
    entry = {
        "key": run_key,
        "run": data.get("run", run_key),
        "imgsz": data.get("imgsz"),
        "val": data["val"]["overall"],
        "test": data["test"]["overall"],
        "per_class_test": {r["class"]: r for r in data["test"]["per_class"]},
        "per_class_val": {r["class"]: r for r in data["val"]["per_class"]},
    }
    if summary.exists():
        s = json.loads(summary.read_text())
        entry["training"] = {
            "epochs_completed": s.get("epochs_completed"),
            "best_epoch": s.get("best_epoch"),
            "training_hours": s.get("training_hours"),
            "batch": s.get("batch"),
            "imgsz": s.get("imgsz", entry["imgsz"]),
        }
        # Run 001's summary carries its own size recall; later runs use the CSV.
        if "size_recall" in s and not sizes.exists():
            entry["sizes"] = {r["bucket"]: r for r in s["size_recall"]}
    if sizes.exists():
        entry["sizes"] = {r["bucket"]: {k: (float(v) if k in ("recall",) else v)
                                        for k, v in r.items()}
                          for r in csv.DictReader(open(sizes))}
    if manifest and manifest.exists():
        m = json.loads(manifest.read_text())
        entry["dataset"] = {
            "train_images": m["dataset_counts"]["train"]["images"],
            "train_boxes": m["dataset_counts"]["train"]["boxes"],
            "total_images": sum(v["images"] for v in m["dataset_counts"].values()),
            "total_boxes": sum(v["boxes"] for v in m["dataset_counts"].values()),
        }
    return entry


def delta(new: float | None, old: float | None) -> tuple[str, str]:
    if new is None or old is None:
        return "—", "—"
    absolute = new - old
    relative = (absolute / old * 100) if old else float("inf")
    rel = "—" if old == 0 else f"{relative:+.1f}%"
    return f"{absolute:+.4f}", rel


def pairwise_report(a: dict, b: dict, path_stem: pathlib.Path, caveat: str) -> None:
    """Writes the A-vs-B comparison in CSV and Markdown."""
    rows = []
    for split in ("val", "test"):
        for key in OVERALL_KEYS:
            old, new = a[split].get(key), b[split].get(key)
            abs_d, rel_d = delta(new, old)
            rows.append({"scope": f"overall/{split}", "metric": key,
                         a["key"]: old, b["key"]: new,
                         "absolute_delta": abs_d, "relative_delta": rel_d})
    for bucket in BUCKETS:
        old = a.get("sizes", {}).get(bucket, {}).get("recall")
        new = b.get("sizes", {}).get(bucket, {}).get("recall")
        old = float(old) if old is not None else None
        new = float(new) if new is not None else None
        abs_d, rel_d = delta(new, old)
        rows.append({"scope": "size_recall/test", "metric": bucket,
                     a["key"]: old, b["key"]: new,
                     "absolute_delta": abs_d, "relative_delta": rel_d})
    classes = sorted(set(a["per_class_test"]) | set(b["per_class_test"]))
    for name in classes:
        for metric in ("ap50_95", "recall"):
            old = a["per_class_test"].get(name, {}).get(metric)
            new = b["per_class_test"].get(name, {}).get(metric)
            abs_d, rel_d = delta(new, old)
            rows.append({"scope": f"per_class/test/{metric}", "metric": name,
                         a["key"]: old, b["key"]: new,
                         "absolute_delta": abs_d, "relative_delta": rel_d})
    for field, label in (("total_images", "dataset images"),
                         ("total_boxes", "dataset annotations"),
                         ("train_images", "training images"),
                         ("train_boxes", "training annotations")):
        old = a.get("dataset", {}).get(field)
        new = b.get("dataset", {}).get(field)
        abs_d, rel_d = delta(new, old)
        rows.append({"scope": "dataset", "metric": label,
                     a["key"]: old, b["key"]: new,
                     "absolute_delta": abs_d, "relative_delta": rel_d})
    for field in ("epochs_completed", "best_epoch", "training_hours"):
        old = a.get("training", {}).get(field)
        new = b.get("training", {}).get(field)
        abs_d, rel_d = delta(new, old)
        rows.append({"scope": "training", "metric": field,
                     a["key"]: old, b["key"]: new,
                     "absolute_delta": abs_d, "relative_delta": rel_d})

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    with open(path_stem.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "scope", "metric", a["key"], b["key"], "absolute_delta", "relative_delta"])
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    md = [f"# {a['run']} vs {b['run']}", "", caveat, "",
          "## Overall — test split", "",
          f"| Metric | {a['key']} | {b['key']} | Δ abs | Δ rel |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "overall/test":
            md.append(f"| {row['metric']} | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} | {row['relative_delta']} |")
    md += ["", "## Overall — validation split", "",
           f"| Metric | {a['key']} | {b['key']} | Δ abs | Δ rel |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "overall/val":
            md.append(f"| {row['metric']} | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} | {row['relative_delta']} |")
    md += ["", "## Recall by object size — test split", "",
           "Identical bucket definitions to Run 001: tiny < 1% of frame area, "
           "small < 5%, medium < 20%, large above.", "",
           f"| Bucket | {a['key']} | {b['key']} | Δ abs | Δ rel |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "size_recall/test":
            md.append(f"| {row['metric']} | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} | {row['relative_delta']} |")
    md += ["", "## Per class — test AP50-95", "",
           f"| Class | {a['key']} | {b['key']} | Δ abs | Δ rel |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "per_class/test/ap50_95":
            md.append(f"| `{row['metric']}` | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} | {row['relative_delta']} |")
    md += ["", "## Per class — test recall", "",
           f"| Class | {a['key']} | {b['key']} | Δ abs | Δ rel |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "per_class/test/recall":
            md.append(f"| `{row['metric']}` | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} | {row['relative_delta']} |")
    md += ["", "## Dataset and training cost", "",
           f"| | {a['key']} | {b['key']} | Δ abs |", "|---|---:|---:|---:|"]
    for row in rows:
        if row["scope"] in ("dataset", "training"):
            md.append(f"| {row['metric']} | {fmt(row[a['key']])} | {fmt(row[b['key']])} "
                      f"| {row['absolute_delta']} |")
    md += ["", "No classification accuracy appears here. These are detection runs and "
           "a classification-style accuracy would describe neither localisation nor "
           "multi-object frames.", ""]
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {path_stem.with_suffix('.md').relative_to(ROOT)}")


def ablation_table(entries: list[dict], winner: str | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        bench = REPORTS / (f"pytorch_mps_benchmark_{e['key']}.json" if e["key"] != "run_001"
                           else "pytorch_mps_benchmark_run_001.json")
        p50 = None
        if bench.exists():
            p50 = json.loads(bench.read_text()).get("median_ms")
        rows.append({
            "run": e["run"],
            "dataset": ("partial TACO (509 img)" if e["key"] == "run_001"
                        else f"recovered TACO ({e.get('dataset', {}).get('total_images', '?')} img)"),
            "imgsz": e.get("training", {}).get("imgsz", e.get("imgsz")),
            "precision": round(e["test"]["precision"], 4),
            "recall": round(e["test"]["recall"], 4),
            "map50": round(e["test"]["map50"], 4),
            "map50_95": round(e["test"]["map50_95"], 4),
            "tiny_recall": (round(float(e.get("sizes", {}).get("tiny", {}).get("recall", 0)), 4)
                            if e.get("sizes") else None),
            "small_recall": (round(float(e.get("sizes", {}).get("small", {}).get("recall", 0)), 4)
                             if e.get("sizes") else None),
            "mps_p50_ms": p50,
            "winner": "yes" if winner == e["key"] else "",
        })
    with open(OUT / "model_ablation_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = ["# Model ablation table", "",
          "Every row is a real trained run, evaluated on its own untouched test split "
          "with `best.pt`. Latency is Mac/MPS at batch 1 — **not** iPhone latency.", "",
          "| Run | Dataset | imgsz | P | R | mAP50 | mAP50-95 | Tiny R | Small R | MPS p50 | Winner |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        md.append(
            f"| {r['run']} | {r['dataset']} | {r['imgsz']} | {r['precision']:.4f} "
            f"| {r['recall']:.4f} | {r['map50']:.4f} | {r['map50_95']:.4f} "
            f"| {r['tiny_recall'] if r['tiny_recall'] is not None else '—'} "
            f"| {r['small_recall'] if r['small_recall'] is not None else '—'} "
            f"| {str(r['mps_p50_ms']) + ' ms' if r['mps_p50_ms'] else '—'} "
            f"| {'**' + r['winner'] + '**' if r['winner'] else ''} |")
    md += ["",
           "**Rows are not scored on the same test set.** Run 001's test split came "
           "from a 509-image pool; later runs from the recovered 1082-image pool, whose "
           "small-object share is 9.3 pp higher. Compare across rows with that in mind: "
           "an equal number on a harder split is an improvement.", ""]
    (OUT / "model_ablation_table.md").write_text("\n".join(md) + "\n")
    print(f"wrote {(OUT / 'model_ablation_table.md').relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="run_001,run_002",
                        help="comma-separated run keys, in chronological order")
    parser.add_argument("--winner", default=None)
    parser.add_argument("--ablation-only", action="store_true")
    args = parser.parse_args()

    keys = [k.strip() for k in args.runs.split(",") if k.strip()]
    entries = []
    for key in keys:
        entry = load(key)
        if entry is None:
            print(f"skipping {key}: no persisted metrics")
            continue
        entries.append(entry)
    if len(entries) < 1:
        print("ERROR: nothing to compare")
        return 1

    caveats = {
        ("run_001", "run_002"):
            "> **The two runs were scored on different test splits.** Recovery enlarged "
            "the pool from 509 to 1082 images and raised the COCO-small share from "
            "29.1% to 38.4%, so Run 002's test set is harder as well as different. "
            "These deltas measure model-plus-data, not the model alone; an equal number "
            "on a harder split is a real gain.",
        ("run_002", "run_003"):
            "> **These two runs share the same dataset and the same splits.** The only "
            "intended difference is input resolution, so — unlike Run 001 vs Run 002 — "
            "these deltas are a clean controlled comparison. Any batch-size change "
            "forced by memory is noted in the training row and is a confound.",
    }
    if not args.ablation_only:
        for a, b in zip(entries, entries[1:]):
            caveat = caveats.get((a["key"], b["key"]),
                                 "> Comparison between two persisted runs.")
            pairwise_report(a, b, OUT / f"{a['key']}_vs_{b['key']}", caveat)

    ablation_table(entries, args.winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
