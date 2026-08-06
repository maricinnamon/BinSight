#!/usr/bin/env python
"""Error analysis, size-bucket performance, threshold sweep and prediction renders.

mAP says how good the model is. It does not say *how* it fails, and for a phone
app pointed at someone's rubbish, the failure mode matters more than the
headline: a confident wrong label is worse than no label.

Matching is greedy by descending confidence at IoU >= 0.50 (stated explicitly,
because every count below depends on it).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
PREPARED = ROOT / "data" / "prepared"

IOU_MATCH = 0.50
LOW_IOU = 0.30          # localised but poorly
SIZE_BUCKETS = [("tiny", 0.0, 0.01), ("small", 0.01, 0.05),
                ("medium", 0.05, 0.20), ("large", 0.20, 1.01)]
PALETTE = ["#FF5C8A", "#5DE2B8", "#8B5CF6", "#D7FF5F",
           "#F5A524", "#4CC9F0", "#FF8A5C", "#B8A8FF"]


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def bucket_for(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return "large"


def load_truth(split: str, classes, prepared: pathlib.Path | None = None):
    """Ground truth in absolute xyxy, per image.

    `prepared` defaults to Run 001's dataset so existing callers are unchanged.
    """
    from PIL import Image
    prepared = prepared or PREPARED
    truth = {}
    for label in sorted((prepared / "labels" / split).glob("*.txt")):
        matches = list((prepared / "images" / split).glob(label.stem + ".*"))
        if not matches:
            continue
        image_path = matches[0]
        with Image.open(image_path) as im:
            w, h = im.size
        boxes = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            boxes.append({
                "class_id": cid, "class": classes[cid],
                "xyxy": [(xc - bw / 2) * w, (yc - bh / 2) * h,
                         (xc + bw / 2) * w, (yc + bh / 2) * h],
                "rel_area": bw * bh,
                "bucket": bucket_for(bw * bh),
            })
        truth[image_path] = boxes
    return truth


def predict(model, images, imgsz, device, conf):
    predictions = {}
    for path in images:
        result = model.predict(str(path), imgsz=imgsz, device=device, conf=conf,
                               verbose=False)[0]
        rows = []
        for box in result.boxes:
            rows.append({"class_id": int(box.cls.item()),
                         "conf": float(box.conf.item()),
                         "xyxy": [float(v) for v in box.xyxy[0].tolist()]})
        rows.sort(key=lambda r: -r["conf"])
        predictions[path] = rows
    return predictions


def match(truth_boxes, pred_boxes, threshold=IOU_MATCH):
    """Greedy match by descending confidence. Returns tp, fp, fn, confusions, low_iou."""
    used = set()
    tp, fp, confusions, low_iou = [], [], [], []
    for pred in pred_boxes:
        best_index, best_iou = None, 0.0
        for index, gt in enumerate(truth_boxes):
            if index in used:
                continue
            value = iou(pred["xyxy"], gt["xyxy"])
            if value > best_iou:
                best_index, best_iou = index, value
        if best_index is not None and best_iou >= threshold:
            used.add(best_index)
            gt = truth_boxes[best_index]
            if gt["class_id"] == pred["class_id"]:
                tp.append((pred, gt, best_iou))
            else:
                confusions.append((pred, gt, best_iou))
        else:
            if best_index is not None and LOW_IOU <= best_iou < threshold \
                    and truth_boxes[best_index]["class_id"] == pred["class_id"]:
                low_iou.append((pred, truth_boxes[best_index], best_iou))
            fp.append((pred, best_iou))
    fn = [gt for index, gt in enumerate(truth_boxes) if index not in used]
    return tp, fp, fn, confusions, low_iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="yolo26n_run_001")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=0.25)
    # Everything below defaults to the Run 001 invocation, so the original
    # command keeps writing exactly the reports it always wrote.
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--out-dir", default="reports/error_analysis")
    parser.add_argument("--figures-dir", default="reports/figures/run_001")
    parser.add_argument("--size-report", default="reports/object_size_performance_run_001")
    parser.add_argument("--sweep-report", default="reports/confidence_threshold_sweep.csv")
    parser.add_argument("--json-report", default="reports/error_analysis_run_001.json")
    parser.add_argument("--label", default="Run 001")
    parser.add_argument("--thresholds",
                        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70")
    args = parser.parse_args()

    prepared = ROOT / args.prepared

    from ultralytics import YOLO

    classes = (CONFIGS / "classes.txt").read_text().split()
    best = ROOT / "runs" / args.run / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: {best} not found")
        return 1

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    figures = ROOT / args.figures_dir
    figures.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- test errors
    model = YOLO(str(best))
    truth = load_truth("test", classes, prepared)
    preds = predict(model, list(truth), args.imgsz, args.device, args.conf)

    fps, fns, confusions, low_ious, tps = [], [], [], [], []
    bucket_stats = collections.defaultdict(lambda: {"gt": 0, "found": 0})
    for path, gts in truth.items():
        tp, fp, fn, conf_pairs, low = match(gts, preds[path])
        for pred, gt, value in tp:
            tps.append({"image": path.name, "class": classes[pred["class_id"]],
                        "conf": round(pred["conf"], 4), "iou": round(value, 4)})
            bucket_stats[gt["bucket"]]["found"] += 1
        for pred, value in fp:
            fps.append({"image": path.name, "predicted_class": classes[pred["class_id"]],
                        "confidence": round(pred["conf"], 4),
                        "best_iou_with_any_gt": round(value, 4)})
        for gt in fn:
            fns.append({"image": path.name, "ground_truth_class": gt["class"],
                        "rel_area": round(gt["rel_area"], 6), "size_bucket": gt["bucket"]})
        for pred, gt, value in conf_pairs:
            confusions.append({"image": path.name, "ground_truth_class": gt["class"],
                               "predicted_class": classes[pred["class_id"]],
                               "confidence": round(pred["conf"], 4), "iou": round(value, 4)})
        for pred, gt, value in low:
            low_ious.append({"image": path.name, "class": gt["class"],
                             "confidence": round(pred["conf"], 4), "iou": round(value, 4)})
        for gt in gts:
            bucket_stats[gt["bucket"]]["gt"] += 1

    def write_csv(name, rows, headers):
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("false_positives.csv", fps,
              ["image", "predicted_class", "confidence", "best_iou_with_any_gt"])
    write_csv("false_negatives.csv", fns,
              ["image", "ground_truth_class", "rel_area", "size_bucket"])
    write_csv("class_confusions.csv", confusions,
              ["image", "ground_truth_class", "predicted_class", "confidence", "iou"])
    write_csv("low_iou_detections.csv", low_ious, ["image", "class", "confidence", "iou"])

    pairs = collections.Counter((c["ground_truth_class"], c["predicted_class"])
                                for c in confusions)
    fp_by_class = collections.Counter(f["predicted_class"] for f in fps)
    fn_by_class = collections.Counter(f["ground_truth_class"] for f in fns)

    summary = [
        f"# Error analysis — {args.label} (test split)", "",
        f"Matching: greedy by descending confidence, **IoU >= {IOU_MATCH}**, "
        f"prediction confidence >= {args.conf}.", "",
        "| | |", "|---|---:|",
        f"| Ground-truth objects | {sum(len(v) for v in truth.values())} |",
        f"| Predictions | {sum(len(v) for v in preds.values())} |",
        f"| True positives | {len(tps)} |",
        f"| False positives | {len(fps)} |",
        f"| False negatives | {len(fns)} |",
        f"| Class confusions (right box, wrong class) | {len(confusions)} |",
        f"| Poor localisation ({LOW_IOU} <= IoU < {IOU_MATCH}, right class) | {len(low_ious)} |",
        "", "## False negatives by class", "", "| Class | Missed |", "|---|---:|"]
    summary += [f"| {k} | {v} |" for k, v in fn_by_class.most_common()]
    summary += ["", "## False positives by predicted class", "",
                "| Predicted class | Count |", "|---|---:|"]
    summary += [f"| {k} | {v} |" for k, v in fp_by_class.most_common()]
    if pairs:
        summary += ["", "## Confusion pairs", "",
                    "| Ground truth | Predicted as | Count |", "|---|---|---:|"]
        summary += [f"| {a} | {b} | {n} |" for (a, b), n in pairs.most_common(12)]
    summary += ["", "## Misses by object size", "",
                "| Bucket | Ground truth | Detected | Recall |", "|---|---:|---:|---:|"]
    size_rows = []
    for name, _, _ in SIZE_BUCKETS:
        stat = bucket_stats[name]
        recall = stat["found"] / stat["gt"] if stat["gt"] else 0.0
        summary.append(f"| {name} | {stat['gt']} | {stat['found']} | {recall:.3f} |")
        size_rows.append({"bucket": name, "ground_truth": stat["gt"],
                          "detected": stat["found"], "recall": round(recall, 4)})
    summary += ["",
                "Small-object recall is the number that matters for BinSight: the eventual",
                "camera sees objects at a distance, and TACO's median object is under 1% of",
                "the frame.", ""]
    (out / "summary.md").write_text("\n".join(summary) + "\n")

    size_stem = ROOT / args.size_report
    size_stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv_path = size_stem.with_suffix(".csv")
    with open(write_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", "ground_truth", "detected", "recall"])
        writer.writeheader()
        writer.writerows(size_rows)
    size_stem.with_suffix(".md").write_text(
        "\n".join([f"# Detection performance by object size — {args.label}", "",
                   f"Test split, IoU >= {IOU_MATCH}, confidence >= {args.conf}.", "",
                   "| Bucket | Range | Ground truth | Detected | Recall |",
                   "|---|---|---:|---:|---:|"] +
                  [f"| {r['bucket']} | {dict((n,(l,h)) for n,l,h in SIZE_BUCKETS)[r['bucket']]} "
                   f"| {r['ground_truth']} | {r['detected']} | {r['recall']:.3f} |"
                   for r in size_rows]) + "\n")

    # ------------------------------------------------------- threshold sweep
    val_truth = load_truth("val", classes, prepared)
    val_preds = predict(model, list(val_truth), args.imgsz, args.device, 0.01)
    sweep = []
    for threshold in [float(t) for t in args.thresholds.split(",")]:
        tp_n = fp_n = fn_n = pred_n = 0
        for path, gts in val_truth.items():
            filtered = [p for p in val_preds[path] if p["conf"] >= threshold]
            pred_n += len(filtered)
            tp, fp, fn, conf_pairs, _ = match(gts, filtered)
            tp_n += len(tp)
            fp_n += len(fp)
            fn_n += len(fn)
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 0.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sweep.append({"threshold": threshold, "predictions": pred_n,
                      "true_positives": tp_n, "false_positives": fp_n,
                      "false_negatives": fn_n, "precision": round(precision, 4),
                      "recall": round(recall, 4), "f1": round(f1, 4)})
    sweep_path = ROOT / args.sweep_report
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [s["threshold"] for s in sweep]
    ax.plot(xs, [s["precision"] for s in sweep], "-o", label="precision", color="#8B5CF6")
    ax.plot(xs, [s["recall"] for s in sweep], "-o", label="recall", color="#5DE2B8")
    ax.plot(xs, [s["f1"] for s in sweep], "-o", label="F1", color="#FF5C8A", linewidth=2.4)
    best_f1 = max(sweep, key=lambda s: s["f1"])
    ax.axvline(best_f1["threshold"], color="grey", linestyle="--", linewidth=1)
    ax.text(best_f1["threshold"], 0.02, f" best F1 @ {best_f1['threshold']}", fontsize=9, color="grey")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_title("Confidence threshold trade-off (validation split)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "confidence_threshold_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    json.dump({"iou_match": IOU_MATCH, "conf": args.conf,
               "counts": {"tp": len(tps), "fp": len(fps), "fn": len(fns),
                          "confusions": len(confusions), "low_iou": len(low_ious)},
               "confusion_pairs": [{"gt": a, "pred": b, "count": n}
                                   for (a, b), n in pairs.most_common()],
               "size_buckets": size_rows, "threshold_sweep": sweep},
              open(ROOT / args.json_report, "w"), indent=2)

    print(f"TP={len(tps)} FP={len(fps)} FN={len(fns)} confusions={len(confusions)} "
          f"low_iou={len(low_ious)}")
    print("best F1 threshold:", best_f1)
    print("size recall:", {r["bucket"]: r["recall"] for r in size_rows})
    return 0


if __name__ == "__main__":
    sys.exit(main())
