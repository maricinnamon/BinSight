#!/usr/bin/env python
"""PRODUCT DIAGNOSTIC — does BinSight's "point at one item" UX rescue small objects?

This is **not** a test-set result and must never be quoted as one. The canonical
number for this model is the full-frame test evaluation; that stays untouched.
What this measures is a different question: given the same weights, does the app
detect more when the camera frames one item instead of a whole scene?

Three framings, same model, no weight changes, nothing trained:

* **full frame** — the canonical baseline, whole test image.
* **object-guided ROI** — a window ~2.5x the target box, i.e. the best case where
  the user aims well. Scored per object: was *this* object found, in this framing?
* **fixed central ROI** — the honest approximation of what an iOS camera UI can
  actually do without knowing where the object is: crop the central 90/75/60%.

The per-object comparison is deliberate. Recall computed over different image
sets is not comparable, so instead every ground-truth object is followed through
both framings and asked the same question, which makes the delta meaningful.

Size buckets always use the object's area in the ORIGINAL full frame — that is
the property of the scene the user encounters, and keeping it fixed is what makes
"tiny recall" comparable across framings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

from PIL import Image, ImageOps

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared_run_002"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

IOU_MATCH = 0.50
SIZE_BUCKETS = [("tiny", 0.0, 0.01), ("small", 0.01, 0.05),
                ("medium", 0.05, 0.20), ("large", 0.20, 1.01)]
GUIDED_CONTEXT = 2.5          # window = 2.5x the target box, inside the 2x-3x brief
MIN_ROI_PX = 224
KEEP_VISIBLE = 0.35           # same rule the training crops used
CENTRAL_FRACTIONS = [0.90, 0.75, 0.60]


def bucket_for(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return "large"


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def load_test(classes):
    """Ground truth in absolute pixels, per test image."""
    data = []
    for label in sorted((PREPARED / "labels" / "test").glob("*.txt")):
        matches = list((PREPARED / "images" / "test").glob(label.stem + ".*"))
        if not matches:
            continue
        path = matches[0]
        with Image.open(path) as im:
            width, height = ImageOps.exif_transpose(im).size
        boxes = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
            boxes.append({
                "class_id": cid, "class": classes[cid],
                "xyxy": [(xc - w / 2) * width, (yc - h / 2) * height,
                         (xc + w / 2) * width, (yc + h / 2) * height],
                "rel_area": w * h, "bucket": bucket_for(w * h),
            })
        data.append({"path": path, "width": width, "height": height, "boxes": boxes})
    return data


def predict(model, image, imgsz, device, conf):
    result = model.predict(image, imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
    rows = [{"class_id": int(b.cls.item()), "conf": float(b.conf.item()),
             "xyxy": [float(v) for v in b.xyxy[0].tolist()]} for b in result.boxes]
    rows.sort(key=lambda r: -r["conf"])
    return rows


def match(truth, preds, threshold=IOU_MATCH):
    """Greedy by descending confidence. Returns matched-truth indices and TP flags."""
    used, tp_flags = set(), []
    for pred in preds:
        best_i, best_v = None, 0.0
        for i, gt in enumerate(truth):
            if i in used:
                continue
            v = iou(pred["xyxy"], gt["xyxy"])
            if v > best_v:
                best_i, best_v = i, v
        if best_i is not None and best_v >= threshold and \
                truth[best_i]["class_id"] == pred["class_id"]:
            used.add(best_i)
            tp_flags.append(True)
        else:
            tp_flags.append(False)
    return used, tp_flags


def average_precision(records, n_truth):
    """AP50 by 101-point interpolation over confidence-ranked detections."""
    if not records or n_truth == 0:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    points = []
    for _, is_tp in records:
        tp += int(is_tp)
        fp += int(not is_tp)
        points.append((tp / n_truth, tp / max(tp + fp, 1)))
    ap = 0.0
    for r in [i / 100 for i in range(101)]:
        precisions = [p for rec, p in points if rec >= r]
        ap += max(precisions) if precisions else 0.0
    return ap / 101


def score(per_class_records, truth_counts, matched_by_bucket, truth_by_bucket,
          tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    aps = [average_precision(per_class_records.get(c, []), n)
           for c, n in truth_counts.items() if n]
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) else 0.0,
        "map50": round(sum(aps) / len(aps), 4) if aps else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "bucket_recall": {b: round(matched_by_bucket[b] / truth_by_bucket[b], 4)
                          if truth_by_bucket[b] else None
                          for b, _, _ in SIZE_BUCKETS},
        "bucket_truth": {b: truth_by_bucket[b] for b, _, _ in SIZE_BUCKETS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO

    classes = list(yaml.safe_load(
        (CONFIGS / "class_mapping_run_002.yaml").read_text())["target_classes"])
    model = YOLO(str(ROOT / args.weights))
    data = load_test(classes)
    results: dict = {
        "disclaimer": ("PRODUCT DIAGNOSTIC. Only the full-frame row is a test-set "
                       "result. ROI rows measure a different question — how the model "
                       "behaves under BinSight's guided-camera UX — and must never be "
                       "reported as standard test performance."),
        "weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
        "test_images": len(data),
        "test_objects": sum(len(d["boxes"]) for d in data),
    }

    # ------------------------------------------------------------ full frame
    tp = fp = 0
    per_class_records = collections.defaultdict(list)
    truth_counts = collections.Counter()
    matched_bucket, truth_bucket = collections.Counter(), collections.Counter()
    fullframe_found: dict[tuple, bool] = {}
    for item in data:
        preds = predict(model, str(item["path"]), args.imgsz, args.device, args.conf)
        used, flags = match(item["boxes"], preds)
        for pred, is_tp in zip(preds, flags):
            per_class_records[pred["class_id"]].append((pred["conf"], is_tp))
            tp += int(is_tp)
            fp += int(not is_tp)
        for i, gt in enumerate(item["boxes"]):
            truth_counts[gt["class_id"]] += 1
            truth_bucket[gt["bucket"]] += 1
            found = i in used
            fullframe_found[(item["path"].name, i)] = found
            if found:
                matched_bucket[gt["bucket"]] += 1
    fn = sum(truth_counts.values()) - tp
    results["full_frame"] = score(per_class_records, truth_counts,
                                  matched_bucket, truth_bucket, tp, fp, fn)
    print("full frame:", json.dumps(results["full_frame"], indent=2))

    # ------------------------------------------------- object-guided ROI
    guided_found: dict[tuple, bool] = {}
    area_gain = []
    tp = fp = 0
    per_class_records = collections.defaultdict(list)
    truth_counts = collections.Counter()
    matched_bucket, truth_bucket = collections.Counter(), collections.Counter()
    for item in data:
        with Image.open(item["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        for i, gt in enumerate(item["boxes"]):
            x0, y0, x1, y1 = gt["xyxy"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(max(x1 - x0, y1 - y0) * GUIDED_CONTEXT, MIN_ROI_PX)
            side = min(side, float(min(item["width"], item["height"])))
            rx0 = min(max(cx - side / 2, 0.0), item["width"] - side)
            ry0 = min(max(cy - side / 2, 0.0), item["height"] - side)
            window = (int(rx0), int(ry0), int(rx0 + side), int(ry0 + side))
            roi = image.crop(window)
            rw, rh = roi.size

            local = []
            for j, other in enumerate(item["boxes"]):
                ox0, oy0, ox1, oy1 = other["xyxy"]
                original = max((ox1 - ox0) * (oy1 - oy0), 1e-9)
                ix0, iy0 = max(ox0, window[0]), max(oy0, window[1])
                ix1, iy1 = min(ox1, window[2]), min(oy1, window[3])
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                if ((ix1 - ix0) * (iy1 - iy0)) / original < KEEP_VISIBLE:
                    continue
                local.append({"index": j, "class_id": other["class_id"],
                              "xyxy": [ix0 - window[0], iy0 - window[1],
                                       ix1 - window[0], iy1 - window[1]],
                              "bucket": other["bucket"]})
            target = next((b for b in local if b["index"] == i), None)
            if target is None:
                continue

            preds = predict(model, roi, args.imgsz, args.device, args.conf)
            used, flags = match(local, preds)
            for pred, is_tp in zip(preds, flags):
                per_class_records[pred["class_id"]].append((pred["conf"], is_tp))
                tp += int(is_tp)
                fp += int(not is_tp)
            truth_counts[gt["class_id"]] += 1
            truth_bucket[gt["bucket"]] += 1
            found = local.index(target) in used
            guided_found[(item["path"].name, i)] = found
            if found:
                matched_bucket[gt["bucket"]] += 1

            tw = (target["xyxy"][2] - target["xyxy"][0]) / rw
            th = (target["xyxy"][3] - target["xyxy"][1]) / rh
            area_gain.append({
                "image": item["path"].name, "object": i, "class": gt["class"],
                "original_rel_area": round(gt["rel_area"], 8),
                "roi_rel_area": round(tw * th, 8),
                "gain": round((tw * th) / max(gt["rel_area"], 1e-12), 1),
                "original_bucket": gt["bucket"], "roi_bucket": bucket_for(tw * th),
                "detected_full_frame": fullframe_found.get((item["path"].name, i)),
                "detected_guided_roi": found,
            })
    fn = sum(truth_counts.values()) - tp
    results["object_guided_roi"] = score(per_class_records, truth_counts,
                                         matched_bucket, truth_bucket, tp, fp, fn)
    gains = sorted(r["gain"] for r in area_gain)
    results["object_guided_roi"]["target_area_gain"] = {
        "median": gains[len(gains) // 2] if gains else None,
        "mean": round(sum(gains) / len(gains), 1) if gains else None,
        "context_factor": GUIDED_CONTEXT,
    }
    print("object-guided ROI:", json.dumps(results["object_guided_roi"], indent=2))

    # --------------------------------------------------- fixed central ROI
    results["central_roi"] = {}
    for fraction in CENTRAL_FRACTIONS:
        tp = fp = 0
        per_class_records = collections.defaultdict(list)
        truth_counts = collections.Counter()
        matched_bucket, truth_bucket = collections.Counter(), collections.Counter()
        objects_outside = 0
        for item in data:
            with Image.open(item["path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            cw, ch = item["width"] * fraction, item["height"] * fraction
            wx0, wy0 = (item["width"] - cw) / 2, (item["height"] - ch) / 2
            window = (int(wx0), int(wy0), int(wx0 + cw), int(wy0 + ch))
            roi = image.crop(window)

            local = []
            for gt in item["boxes"]:
                ox0, oy0, ox1, oy1 = gt["xyxy"]
                original = max((ox1 - ox0) * (oy1 - oy0), 1e-9)
                ix0, iy0 = max(ox0, window[0]), max(oy0, window[1])
                ix1, iy1 = min(ox1, window[2]), min(oy1, window[3])
                if ix1 <= ix0 or iy1 <= iy0 or \
                        ((ix1 - ix0) * (iy1 - iy0)) / original < KEEP_VISIBLE:
                    objects_outside += 1
                    continue
                local.append({"class_id": gt["class_id"],
                              "xyxy": [ix0 - window[0], iy0 - window[1],
                                       ix1 - window[0], iy1 - window[1]],
                              # bucket by ORIGINAL frame area, so it stays comparable
                              "bucket": gt["bucket"]})
            if not local:
                continue
            preds = predict(model, roi, args.imgsz, args.device, args.conf)
            used, flags = match(local, preds)
            for pred, is_tp in zip(preds, flags):
                per_class_records[pred["class_id"]].append((pred["conf"], is_tp))
                tp += int(is_tp)
                fp += int(not is_tp)
            for i, gt in enumerate(local):
                truth_counts[gt["class_id"]] += 1
                truth_bucket[gt["bucket"]] += 1
                if i in used:
                    matched_bucket[gt["bucket"]] += 1
        fn = sum(truth_counts.values()) - tp
        entry = score(per_class_records, truth_counts, matched_bucket,
                      truth_bucket, tp, fp, fn)
        entry["objects_lost_outside_roi"] = objects_outside
        entry["objects_retained"] = sum(truth_counts.values())
        results["central_roi"][f"{int(fraction*100)}%"] = entry
        print(f"central {int(fraction*100)}%:", json.dumps(entry, indent=2))

    # ------------------------------------------------------------- write out
    out = REPORTS / "guided_framing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "guided_framing_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(out / "object_guided_per_object.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(area_gain[0].keys()))
        writer.writeheader()
        writer.writerows(area_gain)

    # Per-object paired comparison: same object, two framings.
    paired = collections.Counter()
    by_bucket = collections.defaultdict(lambda: collections.Counter())
    for row in area_gain:
        key = (bool(row["detected_full_frame"]), bool(row["detected_guided_roi"]))
        paired[key] += 1
        by_bucket[row["original_bucket"]][key] += 1
    results["paired_per_object"] = {
        "missed_both": paired[(False, False)],
        "rescued_by_guided_roi": paired[(False, True)],
        "lost_by_guided_roi": paired[(True, False)],
        "found_both": paired[(True, True)],
        "by_bucket": {b: dict(("_".join(map(str, k)), v) for k, v in c.items())
                      for b, c in by_bucket.items()},
    }
    (out / "guided_framing_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\npaired per-object:", json.dumps(results["paired_per_object"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
