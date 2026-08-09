#!/usr/bin/env python3
"""Evaluates the trained BinSight detector on validation and the held-out test split.

`best.pt`, never `last.pt` — the best checkpoint is the model-quality candidate.
Validation and test are reported separately and never mixed: validation guided
checkpoint selection, so quoting it as final performance would be reporting a
number the model was selected on. Test is the honest figure.

Also produces prediction previews on real test images, so the boxes can be
judged by eye rather than only through aggregate metrics.

Memory: one evaluation at a time, model reloaded between splits (Ultralytics
mutates model state during `val`), previews rendered one image at a time.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ["paper", "plastic", "metal"]
COLOURS = {0: (255, 92, 138), 1: (93, 226, 184), 2: (139, 92, 246)}


def collect(metrics) -> dict:
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    out = {
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
    }
    try:
        out["map75"] = round(float(box.map75), 4)
    except Exception:  # noqa: BLE001
        out["map75"] = None
    per_class = []
    for position, cid in enumerate(box.ap_class_index):
        cp, cr, ap50, ap = box.class_result(position)
        cp, cr = float(cp), float(cr)
        per_class.append({
            "class_id": int(cid), "class": CLASSES[int(cid)],
            "precision": round(cp, 4), "recall": round(cr, 4),
            "f1": round(2 * cp * cr / (cp + cr), 4) if (cp + cr) else 0.0,
            "ap50": round(float(ap50), 4), "ap50_95": round(float(ap), 4),
        })
    per_class.sort(key=lambda r: r["class_id"])
    out["per_class"] = per_class
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/detect/binsight_yolo26n_pretrained_final")
    parser.add_argument("--data", default="datasets/binsight_waste_3class/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--previews", type=int, default=18)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    run = ROOT / args.run
    best = run / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: {best} not found")
        return 1

    from ultralytics import YOLO

    results: dict = {"weights": str(best.relative_to(ROOT)), "imgsz": args.imgsz}
    for split in ("val", "test"):
        model = YOLO(str(best))          # reload: val mutates model state
        m = model.val(data=str(ROOT / args.data), split=split, imgsz=args.imgsz,
                      device=args.device, plots=True, verbose=False,
                      project=str(run.parent), name=f"{run.name}_{split}", exist_ok=True)
        results[split] = collect(m)
        o = results[split]
        print(f"{split:<5} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f} "
              f"mAP50={o['map50']:.4f} mAP75={o['map75']} mAP50-95={o['map50_95']:.4f}")
        del model

    (run / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")

    rows = []
    for split in ("val", "test"):
        for c in results[split]["per_class"]:
            rows.append({"split": split, **c})
    with open(run / "per_class_metrics.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------ previews
    out = ROOT / "artifacts" / "test_predictions"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()

    from PIL import Image, ImageDraw
    test_images = sorted((ROOT / "datasets/binsight_waste_3class/images/test").iterdir())
    rng = random.Random(42)
    rng.shuffle(test_images)
    model = YOLO(str(best))
    written = 0
    for path in test_images:
        if written >= args.previews:
            break
        r = model.predict(str(path), imgsz=args.imgsz, device=args.device,
                          conf=args.conf, verbose=False)[0]
        gt_path = ROOT / "datasets/binsight_waste_3class/labels/test" / f"{path.stem}.txt"
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            # Ground truth dashed-white first, predictions solid colour on top.
            if gt_path.exists():
                for line in gt_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    _, x, y, w, h = line.split()
                    x, y, w, h = float(x)*W, float(y)*H, float(w)*W, float(h)*H
                    d.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline=(255, 255, 255), width=1)
            for b in r.boxes:
                cid = int(b.cls); conf = float(b.conf)
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                col = COLOURS.get(cid, (255, 255, 255))
                d.rectangle([x1, y1, x2, y2], outline=col, width=3)
                tag = f"{CLASSES[cid]} {conf:.2f}"
                d.rectangle([x1, max(y1-15, 0), x1 + 8*len(tag), max(y1, 15)], fill=col)
                d.text((x1+3, max(y1-13, 2)), tag, fill=(0, 0, 0))
            im.save(out / f"{written:02d}_{path.stem[:36]}.jpg", "JPEG", quality=88)
            del d
        written += 1
    print(f"\npreviews: {written} -> {out.relative_to(ROOT)}")
    print(f"reports : {(run / 'evaluation.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
