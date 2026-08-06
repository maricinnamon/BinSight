#!/usr/bin/env python
"""Renders REAL model predictions on the untouched test split.

Predictions are drawn solid; ground truth, where shown, is dashed white. These
are model outputs, not ground-truth labels relabelled as predictions.
"""
from __future__ import annotations
import argparse, collections, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyse_errors import IOU_MATCH, load_truth, match, predict  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PALETTE = ["#FF5C8A", "#5DE2B8", "#8B5CF6", "#D7FF5F",
           "#F5A524", "#4CC9F0", "#FF8A5C", "#B8A8FF"]


def panel(ax, path, preds, gts, classes):
    with Image.open(path) as im:
        ax.imshow(im.convert("RGB"))
    for gt in gts:
        x0, y0, x1, y1 = gt["xyxy"]
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.6,
                                       edgecolor="white", facecolor="none", linestyle="--"))
    for p in preds:
        x0, y0, x1, y1 = p["xyxy"]
        colour = PALETTE[p["class_id"] % len(PALETTE)]
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.4,
                                       edgecolor=colour, facecolor="none"))
        ax.text(x0, max(y0 - 6, 12), f"{classes[p['class_id']]} {p['conf']:.2f}",
                fontsize=8, color="black",
                bbox=dict(facecolor=colour, edgecolor="none", pad=1.4, alpha=0.95))
    ax.axis("off")


def grid(items, classes, path, title, columns=3):
    if not items:
        return
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (image_path, preds, gts) in zip(axes, items):
        panel(ax, image_path, preds, gts, classes)
    fig.legend(handles=[Line2D([0], [0], color="white", ls="--", label="ground truth"),
                        Line2D([0], [0], color=PALETTE[0], lw=2.4, label="prediction")],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)} ({len(items)} images)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="yolo26n_run_001")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    # Defaults reproduce the Run 001 invocation.
    ap.add_argument("--prepared", default="data/prepared")
    ap.add_argument("--out", default="reports/samples/run_001")
    a = ap.parse_args()

    out = ROOT / a.out

    from ultralytics import YOLO
    classes = (ROOT / "configs" / "classes.txt").read_text().split()
    best = ROOT / "runs" / a.run / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: {best} missing"); return 1

    model = YOLO(str(best))
    truth = load_truth("test", classes, ROOT / a.prepared)
    preds = predict(model, list(truth), a.imgsz, a.device, a.conf)

    buckets = collections.defaultdict(list)
    for path, gts in truth.items():
        p = preds[path]
        tp, fp, fn, conf_pairs, low = match(gts, p)
        entry = (path, p, gts)
        if tp:
            best_conf = max(t[0]["conf"] for t in tp)
            buckets["high_confidence_correct" if best_conf >= 0.6
                    else "low_confidence_correct"].append(entry)
        if fp and not tp:
            buckets["false_positives"].append(entry)
        if fn and not p:
            buckets["false_negatives"].append(entry)
        if conf_pairs:
            buckets["class_confusions"].append(entry)
        if len(gts) >= 4:
            buckets["crowded_scenes"].append(entry)
        if gts and all(g["bucket"] in ("tiny", "small") for g in gts):
            buckets["small_objects"].append(entry)
        # Split out the tiny bucket on its own: it is where Run 001 failed
        # hardest (recall 0.051) and it is the whole point of a 768 ablation.
        if gts and any(g["bucket"] == "tiny" for g in gts):
            buckets["tiny_objects"].append(entry)

    titles = {
        "high_confidence_correct": "Correct detections, high confidence",
        "low_confidence_correct": "Correct detections, low confidence",
        "false_positives": "False positives — predicted where ground truth has nothing",
        "false_negatives": "False negatives — ground truth present, nothing predicted",
        "class_confusions": "Class confusions — box right, class wrong",
        "crowded_scenes": "Crowded scenes (4+ ground-truth objects)",
        "small_objects": "Small and tiny objects only",
        "tiny_objects": "Scenes containing tiny objects (< 1% of frame)",
    }
    for key, title in titles.items():
        grid(buckets[key][:6], classes, out / f"{key}.jpg", title)
    missing = [k for k in titles if not buckets[k]]
    if missing:
        print("no examples for:", missing)
    print("samples ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
