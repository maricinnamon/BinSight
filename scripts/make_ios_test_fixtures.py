#!/usr/bin/env python3
"""Builds the fixtures the Swift static-inference test asserts against.

The Swift test needs two things the iOS side cannot produce for itself: images
that are known to detect, and the answer to compare against. Both come from the
same CoreML package the app bundles, run here through coremltools — so a Swift
failure means the *Swift* path is wrong, not that the model changed.

Two kinds of fixture, for two different failure modes:

* **square 416x416** — the held-out test images used in the CoreML parity check.
  Letterboxing these to 640 needs no padding at all (416 is square), so they
  isolate output decoding, class mapping and the xyxy interpretation.
* **portrait 720x1280** — a composed frame the exact shape the capture session
  produces. Here the letterbox pads: `scale 0.5, padX 140, padY 0`. Nothing in
  the square fixtures would catch a wrong pad offset; these do.

Expected boxes are written in **source-image pixels**, which is the space
`Detection.boundingBox` is defined in, so the Swift test compares like with like.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "datasets/binsight_waste_3class/images/test"
OUT = ROOT / "BinSightTests/Resources/DetectionFixtures"
PACKAGE = ROOT / "models/coreml/BinSightYOLO26n.mlpackage"
CLASSES = {0: "paper", 1: "plastic", 2: "metal"}
NET = 640
PAD = 114
CONF = 0.25

# Chosen because each is the most confident example of its class found during
# the CoreML parity check. Source filenames carry the *original* Roboflow class
# folder, which is often not the class actually annotated inside — hence the
# rename: `cardboard1099` really does contain paper.
SQUARE_SOURCES = {
    "confident_paper.jpg": "cardboard1099_jpeg.rf.0b5e1b1d1570363e62d6a944f3e0069e.jpg",
    "confident_plastic.jpg": "glass810_jpg.rf.4fe9533ff569f486696512a4e1107d74.jpg",
    "confident_metal.jpg": "metal1253_jpg.rf.c93299f221d8b9614552891bf14f23be.jpg",
}


def letterbox(bgr, net: int = NET):
    """Ultralytics-compatible: uniform scale, centred, grey padding."""
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    scale = min(net / w, net / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((net, net, 3), PAD, dtype=np.uint8)
    pad_x, pad_y = (net - new_w) // 2, (net - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    # Return the *exact* float scale/pads the Swift transform computes, not the
    # rounded integers used for pixel placement.
    return canvas, min(net / w, net / h), (net - w * scale) / 2, (net - h * scale) / 2


def summarise(detections) -> str:
    """Short one-line description for the console, no f-string gymnastics."""
    parts = [f"{d['className']} {d['confidence']:.2f}" for d in detections[:3]]
    return ", ".join(parts) if parts else "none"


def predict(model, output_name, bgr):
    import cv2
    import numpy as np
    from PIL import Image

    h, w = bgr.shape[:2]
    canvas, scale, pad_x, pad_y = letterbox(bgr)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    rows = np.asarray(model.predict({"image": pil})[output_name]).reshape(-1, 6)

    out = []
    for x1, y1, x2, y2, conf, cid in rows:
        if float(conf) < CONF:
            continue
        cid = int(round(float(cid)))
        if cid not in CLASSES:
            continue
        sx1 = min(max((float(x1) - pad_x) / scale, 0.0), w)
        sy1 = min(max((float(y1) - pad_y) / scale, 0.0), h)
        sx2 = min(max((float(x2) - pad_x) / scale, 0.0), w)
        sy2 = min(max((float(y2) - pad_y) / scale, 0.0), h)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        out.append({
            "className": CLASSES[cid],
            "classID": cid,
            "confidence": round(float(conf), 4),
            "x": round(sx1, 2), "y": round(sy1, 2),
            "width": round(sx2 - sx1, 2), "height": round(sy2 - sy1, 2),
        })
    out.sort(key=lambda d: -d["confidence"])
    return out


def main() -> int:
    import cv2
    import numpy as np
    import coremltools as ct

    if not PACKAGE.exists():
        print(f"ERROR: {PACKAGE} not found")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT.glob("*.jpg")) + list(OUT.glob("*.json")):
        stale.unlink()

    model = ct.models.MLModel(str(PACKAGE))
    output_name = model.get_spec().description.output[0].name

    fixtures = []

    # ---------------------------------------------------------------- square
    for name, source in SQUARE_SOURCES.items():
        src = TEST_IMAGES / source
        if not src.exists():
            print(f"ERROR: source image missing: {src}")
            return 1
        shutil.copy2(src, OUT / name)
        bgr = cv2.imread(str(OUT / name))
        h, w = bgr.shape[:2]
        expected = predict(model, output_name, bgr)
        fixtures.append({
            "image": name, "sourceWidth": w, "sourceHeight": h,
            "note": "square 416x416 — letterbox needs no padding",
            "expected": expected,
        })
        print(f"  {name}: {w}x{h} -> {len(expected)} detections ({summarise(expected)})")

    # -------------------------------------------------------------- portrait
    # The shape the capture session actually delivers, so the pad offset is
    # exercised: 720x1280 -> scale 0.5, padX 140, padY 0.
    metal = cv2.imread(str(OUT / "confident_metal.jpg"))
    canvas = np.full((1280, 720, 3), PAD, dtype=np.uint8)
    scaled = cv2.resize(metal, (720, 720), interpolation=cv2.INTER_LINEAR)
    canvas[280:1000, 0:720] = scaled
    name = "portrait_frame.jpg"
    cv2.imwrite(str(OUT / name), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    bgr = cv2.imread(str(OUT / name))
    h, w = bgr.shape[:2]
    expected = predict(model, output_name, bgr)
    fixtures.append({
        "image": name, "sourceWidth": w, "sourceHeight": h,
        "note": "portrait 720x1280 — the capture shape; letterbox pads x by 140",
        "expected": expected,
    })
    print(f"  {name}: {w}x{h} -> {len(expected)} detections ({summarise(expected)})")

    manifest = {
        "generatedBy": "scripts/make_ios_test_fixtures.py",
        "package": str(PACKAGE.relative_to(ROOT)),
        "outputFeature": output_name,
        "confidenceThreshold": CONF,
        "coordinateSpace": "source-image pixels, origin top-left",
        "fixtures": fixtures,
    }
    (OUT / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {(OUT / 'expected.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
