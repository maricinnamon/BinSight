#!/usr/bin/env python
"""Runs the exported model on a single image — the same contract iOS uses.

Reproduces the deployment path exactly: bilinear resize to 224x224, raw [0,255]
RGB, no client-side normalisation, then the confidence threshold that decides
between naming a class and saying "Not sure".
"""
from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image

from binsight_training import export as export_module
from binsight_training.config import CLASS_NAMES, DISPLAY_NAMES, load_config
from binsight_training.utils import get_logger

LOGGER = get_logger()


def load_image(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.BILINEAR)
    # [0, 255] float32 RGB — preprocessing is baked into the graph.
    return np.asarray(rgb, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="defaults to the selected model")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = (
        config.tflite_dir / "binsight_trashnet.tflite" if args.model is None
        else __import__("pathlib").Path(args.model)
    )
    if not model_path.exists():
        LOGGER.error("No model at %s — run export + verify first.", model_path)
        return 1

    image = load_image(args.image, config.data.image_size)
    probabilities = export_module.run_tflite(model_path.read_bytes(), image[None, ...])[0]

    order = np.argsort(-probabilities)
    top_index = int(order[0])
    top_score = float(probabilities[top_index])
    threshold = config.inference.confidence_threshold
    confident = top_score >= threshold

    payload = {
        "image": args.image,
        "model": str(model_path),
        "threshold": threshold,
        "confident": confident,
        "prediction": CLASS_NAMES[top_index] if confident else None,
        "display": DISPLAY_NAMES[CLASS_NAMES[top_index]] if confident else "Not sure",
        "confidence": round(top_score, 4),
        "scores": {CLASS_NAMES[i]: round(float(probabilities[i]), 6) for i in range(len(CLASS_NAMES))},
        "ranked": [
            {"label": CLASS_NAMES[i], "display": DISPLAY_NAMES[CLASS_NAMES[i]],
             "score": round(float(probabilities[i]), 6)}
            for i in order
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n{args.image}")
        print(f"  verdict    : {payload['display']}"
              + ("" if confident else f"  (top score {top_score:.3f} < {threshold})"))
        print("  full probability vector:")
        for entry in payload["ranked"]:
            bar = "#" * int(entry["score"] * 40)
            print(f"    {entry['label']:<10} {entry['score']:.4f}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
