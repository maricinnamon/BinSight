#!/usr/bin/env python3
"""Exports the accepted BinSight YOLO26n detector to CoreML for iOS.

Source is the final `best.pt` and nothing else — not `last.pt`, not the original
pretrained `yolo26n.pt`. The script verifies the checkpoint's identity (task,
class count, class names, SHA-256) before it exports, and refuses otherwise; an
export is only meaningful if it provably came from the model that was evaluated.

Two YOLO26 specifics drive everything here, both read from the loaded model
rather than assumed from YOLOv5/v8 tutorials:

  * `head.end2end is True`. YOLO26 carries a one-to-one head and is NMS-free.
    Ultralytics *forces* `nms=False` for such models (exporter.py: "'nms=True'
    is not available for end2end models"), so no CoreML NMS pipeline is built
    and none is needed — the model emits already-deduplicated detections.
  * The export graph therefore emits a single `(1, 300, 6)` tensor, where 300 is
    `head.max_det` and the 6 columns are `[x1, y1, x2, y2, confidence, class_id]`
    in 640x640 input-pixel space. Verified empirically against Ultralytics'
    own `predict()` output, not taken from the `postprocess` docstring, which
    describes the columns as xywh — for this checkpoint they are xyxy.

Precision: `mlprogram` is FP16 by default in coremltools, which is what we want
for Apple Silicon. `half=True` is passed for explicitness but is a no-op on this
path. INT8/palettisation is deliberately NOT used — no calibration data exists
and quantisation is out of scope for this phase.

Environment: Ultralytics pins `numpy<=2.3.5` for CoreML export and that pin is
real, not defensive. Under numpy 2.4.x, `int(np.array([400]))` — a size-1 but
1-dimensional array — raises `TypeError: only 0-dimensional arrays can be
converted to Python scalars`, and coremltools' `_cast` op handler depends on that
conversion. The export dies converting the attention block's `int` node. numpy
was therefore pinned down to 2.3.5 in the environment before exporting.

`ULTRALYTICS_AUTOINSTALL` is disabled below all the same: the requirement is
satisfied up front and deliberately, so no pip install should ever run from
inside a traced export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

# Must precede the ultralytics import — checked at import time.
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = "runs/detect/binsight_yolo26n_pretrained_final/weights/best.pt"
DEFAULT_DEST = "models/coreml/BinSightYOLO26n.mlpackage"
EXPECTED_NAMES = {0: "paper", 1: "plastic", 2: "metal"}
EXPECTED_SHA256 = "958e49a5fc8358442497bcb9ce98ca9b8e48a6a1da3bfb92539ac9d14f58ed67"


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: pathlib.Path) -> tuple[str, int, int]:
    """An .mlpackage is a directory. Hash it as (relative path, content) pairs.

    A plain per-file hash would be blind to renames and to files appearing or
    disappearing, so the path is folded in alongside the bytes.
    """
    h = hashlib.sha256()
    total = count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(sha256_file(path).encode())
        total += path.stat().st_size
        count += 1
    return h.hexdigest(), total, count


def verify_source(model, weights: pathlib.Path, expected_sha: str,
                  allow_any_name: bool = False) -> dict:
    """Refuses to export anything that is not the accepted detector."""
    from ultralytics.nn.modules.head import Classify, Detect, OBB, Pose, Segment
    from ultralytics.nn.tasks import DetectionModel

    core = model.model
    head = core.model[-1]
    names = {int(k): str(v) for k, v in model.names.items()}
    digest = sha256_file(weights)
    failures = []

    if weights.name != "best.pt" and not allow_any_name:
        failures.append(f"source is {weights.name!r}, not best.pt")
    if digest != expected_sha:
        failures.append(f"SHA-256 {digest} != expected {expected_sha}")
    if model.task != "detect":
        failures.append(f"task is {model.task!r}, not 'detect'")
    if not isinstance(core, DetectionModel):
        failures.append(f"model is {type(core).__name__}, not DetectionModel")
    for wrong in (Segment, Pose, OBB, Classify):
        if isinstance(head, wrong):
            failures.append(f"head is {wrong.__name__}, not a plain Detect head")
    if not isinstance(head, Detect):
        failures.append(f"head is {type(head).__name__}, not Detect")
    if int(head.nc) != 3:
        failures.append(f"nc is {head.nc}, not 3")
    if names != EXPECTED_NAMES:
        failures.append(f"names {names} != {EXPECTED_NAMES}")

    info = {
        "weights": str(weights.relative_to(ROOT)),
        "sha256": digest,
        "size_bytes": weights.stat().st_size,
        "task": model.task,
        "model_class": type(core).__name__,
        "head_class": type(head).__name__,
        "nc": int(head.nc),
        "names": names,
        "parameters": sum(p.numel() for p in core.parameters()),
        "end2end": bool(getattr(head, "end2end", False)),
        "max_det": int(getattr(head, "max_det", 0)),
        "reg_max": int(getattr(head, "reg_max", 0)),
        "strides": [int(s) for s in head.stride.tolist()],
    }
    if failures:
        for f in failures:
            print(f"  REFUSING: {f}")
        raise SystemExit(2)
    return info


def describe_package(path: pathlib.Path) -> dict:
    """Reads the exported package back with coremltools and reports its schema."""
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(path))
    spec = mlmodel.get_spec()

    def io_entry(feature) -> dict:
        kind = feature.type.WhichOneof("Type")
        entry = {"name": feature.name, "kind": kind}
        if kind == "imageType":
            it = feature.type.imageType
            entry.update(
                width=int(it.width),
                height=int(it.height),
                colorspace=it.ColorSpace.Name(it.colorSpace),
            )
        elif kind == "multiArrayType":
            mt = feature.type.multiArrayType
            entry.update(
                shape=[int(d) for d in mt.shape],
                dtype=mt.ArrayDataType.Name(mt.dataType),
            )
        if feature.shortDescription:
            entry["description"] = feature.shortDescription
        return entry

    meta = dict(spec.description.metadata.userDefined)
    return {
        "path": str(path.relative_to(ROOT)),
        "spec_version": spec.specificationVersion,
        "inputs": [io_entry(f) for f in spec.description.input],
        "outputs": [io_entry(f) for f in spec.description.output],
        "short_description": spec.description.metadata.shortDescription,
        "author": spec.description.metadata.author,
        "license": spec.description.metadata.license,
        "user_defined_metadata": meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--expect-sha256", default=EXPECTED_SHA256,
                        help="SHA-256 the source checkpoint must have. Defaults to "
                             "the currently accepted model, so exporting the wrong "
                             "file stays an error rather than a surprise. Pass the "
                             "new digest deliberately when a different checkpoint "
                             "has been accepted.")
    parser.add_argument("--allow-any-name", action="store_true",
                        help="accept a source filename other than best.pt, e.g. a "
                             "checkpoint already copied into models/pytorch/.")
    args = parser.parse_args()

    weights = (ROOT / args.weights).resolve()
    dest = (ROOT / args.dest).resolve()
    if not weights.exists():
        print(f"ERROR: {weights} not found")
        return 1

    from ultralytics import YOLO

    print("=== source verification ===")
    model = YOLO(str(weights))
    source = verify_source(model, weights, args.expect_sha256, args.allow_any_name)
    for k, v in source.items():
        print(f"  {k}: {v}")

    print("\n=== export ===")
    produced = pathlib.Path(
        model.export(
            format="coreml",
            imgsz=args.imgsz,
            half=True,      # mlprogram is FP16 already; explicit for the record
            int8=False,     # no calibration data in this phase, deliberately
            nms=False,      # forced anyway: end2end models carry no NMS
            batch=1,
            dynamic=False,  # fixed 640x640 keeps the iOS side simple
            device="cpu",   # tracing on CPU avoids MPS dtype quirks
            verbose=False,
        )
    ).resolve()
    print(f"  ultralytics wrote: {produced}")

    # Canonical location. One "final" model, not several.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.copytree(produced, dest) if produced.is_dir() else shutil.copy2(produced, dest)
    print(f"  canonical copy   : {dest.relative_to(ROOT)}")

    if weights.stat().st_size != source["size_bytes"] or sha256_file(weights) != source["sha256"]:
        print("  REFUSING: best.pt changed during export")
        return 2
    print("  best.pt unchanged after export: yes")

    print("\n=== package schema ===")
    described = describe_package(dest)
    digest, total, files = tree_digest(dest)
    described.update(tree_sha256=digest, size_bytes=total, file_count=files)
    for k in ("spec_version", "size_bytes", "file_count", "tree_sha256"):
        print(f"  {k}: {described[k]}")
    for f in described["inputs"]:
        print(f"  INPUT  {f}")
    for f in described["outputs"]:
        print(f"  OUTPUT {f}")

    summary = {
        "source": source,
        "export_options": {
            "format": "coreml", "imgsz": args.imgsz, "half": True, "int8": False,
            "nms": False, "nms_reason": "forced False by Ultralytics for end2end models",
            "batch": 1, "dynamic": False, "device": "cpu",
        },
        "coreml": described,
        "ultralytics_wrote": str(produced.relative_to(ROOT)) if produced.is_relative_to(ROOT) else str(produced),
    }
    out = ROOT / "models" / "coreml" / "export_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
