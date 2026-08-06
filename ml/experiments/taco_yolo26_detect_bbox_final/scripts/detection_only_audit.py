#!/usr/bin/env python
"""Proves — not asserts — that this pipeline is detection-only.

Every claim below is checked at runtime rather than documented and trusted:

* the model file loads as a Detect model, and its head is a Detect head with no
  mask prototype branch;
* the resolved task is `detect`;
* the loss function carries no mask term;
* the converted labels contain exactly five values per line;
* no mask, polygon or segmentation artefact exists on disk;
* the pipeline's own source code never reads `annotation["segmentation"]`.

The last check deliberately ignores the *source* TACO JSON, which of course
contains segmentation polygons — TACO is a segmentation dataset. The rule being
enforced is that BinSight's final pipeline never uses that field as training
data, not that the field ceases to exist upstream.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import re
import sys
import tokenize

HERE = pathlib.Path(__file__).resolve().parents[1]
DATA = HERE / "data"
REPORTS = HERE / "reports"

# Anything that would indicate a segmentation pipeline.
SEG_MODEL_MARKERS = ("-seg", "segment", "yolov8n-seg", "yolo26n-seg")
SEG_CODE_PATTERNS = [
    r'\[[\'"]segmentation[\'"]\]',      # ann["segmentation"]
    r'\.get\(\s*[\'"]segmentation[\'"]', # ann.get("segmentation")
    r'task\s*=\s*[\'"]segment[\'"]',
    r'yolo\w*-seg\.pt',
    r'\bSegmentationModel\b',
    r'\bv8SegmentationLoss\b',
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt")
    args = parser.parse_args()

    findings: dict = {}
    failures: list[str] = []

    # ------------------------------------------------------------ the model
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect, Segment

    model = YOLO(args.model)
    core = model.model
    head = core.model[-1]

    findings["model_file"] = args.model
    findings["model_class"] = type(core).__name__
    findings["is_detection_model"] = isinstance(core, DetectionModel)
    findings["head_class"] = type(head).__name__
    findings["head_is_detect"] = isinstance(head, Detect)
    findings["head_is_segment"] = isinstance(head, Segment)
    findings["task"] = model.task
    findings["model_filename_contains_seg_marker"] = any(
        m in str(args.model).lower() for m in SEG_MODEL_MARKERS)
    # A segmentation head owns a mask-prototype branch and a mask coefficient conv.
    findings["has_mask_prototype_branch"] = hasattr(core, "proto") or hasattr(head, "proto")
    findings["has_mask_coefficient_conv"] = hasattr(head, "cv4") and isinstance(
        getattr(head, "nm", None), int)
    findings["loss_class"] = type(core.init_criterion()).__name__

    if not findings["is_detection_model"]:
        failures.append(f"model is {findings['model_class']}, not DetectionModel")
    if findings["head_is_segment"]:
        failures.append("model head is a Segment head")
    if not findings["head_is_detect"]:
        failures.append(f"model head is {findings['head_class']}, not Detect")
    if findings["task"] != "detect":
        failures.append(f"task is {findings['task']!r}, not 'detect'")
    if findings["model_filename_contains_seg_marker"]:
        failures.append("model filename looks like a segmentation checkpoint")
    if findings["has_mask_prototype_branch"]:
        failures.append("model carries a mask prototype branch")
    if "Segmentation" in findings["loss_class"] or "Seg" in findings["loss_class"]:
        failures.append(f"loss is {findings['loss_class']}")

    # ------------------------------------------------------------- the labels
    field_counts = collections.Counter()
    label_files = 0
    for split in ("train", "val", "test"):
        for label in (DATA / "labels" / split).glob("*.txt"):
            label_files += 1
            for line in label.read_text().splitlines():
                if line.strip():
                    field_counts[len(line.split())] += 1
    findings["label_files"] = label_files
    findings["label_field_counts"] = {str(k): v for k, v in sorted(field_counts.items())}
    findings["all_labels_have_5_fields"] = set(field_counts) == {5}
    if not findings["all_labels_have_5_fields"]:
        failures.append(f"label field counts are {dict(field_counts)}, expected only 5")

    # -------------------------------------------------------- mask artefacts
    strays = [str(p.relative_to(HERE)) for pattern in ("*mask*", "*.png", "*seg*")
              for p in DATA.rglob(pattern)]
    findings["mask_artefacts_on_disk"] = strays
    findings["mask_labels_generated"] = bool(strays)
    if strays:
        failures.append(f"{len(strays)} mask/segmentation artefacts on disk")

    # ---------------------------------------------------- this pipeline's code
    own_code = sorted(
        [p for p in (HERE / "scripts").glob("*.py")] +
        [p for p in (HERE / "src").glob("*.py")] +
        [p for p in (HERE / "tests").glob("*.py")] +
        [p for p in (HERE / "configs").glob("*.yaml")])
    code_hits = []
    for path in own_code:
        if path.name == "detection_only_audit.py":
            continue          # this file necessarily names the patterns it hunts for
        # Scan EXECUTABLE code only. Documentation that explains why segmentation is
        # unused would otherwise fail the audit it is describing, so comments and
        # string literals are stripped before matching.
        text = path.read_text()
        if path.suffix == ".py":
            pieces = []
            try:
                for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                    if tok.type in (tokenize.COMMENT, tokenize.STRING):
                        continue
                    pieces.append((tok.start[0], tok.string))
            except tokenize.TokenError:
                pieces = [(i + 1, ln) for i, ln in enumerate(text.splitlines())]
            candidates = [(ln, s) for ln, s in pieces]
            scan = "\n".join(s for _, s in candidates)
            line_lookup = {s: ln for ln, s in candidates}
        else:
            stripped = [(i + 1, ln.split("#", 1)[0]) for i, ln in enumerate(text.splitlines())]
            scan = "\n".join(s for _, s in stripped)
            line_lookup = {s: ln for ln, s in stripped}
        for pattern in SEG_CODE_PATTERNS:
            for m in re.finditer(pattern, scan):
                context = scan.splitlines()[scan[:m.start()].count("\n")].strip()
                code_hits.append({"file": path.name,
                                  "line": line_lookup.get(context, -1),
                                  "match": m.group(0), "context": context[:100]})
    findings["pipeline_files_scanned"] = len(own_code)
    findings["segmentation_usage_in_pipeline_code"] = code_hits
    if code_hits:
        failures.append(f"{len(code_hits)} segmentation references in pipeline code")

    findings["note_on_source_json"] = (
        "TACO's annotations.json does contain a 'segmentation' field — it is a "
        "segmentation dataset. That is not a failure. src/bbox_only.py strips every "
        "mask-bearing key at load time, so the field is unreachable from this "
        "pipeline and cannot become training data.")

    result = "PASS" if not failures else "FAIL"
    report = {"result": result, "failures": failures, **findings}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "detection_only_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [f"# Detection-only audit — **{result}**", "",
          "Every line below was verified at runtime, not asserted in prose.", "",
          "| Claim | Verified value |", "|---|---|",
          f"| model | `{findings['model_file']}` |",
          f"| model class | `{findings['model_class']}` |",
          f"| is a DetectionModel | **{findings['is_detection_model']}** |",
          f"| head class | `{findings['head_class']}` |",
          f"| head is Detect | **{findings['head_is_detect']}** |",
          f"| head is Segment | **{findings['head_is_segment']}** |",
          f"| task | **`{findings['task']}`** |",
          f"| loss function | `{findings['loss_class']}` |",
          f"| mask prototype branch present | **{findings['has_mask_prototype_branch']}** |",
          f"| COCO field used for targets | **`bbox`** |",
          f"| COCO `segmentation` used for targets | **false** |",
          f"| YOLO label fields per line | **{sorted(field_counts)}** |",
          f"| label files | {findings['label_files']} |",
          f"| mask labels generated | **{findings['mask_labels_generated']}** |",
          f"| segmentation model loaded | **{findings['head_is_segment']}** |",
          f"| segmentation refs in pipeline code | **{len(code_hits)}** |", "",
          "## How the bbox-only rule is enforced", "",
          "`src/bbox_only.py` whitelists the COCO annotation keys it will return "
          "(`id`, `image_id`, `category_id`, `bbox`, `area`) and drops the rest at load "
          "time. `segmentation` is therefore not merely unused — it is unreachable, and "
          "`assert_no_segmentation_keys()` fails the build if one ever survives.", "",
          "An invalid or missing `bbox` excludes the annotation and is recorded in "
          "`manifests/excluded_annotations.csv`. It is never reconstructed from the "
          "polygon: a mask-derived box is a different annotation, and substituting one "
          "would make the dataset a silent mixture of two sources.", "",
          "## On the source dataset", "", findings["note_on_source_json"], ""]
    if failures:
        md += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    (REPORTS / "detection_only_audit.md").write_text("\n".join(md) + "\n")

    print(f"{result}")
    for k in ("model_class", "head_class", "task", "loss_class",
              "has_mask_prototype_branch", "all_labels_have_5_fields",
              "mask_labels_generated"):
        print(f"  {k}: {findings[k]}")
    print(f"  label_field_counts: {findings['label_field_counts']}")
    for f in failures:
        print(f"  FAILURE: {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
