# BinSight YOLO26n — CoreML export report

> ## Superseded — re-exported after a 40-epoch continuation run
>
> The figures below describe the **first** export, from the 25-epoch model
> (`958e49a5…f58ed67`). The shipped model is now the 65-epoch continuation
> (`f6ca7233…276c96`), re-exported with the same script and the same options.
>
> The exported **contract is unchanged**: input `image` 640×640 RGB with
> `scale = 1/255`, one MultiArray output `[1, 300, 6]` of
> `[x1, y1, x2, y2, confidence, class_id]`, end2end and NMS-free. No Swift code
> needed changing.
>
> Current parity, from `models/coreml/parity_report.json`:
>
> | | first export | **current** |
> |---|---:|---:|
> | PyTorch / CoreML detections | 27 / 29 | 40 / 40 |
> | Geometry-matched | 27 / 27 | 38 / 40 |
> | Class agreement | 96.30% | **100%** |
> | Mean matched-box IoU | 0.9928 | 0.9924 (min 0.9685) |
> | Mean confidence delta | 0.017 paired | **0.0168 distribution** |
>
> Two measurement flaws were found and fixed while verifying the new export, and
> both are worth knowing before reading the numbers below:
>
> 1. **Greedy matching is wrong in dense scenes.** With 19 boxes overlapping at
>    IoU > 0.99 on a pile of screws, it paired detections with the wrong twin and
>    invented confidence deltas of 0.24. Replaced with optimal assignment.
> 2. **Pairwise confidence delta is not a sound metric at all there.** Optimal
>    assignment made it *worse* (0.24 → 0.39), because many assignments score
>    almost the same total IoU. Confidence is now compared as a sorted
>    distribution, which is immune to which twin was paired with which.
>
> The single remaining outlier (0.1619) is real and benign: PyTorch emitted two
> near-identical boxes on one bottle, splitting confidence across 0.762 and
> 0.537, where CoreML suppressed the duplicate into one 0.924 detection.


## Source model

| | |
|---|---|
| Path | `runs/detect/binsight_yolo26n_pretrained_final/weights/best.pt` |
| SHA-256 | `958e49a5fc8358442497bcb9ce98ca9b8e48a6a1da3bfb92539ac9d14f58ed67` |
| Size | 5,385,157 bytes |
| Task | `detect` (DetectionModel / Detect head) |
| Classes | **3** — `0 paper · 1 plastic · 2 metal` |
| Parameters | 2,504,970 |

The exporter refuses to run unless the filename is `best.pt`, the SHA-256 matches
the accepted checkpoint, the task is `detect`, `nc == 3` and the names match
exactly. `last.pt` and the original pretrained `yolo26n.pt` cannot be substituted
by accident. `best.pt` was re-hashed after the export and is unchanged.

## Export

| | |
|---|---|
| Ultralytics | 8.4.9 |
| coremltools | 9.0 |
| torch | 2.13.0 |
| numpy | 2.3.5 (see below) |
| macOS | 26.5.2 (25F84), arm64 (Apple M1) |
| Script | `scripts/export_yolo26n_coreml.py` |

Options: `format=coreml, imgsz=640, half=True, int8=False, nms=False, batch=1,
dynamic=False, device=cpu`.

**Precision.** The package is FP16. `mlprogram` conversion is FP16 by default in
coremltools, so `half=True` is explicit rather than load-bearing on this path.
INT8 was **not** used: no calibration data exists and quantisation is out of
scope for this phase.

**numpy had to be pinned down to 2.3.5.** Ultralytics declares
`numpy>=1.14.5,<=2.3.5` for CoreML export, and that ceiling is real. Under the
numpy 2.4.6 originally installed, conversion aborts inside the attention block
with `TypeError: only 0-dimensional arrays can be converted to Python scalars`:
coremltools' `_cast` handler calls `int(x.val)` on `array([400])`, a size-1 but
1-dimensional array, which numpy 2.4 refuses. Only `coremltools` and this numpy
pin were changed; torch, Ultralytics and MPS were re-verified working afterwards.

## CoreML model

| | |
|---|---|
| Path | `models/coreml/BinSightYOLO26n.mlpackage` |
| Format | `.mlpackage` — ML Program, specification version 6 (iOS 16+) |
| Size | 5,005,354 bytes (5.01 MB), 3 files |
| Content SHA-256 | `161cad1b14ff12238129636d06d03ffc0c9a0648ac5e6d286572e82b5969e088` |

The package is a directory, so the digest above is over sorted
`(relative path, file SHA-256)` pairs rather than a single file — a per-file hash
alone would be blind to renames or to files appearing and disappearing.

### Input

| | |
|---|---|
| Name | `image` |
| Kind | **Image** (RGB) |
| Size | 640 x 640, fixed |
| Scaling | built in: `scale = 1/255`, `bias = [0, 0, 0]` |

The input is a CoreML **image**, not a raw array. Normalisation is inside the
model — the caller must not divide by 255 itself.

### Output

| | |
|---|---|
| Name | `var_1441` |
| Kind | MultiArray, FLOAT32 |
| Shape | `[1, 300, 6]` |

### Class metadata

Carried in the package's `user_defined_metadata`, so iOS can read labels from the
model instead of hardcoding them:

```
names   = {0: 'paper', 1: 'plastic', 2: 'metal'}
task    = detect
end2end = True
imgsz   = [640, 640]
stride  = 32
```

License metadata reads `AGPL-3.0 License (https://ultralytics.com/license)`, inherited from Ultralytics —
relevant if this app is ever distributed.

The `shortDescription` field embeds the absolute local dataset path from the
training machine. Harmless, but worth rewriting before any public release.

## YOLO26 output semantics

This was determined from the installed Ultralytics and the exported model, not
from YOLOv5/v8 convention — and it differs from both.

**The model is end-to-end (NMS-free).** `head.end2end` is `True` and the
checkpoint carries a one-to-one head, so it emits already-deduplicated
detections. Ultralytics enforces this at export time: `'nms=True' is not
available for end2end models. Forcing 'nms=False'` (`exporter.py:443`).

**No post-processing is embedded in the package.** It is a plain ML Program, not
a pipeline, with a single output and no classifier head:

- `spec.WhichOneof('Type')` = `mlProgram`, not `pipeline`
- output count = 1
- `predictedFeatureName` = empty

**The single `[1, 300, 6]` tensor is `[x1, y1, x2, y2, confidence, class_id]`**,
where 300 is `head.max_det`:

| Column | Meaning |
|---|---|
| 0-3 | `x1, y1, x2, y2` — **corner** coordinates in 640x640 **input-pixel** space |
| 4 | confidence, already the max class probability (0-1) |
| 5 | class id as a float — `0.0` paper, `1.0` plastic, `2.0` metal |

Two traps worth recording:

1. **The boxes are xyxy, not xywh.** Ultralytics' own `Detect.postprocess`
   docstring describes the columns as `[x, y, w, h, max_class_prob, class_index]`.
   For this checkpoint that is wrong. Decoding as xywh yields negative
   coordinates; decoding as xyxy reproduces Ultralytics' `predict()` to within a
   pixel. This was verified empirically, and the empirical result was taken over
   the docstring.
2. **Rows are not confidence-filtered.** All 300 slots are always returned,
   padded with low-scoring entries. The application must apply its own confidence
   threshold.

Rows are ordered by descending confidence, and coordinates are **not** clamped to
the image — boxes may extend slightly outside `[0, 640]`.

## Parity results

10 held-out test images, deterministic (seed 42, same procedure as
`evaluate_yolo26n.py`), confidence threshold 0.25, IoU floor
0.5.

| | |
|---|---|
| Test images | 10 |
| PyTorch detections | 27 |
| CoreML detections | 29 |
| Matched | **27 / 27** |
| Class agreement | **96.3%** (26/27) |
| Mean matched-box IoU | **0.9928** (min 0.9771) |
| Mean absolute confidence delta | 0.01717 (max 0.09654) |
| Unmatched PyTorch | 0 |
| Unmatched CoreML | 2 |

**Matching is on geometry alone**, with class compared afterwards. An earlier
version of this check required the class to agree *before* pairing, which makes
"class agreement" tautologically 100% — a disagreeing pair simply never becomes a
pair and reappears as one unmatched detection on each side. The 96.3% above is
the honest figure.

Preprocessing was held identical so the comparison isolates the export: every
test image is 416x416 square, so the letterbox to 640 is a pure uniform scale
with no padding, and the CoreML side resizes with cv2 `INTER_LINEAR` exactly as
Ultralytics' `LetterBox` does. Resizing with PIL instead shifts confidences by
~0.004, which would otherwise be misread as export error.

### Notable differences — all benign

**One class flip**, at IoU 0.986 — the same box, relabelled:
`paper 0.278` (PyTorch) → `plastic 0.284` (CoreML), on a translucent
plastic bag in `plastic510_jpg.rf.f45cf2c491141e41…`. The model is genuinely undecided
between the two labels here; FP16 rounding tips a near-tie.

**Two extra CoreML detections**, at confidence 0.284 and 0.252 — both within
0.04 of the 0.25 cutoff, pushed across it by FP16 rounding.

Geometry shows no drift at all: every matched pair exceeds IoU 0.95, and the
worst is 0.9771. Confidence deltas above 0.05 occur in 3 of 27 pairs, all in
one crowded image of screws where every detection sits in the 0.28-0.46 band.

### Confident examples — supplementary

The seed-42 sample is dominated by low-confidence boxes, which is where FP16
noise is worst. The most confident example of each class behaves very differently:

| Class | PyTorch | CoreML | IoU |
|---|---|---|---|
| paper | 0.9615 | 0.9619 | 0.9994 |
| plastic | 0.9588 | 0.9590 | 0.9967 |
| metal | 0.9761 | 0.9756 | 0.9969 |

Confidence deltas below 0.0005. Export error is confined to the low-confidence
regime and does not affect detections the app would act on.

Side-by-side previews: `artifacts/coreml_parity/` (10 images, PyTorch left,
CoreML right).

## iOS integration notes

Factual, from the exported model. Nothing in the app has been changed yet.

**Input.** A 640x640 RGB image — `CVPixelBuffer` or `CGImage`, feature name
`image`. Do not normalise; `scale=1/255` is inside the model. The camera
frame must be resized to a square 640x640. The training images were square, so
letterboxing was never exercised — if the app letterboxes a 16:9 frame instead of
squashing it, the padding offset must be undone when mapping boxes back.

**Output.** One MultiArray `var_1441`, shape `[1, 300, 6]`, FP32.
⚠️ The name is compiler-generated, not stable across re-exports. Read it from the
model description rather than hardcoding the string.

**Can Vision wrap it cleanly? No.** `VNCoreMLRequest` only produces
`VNRecognizedObjectObservation` for models shaped as CoreML detector pipelines,
with `confidence` and `coordinates` outputs. This model has neither — a single
unnamed MultiArray. Vision will return a `VNCoreMLFeatureValueObservation`
wrapping the raw array, so Vision buys nothing here beyond image conversion.
Driving `MLModel` directly is the cleaner path.

**Manual decoding required: YES.** For each of the 300 rows: read
`[x1, y1, x2, y2, conf, cls]`, drop rows below the confidence threshold, scale
`xyxy` from 640-space to the displayed image, and clamp to bounds.

**NMS required: NO.** The head is end-to-end and the rows are already
deduplicated. Adding NMS would be actively wrong — it would suppress legitimately
overlapping objects, which are common in this dataset (the crowded screws image
carries 11 valid overlapping `metal` boxes).

**Coordinate system.** Model space is 640x640 pixels, origin top-left, y down —
which matches UIKit but **not** Vision's normalised bottom-left convention. There
is no normalisation step, so a Vision-style `VNImageRectForNormalizedRect` would
be wrong. Boxes may fall slightly outside `[0, 640]` and need clamping.

**Suggested threshold.** 0.25 was used throughout evaluation. Given that test
recall is 0.578 and precision 0.682, the app may want a higher bar for a
confident single-label verdict.

## Acceptance gate

| Check | Result |
|---|---|
| source is final `best.pt` (SHA verified) | PASS |
| task = detect | PASS |
| nc = 3 | PASS |
| classes = paper/plastic/metal | PASS |
| `.mlpackage` exported | PASS |
| CoreML model loads unmodified | PASS |
| input schema understood | PASS |
| output schema understood | PASS |
| inference runs on real test images | PASS |
| CoreML produces real detections | PASS (29 on 10 images) |
| classes broadly match PyTorch | PASS (96.3%, 1 near-threshold flip) |
| boxes broadly match PyTorch | PASS (mean IoU 0.9928) |
| no systematic coordinate error | PASS (min IoU 0.9771) |
| no segmentation/classification output | PASS (single detect tensor) |
| report created | PASS |

## Files

| Path | |
|---|---|
| `scripts/export_yolo26n_coreml.py` | export + source verification |
| `scripts/verify_coreml_parity.py` | PyTorch vs CoreML parity |
| `models/coreml/BinSightYOLO26n.mlpackage` | **the deliverable** |
| `models/coreml/export_summary.json` | schema + metadata, machine-readable |
| `models/coreml/parity_report.json` | full per-detection parity record |
| `artifacts/coreml_parity/` | side-by-side previews (git-ignored) |

Not done in this phase: no Swift or Xcode changes, no TFLite export, nothing
committed or pushed.
