# BinSight detection pipeline

How a camera frame becomes boxes on screen. This is the whole production
inference path — there is nothing else.

```
BinSight camera  (AVCaptureSession, 1280x720, rotated to portrait 720x1280 BGRA)
      │          throttled to 4 fps at the capture boundary (FrameSampler)
      ▼
letterbox to 640x640   scale = min(640/w, 640/h), centred, padded RGB(114,114,114)
      │                for 720x1280: scale 0.5, padX 140, padY 0
      ▼
BinSightYOLO26n.mlpackage   CoreML ML Program, FP16, computeUnits = .all
      │                     input `image` carries scale = 1/255 internally
      ▼
[1, 300, 6]   rows of [x1, y1, x2, y2, confidence, class_id]
      │        in 640x640 model-input pixels
      ▼
confidence threshold  >= 0.25, plus validity checks (finite, x2 > x1, y2 > y1,
      │               known class id, non-degenerate after clamping)
      ▼
xyxy un-letterbox   (v - pad) / scale, clamped to the frame
      │
      ▼
[Detection]   paper / plastic / metal, boxes in camera-buffer pixels
      │
      ▼
SwiftUI overlay   buffer pixels -> aspect-fill preview points
```

## The five things that are easy to get wrong

**No separate classifier.** The model localises *and* classifies. There is no
second stage and no `WasteClassifying` implementation in the production path.

**No segmentation.** Detect head only — boxes, not masks.

**One box on screen, by policy.** The decoder returns everything above
threshold; `LiveDetectionEngine` then surfaces only the strongest detection
(`singleObjectMode`). This is a *display* decision, not a model property — a
detector cannot be trained to emit exactly one box. It exists because test
precision at the 0.25 threshold is 0.7533, so roughly one shown box in four would
otherwise be wrong.

**No NMS.** YOLO26 is end-to-end (`end2end = True` in the package metadata, a
one-to-one head in the checkpoint), so the rows are already deduplicated.
Ultralytics *forces* `nms=False` when exporting such a model. Adding NMS on the
app side would suppress genuinely overlapping objects — the evaluation set has an
image with 19 valid overlapping `metal` boxes.

**No manual normalisation.** The exported input layer carries `scale = 1/255`
and `bias = [0,0,0]`. Dividing by 255 in Swift would hand the model a
near-black image and it would detect nothing. Pixels go in as 8-bit BGRA.

**Boxes are xyxy, not xywh.** Ultralytics' own `Detect.postprocess` docstring
says `[x, y, w, h, ...]`; for this checkpoint that is wrong. Decoding as xywh
produces negative coordinates. Verified empirically against the PyTorch model
during the CoreML export phase.

## Model

| | |
|---|---|
| Bundled at (**the committed copy**) | `BinSight/Resources/Models/BinSightYOLO26n.mlpackage` |
| Export output (not committed) | `models/coreml/BinSightYOLO26n.mlpackage` |
| PyTorch release artifact | `models/pytorch/BinSightYOLO26n.pt` — sha256 `f6ca7233…276c96` |
| Training run (not committed) | `runs/detect/binsight_yolo26n_continued/` |
| Format | `.mlpackage`, ML Program, spec v6 (iOS 16+), FP16, 4.8 MB |
| Input | `image`, 640x640 RGB, `scale = 1/255` built in |
| Output | one MultiArray, `[1, 300, 6]`, FP32 |
| Classes | `0 paper · 1 plastic · 2 metal` |
| Threshold | 0.25 (`DetectionConfiguration.confidenceThreshold`) |
| Displayed boxes | 1 — `DetectionConfiguration.singleObjectMode` |

`scripts/export_yolo26n_coreml.py` writes to `models/coreml/`, which is the
export workspace; the package is then copied into `BinSight/Resources/Models/`,
which is what Xcode compiles and bundles. The two were verified byte-identical by
content hash (`afab107f…7a1a45`), and **only the app's copy is committed** —
tracking both would put two identical 4.8 MB blobs in Git. `models/coreml/` keeps
its `export_summary.json` and `parity_report.json`, which are small and are the
verification record.

After any re-export:

```bash
rm -rf BinSight/Resources/Models/BinSightYOLO26n.mlpackage && cp -R models/coreml/BinSightYOLO26n.mlpackage BinSight/Resources/Models/
python scripts/make_ios_test_fixtures.py
```

**The output feature name is not hardcoded.** The exported tensor is currently
called `var_1441`, which is compiler-generated and unstable across re-exports.
`YOLODetectionService` resolves the single MultiArray output from
`MLModelDescription` at load time, and fails loudly if the *shape* is wrong.

## Where each piece lives

| File | Responsibility |
|---|---|
| `Core/Detection/Detection.swift` | `Detection`, `DetectedClass` (3 classes, not `WasteCategory`'s 6) |
| `Core/Detection/LetterboxTransform.swift` | aspect-preserving fit and its inverse |
| `Core/Detection/DetectionDecoder.swift` | `[1,300,6]` → `[Detection]`, pure, model-free |
| `Core/Detection/DetectionConfiguration.swift` | thresholds and the tensor contract |
| `Core/Detection/YOLODetectionService.swift` | actor: loads CoreML once, preprocesses, infers |
| `Features/Scanner/Live/LiveDetectionEngine.swift` | frame loop, backpressure, published state |
| `Features/Scanner/Components/DetectionOverlayView.swift` | boxes, labels, confidence |
| `Features/Scanner/Camera/PreviewCropGeometry.swift` | the one coordinate-transform utility |

## Coordinate systems

Three spaces, and every hop is explicit:

1. **Model space** — 640x640 input pixels, origin top-left, y down. Boxes are
   *not* clamped by the model and can overhang.
2. **Buffer space** — 720x1280 camera pixels, origin top-left. `Detection.boundingBox`
   is always in this space. `LetterboxTransform.unletterbox` converts and clamps.
3. **Preview space** — SwiftUI points. The preview is `.resizeAspectFill`, so it
   shows a *centre crop* of the buffer.
   `PreviewCropGeometry.previewRect(forBufferPixelRect:bufferSize:previewSize:)`
   is the only place this conversion happens.

Scaling by `previewSize / bufferSize` is wrong whenever the aspect ratios differ
— which here is always. It is wrong by a translation as well as a scale, so
boxes drift toward the centre. `DetectionPreviewProjectionTests` pins this,
including a test that the naive version is measurably wrong.

Boxes for objects the preview has cropped away are drawn off-screen and clipped,
not clamped: pinning a box to the border would claim the object is at the edge
of the frame when it is outside it.

Nothing uses Vision, so Vision's normalised, bottom-left-origin convention never
enters. `VNCoreMLRequest` cannot wrap this model cleanly anyway — it has no
`confidence`/`coordinates` outputs, so it is not a CoreML detector pipeline.

## Frame scheduling

One inference in flight, newest frame only, bounded memory — with no explicit
"busy" flag:

* `CameraSessionController.makeFrameStream()` uses
  `AsyncStream(bufferingPolicy: .bufferingNewest(1))`, so at most one pixel
  buffer waits and a slow consumer drops old frames instead of growing a queue.
* `LiveDetectionEngine`'s loop `await`s each inference before pulling the next
  frame, so the loop body is sequential by construction.
* `AVCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = true` at the source.
* `FrameSampler` throttles to 4 fps before any work happens.

Every run carries a generation number, so a result landing after `stop()` is
discarded rather than painting boxes over a stopped preview.

The model, the `CIContext`, the `CVPixelBufferPool` and the output scratch array
are created **once** in the actor's initialiser, never per frame.

## Retired: the classification path

`LiveScanEngine`, `WasteClassifying`, `MockWasteClassifier`, `PredictionSmoother`
and `ScoreVector` implemented the previous six-class TrashNet classifier. They
are **disconnected from the production runtime** — nothing in the shipping path
constructs them, and no model backs them — but they are still compiled and
tested, because their tests document behaviour that would otherwise be deleted
without replacement.

`ScannerDisplayState` and `ResultCard` are still live, but only for the pinned
and unavailable screens: camera denied, camera unavailable, model unavailable,
and the DEBUG/UI-test scenarios. When the detector is running, the bottom
surface is `DetectionSummaryCard`.

## Tests

| Suite | Covers |
|---|---|
| `LetterboxTransformTests` | scale/pad for 416², 720x1280, 1920x1080, 1080x1920; round-trip; clamping; NaN |
| `DetectionDecoderTests` | class mapping, threshold boundary at 0.25, invalid boxes, xyxy vs xywh, no NMS, caps |
| `DetectionPreviewProjectionTests` | buffer → preview, inverse of the aspect-fill crop, naive-scaling counter-example |
| `YOLODetectionServiceTests` | **static inference against the real bundled model** |

The static test is the one that catches what unit tests structurally cannot:
double normalisation, a vertically flipped letterbox, wrong output indexing and
transposed classes. Its fixtures and expected values are generated by
`scripts/make_ios_test_fixtures.py`, which runs the *same* `.mlpackage` through
coremltools — so a failure means the Swift path is wrong, not that the model
changed.

Regenerate fixtures after any re-export:

```bash
python scripts/make_ios_test_fixtures.py
```

## Measured latency

On the iOS Simulator (no Neural Engine), 720x1280 input:

| | |
|---|---|
| Letterbox (CoreImage) | ~2–4 ms |
| CoreML inference | ~32–136 ms |
| End-to-end | ~42–191 ms (≈ 5.2–23.9 /s) |

Comfortably above the 4 fps the pipeline asks for, and the simulator is the
pessimistic case — a physical device runs the model on the Neural Engine.
