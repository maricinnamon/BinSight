# BinSight

**On-device waste detection for iOS.** BinSight points the iPhone camera at a
scene and draws live bounding boxes around **paper**, **plastic** and **metal**,
using a YOLO26n object detector fine-tuned from COCO-pretrained weights and
exported to CoreML. Everything runs on the phone: no server, no API key, no
frame ever leaves the device.

The detector localises *and* classifies in a single pass. There is no second
classification stage, no segmentation, and — because YOLO26 is end-to-end — no
non-maximum suppression anywhere in the app.

---

## Demo

Not yet recorded. A screenshot or capture will be added once the app has been
filmed on a physical device; nothing is linked here that does not exist.

---

## Features

- Real-time detection from the live camera at ~4 inferences/second
- Bounding boxes with class name and confidence (`Plastic 81%`)
- Three materials: paper, plastic, metal
- Fully on-device CoreML inference — no network, no server, no account
- Bounded memory: one inference in flight, newest frame only, no frame backlog
- Honest failure states: camera denied, camera unavailable, model unavailable

---

## ML pipeline

```
Roboflow object-detection dataset (6 classes)
   ↓  deterministic filtering + class remapping (seed 42, grouped split)
3-class detection dataset — 5,074 images / 16,173 boxes
   ↓  fine-tuning from COCO-pretrained yolo26n.pt
YOLO26n detector, nc=3
   ↓  held-out test evaluation
   ↓  FP16 CoreML export (.mlpackage)
   ↓  PyTorch ↔ CoreML parity validation
iOS app — AVCaptureSession → letterbox → CoreML → SwiftUI overlay
```

---

## Dataset

| | |
|---|---|
| Images | **5,074** |
| Bounding boxes | **16,173** |
| Classes | `0 paper` · `1 plastic` · `2 metal` |
| Split | 4,059 train / 507 val / 508 test |
| Strategy | 80/10/10, seed 42, stratified by dominant class, grouped by source image |

Boxes per class: paper 4,387 · plastic 5,945 · metal 5,841.

The split is grouped by source-image identity so that augmented variants of the
same photo cannot straddle train and test. `scripts/validate_waste_dataset.py`
re-reads everything from disk independently of the preparation script and checks
label geometry, field counts, orphans, corrupt images and cross-split content
hashes. It reports 0 corrupt images, 0 invalid labels and 0 leakage.

### Source and attribution

This project uses a **filtered and re-mapped subset** of a third-party dataset.
It does not claim ownership of the source data.

> **Garbage Classification 3** (v2) — Roboflow Universe workspace
> `material-identification`.
> <https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2>
> Licensed **CC BY 4.0**, read from the `roboflow:` block of the source
> `data.yaml` rather than assumed.

The source has six classes. Three are excluded entirely and three are kept and
renumbered — the source order is *not* the final order:

| Source ID | Source name | Action | Final ID |
|---:|---|---|---:|
| 0 | BIODEGRADABLE | excluded | — |
| 1 | CARDBOARD | excluded | — |
| 2 | GLASS | excluded | — |
| 3 | METAL | kept | **2** |
| 4 | PAPER | kept | **0** |
| 5 | PLASTIC | kept | **1** |

`CARDBOARD` is excluded rather than merged into `paper`: the two are visually
adjacent, and folding one into the other would have made the remaining `paper`
class incoherent. `scripts/prepare_waste_dataset.py` reads the source class order
from the source `data.yaml` instead of hardcoding it, because hardcoding the
final order onto source ids would have mislabelled the dataset silently.

**The raw dataset is not committed** — 123 MB of images and labels. The
preparation and validation scripts, `data.yaml`, the manifest and the reports
are, so the dataset can be rebuilt. See [Reproducing the ML work](#reproducing-the-ml-work).

Full detail: [`datasets/binsight_waste_3class/dataset_report.md`](datasets/binsight_waste_3class/dataset_report.md).

---

## Model

**YOLO26n, fine-tuned from official COCO-pretrained weights. Not trained from
scratch.**

| | |
|---|---|
| Architecture | YOLO26n Detect (Ultralytics 8.4.9), 2.5 M parameters |
| Initialisation | `yolo26n.pt`, 80-class COCO — **606/708 tensors transferred**, head adapted 80 → 3 |
| Training | 25 epochs, batch 4, imgsz 640, AdamW, seed 42, MPS (Apple M1) |
| Best epoch | 25 (no early stopping; 22 of 25 epochs set a new best) |
| Duration | 6.85 h |

The training script refuses to start unless the checkpoint is genuinely
pretrained: it asserts the checkpoint dict is present, the head is `Detect`, and
the BatchNorm running statistics are non-trivial — an untrained model has exactly
zero means and unit variances.

Full detail: [`reports/yolo26n_training_report.md`](reports/yolo26n_training_report.md).

---

## Results

Validation guided checkpoint selection, so **the held-out test split is the
headline number** — quoting validation as final performance would report a figure
the model was selected on.

| Metric | Validation | **Held-out test** |
|---|---:|---:|
| Precision | 0.7086 | **0.6817** |
| Recall | 0.6045 | **0.5777** |
| mAP50 | 0.6779 | **0.6480** |
| mAP75 | 0.5450 | **0.4809** |
| mAP50-95 | 0.5070 | **0.4620** |

Per class, on the held-out test split:

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| paper | 0.6519 | 0.5000 | 0.5713 | 0.4386 |
| plastic | 0.6822 | 0.5629 | 0.6568 | 0.4349 |
| metal | 0.7108 | 0.6700 | 0.7159 | **0.5126** |

The val→test drop (mAP50-95 0.5070 → 0.4620) is about 0.045 — modest, and
consistent with mild selection bias rather than overfitting.

`metal` is the strongest class on both splits: cans and foil have hard edges and
consistent specular highlights. `paper` is the weakest, with recall of exactly
0.50 — it is deformable, often crumpled, and shades into cardboard, a class the
taxonomy deliberately excludes.

---

## CoreML parity

Converting PyTorch → CoreML can silently transpose coordinates, reorder classes
or quantise a model into uselessness, and none of that shows up as an error. So
the exported package was checked against the PyTorch model it came from, on the
same 10 deterministic held-out images:

| | |
|---|---|
| PyTorch detections | 27 |
| CoreML detections | 29 |
| Geometry-matched | **27 / 27** |
| Class agreement | **96.30%** (26/27) |
| Mean matched-box IoU | **0.9928** (min 0.9771) |
| Mean confidence delta | 0.017 (max 0.097) |

Matching is on **geometry alone**, with class compared afterwards. Requiring the
class to agree *before* pairing would make "class agreement" tautologically 100%
— a disagreeing pair simply never becomes a pair. The one disagreement is a
translucent plastic bag labelled `paper 0.278` by PyTorch and `plastic 0.284` by
CoreML: the same box, a genuine near-tie tipped by FP16 rounding. The two extra
CoreML detections both sit within 0.04 of the 0.25 threshold.

Confident detections are unaffected — the strongest example of each class
reproduces at IoU ≥ 0.9967 with confidence deltas under 0.0005.

Full detail: [`reports/coreml_export_report.md`](reports/coreml_export_report.md).

---

## iOS inference architecture

```
AVCaptureSession        1280×720, rotated to portrait 720×1280 BGRA, 4 fps
        ↓
letterbox to 640×640    scale = min(640/w, 640/h), centred, RGB(114,114,114) pad
        ↓               for 720×1280: scale 0.5, padX 140, padY 0
BinSightYOLO26n         CoreML ML Program, FP16, computeUnits = .all
        ↓
[1, 300, 6]             rows of [x1, y1, x2, y2, confidence, class_id]
        ↓
confidence ≥ 0.25       plus finite / x2>x1 / y2>y1 / known-class checks
        ↓
xyxy un-letterbox       (v − pad) / scale, clamped to the frame
        ↓
preview projection      buffer pixels → aspect-fill preview points
        ↓
SwiftUI overlay         box + class + confidence
```

Four things that are easy to get wrong, and how this app handles them:

- **No app-side NMS.** YOLO26 is end-to-end (`end2end = True`); the rows are
  already deduplicated and Ultralytics forces `nms=False` when exporting. Adding
  NMS would suppress genuinely overlapping objects — one evaluation image has 11
  valid overlapping `metal` boxes.
- **No manual normalisation.** The exported input layer carries `scale = 1/255`.
  Dividing again in Swift would hand the model a near-black image.
- **Boxes are xyxy, not xywh.** Ultralytics' own `postprocess` docstring says
  xywh; for this checkpoint that is wrong, and decoding as xywh yields negative
  coordinates. Verified empirically against PyTorch.
- **The output feature name is not hardcoded.** The exported tensor is called
  `var_1441`, which is compiler-generated and unstable across re-exports; it is
  resolved from `MLModelDescription` at load time.

The overlay's coordinate mapping is the one place buffer pixels become preview
points. The preview is `.resizeAspectFill`, so it shows a *centre crop* — scaling
by `previewSize / bufferSize` is wrong by a translation as well as a scale, and a
unit test exists specifically to prove the naive version is wrong.

Full detail: [`docs/DETECTION_PIPELINE.md`](docs/DETECTION_PIPELINE.md).

---

## Performance

> **Measured on the iOS Simulator, which has no Neural Engine.** These are not
> iPhone figures. Physical-device latency has not been measured yet.

Observed across several runs of the `inferenceLatency` test (10 iterations after
a warm-up), on an M1 host:

| | |
|---|---|
| Letterbox (CoreImage) | 3–4 ms |
| CoreML inference | 58–136 ms |
| End-to-end | 116–191 ms (≈ 5.2–8.6 /s) |

The spread is wide because the Simulator shares the host CPU with whatever else
is running; the faster end of the range is the less contended case.

The pipeline targets 4 inferences/second, so even the simulator has headroom.
The model, `CIContext`, pixel-buffer pool and output buffer are created once and
reused; nothing is allocated per frame.

---

## Tech stack

**iOS** — Swift 5, SwiftUI, AVFoundation, CoreML, CoreImage, CoreVideo,
swift-testing, XCTest (UI tests). iOS 17+.

**ML** — Python 3.11, PyTorch 2.13 (MPS), Ultralytics 8.4.9, coremltools 9.0,
NumPy, OpenCV, Pillow, PyYAML.

---

## Repository structure

```
BinSight/                      iOS app
  App/                         entry point
  Core/Detection/              Detection, LetterboxTransform, DetectionDecoder,
                               DetectionConfiguration, YOLODetectionService
  Core/Classification/         retired 6-class classifier types (see below)
  Core/DesignSystem/           theme, palette, glass surfaces
  Features/Scanner/            camera, live engine, overlay, UI components
  Resources/Models/            BinSightYOLO26n.mlpackage  ← bundled model
BinSightTests/                 unit + static CoreML integration tests
BinSightUITests/               scanner screen UI tests
scripts/                       dataset prep, training, evaluation, export, parity
datasets/binsight_waste_3class/  data.yaml + reports (images/labels not committed)
models/coreml/                 CoreML export workspace (package not committed)
reports/                       training and CoreML export reports
docs/                          architecture, model card, device testing
```

### Release artifacts

Two, both committed, and they are the only two:

| Artifact | Path | Size | Role |
|---|---|---:|---|
| **PyTorch** | `models/pytorch/BinSightYOLO26n.pt` | 5.1 MB | the accepted training checkpoint |
| **CoreML** | `BinSight/Resources/Models/BinSightYOLO26n.mlpackage` | 4.8 MB | what the iOS app bundles and runs |

`models/pytorch/BinSightYOLO26n.pt` is byte-identical to the training run's
`best.pt` — SHA-256 `958e49a5fc8358442497bcb9ce98ca9b8e48a6a1da3bfb92539ac9d14f58ed67`,
5,385,157 bytes. It is copied out of the run directory so the release model
survives in Git without the 460 MB of training output around it. With it
committed, CoreML can be re-exported, and the model re-evaluated, **without
retraining**.

**`runs/` is not committed** — 460 MB of `last.pt`, 25 per-epoch checkpoints from
`save_period=1`, batch previews and plots. Only the final checkpoint above is
preserved.

`scripts/export_yolo26n_coreml.py` writes its output to `models/coreml/`, which is
then copied into `BinSight/Resources/Models/`. Only the app's copy is tracked —
committing both would put two byte-identical 4.8 MB blobs in Git. The small JSON
files in `models/coreml/` are the export and parity verification record and *are*
committed.

**Retired code.** `LiveScanEngine`, `WasteClassifying`, `MockWasteClassifier`,
`PredictionSmoother` and `ScoreVector` implemented an earlier six-class TrashNet
classifier. They are disconnected from the production runtime — nothing in the
shipping path constructs them and no model backs them — but they still compile
and their tests still pass, so they were left in place rather than deleted along
with the behaviour those tests document.

---

## Reproducing the ML work

Requires a Python environment with the packages listed under
[Tech stack](#tech-stack), and the source archive from Roboflow (link above)
placed at `datasets/_tmp/archive.zip`.

```bash
python scripts/prepare_waste_dataset.py      # build the 3-class dataset
python scripts/validate_waste_dataset.py     # independent re-read validation
python scripts/preview_waste_dataset.py      # render labels onto images to eyeball
python scripts/train_yolo26n.py --name binsight_yolo26n_pretrained_final \
       --batch 4 --epochs 25 --patience 7 --close-mosaic 5
python scripts/evaluate_yolo26n.py           # val + held-out test metrics
python scripts/export_yolo26n_coreml.py      # FP16 CoreML export
python scripts/verify_coreml_parity.py       # PyTorch ↔ CoreML parity
python scripts/make_ios_test_fixtures.py     # regenerate the Swift test fixtures
```

Steps up to and including training are only needed to reproduce the model from
scratch. Because `models/pytorch/BinSightYOLO26n.pt` is committed, the export,
parity and fixture steps can be run directly against it:

```bash
python scripts/export_yolo26n_coreml.py --weights models/pytorch/BinSightYOLO26n.pt
```

Then copy the exported package into the app target:

```bash
rm -rf BinSight/Resources/Models/BinSightYOLO26n.mlpackage && cp -R models/coreml/BinSightYOLO26n.mlpackage BinSight/Resources/Models/
```

**Training is not bit-reproducible.** The seed is fixed at 42 and the split is
deterministic, but PyTorch's MPS backend does not guarantee deterministic
reductions, so metrics will land close to — not identically on — the figures
above. The dataset build *is* deterministic and its manifest is committed.

---

## Running the iOS app

Requires Xcode 16+ and a device or simulator on iOS 17+.

```bash
open BinSight.xcworkspace
```

Build the `BinSight` scheme. Open the **workspace**, not the `.xcodeproj`.

The camera is only available on a physical device — the simulator has none, so
the app there shows its camera-unavailable state, which is the designed
behaviour. On first launch the app asks for camera permission
(`NSCameraUsageDescription`: *"BinSight uses the camera to classify waste items
on this device. Frames are not uploaded."*). Signing uses automatic provisioning;
select your own team.

Run the tests:

```bash
xcodebuild -workspace BinSight.xcworkspace -scheme BinSight -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test
```

---

## Limitations

- **Three materials only.** Paper, plastic and metal. Cardboard, glass and
  organic waste are outside the taxonomy — the detector will either miss them or
  force them into one of the three classes.
- **This is not a recycling authority.** Guidance shown in the app is generic
  material advice, never a particular council's rules.
- **Recall is 0.578 on the held-out test split.** Roughly two in five annotated
  objects are missed at the 0.25 threshold. It finds things reliably; it does not
  find everything.
- **Paper is the hardest class** (AP50-95 0.4386, recall 0.50) and is the one
  most often confused with plastic on translucent or crumpled items.
- **Domain gap.** Training imagery is product-style photography at 416×416, not
  handheld camera frames at arm's length under kitchen lighting. Live performance
  can be expected to fall short of the test figures.
- **Device latency is unmeasured.** The numbers above are simulator numbers.
- **Live behaviour has been confirmed only qualitatively.** A signed build was
  run on an iPhone 15: boxes appear, track objects, stay aligned near the frame
  edges, and paper/plastic/metal are labelled correctly. No quantitative
  device-side accuracy or latency measurement has been taken.

---

## Future work

- Expand the taxonomy — cardboard and glass are the obvious next two
- Collect real handheld camera-domain images to close the domain gap
- Measure and optimise latency on device; consider INT8 if it is justified
- Record a demo capture

---

## Licence and attribution

The dataset is third-party and separately licensed — see
[Source and attribution](#source-and-attribution) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Ultralytics YOLO26 and the weights derived from it are **AGPL-3.0**, which the
exported CoreML package inherits. That is a genuine constraint on redistributing
this app, and it is recorded rather than glossed over.
