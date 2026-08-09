# Model card — BinSightYOLO26n

Every figure here is measured. Nothing is estimated, borrowed from a paper, or
carried over from an earlier experiment.

## Model details

| | |
|---|---|
| **Name** | BinSightYOLO26n |
| **Task** | Object detection (bounding box + class + confidence) |
| **Architecture** | YOLO26n Detect, Ultralytics 8.4.9 |
| **Parameters** | 2,504,970 |
| **Training type** | **Transfer learning / fine-tuning from COCO-pretrained `yolo26n.pt`. NOT trained from scratch.** |
| **Classes** | 3 — `0 paper`, `1 plastic`, `2 metal` |
| **Input** | 640×640 RGB |
| **Deployed format** | CoreML `.mlpackage`, ML Program, FP16, spec v6 (iOS 16+), 4.8 MB |
| **PyTorch release artifact** | `models/pytorch/BinSightYOLO26n.pt` — 5,385,157 bytes, SHA-256 `958e49a5fc8358442497bcb9ce98ca9b8e48a6a1da3bfb92539ac9d14f58ed67` |
| **CoreML release artifact** | `BinSight/Resources/Models/BinSightYOLO26n.mlpackage` — 4.8 MB |
| **Provenance** | byte-identical copy of the training run's `best.pt`; `runs/` itself is not committed |
| **Framework at training** | PyTorch 2.13.0, MPS (Apple M1) |

### End-to-end / NMS-free

`head.end2end = True`. The checkpoint carries a one-to-one head and emits
already-deduplicated detections, so no non-maximum suppression is applied at
export or in the app. Ultralytics enforces this: `'nms=True' is not available for
end2end models. Forcing 'nms=False'`.

### Output contract

One MultiArray, `[1, 300, 6]`, FP32. Each row is
`[x1, y1, x2, y2, confidence, class_id]` in 640×640 input-pixel space, where 300
is `head.max_det`.

Boxes are **xyxy corners**, not xywh — Ultralytics' own `Detect.postprocess`
docstring says xywh, and for this checkpoint that is wrong. Confidence is already
the maximum class probability; no sigmoid or softmax is applied. All 300 rows are
always returned, padded with low-scoring entries, so a confidence threshold is
required.

## Intended use

A portfolio demonstration of on-device detection: pointing an iPhone at everyday
waste and seeing paper, plastic and metal localised in real time.

**Not intended** as a recycling authority. Guidance shown in the app is generic
material advice and never a particular council's rules. Not validated for
industrial sorting, waste auditing, or any decision with a material consequence.

## Training data

| | |
|---|---|
| Source | Roboflow Universe `material-identification/garbage-classification-3` v2, CC BY 4.0 |
| Prepared images | 5,074 |
| Bounding boxes | 16,173 |
| Split | 4,059 train / 507 val / 508 test |
| Split method | 80/10/10, seed 42, stratified by dominant class, grouped by source image |

Three of six source classes are excluded (`BIODEGRADABLE`, `CARDBOARD`, `GLASS`);
the rest are renumbered. The split is grouped by source-image identity so that
augmented variants of one photo cannot straddle train and test. Independent
validation reports 0 corrupt images, 0 invalid labels and 0 cross-split leakage.

## Training configuration

| | |
|---|---|
| Epochs | 25 (completed; no early stopping) |
| Best epoch | 25 |
| Batch | 4 |
| Image size | 640 |
| Optimizer | AdamW (auto), cosine decay |
| Augmentation | mosaic 0.3, disabled for the final 5 epochs (`close_mosaic=5`) |
| Seed | 42 |
| Duration | 6.85 h |

606 of 708 tensors transferred from the pretrained checkpoint; the 102 skipped
are the class-count-dependent head, adapted 80 → 3.

## Evaluation

Validation guided checkpoint selection, so **the held-out test split is the
reportable figure**.

| Metric | Validation | **Held-out test** |
|---|---:|---:|
| Precision | 0.7086 | **0.6817** |
| Recall | 0.6045 | **0.5777** |
| mAP50 | 0.6779 | **0.6480** |
| mAP75 | 0.5450 | **0.4809** |
| mAP50-95 | 0.5070 | **0.4620** |

Per class, held-out test (508 images, 1,666 instances):

| Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| paper | 0.6519 | 0.5000 | 0.5660 | 0.5713 | 0.4386 |
| plastic | 0.6822 | 0.5629 | 0.6169 | 0.6568 | 0.4349 |
| metal | 0.7108 | 0.6700 | 0.6898 | 0.7159 | 0.5126 |

Evaluation threshold: confidence 0.25 — the same value the app ships with.

## PyTorch → CoreML agreement

10 deterministic held-out images (seed 42), matched on geometry alone with class
compared afterwards:

| | |
|---|---|
| PyTorch detections | 27 |
| CoreML detections | 29 |
| Geometry-matched | 27 / 27 |
| Class agreement | 96.30% (26/27) |
| Mean matched-box IoU | 0.9928 (min 0.9771) |
| Mean confidence delta | 0.017 (max 0.097) |

The single class disagreement is a translucent plastic bag at IoU 0.986 —
`paper 0.278` in PyTorch, `plastic 0.284` in CoreML — a genuine near-tie tipped
by FP16 rounding. Both extra CoreML detections sit within 0.04 of the threshold.
Confident detections reproduce at IoU ≥ 0.9967 with confidence deltas below
0.0005, so export error is confined to the low-confidence band.

## Performance

Measured on the **iOS Simulator**, which has no Neural Engine. Physical-device
latency has **not** been measured.

| | |
|---|---|
| Letterbox (CoreImage) | 3–4 ms |
| CoreML inference | 58–136 ms |
| End-to-end | 116–191 ms (≈ 5.2–8.6 /s) |

The range is wide because the Simulator shares the host CPU; it is reported as a
range rather than a single figure because a single figure would be a fiction.

## Limitations

**Three materials only.** Cardboard, glass and organic waste are outside the
taxonomy. Cardboard is the sharpest risk: it is visually adjacent to paper and
was deliberately excluded, so cardboard in frame may be detected as `paper`.

**Recall is 0.578 on held-out test.** Roughly two in five annotated objects are
missed at the 0.25 threshold.

**`paper` is the weakest class** (AP50-95 0.4386, recall exactly 0.50). It is
deformable, often crumpled, and shades into both cardboard and translucent
plastic.

**Domain shift is the dominant limitation.** The training imagery is
product-style photography at 416×416 — largely single objects, clean
backgrounds, even lighting. The app runs on handheld camera frames at arm's
length under whatever light is available, with clutter and motion blur. Live
performance should be expected to fall short of the test figures, and the test
figures should not be quoted as in-app accuracy.

**Crowded scenes produce many overlapping boxes.** This is correct behaviour for
an NMS-free detector — one evaluation image legitimately contains 11 overlapping
`metal` detections — but the app caps the overlay at 10 boxes for legibility.

**Not evaluated for fairness or demographic bias.** The subject matter is objects
rather than people, but no analysis of geographic or packaging-market bias in the
source dataset has been done, and packaging design varies considerably by region.

## Provenance

- Training: [`reports/yolo26n_training_report.md`](../reports/yolo26n_training_report.md)
- CoreML export and parity: [`reports/coreml_export_report.md`](../reports/coreml_export_report.md)
- Dataset: [`datasets/binsight_waste_3class/dataset_report.md`](../datasets/binsight_waste_3class/dataset_report.md)
- iOS pipeline: [`docs/DETECTION_PIPELINE.md`](DETECTION_PIPELINE.md)

## Licence

The model derives from Ultralytics COCO-pretrained weights and inherits
**AGPL-3.0**. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
