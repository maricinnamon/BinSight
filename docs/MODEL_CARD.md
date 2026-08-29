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
| **PyTorch release artifact** | `models/pytorch/BinSightYOLO26n.pt` — 5,387,013 bytes, SHA-256 `f6ca723341610b4d5f19856fcecbb30528193e2f7bed47c5d21deda539276c96` |
| **CoreML release artifact** | `BinSight/Resources/Models/BinSightYOLO26n.mlpackage` — 4.8 MB |
| **Provenance** | byte-identical copy of the continuation run's `best.pt`; `runs/` itself is not committed |
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
| Epochs | **65 total** — 25 initial + 40 continuation (both completed; no early stopping) |
| Best epoch | 40 of the continuation run — i.e. its last, so still improving |
| Batch | 4 |
| Image size | 640 |
| Optimizer | AdamW (auto), cosine decay |
| Augmentation | mosaic 0.3, disabled for the final epochs (`close_mosaic=5` initial, `10` continuation) |
| Seed | 42 |
| Duration | 6.85 h + 2.86 h |

606 of 708 tensors transferred from COCO in the first stage; the 102 skipped are
the class-count-dependent head, adapted 80 → 3. The continuation started from the
first stage's own checkpoint, so it transferred 708/708 — the head was already
3-class.

Disabling mosaic did **not** cause the continuation's gain, contrary to what the
first run suggested: the model passed the old checkpoint at epoch 27, while
mosaic was still on, and the epoch mosaic was disabled moved mAP50-95 by −0.0012.

## Evaluation

Validation guided checkpoint selection, so **the held-out test split is the
reportable figure**.

| Metric | Validation | **Held-out test** |
|---|---:|---:|
| Precision | 0.7502 | **0.7533** |
| Recall | 0.6485 | **0.5744** |
| mAP50 | 0.7159 | **0.6740** |
| mAP75 | 0.5897 | **0.5266** |
| mAP50-95 | 0.5425 | **0.4976** |

Against the 25-epoch model the continuation gained **+0.0716 precision** at
essentially unchanged recall (−0.0033), and +0.0356 mAP50-95.

Per class, held-out test (508 images, 1,666 instances):

| Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| paper | 0.7395 | 0.4693 | 0.5742 | 0.5818 | 0.4578 |
| plastic | 0.7421 | 0.5873 | 0.6557 | 0.7001 | 0.4781 |
| metal | 0.7784 | 0.6667 | 0.7182 | 0.7402 | 0.5568 |

Evaluation threshold: confidence 0.25 — the same value the app ships with.

## PyTorch → CoreML agreement

10 deterministic held-out images (seed 42), matched on geometry alone with class
compared afterwards:

| | |
|---|---|
| PyTorch detections | 40 |
| CoreML detections | 40 |
| Geometry-matched | 38 / 40 |
| Class agreement | 100% (38/38) |
| Mean matched-box IoU | 0.9924 (min 0.9685) |
| Mean confidence delta (distribution) | 0.0168 (max 0.1619) |

No class disagreements. Confidence is compared as a *distribution* rather than
pairwise, because in dense scenes — one test image yields 19 boxes overlapping at
IoU > 0.99 — no geometry-only matcher can decide which box corresponds to which
twin, and pairwise deltas then measure the arbitrariness of the pairing.

The one outlier (0.1619) was traced: PyTorch emitted two near-identical boxes on
a single bottle, splitting its confidence across 0.762 and 0.537, while CoreML
suppressed the duplicate into one 0.924 detection. Same object, same class, boxes
agreeing at IoU 0.985.

## Performance

Measured on the **iOS Simulator**, which has no Neural Engine. Physical-device
latency has **not** been measured.

| | |
|---|---|
| Letterbox (CoreImage) | 2–4 ms |
| CoreML inference | 32–136 ms |
| End-to-end | 42–191 ms (≈ 5.2–23.9 /s) |

The range is wide because the Simulator shares the host CPU; it is reported as a
range rather than a single figure because a single figure would be a fiction.

## Limitations

**Three materials only.** Cardboard, glass and organic waste are outside the
taxonomy. Cardboard is the sharpest risk: it is visually adjacent to paper and
was deliberately excluded, so cardboard in frame may be detected as `paper`.

**Recall is 0.574 on held-out test.** Roughly two in five annotated objects are
missed at the 0.25 threshold. The app compounds this deliberately by showing only
the single strongest detection, so a frame containing two materials names one.

**`paper` is the weakest class** (AP50-95 0.4578, recall 0.4693). It is
deformable, often crumpled, and shades into both cardboard and translucent
plastic — and its recall *fell* in the continuation run even as precision rose.

**The model is not converged.** Both training stages ended on their best epoch.

**Domain shift is the dominant limitation.** The training imagery is
product-style photography at 416×416 — largely single objects, clean
backgrounds, even lighting. The app runs on handheld camera frames at arm's
length under whatever light is available, with clutter and motion blur. Live
performance should be expected to fall short of the test figures, and the test
figures should not be quoted as in-app accuracy.

**Crowded scenes produce many overlapping boxes.** This is correct behaviour for
an NMS-free detector — one evaluation image legitimately contains 19 overlapping
`metal` detections — but the app surfaces only the single strongest box.

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
