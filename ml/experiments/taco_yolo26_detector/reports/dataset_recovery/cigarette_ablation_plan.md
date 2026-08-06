# Cigarette ablation plan — how to decide, not what to decide

`cigarette` was Run 001's total failure: **AP@50 0.0039, precision 0.0, recall
0.0** on the test split, and 0.0/0.0 at the 0.25 confidence threshold on val too.
It is also the class recovery helped most — 186 → 537 annotations, 2.89×.

The obvious move is to drop it. Eight classes become seven, the headline mAP
jumps, and nothing about the detector has improved. This document exists so that
decision is measured rather than assumed.

**The class stays in Run 002.** The ablation is a separate, later run.

## Why cigarette fails, as far as the data can say

Three causes, and they do not have the same remedy:

**1. Size.** 96.5% of cigarette annotations fall in the COCO *small* bucket;
median area is **77.9 px² at 640** — roughly 9×9 pixels — and the 10th percentile
is 20.3 px². Run 001's size analysis shows recall collapsing with size across all
classes (tiny 0.051, small 0.174, medium 0.308, large 0.500), so cigarette is the
extreme of a general problem, not a special one. YOLO26n's stride-8 head has a
limited amount to work with at this input size.

**2. Volume.** 186 annotations across 65 images in Run 001. That alone could
explain a zero. Recovery raises it to 537 across 173 images, which is what
Run 002 tests.

**3. Incomplete ground truth.** The zoomed crops in
`reports/samples/run_002_dataset/tiny_objects/cigarette_*.jpg` show images with
several visible butts where only one carries a box. A correct detection on an
unlabelled butt scores as a false positive, so measured precision has a ceiling
below 1.0. This is a qualitative observation from the renders — no rate has been
measured, and measuring one would mean re-annotating.

Causes 1 and 3 are not fixed by more data. Cause 2 is. Run 002 separates them.

## Sequence

**Step 1 — Run 002, cigarette included, 640 px.** Already configured
(`configs/train_run_002_planned.yaml`). Read cigarette AP@50 and recall against
Run 001's 0.0039 / 0.0.

**Step 2 — decide from the result, using thresholds fixed now:**

| Run 002 cigarette AP@50 | Reading | Next |
|---|---|---|
| **> 0.15** | Volume was the binding constraint | Keep it. No ablation needed. |
| **0.03 – 0.15** | Data helped but size dominates | Run 003 at 768/1024 before any ablation |
| **< 0.03 (still ~zero)** | 2.89× the data changed nothing | Run the ablation below |

**Step 3 — the ablation itself, only if Step 2 lands in the bottom row.** Two
runs, identical in every other respect:

- **A:** 8 classes, Run 002's exact configuration (already have it).
- **B:** 7 classes, cigarette removed from `class_mapping`, everything else —
  seed, epochs, imgsz, batch, augmentation — unchanged. New split manifests,
  since removing a class removes some images entirely.

Compare **per-class AP for the seven shared classes**, never the mAP. mAP will
rise in B by arithmetic alone; that is not evidence of anything.

## How to read the ablation

- **The seven shared classes improve in B.** Cigarette was actively harming
  them — plausibly by consuming the small-object capacity of a nano model.
  Removing it is a real gain and is reported as a scope decision: "BinSight does
  not detect cigarettes."
- **The seven are unchanged in B.** Cigarette costs nothing to carry. Keep it.
  A class at low AP that harms nothing is honest breadth, not a defect, and
  removing it would be presentation rather than engineering.
- **The seven get worse in B.** Unlikely, but it would mean cigarette images
  were contributing useful background variety. Keep the class and say why.

## Rules for this ablation

1. **Both runs from `yolo26n.pt`.** Never fine-tune B from A's weights.
2. **Do not compare mAP across a different number of classes.** The only valid
   comparison is per class.
3. **Removing a class is a product decision, not a metric fix.** If cigarette is
   dropped, the README says the model does not detect cigarettes — it does not
   quietly report a seven-class mAP as if it were the eight-class one.
4. **Do not run the ablation before Step 1.** Ablating against a zero from a
   509-image dataset would measure the rate limiter, not the class.

## Status

Nothing here has been executed. Run 001's cigarette numbers above are measured
and come from `reports/test_metrics_run_001.csv`; every Run 002 and ablation
figure in this document is a threshold set in advance, not a result.
