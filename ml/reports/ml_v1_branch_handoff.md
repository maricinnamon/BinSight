# BinSight ML v1 — branch handoff

Archived 2026-08-06. Everything below is preserved, including the runs
that failed. Nothing was deleted.

Full machine-readable record:
[`experiments/taco_yolo26_detector/reports/ml_v1_archive_manifest.json`](../experiments/taco_yolo26_detector/reports/ml_v1_archive_manifest.json).

---

## TrashNet TensorFlow baseline

- **Task:** whole-image classification, 6 classes
- **Dataset:** TrashNet, ~2 500 images of isolated objects on clean backgrounds
- **Model:** MobileNetV3Small, two-stage transfer learning
- **Result:** **83.38%** test accuracy, macro F1 **0.8046**, weighted F1 **0.8341**
- **Shipped:** yes — the exported `.tflite` is what the iOS app currently bundles

**Why it was not selected going forward.** The accuracy is real but answers the
wrong question. A whole-frame classifier cannot localise, cannot handle two items
in shot, and cannot abstain when the frame holds nothing. Its per-class weakness
was also telling: `trash` reached only F1 0.5946, i.e. the catch-all category was
the one it could not learn. In live camera use the in-distribution accuracy did
not carry over.

---

## TACO Run 001 — YOLO26n Detect, 640 px

- **Dataset:** partial TACO — 509 images / 1 290 boxes. Not a choice: Flickr
  returned HTTP 429 for 898 of 1 500 image URLs.
- **Config:** `yolo26n.pt`, imgsz 640, batch 8, MPS, seed 42
- **Completed:** 64 of 100 epochs, best epoch 44, early stopping worked normally,
  1.59 h — **the only run that finished as intended**

| Split | P | R | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| test | 0.2696 | 0.1853 | 0.2197 | 0.2007 | 0.1696 | 0.1586 |

Size recall: tiny 0.051, small 0.174, medium 0.308, large 0.500 (large n=2 — that
figure means nothing). Verdict at the time: needs iteration, starved of data.

---

## TACO Run 002 — recovered dataset, 640 px

- **Dataset:** 1 082 images / 2 671 boxes, recovered to 86.7% of the official
  record from the archived Zenodo release (MD5-verified) merged with what Flickr
  served. Labelled a *maximally recovered official subset*, never "TACO".
- **Config:** identical to Run 001 except the data — the controlled variable
- **State:** `incomplete_due_to_host_memory_pressure` — **24 of 100 epochs**, best epoch 20, 2.18 h across
  three process segments

| Split | P | R | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| test | 0.3158 | 0.2470 | 0.2772 | 0.2105 | 0.1637 | 0.1453 |

Against Run 001: precision +0.046, recall +0.062, F1 +0.058, tiny recall
0.051 → 0.095. mAP50-95 fell 0.013, but the test split also got harder — the
COCO-small share rose from 29.1% to 38.4%, because the images Flickr had refused
were not a random sample.

**This is the selected deployment candidate**, chosen on validation against every
other checkpoint. It won on fitness, mAP50-95, recall, precision, and per-class
robustness (one failing class against Run 001's three) — but it is an *interrupted*
run, selected because it was the best available, not because it finished.

- `best.pt` — `runs/yolo26n_run_002_full_taco_640/weights/best.pt`
  SHA-256 `3f470d1abaf8bd615783a68836eefcf482bd771a437a91254d1a6f07c2ed276f`
- `last.pt` — `runs/yolo26n_run_002_full_taco_640/weights/last.pt`
  SHA-256 `f5da05b54fb23c34747c1370b11bbe2b6d802f9f4fef0814dffd6b94fa8daf68`

---

## 768 px + object-crop attempt

The designed answer to Run 002's diagnosis that **86.5% of false negatives were
tiny or small objects**.

- **Dataset:** 748 original scenes **plus 1 056 object-centric crops** generated
  only from training images, making each target a median **29× larger** in the
  training signal. Val and test stayed byte-identical originals; zero leakage,
  tested rather than assumed.
- **Config:** `yolo26n.pt`, imgsz 768, batch 4, mosaic 0.5, scale 0.35, cos_lr,
  workers 0, cache off — deliberately memory-conservative
- **State:** `incomplete_due_to_host_memory_pressure` — **14 of 70 epochs**, best epoch 14, 1.58 h
- **Best validation:** P 0.4273, R 0.2516, mAP50 0.2351, mAP50-95 0.1703
- **Never evaluated on test** — an incomplete run is not a result

- `best.pt` — `runs/yolo26n_final_taco_context_plus_crops_768/weights/best.pt`
  SHA-256 `4e2e8cdc75497e21890190b999f5dbee714a5f67c562b5027659362a989bf8fd`

**Why it was not continued.** It was killed at 99% of epoch 15 (batch 447 of 451),
the fourth such kill in the project. Continuing would have meant an indefinite
cycle of resume-and-be-killed on a machine that cannot hold this configuration
alongside normal use.

Worth recording so the recipe is not written off: at epoch 10 it was **ahead** of
Run 002 on validation — recall +0.068, a 31% relative gain, exactly the axis the
crops targeted. It looked promising. It could not be tested to conclusion here.

---

## The memory-pressure observation

Stated carefully, because the evidence supports an observation and not a diagnosis.

**What was observed:** four training processes across two runs stopped mid-batch
with **no Python traceback, no MPS error and no OOM message** in any log. That
pattern is consistent with SIGKILL from the macOS memory-pressure handler. Swap
use during training ranged from 1.9 GB to 5.2 GB on a host with ~8.6 GB unified
memory, and kills clustered when other workloads (iOS Simulator, `xcodebuild`,
Chrome) were competing. Epoch time also degraded roughly 2× under that contention
— 205 s to 450 s — and recovered when the machine was quiet.

**What was NOT established:** no PyTorch or MPS memory leak was proven. Doing so
would need per-process RSS sampled over a full run against a controlled baseline,
which was never collected. Long-running MPS training on this host showed
**growing memory and swap pressure**; the mechanism is unattributed.

---

## Main lessons

**The application needs object detection, not whole-image classification.** The
TrashNet baseline settled this. A classifier cannot localise, cannot handle
multiple items, and has no way to abstain.

**TACO's segmentation polygons are unnecessary for BinSight.** Every pipeline here
consumed COCO `bbox` fields only; the polygons were never read. The product needs
a box to draw and a class to name — nothing more. The next implementation will use
**bounding boxes only, explicitly ignoring segmentation**, which removes a whole
category of conversion bugs.

**The next trainer must be designed for ~9 GB of unified memory**, not adapted to
it afterwards. Concretely: batch 4 at 768, `workers=0`, `cache=false`, and no
assumption that a multi-hour process survives.

**Long-lived MPS training processes showed growing memory and swap pressure.** The
practical consequence matters more than the cause: a 70-epoch single process is
not a safe unit of work on this host.

**Future training will use controlled process restarts and checkpoint resume.** A
supervisor should run training in a child process, stop it every N epochs, and
restart from `last.pt`. Ultralytics' resume restores weights, EMA, optimiser
moments, AMP scaler, LR schedule position and epoch counter — verified repeatedly
in this branch, where resume worked correctly every time it was used. Deliberate
restarts convert an unpredictable kill into a scheduled, cheap event.

**Two smaller lessons worth carrying:**

- *Look at the rendered images.* The EXIF-orientation defect that misplaced boxes
  on a quarter of the dataset was found by eye. Every coordinate was programmatically
  valid.
- *Guided framing does not rescue tiny objects.* Measured, not assumed: an oracle
  ROI centred on each known object rescued 38 and lost 40. Tiny recall stayed
  between 0.099 and 0.107 under every framing. The remaining gap is **domain
  mismatch** — TACO is standing-height wide scenes, BinSight is a close-up
  single-item scanner — and closing it needs real phone-camera photographs, not
  another hyperparameter configuration.

---

## Next branch

**YOLO26 Bounding-Box-Only + Memory-Safe Training** — see the ML README.

Not started. Nothing has been exported to CoreML or TFLite; the iOS app is
unchanged and still bundles the TrashNet classifier.
