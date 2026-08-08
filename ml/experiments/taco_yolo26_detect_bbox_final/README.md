# TACO YOLO26n Detect — bbox-only pipeline

```
STATUS: pipeline complete and verified — FINAL MODEL NOT TRAINED
Reason: source imagery was deleted to reclaim disk before the run started
Deployment candidate remains: ../taco_yolo26_detector/models/final/binsight_yolo26n_deployment.pt
```

## What this experiment is

A clean, detection-only training pipeline built to replace the earlier
experimental branches. TACO ships instance-segmentation polygons; BinSight does
not do segmentation, because the iOS app needs only

```
x1, y1, x2, y2 · class_id · confidence
```

and a mask head would cost parameters and latency on a phone for output the
product discards.

```
Official TACO COCO annotations
        ↓  annotation["bbox"] ONLY — segmentation stripped at load time
YOLO Detect labels: class_id x_center y_center width height   (exactly 5 values)
        ↓
YOLO26n Detect, imgsz 640, fresh yolo26n.pt
        ↓
bounding rectangle + class + confidence
        ↓
future CoreML export and iPhone integration
```

## How bbox-only is enforced, not merely promised

`src/bbox_only.py` whitelists the COCO annotation keys it will return — `id`,
`image_id`, `category_id`, `bbox`, `area` — and drops everything else at load
time. `segmentation` is therefore **unreachable** from this pipeline rather than
just unused, and `assert_no_segmentation_keys()` fails the build if one survives.

An invalid or missing `bbox` **excludes** the annotation and records it in
`manifests/excluded_annotations.csv`. It is never reconstructed from the polygon:
a mask-derived box is a different annotation, and silently substituting one would
make the dataset an untraceable mixture of two sources.

`reports/detection_only_audit.md` verifies all of this at runtime rather than
asserting it in prose.

## Verification gates — all passed

| Gate | Result |
|---|---|
| Label validation | **PASS** — 2671 boxes, field-count histogram `{5: 2671}`, worst COCO→YOLO→COCO round trip 1.0 px |
| Detection-only audit | **PASS** — `DetectionModel` / `Detect` head / `task=detect` / `E2ELoss`, no mask prototype branch, no mask artefacts, no segmentation reference in executable pipeline code |
| Split leakage | **PASS** — zero by image id, source path and content SHA-256; 1082 unique hashes for 1082 images |
| Visual bbox check | **PASS** — all 8 classes rendered as plain rectangles, boxes on target (`reports/bbox_visual_validation.jpg`) |
| Supervisor mechanics | **PASS** — one logical run, `results.csv` continuing 1→2→3→4, no duplicate epochs, optimizer and EMA surviving the recycle |
| Unit tests | **PASS** — 20 tests |

## Dataset (as built and measured)

1082 images / 2671 boxes from the official reviewed TACO, reusing the already
verified image recovery, 8-class mapping and deterministic splits so the numbers
stay comparable with the archived runs.

| Split | Images | Boxes |
|---|---:|---:|
| train | 748 | 1861 |
| val | 168 | 405 |
| test | 166 | 405 |

Per class: `bag_wrapper` 762 · `cigarette` 537 · `bottle` 349 · `bottle_cap` 238 ·
`can` 235 · `carton` 221 · `cup` 176 · `straw` 153 — imbalance 4.98:1.

**Zero annotations were excluded for an invalid bbox**, so nothing was dropped
and nothing needed a polygon fallback. 83.5% of boxes are tiny or small, which is
the defining difficulty of TACO: street litter photographed from standing height.

## The memory-safe trainer

Four training processes died to SIGKILL in the archived experiments, each
mid-batch with no traceback — the signature of macOS reclaiming memory from a
long-lived process on an ~8.6 GB machine.

**This is observed host behaviour. No PyTorch or MPS memory leak was diagnosed,
and none is claimed.**

`scripts/train_supervisor.py` turns that unpredictable kill into a scheduled
event: a supervisor holding no torch launches a disposable child, the child
trains N epochs and exits cleanly *after* its checkpoint is written, and a new
child resumes from the same `last.pt`. One logical run, one directory, no
`run2`. Swap is sampled continuously to `reports/memory_log.csv`, with an early
recycle at 3.5 GB and immediate termination at 5 GB rather than waiting to be
killed.

The mechanics are verified end to end in `reports/supervisor_smoke_test.md`.

## Why the final model was not trained

The development Mac ran out of disk. Reclaiming space meant deleting the source
imagery — `data/raw` (2.4 GB of Flickr downloads) and `data/official_complete`
(1.0 GB Zenodo archive) — which the prepared dataset symlinks into. That was a
deliberate trade made with full knowledge of the consequence, not an accident.

`data/raw` is **not recoverable**: those images came from a Flickr download in
which 898 of 1500 URLs returned HTTP 429, and re-fetching them would hit the same
limit. `data/official_complete` is recoverable from Zenodo (MD5
`5c674548402b142d5a27a1f7b6a653f3`).

Everything except the pixels survives: 1082 label files, the annotation
manifests, all reports, all configs, the full trainer, and the tests.

## To finish this experiment later

```bash
# 1. re-download the Zenodo archive (recovers ~715 of the 1082 images)
python ../taco_yolo26_detector/scripts/recover_official_dataset.py

# 2. rebuild the bbox dataset and re-run every gate
python scripts/build_bbox_dataset.py
python scripts/validate_bbox_labels.py
python scripts/detection_only_audit.py
python scripts/audit_dataset.py
python scripts/render_bbox_validation.py
python -m pytest tests -q

# 3. train
python scripts/write_dataset_yaml.py
python scripts/train_supervisor.py --total-epochs 60 --epochs-per-child 5
```

Coverage will be lower than the 1082 images measured here unless the Flickr
images can also be recovered.

## Current deployment candidate

Not from this experiment. The model the project ships with is
`../taco_yolo26_detector/models/final/binsight_yolo26n_deployment.pt` — the Run
002 checkpoint, classified **`research_only`**: full-frame test precision 0.3158,
recall 0.2470, mAP50-95 0.1453, tiny-object recall 0.095. It finds roughly a
quarter of objects and misses about nine tiny objects in ten.

The honest limitation is not a hyperparameter: TACO is wide litter scenes, and
BinSight users hold a phone close to one item. The guided-framing diagnostic in
the archived experiment measured that gap and showed framing does **not** rescue
tiny objects. Real improvement needs real phone-camera photographs.
