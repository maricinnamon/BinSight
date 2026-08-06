# BinSight ML

BinSight is an on-device iOS waste-recognition application. The ML work explores
compact computer-vision models suitable for real-time inference on mobile
hardware.

Everything here is organised as a sequence of **experiments**. Each one is
self-contained — its own environment, code, reports and models — so a result can
be read, audited and reproduced without reconstructing the state of the
repository at the time.

## Experiments

| Experiment | Dataset | Task | Model | Status | Key result |
|---|---|---|---|---|---|
| [TrashNet TensorFlow Baseline](experiments/trashnet_tensorflow_baseline/) | TrashNet | Image classification | MobileNetV3Small | Baseline completed | **83.38%** test accuracy, macro F1 **0.8046** |
| [TACO YOLO26 Detector — Run 001](experiments/taco_yolo26_detector/) | TACO, 509 images | Object detection | YOLO26n | Trained & evaluated, **frozen** | test mAP50 **0.2007**, mAP50-95 **0.1586** |
| [TACO YOLO26 Detector — Run 002](experiments/taco_yolo26_detector/#run-002--recovered-taco-640) | TACO, 1 082 images (86.7% recovered) | Object detection | YOLO26n | Interrupted at epoch 24/100 — **SELECTED** | test mAP50 **0.2105**, mAP50-95 **0.1453**, recall **0.2470** |
| [TACO YOLO26 — 768 + object crops](experiments/taco_yolo26_detector/#run-003--768--object-centric-crops-incomplete) | 748 scenes + 1 056 crops | Object detection | YOLO26n | **Incomplete** — host memory limit | epoch 14/70, val mAP50-95 0.1703; not a result |

## Architecture evolution

```
Experiment 1
TrashNet
    ↓
TensorFlow image classifier (MobileNetV3Small)
    ↓
whole-frame classification
    ↓
83.38% in-distribution accuracy
    ↓
poor robustness on live-camera scenes

Experiment 2 — three runs, one selected
TACO (1 300 of 1 500 official images recovered)
    ↓
8 object classes (object-oriented mapping, no catch-all)
    ↓
YOLO26n detector, 640px, Apple MPS
    ↓
bounding box + class + confidence
    ↓
test mAP50 0.211, mAP50-95 0.145, recall 0.247
    ↓
tiny-object recall 0.095 — data and domain limited, not hyperparameter limited
```

The baseline was useful engineering, not a dead end. It established the six
classes, the deployment contract, the confidence policy, the export-and-verify
discipline, and a measured number to beat. It also surfaced the finding that
drives everything after it: **in-distribution accuracy did not translate to
uncontrolled camera scenes**, and a whole-frame classifier cannot localise an
object even in principle.

## Design goals

- fully on-device inference — no camera-frame uploads, no backend, no account;
- mobile-friendly model size and latency;
- robust recognition in real scenes, not only in dataset-like conditions;
- explicit uncertainty handling — abstaining is better than a confident wrong
  answer about someone's bin;
- reproducible training and evaluation, from pinned environments and fixed seeds;
- a measurable transition from experiment to production, so every shipped model
  can be traced to a verified artefact.

## Current direction

**Training is finished. No further model training is planned.** The deployment
candidate is `ml/experiments/taco_yolo26_detector/models/final/binsight_yolo26n_deployment.pt`
— the Run 002 checkpoint, selected on validation against every other available
checkpoint. It is classified **research_only**.

### What the detector actually does

Full-frame test, 166 untouched images / 405 objects: precision 0.3158, recall
0.2470, F1 0.2772, mAP50 0.2105, mAP50-95 0.1453. About a quarter of objects are
found. `cigarette` (AP50-95 0.021) and `straw` (0.012) do not work.

### Why training stopped

Three things converged.

**Unrestricted wide-scene litter detection is genuinely hard at this scale**, and
hardest for the objects TACO has most of. Tiny objects — under 1% of frame area —
are 62% of the test set, and recall on them is 0.095.

**The 768 px attempt could not complete on this hardware.** Run 002 was killed
three times and the 768 run once, always by macOS memory pressure with no Python
traceback: SIGKILL on a ~8.6 GB unified-memory machine shared with everyday
applications. The 768 run reached epoch 14 of 70 and is preserved as
`incomplete_due_to_host_memory_constraints`. It is a hardware limit, not a result.

**Guided framing was measured, and it does not rescue small objects.** See
[`reports/guided_framing_comparison.md`](experiments/taco_yolo26_detector/reports/guided_framing_comparison.md).
An oracle ROI centred on each known object rescued 38 objects and lost 40 — a net
of −2. Tiny recall stayed between 0.099 and 0.107 under *every* framing tested.
Cropping cannot add detail that the camera never captured.

### What BinSight actually is

**A guided single-item scanner, not a wide-scene litter detector.** The product
asks the user to point at one item, so the model never has to solve unrestricted
scene detection. Deployment uses a **central 75% ROI**: recall 0.1531 → 0.1778 and
small-object recall 0.2250 → 0.3125, for a precision cost of 0.470 → 0.424. A
fixed central crop is also all an iOS camera UI can honestly implement — it does
not know where the object is. Object-guided cropping is explicitly **not**
recommended: its best case underperforms the full frame.

### Where improvement has to come from

**Real phone-camera photographs of single items in the intended use case** — not
another hyperparameter run. TACO is litter shot from standing height across whole
scenes; a BinSight user holds a phone close to one object. That gap is the binding
constraint, and it is a data problem. A few hundred genuine in-app photographs
would very likely be worth more than every training run in this repository
combined.

## Planned next branch — YOLO26 Bounding-Box-Only + Memory-Safe Training

**Status: `next branch`.** Not started.

Two changes, both driven by what v1 actually ran into rather than by a hunch.

**Bounding boxes only.** TACO ships segmentation polygons; BinSight never needed
them and every pipeline in v1 read COCO `bbox` fields exclusively. The next
implementation makes that explicit — polygons are ignored outright. The product
draws a box and names a class; carrying mask data through the pipeline only adds
conversion surface where bugs hide.

**A memory-safe trainer.** Four training processes in v1 were killed mid-batch by
macOS memory pressure on a ~8.6 GB machine, twice costing hours. The next trainer
treats that as a design constraint rather than an accident:

- a lightweight supervisor owns the run and launches training in a **child
  process**;
- the child trains for a bounded number of epochs, then exits **deliberately**;
- the supervisor relaunches it, **resuming from `last.pt`**;
- a scheduled restart replaces an unpredictable kill, so long-running memory
  accumulation can never cost more than one epoch.

Resume was exercised repeatedly in v1 and restored weights, EMA, optimiser
moments, AMP scaler, LR-schedule position and epoch counter correctly every time,
so the mechanism is known to work here.

Handoff document: [`reports/ml_v1_branch_handoff.md`](reports/ml_v1_branch_handoff.md).

## Experiment progression

```
TrashNet TensorFlow classification baseline    83.38% accuracy, macro F1 0.8046
              |
TACO YOLO26 Run 001 — partial data (509 img)   test mAP50-95 0.1586
              |
TACO YOLO26 Run 002 — recovered data (1082)    test mAP50-95 0.1453, recall +0.062
              |                                 SELECTED (interrupted at epoch 24/100)
              |
Run 003 (768 + object crops)                    incomplete — host memory limit
              |                                 epoch 14/70, preserved not deleted
              v
guided-framing diagnostic                       central 75% ROI recommended
              |                                 does NOT solve tiny objects
              v
selected detector: binsight_yolo26n_deployment.pt   research_only
```

Neither Run 001 nor Run 002 is a required production stage. They are diagnostic
history, kept so the reasoning is auditable.

**Next phase: CoreML/TFLite export and iPhone integration.** Nothing has been
exported yet.

## Layout

```
ml/
├── README.md                              this file
└── experiments/
    │
    ├── trashnet_tensorflow_baseline/
    │   ├── README.md                      full experiment write-up
    │   ├── experiment.json                machine-readable summary
    │   ├── Makefile                       one target per pipeline stage
    │   ├── environment.yml                pinned Conda environment
    │   ├── labels.txt                     output-index contract
    │   ├── model_contract.json            deployment contract
    │   ├── configs/ scripts/ tests/
    │   ├── src/binsight_training/         the library the scripts call
    │   ├── notebooks/                     earlier, never-executed Colab attempt
    │   ├── models/{keras,tflite}/         trained model + conversion candidates
    │   └── reports/                       metrics, figures, predictions, logs
    │
    └── taco_yolo26_detector/
            ├── README.md                  experiment write-up
            ├── environment.yml            BinSight_YOLO (PyTorch, no TensorFlow)
            ├── configs/                   dataset*.yaml, class_mapping*.yaml,
            │                              train_run_001.yaml, train_run_002_planned.yaml
            ├── data/
            │   ├── source/ raw/           TACO repo + Flickr download (gitignored)
            │   ├── official_complete/     recovered Zenodo archive (gitignored)
            │   ├── partial_run_001/       symlinks naming Run 001's frozen dataset
            │   ├── prepared/ splits/      Run 001 dataset and manifests
            │   ├── prepared_run_002/      Run 002 dataset
            │   └── splits/run_002/        Run 002 manifests
            ├── scripts/                   download, recover, audit, map, convert,
            │                              validate, train, evaluate, compare
            ├── reports/
            │   ├── dataset_recovery/      recovery audit, manifests, Run 002 plans
            │   └── …                      taxonomy, validation, sample renders
            ├── runs/yolo26n_run_001/      frozen: weights, metrics, plots
            └── tests/
```

The two experiments use **separate Conda environments** — `binsight-trashnet-metal`
(TensorFlow) and `BinSight_YOLO` (PyTorch). Neither has the other's framework
installed.

Datasets, checkpoints and SavedModel exports are gitignored — all are
regenerable from committed artefacts and documented commands. Conda environments
live outside the repository; `environment.yml` is what is committed.
