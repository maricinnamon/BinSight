# BinSight YOLO26n — training report

## Training type

**Transfer learning / fine-tuning from the official pretrained `yolo26n.pt`
detection weights. NOT trained from scratch.**

Verified before the first batch, and the trainer refuses to start otherwise:

| Evidence | Value |
|---|---|
| Source | `ultralytics/assets` v8.4.0 → `yolo26n.pt` |
| Checkpoint SHA-256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Built from | `.pt` weights, checkpoint dict present — never `yolo26n.yaml` |
| Pretrained classes | **80 (COCO)** — `person`, `bicycle`, `car`, … |
| BatchNorm statistics | trained (non-zero means, non-unit variances) |
| **Transferred at load** | **606 / 708 tensors** — the 102 skipped are the nc-dependent head |
| Head adapted | 80 → **3** classes |

After training, all 492 comparable tensors differ from the pretrained values and
359 of 366 AdamW momentum buffers are non-zero — real gradient steps, not a copy.

## Model

**YOLO26n Detect** — bounding box + class + confidence. No segmentation, no
masks, no second classifier. 2,572,280 parameters.

## Dataset

**BinSight Waste 3-Class Detection** — `datasets/binsight_waste_3class/`

| Split | Images | Boxes |
|---|---:|---:|
| train | 4,059 | 13,072 |
| val | 507 | 1,435 |
| test | 508 | 1,666 |
| **total** | **5,074** | **16,173** |

Classes `0 paper · 1 plastic · 2 metal`. Source: Roboflow *garbage-classification-3*
v2 (CC BY 4.0), filtered from 6 classes; `biodegradable`, `cardboard` and `glass`
removed. 0 corrupt images, 0 invalid labels, 0 split leakage.

## Configuration

| | |
|---|---|
| pretrained checkpoint | `yolo26n.pt` |
| task | `detect` |
| device | `mps` (Apple M1) |
| imgsz | 640 |
| batch | 4 |
| epochs | 25 |
| patience | 7 |
| workers | 0 |
| cache | False |
| seed | 42 |
| mosaic | 0.3 |
| close_mosaic | 5 |
| optimizer | AdamW (auto), lr0 → 0.0000714 by cosine decay |

## Training outcome

| | |
|---|---|
| Epochs completed | **25 / 25** |
| Early stopping | **No** — ran to the epoch limit |
| Best epoch | **25** (fitness 0.5240) |
| Total duration | **6.85 h** across 2 process segments |
| `best.pt` | `runs/detect/binsight_yolo26n_pretrained_final/weights/best.pt` — 5,385,157 bytes |
| `last.pt` | `runs/detect/binsight_yolo26n_pretrained_final/weights/last.pt` — 5,385,157 bytes |
| `best.pt` SHA-256 | `958e49a5fc8358442497bcb9ce98ca9b8e48a6a1da3bfb92539ac9d14f58ed67` |

Patience was never seriously challenged: 22 of 25 epochs set a new best, and the
longest gap without improvement was 2 epochs.

### Progression

| epoch | P | R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.3752 | 0.2379 | 0.2333 | 0.1375 |
| 2 | 0.4159 | 0.3515 | 0.3023 | 0.1745 |
| 3 | 0.4677 | 0.3580 | 0.3440 | 0.2185 |
| 4 | 0.5084 | 0.4040 | 0.3996 | 0.2619 |
| 5 | 0.5549 | 0.4203 | 0.4396 | 0.2864 |
| 6 | 0.5416 | 0.4480 | 0.4626 | 0.2998 |
| 7 | 0.6002 | 0.4938 | 0.5059 | 0.3205 |
| 8 | 0.5947 | 0.4739 | 0.4994 | 0.3375 |
| 9 | 0.6227 | 0.4899 | 0.5390 | 0.3615 |
| 10 | 0.6228 | 0.4987 | 0.5406 | 0.3716 |
| 11 | 0.6399 | 0.5199 | 0.5639 | 0.3862 |
| 12 | 0.6756 | 0.4869 | 0.5524 | 0.3648 |
| 13 | 0.6398 | 0.5287 | 0.5801 | 0.4084 |
| 14 | 0.7031 | 0.5644 | 0.6210 | 0.4282 |
| 15 | 0.6957 | 0.5782 | 0.6347 | 0.4578 |
| 16 | 0.6512 | 0.5731 | 0.6237 | 0.4537 |
| 17 | 0.7070 | 0.5574 | 0.6324 | 0.4527 |
| 18 | 0.6777 | 0.5920 | 0.6438 | 0.4697 |
| 19 | 0.7043 | 0.5839 | 0.6554 | 0.4761 |
| 20 | 0.6718 | 0.6031 | 0.6572 | 0.4834 |
| 21 | 0.7183 | 0.5972 | 0.6704 | 0.4944 |
| 22 | 0.6912 | 0.6090 | 0.6714 | 0.4987 |
| 23 | 0.7014 | 0.5999 | 0.6747 | 0.5035 |
| 24 | 0.6824 | 0.6107 | 0.6744 | 0.5029 |
| 25 | 0.7034 | 0.6063 | 0.6776 | 0.5069 |

mAP50-95 rose **0.1375 → 0.5069** (+269%).
Disabling mosaic at epoch 20 produced a visible step: training box loss fell
1.045 → 0.965 and precision rose 0.672 → 0.718 in one epoch.

## Validation metrics (`best.pt`, 507 images / 1,435 boxes)

| Metric | Value |
|---|---:|
| Precision | 0.7086 |
| Recall | 0.6045 |
| F1 | 0.6525 |
| mAP50 | 0.6779 |
| mAP75 | 0.545 |
| mAP50-95 | 0.5070 |

| Class | P | R | F1 | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| `paper` | 0.6350 | 0.5228 | 0.5735 | 0.5816 | 0.4788 |
| `plastic` | 0.7426 | 0.5775 | 0.6497 | 0.6662 | 0.4631 |
| `metal` | 0.7483 | 0.7132 | 0.7303 | 0.7860 | 0.5789 |

## Test metrics — held out (`best.pt`, 508 images / 1,666 boxes)

**This is the honest figure.** Validation guided checkpoint selection, so quoting
it as final performance would report a number the model was selected on.

| Metric | Value |
|---|---:|
| Precision | 0.6817 |
| Recall | 0.5777 |
| F1 | 0.6254 |
| mAP50 | 0.6480 |
| mAP75 | 0.4809 |
| mAP50-95 | 0.4620 |

| Class | P | R | F1 | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| `paper` | 0.6519 | 0.5000 | 0.5660 | 0.5713 | 0.4386 |
| `plastic` | 0.6822 | 0.5629 | 0.6169 | 0.6568 | 0.4349 |
| `metal` | 0.7108 | 0.6700 | 0.6898 | 0.7159 | 0.5126 |

The val→test drop (mAP50-95 0.5070 → 0.4620) is modest and expected —
about 0.045, consistent with mild selection bias rather than overfitting.

### Per-class reading

`metal` is the strongest class on both splits (test AP50-95 0.5126) — cans and
foil trays have hard edges and consistent specular appearance. `paper` is weakest
(0.4386) with recall exactly 0.50, which fits: paper is deformable, often
crumpled, and shades into cardboard, a class deliberately excluded from the
taxonomy. `plastic` sits between the two.

## Memory

| | |
|---|---|
| Peak process RSS | **705 MB** |
| Peak swap observed | **3.85 GB** |
| Interruptions | **Yes** — see below |

Training was interrupted three times and resumed from `last.pt` each time. Two
interruptions were caused by external `ffmpeg` video renders (476% and 316% CPU)
competing for an ~8.6 GB machine; one was a session teardown. **The trainer itself
was never the memory problem** — its RSS stayed between 61 MB and 705 MB
throughout, while the renders held 340–500 MB and saturated the CPU.

`save_period=1` meant no interruption cost more than one epoch, and every resume
restored weights, EMA, optimizer state and epoch counter correctly.

## Artefacts

- `runs/detect/binsight_yolo26n_pretrained_final/` — weights, `results.csv`, plots, `memory_log.csv`
- `runs/detect/binsight_yolo26n_pretrained_final/evaluation.json`, `per_class_metrics.csv`
- `artifacts/test_predictions/` — 18 held-out test predictions with ground truth overlaid
- Preserved diagnostics: `..._3class_b2_partial_stopped/`, `..._3class_b4/`

## Not done here

No CoreML export, no TFLite export, no iOS integration, nothing committed.

---

# Continuation run — 40 further epochs

The run above stopped at its epoch limit, not at convergence: its best epoch was
its 25th and last, and mAP50-95 had risen 0.0235 over the final five epochs. A
40-epoch continuation was therefore trained from its `best.pt`.

`resume=True` was not usable — a completed Ultralytics run stores `epoch: -1` and
refuses to resume — so the continuation started from the finished checkpoint as
initial weights, with a fresh schedule. **Transferred 708/708 items**, as opposed
to 606/708 in the first stage, because the head was already 3-class.

| | |
|---|---|
| Starting weights | `models/pytorch/BinSightYOLO26n.pt` (the 25-epoch model) |
| Epochs | 40 / 40 completed, no early stopping |
| Best epoch | **40** — again the last, so still not converged |
| Batch / imgsz | 4 / 640 |
| close_mosaic | 10 (mosaic disabled from epoch 31) |
| Duration | 2.86 h |
| Peak process RSS | 1,943 MB · peak swap 2.19 GB |
| Result | `runs/detect/binsight_yolo26n_continued/weights/best.pt` |
| SHA-256 | `f6ca723341610b4d5f19856fcecbb30528193e2f7bed47c5d21deda539276c96` |

## Held-out test — the reportable comparison

| Metric | 25-epoch model | **65-epoch model** | Δ |
|---|---:|---:|---:|
| Precision | 0.6817 | **0.7533** | **+0.0716** |
| Recall | 0.5777 | **0.5744** | −0.0033 |
| mAP50 | 0.6480 | **0.6740** | +0.0260 |
| mAP75 | 0.4809 | **0.5266** | +0.0457 |
| mAP50-95 | 0.4620 | **0.4976** | +0.0356 |

Per class, AP50-95: paper 0.4386 → **0.4578**, plastic 0.4349 → **0.4781**,
metal 0.5126 → **0.5568**. All three improved.

The gain is concentrated in **precision** — the model became markedly more
selective at essentially unchanged recall. `paper` recall actually fell, 0.5000 →
0.4693, which is the cost side of that trade.

## Two predictions that were wrong

**The restart hurt before it helped.** Starting a fresh learning-rate schedule on
a converged model knocked validation mAP50-95 from 0.5070 down to 0.3683 by
epoch 3. It took until epoch 27 to pass the old model. Anyone repeating this
should expect the dip and not stop early because of it.

**Disabling mosaic did nothing here.** The first run's jump at `close_mosaic`
(+0.0110 in one epoch) suggested that would be the decisive moment again. It was
not: the model passed the old checkpoint at epoch 27 with mosaic still on, and
the epoch mosaic was actually disabled moved mAP50-95 by **−0.0012**.

## Progression

| epoch | mAP50 | mAP50-95 | |
|---:|---:|---:|---|
| 1 | 0.6242 | 0.4423 |  |
| 3 | 0.5540 | 0.3683 | lowest point |
| 10 | 0.5977 | 0.4175 |  |
| 20 | 0.6593 | 0.4774 |  |
| 25 | 0.6720 | 0.4968 |  |
| 27 | 0.7060 | 0.5176 | passes the 25-epoch model |
| 30 | 0.7057 | 0.5281 |  |
| 31 | 0.7069 | 0.5268 | mosaic disabled |
| 35 | 0.7059 | 0.5325 |  |
| 40 | 0.7161 | 0.5427 | best |

Validation mAP50-95 finished at 0.5425 against the previous model's 0.5070.

## Status

This checkpoint is the accepted release model. It replaced the previous one in
both `models/pytorch/BinSightYOLO26n.pt` and the app's bundled CoreML package,
after passing the parity check in `reports/coreml_export_report.md` and the full
iOS test suite (178 tests).
