# BinSight 3-class waste detection dataset

## Source

| | |
|---|---|
| Dataset | Garbage Classification 3, version 2 |
| Roboflow workspace / project | `material-identification` / `garbage-classification-3` |
| Source URL (from `data.yaml`) | <https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2> |
| Archive supplied | `~/Downloads/archive.zip` → copied to `datasets/_tmp/archive.zip` |
| Archive SHA-256 | `eadbff8813b43163b6753938a5838f999fcaa7bf8e9cbb97bce606e075763192` |
| Archive size | 199,963,613 bytes · 20,929 entries |
| Extracted root | `GARBAGE CLASSIFICATION/` |
| **License** | **CC BY 4.0** — read from the `roboflow:` block of the source `data.yaml`, not guessed |

Attribution is required under CC BY 4.0: credit the *material-identification /
garbage-classification-3* dataset and link the URL above.

> **Note on provenance.** The brief named a different Roboflow dataset
> (`garbage-computer-vision/yolo-waste-detection`). The archive actually supplied
> is `material-identification/garbage-classification-3` v2. This report describes
> what was really processed.

## Task

**Object detection.** Bounding boxes only — no segmentation, no masks, no
polygons. Verified: all 74,090 source label rows carry exactly five fields, and
so do all 16,173 final rows. A YOLO-seg polygon row has a variable field
count and would have failed the parser.

## Classes

Source order, read from the source `data.yaml` rather than assumed:

| Source ID | Source name | Action | Final ID |
|---:|---|---|---:|
| 0 | BIODEGRADABLE | excluded | — |
| 1 | CARDBOARD | excluded | — |
| 2 | GLASS | excluded | — |
| 3 | METAL | kept | **2** |
| 4 | PAPER | kept | **0** |
| 5 | PLASTIC | kept | **1** |

Final mapping:

```
0 paper
1 plastic
2 metal
```

The source order is *not* the final order — PAPER is source id 4 and becomes 0.
Hardcoding the final order onto source ids would have mislabelled the dataset
silently.

## Filtering rules

1. Keep only boxes whose source class is PAPER, PLASTIC or METAL.
2. Remap to the final ids above.
3. Drop geometrically invalid boxes (zero width or height, NaN/Inf, out of frame).
4. If an image retains no boxes, drop the image **and** its label.
5. Never derive a replacement box from anything else.

## Size

| Split | Images | Boxes | Paper | Plastic | Metal |
|---|---:|---:|---:|---:|---:|
| train | 4059 | 13072 | 3509 | 4860 | 4703 |
| val | 507 | 1435 | 394 | 497 | 544 |
| test | 508 | 1666 | 484 | 588 | 594 |
| **TOTAL** | **5,074** | **16,173** | **4,387** | **5,945** | **5,841** |

## Class distribution

| Class | Boxes | Share |
|---|---:|---:|
| paper | 4,387 | 27.13% |
| plastic | 5,945 | 36.76% |
| metal | 5,841 | 36.12% |

**Imbalance is mild — 1.36:1** between the largest
(`plastic`) and smallest (`paper`). No reweighting or resampling is applied; this
is a workable distribution for a first detector and does not need correcting.

## What was removed

| | |
|---|---:|
| Source images | 10,464 |
| Source boxes | 74,090 |
| Boxes removed (excluded classes) | **57,914** |
| Boxes removed (invalid geometry) | 3 |
| Images removed (no target objects left) | **5,390** |
| Duplicate images dropped | 0 |
| **Final images** | **5,074** |
| **Final boxes** | **16,173** |

The three invalid boxes were degenerate — zero width or zero height — and were
dropped rather than repaired. Ambiguous annotations are never guessed at.

## Splits

**The source splits were discarded.** Preserving them was the preference, but they
are unusable for these three classes:

| Source split | paper | plastic | metal |
|---|---:|---:|---:|
| train | 2,981 | 4,146 | 3,948 |
| valid | **33** | 214 | 1,360 |
| test | 1,376 | 1,585 | 533 |

`valid` holds 33 paper boxes against 1,360 metal — 85% metal. Per-class AP for
paper cannot be measured from 33 boxes, and validating on an 85%-metal set while
training on a balanced one produces numbers that mean nothing.

Replaced with a deterministic **80/10/10 re-split, seed 42**, stratified by each
image's dominant class so all three splits share a distribution. Grouped by
source-image identity so no two variants of one photograph can separate — though
in this export every file proved to be a distinct original (5,074 files, 5,074
distinct base identities, zero groups spanning source splits).

## Validation

Run independently by `scripts/validate_waste_dataset.py`, which re-reads
everything from disk rather than trusting the builder's own summary.

| Check | Result |
|---|---|
| All images readable | **yes** — 0 corrupt, 0 zero-size |
| All labels valid | **yes** — 0 malformed rows |
| Field count per row | **exactly 5** for all 16,173 rows |
| Class IDs | **{0, 1, 2}** — nothing else present |
| Excluded classes remaining | **none** |
| Coordinates normalised and in range | **yes** |
| Boxes inside the frame | **yes** |
| Image/label pairs synchronised | **yes** — 0 orphans, 0 missing |
| Empty label files | **0** |
| Duplicate filenames | **0** |
| Identical images across splits (leakage) | **0** |
| `data.yaml` resolves in Ultralytics | **yes** — `nc: 3`, all three split paths found |

**Result: PASS**

## Reproducibility

The image data is **not** committed — thousands of files do not belong in this
repository. To rebuild it exactly:

```bash
# 1. obtain the archive (CC BY 4.0, link above) and place it anywhere readable
mkdir -p datasets/_tmp && cp /path/to/archive.zip datasets/_tmp/
unzip -q datasets/_tmp/archive.zip -d datasets/_tmp/garbage_source

# 2. build the 3-class dataset (deterministic, seed 42)
python scripts/prepare_waste_dataset.py \
    --source "datasets/_tmp/garbage_source/GARBAGE CLASSIFICATION" \
    --output datasets/binsight_waste_3class

# 3. validate and preview
python scripts/validate_waste_dataset.py
python scripts/preview_waste_dataset.py

# 4. remove the temporary source
rm -rf datasets/_tmp
```

Same seed, same inputs, same output. `manifest.csv` records every final file with
its source split, base identity, SHA-256 and dimensions, so any image can be
traced back.

## Not done here

No model was trained. Nothing was exported to CoreML or TFLite. The iOS app was
not touched.
