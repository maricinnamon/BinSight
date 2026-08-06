# Model Card — BinSightWasteClassifier

> ## ⚠️ Status: NOT TRAINED — this card is an empty template
>
> **No model exists yet.** The training notebook
> ([`ml/notebooks/train_binsight_classifier.ipynb`](../ml/notebooks/train_binsight_classifier.ipynb))
> has been written and statically validated but **never executed**.
>
> Every field below marked `NOT MEASURED` is a placeholder. Fill them from
> `ml/artifacts/metrics.json` after a real run — do not estimate, infer from
> published TrashNet results, or copy numbers from a paper. If a number is not
> in your own `metrics.json`, it does not belong on this card.

---

## Model details

| | |
|---|---|
| **Name** | BinSightWasteClassifier |
| **Version** | 1.0.0 (unreleased) |
| **Author** | Maryna Antonevych |
| **Type** | Single-label image classification, 6 classes |
| **Architecture** | EfficientNet-Lite0, ImageNet-pretrained, transfer-learned |
| **Backbone preset** | `efficientnet_lite0_ra_imagenet` (KerasHub 0.30.0) |
| **Format** | LiteRT / TFLite |
| **Quantisation** | `NOT MEASURED` — selected candidate |
| **File size** | `NOT MEASURED` |
| **SHA256** | `NOT MEASURED` |
| **Trained on** | `NOT MEASURED` — date, GPU type, wall-clock duration |

## Intended use

**In scope.** A portfolio demonstration inside the BinSight iOS app: a person
points the camera at *one* discarded item and gets a suggested material category
plus a confidence score, computed entirely on device.

**Out of scope.**

- Anything that determines what actually goes in a bin. Recycling rules are
  local, change often, and are not a function of material alone.
- Regulatory, commercial or industrial waste sorting.
- Multiple items in one frame — the model predicts a single label.
- Hazardous-material identification (batteries, chemicals, medical waste, sharps).
- Any use where a wrong answer has a cost beyond mild inconvenience.

The app labels its guidance as generic and non-local, and this must stay true of
any surface that presents these predictions.

## Classes

Output index order — this is the contract the app depends on, and it is **not**
alphabetical:

| Index | Internal label | App label |
|---:|---|---|
| 0 | `cardboard` | Cardboard |
| 1 | `glass` | Glass |
| 2 | `metal` | Metal |
| 3 | `paper` | Paper |
| 4 | `plastic` | Plastic |
| 5 | `general_waste` | General waste |

`general_waste` is TrashNet's `trash` class, renamed.

## Training data

**TrashNet** — <https://github.com/garythung/trashnet>, `dataset-resized.zip`
(512×384 JPEGs). MIT licence; the repository asks to be cited. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

| Class | Images |
|---|---:|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash → general_waste | 137 |
| **Total** | **2 527** |

*(Counted directly from the archive. The notebook re-counts at run time and
records the result in `metrics.json`; if those numbers ever disagree with this
table, `metrics.json` is authoritative.)*

Split: stratified 70 / 15 / 15, seed 42, asserted disjoint. Manifest at
`ml/artifacts/split_manifest.csv`.

Class imbalance is ≈4.3× (paper vs trash), which triggers balanced class
weighting.

## Evaluation results

**`NOT MEASURED` — the notebook has not been run.**

Fill from `ml/artifacts/metrics.json`:

| Metric | Value |
|---|---|
| Test images | `NOT MEASURED` |
| Accuracy | `NOT MEASURED` |
| Macro F1 | `NOT MEASURED` |
| Weighted F1 | `NOT MEASURED` |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cardboard | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |
| glass | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |
| metal | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |
| paper | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |
| plastic | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |
| general_waste | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` | `NOT MEASURED` |

Plots (produced by a run): `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `training_curves.png` in `ml/artifacts/`.

### Keras vs TFLite agreement

| | |
|---|---|
| Comparison batch size | `NOT MEASURED` |
| Max abs. probability difference | `NOT MEASURED` |
| Mean abs. probability difference | `NOT MEASURED` |
| Top-1 agreement | `NOT MEASURED` |

## Limitations

These hold regardless of what the test accuracy turns out to be.

**Domain shift is the dominant limitation.** TrashNet images are studio-like:
one object, centred, well lit, plain white background, consistent distance. Real
BinSight input is none of those things. Expect a substantial drop on handheld
photos from:

- **Clutter** — several items in frame, patterned worktops. The model has never
  seen a background that carries information.
- **Lighting** — tungsten, evening light, on-device flash, shadows cast by the
  phone. Colour-temperature shifts hit glass-vs-plastic hardest, since both are
  judged largely on transparency and specular highlights.
- **Partial objects** — items held close enough to crop, or half inside a bin.
- **Packaging locality** — TrashNet is one collector's mid-2010s
  American/European packaging. Composites (coffee cups, crisp packets, Tetra Pak)
  barely appear, and those are exactly the items people are unsure about.

**Other limitations.**

- **Small dataset.** 2 527 images total; `general_waste` has 137. That class is
  a residual category rather than a material, so it has both the least data and
  the least coherent visual definition.
- **Single label.** No "several items" or "I don't know" class. The app handles
  uncertainty with a confidence threshold in the UI, not in the model.
- **No fairness or geographic evaluation.** The dataset's provenance is one
  contributor's collection; no assessment has been made of how performance
  varies by region, packaging market, or camera hardware.
- **The confidence score is not calibrated.** Softmax outputs are not
  probabilities of being correct, and no calibration (temperature scaling,
  reliability diagram) has been done.
- **No real-world evaluation set exists.** Measuring the drop described above
  needs a hand-collected BinSight test set. There isn't one.

## Ethical and practical considerations

Giving someone a confident wrong answer about disposal is worse than giving them
none — it can contaminate a recycling stream, and it erodes trust in the whole
category of tool. The app therefore presents guidance as generic, never as local
law, and declines to name a class below its confidence threshold.

Camera frames are processed on device and never leave the phone. There is no
telemetry, no upload, and no account.

## Not production-ready

This is a portfolio demonstration. It has not been evaluated on real-world
input, its confidence is uncalibrated, and its training data does not represent
the packaging any particular user will encounter. It should not be shipped as
authoritative disposal advice.

## Reproducing

See [`ml/README.md`](../ml/README.md). Notebook, pinned environment, split
manifest and fixed seed (42) are all in the repository; the dataset is
downloaded at run time and never committed.
