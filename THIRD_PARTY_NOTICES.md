# Third-party notices

BinSight uses the third-party assets and software listed here.

The **dataset is not redistributed** in this repository — it is prepared from a
separately downloaded archive — and the Python libraries are installed from their
package indexes. The one third-party artifact that *is* committed is the exported
CoreML model, which is derived from Ultralytics YOLO26 weights; see the licence
note below, because it carries a real obligation.

---

## Garbage Classification 3 (dataset)

- **Source:** <https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2>
- **Workspace / project:** `material-identification` / `garbage-classification-3`, version 2
- **Hosted by:** Roboflow Universe
- **Licence:** **CC BY 4.0**

The licence was read from the `roboflow:` block of the source `data.yaml`, not
assumed. CC BY 4.0 requires attribution, which this notice and the README
provide.

**Use in BinSight.** A filtered and re-mapped subset is used. Of the six source
classes, `BIODEGRADABLE`, `CARDBOARD` and `GLASS` are excluded entirely, and
`METAL`, `PAPER` and `PLASTIC` are kept and renumbered to `2`, `0` and `1`
respectively. Only bounding-box coordinates and class ids are used; images are
not modified beyond the resizing and augmentation Ultralytics applies during
training. Neither the source archive nor any image from it is committed —
`scripts/prepare_waste_dataset.py` rebuilds the prepared dataset from the
archive. See
[`datasets/binsight_waste_3class/dataset_report.md`](datasets/binsight_waste_3class/dataset_report.md)
for the full provenance record, including the archive's SHA-256.

BinSight claims no ownership of the source dataset.

---

## Ultralytics YOLO26 (model and library)

- **Source:** <https://github.com/ultralytics/ultralytics> (version 8.4.9)
- **Pretrained weights:** `yolo26n.pt`, COCO-pretrained, from `ultralytics/assets` v8.4.0
- **Licence:** **AGPL-3.0**

**This matters for redistribution.** The bundled model
`BinSight/Resources/Models/BinSightYOLO26n.mlpackage` is a fine-tune of
AGPL-3.0-licensed COCO-pretrained weights, and the exported package records that
licence in its own metadata (`AGPL-3.0 License (https://ultralytics.com/license)`).
The AGPL obligations follow the derived weights. Distributing this app publicly
would require either complying with AGPL-3.0 or obtaining an Ultralytics
Enterprise Licence. This project is a portfolio piece and is not distributed
through the App Store.

---

## Software

| Package | Licence |
|---|---|
| Ultralytics (YOLO26) | AGPL-3.0 |
| PyTorch | BSD 3-Clause |
| coremltools | BSD 3-Clause |
| NumPy | BSD 3-Clause |
| OpenCV (`opencv-python`) | Apache License 2.0 |
| Pillow | MIT-CMU |
| PyYAML | MIT |

The iOS application target has **no third-party runtime dependencies**. It uses
only Apple frameworks — SwiftUI, AVFoundation, CoreML, CoreImage, CoreVideo. The
`Podfile` is retained but declares no pods.

---

## Historical

Earlier phases of this project used a TrashNet-based six-class TensorFlow
classifier with EfficientNet-Lite0 weights, and a TACO-based YOLO26 detector.
Neither ships in the current app and neither is present in the working tree; both
remain in Git history. The notices below are kept so that history stays
attributable.

**TrashNet** — <https://github.com/garythung/trashnet>, Gary Thung and Mindy
Yang, MIT licence. The README asks that users of the dataset cite the repository.

**EfficientNet-Lite0** — preset `efficientnet_lite0_ra_imagenet`, distributed by
[KerasHub](https://keras.io/keras_hub/), Apache License 2.0, pretrained on
ImageNet-1k.

**TACO** — <http://tacodataset.org>, Pedro F. Proença and Pedro Simões,
CC BY 4.0.
