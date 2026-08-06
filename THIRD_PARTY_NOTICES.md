# Third-party notices

BinSight uses the third-party assets and software listed here. None of it is
redistributed in this repository — the dataset is downloaded at training time,
and the libraries are installed from their package indexes.

---

## TrashNet (dataset)

- **Source:** <https://github.com/garythung/trashnet>
- **Archive used:** `data/dataset-resized.zip` (512×384 JPEGs, 42.8 MB, 2 527 images)
- **Mirror of the full-resolution set:** <https://huggingface.co/datasets/garythung/trashnet>
- **Author:** Gary Thung (with Mindy Yang)
- **Licence:** MIT

The TrashNet README requests that users of the dataset cite the repository:

> If you are using the dataset, please give a citation of this repository.

**Citation**

```
Gary Thung and Mindy Yang. TrashNet: dataset of images of trash.
https://github.com/garythung/trashnet
```

**Use in BinSight.** The dataset is downloaded at training time by
`ml/notebooks/train_binsight_classifier.ipynb`. Neither the archive nor any
image from it is committed to this repository. The class folder `trash` is
mapped to the internal label `general_waste` and displayed as "General waste";
no image content is modified beyond resizing and standard augmentation during
training.

---

## EfficientNet-Lite0 pretrained weights

- **Preset:** `efficientnet_lite0_ra_imagenet`
- **Distributed by:** [KerasHub](https://keras.io/keras_hub/) (`keras-hub`)
- **Architecture:** EfficientNet-Lite, from *EfficientNet: Rethinking Model
  Scaling for Convolutional Neural Networks* (Tan & Le, 2019)
- **Pretraining data:** ImageNet-1k, RandAugment recipe
- **Licence:** Apache License 2.0 (KerasHub and its distributed weights)

Weights are downloaded by the notebook at training time and are not committed.

---

## Software

| Package | Licence |
|---|---|
| TensorFlow / LiteRT | Apache License 2.0 |
| Keras | Apache License 2.0 |
| KerasHub | Apache License 2.0 |
| NumPy | BSD 3-Clause |
| pandas | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| Matplotlib | Matplotlib License (BSD-style) |
| Pillow | MIT-CMU |

The iOS application target itself has **no third-party runtime dependencies**.
It uses only Apple frameworks.
