"""Model construction and the two-stage transfer-learning schedule."""
from __future__ import annotations

from typing import Any

from .config import CLASS_NAMES, Config
from .utils import get_logger

LOGGER = get_logger()


def build_model(config: Config):
    """MobileNetV3Small transfer model. [0, 255] RGB in, softmax probabilities out.

    MobileNetV3's Keras implementation carries `include_preprocessing=True`,
    which puts the [0,255] → [-1,1] rescaling *inside* the graph. That is
    exactly what we want for deployment: the exported TFLite model takes raw
    pixel values and the Swift side has no normalisation constants to get wrong
    — historically the most common cause of a model that trains well and then
    underperforms on device.
    """
    import tensorflow as tf

    size = config.data.image_size
    inputs = tf.keras.Input(shape=(size, size, 3), dtype="float32", name="image")

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(size, size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=config.model.bake_preprocessing,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(features)
    dropped = tf.keras.layers.Dropout(config.model.dropout, name="head_dropout")(pooled)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", dtype="float32", name="probabilities"
    )(dropped)

    model = tf.keras.Model(inputs, outputs, name="binsight_trashnet")
    return model, backbone


def compile_model(model, learning_rate: float) -> None:
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )


def unfreeze_backbone_tail(backbone, fraction: float) -> dict[str, int]:
    """Unfreezes the top `fraction` of backbone layers for fine-tuning.

    BatchNormalization layers stay frozen throughout. With ~1 700 training
    images and small batches, letting BN recompute its running statistics
    reliably makes validation accuracy worse — the batch estimates are too
    noisy to improve on ImageNet's.
    """
    import tensorflow as tf

    backbone.trainable = True
    total = len(backbone.layers)
    cut = int(total * (1.0 - fraction))

    frozen_bn = 0
    trainable_layers = 0
    for index, layer in enumerate(backbone.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
            continue
        layer.trainable = index >= cut
        if layer.trainable:
            trainable_layers += 1

    return {
        "total_layers": total,
        "unfrozen_from_index": cut,
        "trainable_layers": trainable_layers,
        "frozen_batchnorm_layers": frozen_bn,
    }


def make_callbacks(config: Config, checkpoint_path, patience: int, log_dir=None) -> list[Any]:
    """Early stopping on validation loss, with best-weight restoration."""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2),
            min_lr=1e-7, verbose=1,
        ),
    ]
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.CSVLogger(str(log_dir), append=True))
    return callbacks


def trainable_parameter_count(model) -> int:
    import numpy as np
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))
