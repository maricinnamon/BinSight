"""Typed configuration, loaded from YAML with sane defaults.

One place for every tunable value so a run can be reproduced from
`configs/default.yaml` alone.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

# This experiment's root: .../ml/experiments/trashnet_tensorflow_baseline
# config.py sits at src/binsight_training/config.py, so two levels up.
SUBPROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
# The repository root: experiment -> experiments -> ml -> repo.
REPO_ROOT = SUBPROJECT_ROOT.parents[2]

# The dataset's own class directory names, in the order the model emits them.
#
# Deliberately explicit: `image_dataset_from_directory` sorts alphabetically,
# which happens to match here, but relying on that is how label mappings break
# silently when a class is renamed. Every loader is passed this list and the
# result is asserted against it.
CLASS_NAMES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# What the iOS app shows a person. `trash` is the residual class, not a
# recyclable one, so it maps to "General waste" and never to a recycling stream.
DISPLAY_NAMES: dict[str, str] = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "General waste",
}

# The internal label the Swift `WasteCategory` enum uses for `trash`.
IOS_LABEL_ALIASES: dict[str, str] = {"trash": "general_waste"}


@dataclasses.dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    cache: bool = True


@dataclasses.dataclass
class ModelConfig:
    backbone: str = "MobileNetV3Small"
    dropout: float = 0.25
    # Input arrives in [0, 255]; the backbone's own preprocessing is baked into
    # the graph so the iOS side has no constants to get wrong.
    bake_preprocessing: bool = True


@dataclasses.dataclass
class StageConfig:
    epochs: int
    learning_rate: float
    patience: int = 5


@dataclasses.dataclass
class TrainConfig:
    stage_a: StageConfig = dataclasses.field(
        default_factory=lambda: StageConfig(epochs=15, learning_rate=1e-3, patience=5)
    )
    stage_b: StageConfig = dataclasses.field(
        default_factory=lambda: StageConfig(epochs=20, learning_rate=1e-5, patience=6)
    )
    # Fraction of backbone layers unfrozen in stage B, counted from the top.
    fine_tune_fraction: float = 0.30
    use_class_weights: bool = True
    # Batch sizes tried in order if the previous one runs out of memory.
    batch_size_ladder: tuple[int, ...] = (32, 16, 8)


@dataclasses.dataclass
class ExportConfig:
    model_name: str = "binsight_trashnet"
    model_version: str = "1.0.0"
    # Maximum absolute degradation a quantised model may show against float32
    # before it is rejected, in accuracy and macro F1 (percentage points).
    max_quantisation_degradation_pp: float = 2.0
    representative_samples: int = 200


@dataclasses.dataclass
class InferenceConfig:
    # Below this the app says "Not sure" rather than naming a class. Matches
    # ScannerConfiguration.minimumConfidence on the iOS side.
    confidence_threshold: float = 0.65


@dataclasses.dataclass
class Config:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    export: ExportConfig = dataclasses.field(default_factory=ExportConfig)
    inference: InferenceConfig = dataclasses.field(default_factory=InferenceConfig)

    @property
    def root(self) -> pathlib.Path:
        return SUBPROJECT_ROOT

    @property
    def repo_root(self) -> pathlib.Path:
        return REPO_ROOT

    # --- Convenience paths --------------------------------------------------
    @property
    def raw_dir(self) -> pathlib.Path:
        return self.root / "data" / "raw"

    @property
    def splits_dir(self) -> pathlib.Path:
        return self.root / "data" / "splits"

    @property
    def reports_dir(self) -> pathlib.Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> pathlib.Path:
        return self.reports_dir / "figures"

    @property
    def predictions_dir(self) -> pathlib.Path:
        return self.reports_dir / "predictions"

    @property
    def models_dir(self) -> pathlib.Path:
        return self.root / "models"

    @property
    def tflite_dir(self) -> pathlib.Path:
        return self.models_dir / "tflite"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir, self.splits_dir, self.reports_dir, self.figures_dir,
            self.predictions_dir, self.reports_dir / "logs",
            self.models_dir / "keras", self.models_dir / "saved_model", self.tflite_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _merge(target: Any, values: dict[str, Any]) -> Any:
    """Overlays a dict onto a dataclass, recursing into nested dataclasses."""
    for key, value in values.items():
        if not hasattr(target, key):
            raise KeyError(f"unknown configuration key: {key!r}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)
    return target


def load_config(path: str | pathlib.Path | None = None) -> Config:
    """Loads configuration, falling back to `configs/default.yaml`."""
    config = Config()
    if path is None:
        path = config.root / "configs" / "default.yaml"
    path = pathlib.Path(path)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        _merge(config, raw)
    return config


def as_dict(config: Config) -> dict[str, Any]:
    return dataclasses.asdict(config)
