"""Dataset discovery, validation, deterministic splitting and tf.data pipelines."""
from __future__ import annotations

import collections
import pathlib
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import CLASS_NAMES, Config
from .utils import get_logger, sha256_file

LOGGER = get_logger()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def extract_archives(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    """Extracts any zip in `raw_dir` that has not been extracted yet.

    Nested archives are handled by re-scanning after each pass, because
    TrashNet has historically shipped as a zip inside a zip.
    """
    extracted: list[pathlib.Path] = []
    for _ in range(3):  # bounded: guards against a pathological archive chain
        archives = [
            p for p in raw_dir.rglob("*.zip")
            if not (p.parent / f".{p.stem}.extracted").exists()
        ]
        if not archives:
            break
        for archive in archives:
            target = archive.parent / archive.stem
            LOGGER.info("Extracting %s", archive.name)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            (archive.parent / f".{archive.stem}.extracted").touch()
            extracted.append(target)
    # macOS sidecar directories confuse class discovery.
    for junk in raw_dir.rglob("__MACOSX"):
        if junk.is_dir():
            import shutil
            shutil.rmtree(junk, ignore_errors=True)
    return extracted


def find_class_directories(raw_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Locates the six class directories at whatever depth they sit.

    The archive's nesting is not trusted: the search is by directory *name*,
    and a candidate only counts if it actually contains images.
    """
    found: dict[str, pathlib.Path] = {}
    for candidate in sorted(raw_dir.rglob("*")):
        if not candidate.is_dir():
            continue
        name = candidate.name.lower()
        if name not in CLASS_NAMES or name in found:
            continue
        has_images = any(
            child.suffix.lower() in IMAGE_SUFFIXES and not child.name.startswith(".")
            for child in candidate.iterdir() if child.is_file()
        )
        if has_images:
            found[name] = candidate
    return found


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def build_manifest(class_dirs: dict[str, pathlib.Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validates every image and returns (manifest, corrupt).

    Every file is fully decoded. `Image.verify()` only checks the header, so a
    truncated JPEG passes it and then explodes mid-training.
    """
    records: list[dict[str, Any]] = []
    corrupt: list[dict[str, Any]] = []

    for class_name in CLASS_NAMES:
        directory = class_dirs.get(class_name)
        if directory is None:
            LOGGER.warning("Class directory missing: %s", class_name)
            continue
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
        )
        for path in files:
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()          # forces a full decode
                    width, height = image.size
                if width < 32 or height < 32:
                    raise ValueError(f"implausibly small image {width}x{height}")
            except Exception as exc:  # noqa: BLE001 - record anything at all
                corrupt.append({
                    "path": str(path),
                    "class": class_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            records.append({
                "path": str(path),
                "filename": path.name,
                "class_name": class_name,
                "label_index": CLASS_NAMES.index(class_name),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    manifest = pd.DataFrame(records)
    return manifest, pd.DataFrame(corrupt)


def find_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    """Groups of files with identical content hashes."""
    if manifest.empty:
        return pd.DataFrame(columns=["sha256", "count", "paths", "classes"])
    counts = manifest.groupby("sha256").size()
    duplicated = counts[counts > 1]
    rows = []
    for digest in duplicated.index:
        subset = manifest[manifest["sha256"] == digest]
        rows.append({
            "sha256": digest,
            "count": int(len(subset)),
            "paths": " | ".join(subset["path"].tolist()),
            "classes": " | ".join(sorted(set(subset["class_name"]))),
        })
    return pd.DataFrame(rows)


def summarise(manifest: pd.DataFrame, corrupt: pd.DataFrame, duplicates: pd.DataFrame) -> dict[str, Any]:
    per_class = manifest.groupby("class_name").size().reindex(CLASS_NAMES).fillna(0).astype(int)
    counts = per_class.to_dict()
    imbalance = float(per_class.max() / per_class.min()) if per_class.min() > 0 else None
    return {
        "total_images": int(len(manifest)),
        "class_counts": counts,
        "class_order": CLASS_NAMES,
        "imbalance_ratio": imbalance,
        "largest_class": per_class.idxmax() if len(per_class) else None,
        "smallest_class": per_class.idxmin() if len(per_class) else None,
        "corrupt_count": int(len(corrupt)),
        "duplicate_group_count": int(len(duplicates)),
        "duplicate_file_count": int(duplicates["count"].sum() - len(duplicates)) if len(duplicates) else 0,
        "width": _stats(manifest, "width"),
        "height": _stats(manifest, "height"),
        "aspect_ratio": _stats(manifest, "aspect_ratio"),
    }


def _stats(frame: pd.DataFrame, column: str) -> dict[str, float] | None:
    if frame.empty or column not in frame:
        return None
    series = frame[column]
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": round(float(series.mean()), 4),
        "median": float(series.median()),
    }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def make_splits(manifest: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Deterministic stratified train/val/test split.

    Splitting happens over **hash groups**, not individual files, so byte-identical
    duplicates can never straddle a boundary and inflate the test score.
    """
    data = config.data
    rng = np.random.default_rng(data.seed)

    # One row per unique content hash, carrying its class.
    groups = (
        manifest.groupby("sha256")
        .agg(class_name=("class_name", "first"), size=("path", "size"))
        .reset_index()
    )

    assignments: dict[str, str] = {}
    for class_name in CLASS_NAMES:
        subset = groups[groups["class_name"] == class_name]
        if subset.empty:
            continue
        # Sort first, then shuffle with a seeded generator: stable regardless of
        # filesystem ordering.
        hashes = np.array(sorted(subset["sha256"].tolist()))
        rng.shuffle(hashes)

        total = len(hashes)
        n_train = int(round(total * data.train_fraction))
        n_val = int(round(total * data.val_fraction))
        # Give the remainder to test so the three always sum to `total`.
        n_train = min(n_train, total)
        n_val = min(n_val, total - n_train)

        for digest in hashes[:n_train]:
            assignments[digest] = "train"
        for digest in hashes[n_train:n_train + n_val]:
            assignments[digest] = "validation"
        for digest in hashes[n_train + n_val:]:
            assignments[digest] = "test"

    result = manifest.copy()
    result["split"] = result["sha256"].map(assignments)
    return result


def verify_splits(splits: pd.DataFrame) -> dict[str, Any]:
    """Fails loudly on anything that would invalidate the test metric."""
    problems: list[str] = []

    if splits["split"].isna().any():
        problems.append(f"{int(splits['split'].isna().sum())} rows have no split")

    by_split = {name: set(group["path"]) for name, group in splits.groupby("split")}
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                problems.append(f"{len(overlap)} files shared between {left} and {right}")

    hashes = {name: set(group["sha256"]) for name, group in splits.groupby("split")}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = hashes[left] & hashes[right]
            if overlap:
                problems.append(f"{len(overlap)} duplicate hashes shared between {left} and {right}")

    for split_name in ("train", "validation", "test"):
        present = set(splits[splits["split"] == split_name]["class_name"])
        missing = set(CLASS_NAMES) - present
        if missing:
            problems.append(f"{split_name} is missing classes: {sorted(missing)}")

    missing_paths = [p for p in splits["path"] if not pathlib.Path(p).exists()]
    if missing_paths:
        problems.append(f"{len(missing_paths)} manifest paths do not exist")

    return {
        "ok": not problems,
        "problems": problems,
        "counts": splits.groupby("split").size().to_dict(),
        "per_class": {
            split_name: group.groupby("class_name").size().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
            for split_name, group in splits.groupby("split")
        },
    }


def load_split(config: Config, name: str) -> pd.DataFrame:
    path = config.splits_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_dataset.py first."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# tf.data pipelines
# --------------------------------------------------------------------------

def build_augmenter(seed: int):
    """Training-only augmentation.

    Kept modest on purpose: colour and transparency *are* the material cue that
    separates glass from plastic, so aggressive colour jitter would train the
    model to ignore the signal it needs. Never part of the exported graph.
    """
    import tensorflow as tf
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.12, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.15, seed=seed),
        ],
        name="augmenter",
    )


def make_dataset(
    frame: pd.DataFrame,
    config: Config,
    *,
    shuffle: bool,
    augment: bool,
    batch_size: int | None = None,
):
    """Builds a tf.data pipeline delivering [0, 255] float32 RGB batches.

    Values stay in [0, 255] because MobileNetV3's own preprocessing is baked
    into the model graph — see `model.build_model`.
    """
    import tensorflow as tf

    size = config.data.image_size
    batch = batch_size or config.data.batch_size
    autotune = tf.data.AUTOTUNE

    paths = frame["path"].tolist()
    labels = frame["label_index"].to_numpy(dtype=np.int32)

    def decode(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, [size, size], method="bilinear")
        image = tf.cast(image, tf.float32)   # still 0..255
        image.set_shape([size, size, 3])
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=config.data.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(decode, num_parallel_calls=autotune, deterministic=not shuffle)

    if config.data.cache and not augment:
        # Only the deterministic pipelines are cached; caching augmented data
        # would freeze one set of random transforms for every epoch.
        dataset = dataset.cache()

    dataset = dataset.batch(batch)

    if augment:
        augmenter = build_augmenter(config.data.seed)
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=autotune
        )

    return dataset.prefetch(autotune)


def class_weights(frame: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights from the training split.

    TrashNet's `trash` class is roughly a quarter the size of `paper`, so
    without this the model learns it can safely ignore it.
    """
    counts = collections.Counter(frame["label_index"].tolist())
    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)
    return {
        index: total / (n_classes * counts.get(index, 1))
        for index in range(n_classes)
    }
