"""Seeding, hashing, logging and small shared helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import platform
import random
import subprocess
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"


def get_logger(name: str = "binsight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seeds(seed: int = 42) -> None:
    """Seeds every RNG that affects a run.

    `PYTHONHASHSEED` must be set before the interpreter starts to have full
    effect, so it is also exported by the Makefile; setting it here still helps
    anything spawned later.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Available from TF 2.9; harmless where missing.
        if hasattr(tf.keras.utils, "set_random_seed"):
            tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | pathlib.Path, payload: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: str | pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def package_versions() -> dict[str, str | None]:
    """Versions of everything that can change a result."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name, module in (
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("PIL", "PIL"),
    ):
        try:
            mod = __import__(module)
            versions[name] = getattr(mod, "__version__", None)
        except Exception:
            versions[name] = None
    try:
        from importlib import metadata
        versions["tensorflow-metal"] = metadata.version("tensorflow-metal")
    except Exception:
        versions["tensorflow-metal"] = None
    return versions


def describe_devices() -> dict[str, Any]:
    """Physical devices TensorFlow can see, plus whether Metal is active."""
    try:
        import tensorflow as tf
    except ImportError:
        return {"error": "tensorflow not importable"}
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    return {
        "all": [d.name for d in tf.config.list_physical_devices()],
        "gpus": gpus,
        "metal_available": bool(gpus),
    }


def system_memory_gb() -> float | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        )
        return round(int(out.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        return None


def configure_device(preference: str = "auto") -> dict[str, object]:
    """Chooses CPU or GPU before any TensorFlow op runs.

    `tensorflow-metal` 1.1.0 on this 8 GB Apple Silicon Mac aborted training
    with SIGABRT at batch 32 and SIGSEGV at batch 16, in both cases partway
    through a run rather than at setup. Training on CPU is slower but completes,
    so `auto` deliberately prefers CPU here. Pass "gpu" to force Metal and
    re-test whether a newer plugin has fixed it.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    chosen = preference
    if preference == "auto":
        chosen = "cpu"          # see docstring: Metal is not stable on this host

    if chosen == "cpu" and gpus:
        tf.config.set_visible_devices([], "GPU")

    visible = [d.name for d in tf.config.get_visible_devices()]
    return {
        "preference": preference,
        "chosen": chosen,
        "gpus_present": [d.name for d in gpus],
        "visible_devices": visible,
        "metal_used": chosen == "gpu" and bool(gpus),
    }
