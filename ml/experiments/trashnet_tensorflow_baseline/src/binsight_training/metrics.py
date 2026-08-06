"""Evaluation metrics and figures. Never accuracy alone."""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .config import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy plus the per-class and macro figures that matter on imbalanced data.

    Macro F1 is the headline: with `trash` at roughly a quarter the size of
    `paper`, accuracy can look respectable while the smallest class is being
    predicted almost never.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "n": int(len(y_true)),
    }


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import classification_report
    return classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )


def report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"class": name, **values} for name, values in metrics["per_class"].items()
    ]
    rows.append({
        "class": "macro avg", "precision": None, "recall": None,
        "f1": metrics["macro_f1"], "support": metrics["n"],
    })
    rows.append({
        "class": "weighted avg", "precision": None, "recall": None,
        "f1": metrics["weighted_f1"], "support": metrics["n"],
    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(metrics: dict[str, Any], path: pathlib.Path, normalise: bool = True) -> None:
    plt = _matplotlib()
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    if normalise:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
        matrix = np.nan_to_num(matrix)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="magma" if normalise else "viridis")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test split, row-normalised)" if normalise
                 else "Confusion matrix (test split, counts)")

    threshold = matrix.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}" if normalise else f"{int(value)}",
                    ha="center", va="center", fontsize=9,
                    color="white" if value < threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict[str, list[float]], metric: str, path: pathlib.Path,
                 boundary: int | None = None, title: str | None = None) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if metric in history:
        ax.plot(history[metric], label=f"train {metric}", linewidth=1.8)
    val_key = f"val_{metric}"
    if val_key in history:
        ax.plot(history[val_key], label=f"validation {metric}", linewidth=1.8)
    if boundary:
        ax.axvline(boundary - 0.5, color="grey", linestyle="--", linewidth=1)
        ax.text(boundary - 0.4, ax.get_ylim()[0], " fine-tuning", fontsize=8, color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Training {metric}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(metrics: dict[str, Any], path: pathlib.Path) -> None:
    plt = _matplotlib()
    names = list(metrics["per_class"])
    values = [metrics["per_class"][n]["f1"] for n in names]
    supports = [metrics["per_class"][n]["support"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(names, values, color="#5DE2B8")
    ax.axhline(metrics["macro_f1"], color="#FF5C8A", linestyle="--", linewidth=1.5,
               label=f"macro F1 = {metrics['macro_f1']:.3f}")
    for bar, value, support in zip(bars, values, supports):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                f"{value:.2f}\nn={support}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (test split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_grid(rows: pd.DataFrame, path: pathlib.Path, title: str, columns: int = 4) -> None:
    """A grid of example images with true/predicted labels and confidence."""
    from PIL import Image
    plt = _matplotlib()

    if rows.empty:
        return
    count = min(len(rows), columns * 2)
    subset = rows.head(count)
    n_rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(n_rows, columns, figsize=(3.0 * columns, 3.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, subset.iterrows()):
        try:
            with Image.open(row["path"]) as image:
                ax.imshow(image.convert("RGB"))
        except Exception:
            continue
        correct = row["true_label"] == row["predicted_label"]
        ax.set_title(
            f"true: {row['true_label']}\npred: {row['predicted_label']}\np={row['confidence']:.2f}",
            fontsize=8.5, color="darkgreen" if correct else "darkred",
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
