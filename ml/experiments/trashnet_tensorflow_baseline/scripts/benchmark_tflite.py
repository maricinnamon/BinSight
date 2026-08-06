#!/usr/bin/env python
"""Benchmarks each exported TFLite model on this Mac.

**This is a relative development benchmark, not iPhone latency.** A Mac's CPU,
memory bandwidth and thermal envelope differ from a phone's; use it to compare
candidates against each other, never to predict on-device speed.
"""
from __future__ import annotations

import argparse
import platform
import sys

import _bootstrap  # noqa: F401

from binsight_training import export as export_module
from binsight_training.config import load_config
from binsight_training.utils import get_logger, read_json, write_json

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()

    results: dict[str, dict] = {}
    for name in ("float32", "float16", "int8"):
        path = config.tflite_dir / f"binsight_trashnet_{name}.tflite"
        if not path.exists():
            continue
        LOGGER.info("Benchmarking %s (%d runs)...", name, args.runs)
        stats = export_module.benchmark(
            path.read_bytes(), config.data.image_size, runs=args.runs)
        stats["size_bytes"] = path.stat().st_size
        stats["size_mb"] = round(path.stat().st_size / 1e6, 3)
        results[name] = stats
        LOGGER.info("  mean %.2f ms | median %.2f ms | p95 %.2f ms | %.2f MB",
                    stats["mean_ms"], stats["median_ms"], stats["p95_ms"], stats["size_mb"])

    if not results:
        LOGGER.error("No TFLite models found — run scripts/export_tflite.py first.")
        return 1

    payload = {
        "note": (
            "Development benchmark on the training Mac (CPU, XNNPACK). NOT iPhone "
            "latency — no on-device measurement has been taken."
        ),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "batch_size": 1,
        "runs": args.runs,
        "results": results,
    }
    write_json(config.reports_dir / "tflite_benchmark.json", payload)

    verification_path = config.reports_dir / "tflite_verification.json"
    verification = read_json(verification_path) if verification_path.exists() else {}
    candidates = verification.get("candidates", {})
    selected = verification.get("selected")

    lines = [
        "# TFLite candidate comparison",
        "",
        "> Latency was measured on the **training Mac**, CPU only. It is a relative",
        "> comparison between candidates and says nothing about iPhone performance.",
        "",
        "| model | size | accuracy | macro F1 | top-1 agreement vs Keras | mean ms | p95 ms | input/output |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, stats in results.items():
        info = candidates.get(name, {})
        marker = " **(selected)**" if name == selected else ""
        lines.append(
            f"| {name}{marker} | {stats['size_mb']:.2f} MB "
            f"| {info.get('accuracy', float('nan')):.4f} "
            f"| {info.get('macro_f1', float('nan')):.4f} "
            f"| {info.get('top1_agreement_vs_keras', float('nan')):.4f} "
            f"| {stats['mean_ms']:.2f} | {stats['p95_ms']:.2f} "
            f"| {info.get('input_dtype', '?')}/{info.get('output_dtype', '?')} |"
        )
    if "keras" in verification:
        keras = verification["keras"]
        lines += [
            "",
            f"Keras reference: accuracy {keras['accuracy']:.4f}, macro F1 {keras['macro_f1']:.4f}.",
        ]
    lines += [
        "",
        "## Selection rule",
        "",
        "A quantised model is rejected if it fails contract verification, or if it",
        f"loses more than {config.export.max_quantisation_degradation_pp} percentage points of",
        "accuracy **or** macro F1 against float32. Macro F1 is checked separately because",
        "a quantised model can hold overall accuracy while quietly dropping the smallest",
        "class. Among survivors, a float input/output contract wins (no dequantisation in",
        "Swift), then the smallest file.",
        "",
    ]
    (config.reports_dir / "tflite_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s", config.reports_dir / "tflite_comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
