#!/usr/bin/env python
"""Records the YOLO environment and verifies MPS with real computation.

`torch.backends.mps.is_available()` returning True is not evidence MPS works:
it reports capability, not correctness. This runs an actual matmul and a Conv2d
forward/backward pass and checks the gradients are finite.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"python": sys.version.split()[0], "architecture": platform.machine(),
                    "platform": platform.platform()}
    checks: dict = {}

    try:
        import torch, torchvision, ultralytics
        report.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                       "ultralytics": ultralytics.__version__,
                       "mps_built": torch.backends.mps.is_built(),
                       "mps_available": torch.backends.mps.is_available()})
        checks["imports"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        (REPORTS / "environment_report.md").write_text(
            f"# YOLO environment\n\nImport failed: {report['error']}\n")
        print("ERROR:", report["error"])
        return 1

    if report["mps_available"]:
        device = torch.device("mps")
        a = torch.randn((2048, 2048), device=device)
        b = torch.randn((2048, 2048), device=device)
        start = time.time()
        c = a @ b
        torch.mps.synchronize()
        report["mps_matmul"] = {
            "shape": list(c.shape), "mean": round(c.mean().item(), 6),
            "all_finite": bool(torch.isfinite(c).all().item()),
            "ms": round((time.time() - start) * 1000, 1)}
        checks["mps_matmul"] = report["mps_matmul"]["all_finite"]

        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16),
            torch.nn.SiLU(), torch.nn.Conv2d(16, 8, 3, padding=1)).to(device)
        x = torch.randn(4, 3, 128, 128, device=device, requires_grad=True)
        y = net(x)
        loss = y.pow(2).mean()
        loss.backward()
        torch.mps.synchronize()
        report["mps_conv_backward"] = {
            "output_shape": list(y.shape), "loss": round(loss.item(), 6),
            "grad_finite": bool(torch.isfinite(x.grad).all().item()),
            "grad_norm": round(x.grad.norm().item(), 6)}
        checks["mps_conv_backward"] = report["mps_conv_backward"]["grad_finite"]
    else:
        checks["mps_matmul"] = False
        checks["mps_conv_backward"] = False
        report["mps_note"] = "MPS unavailable — dataset preparation continues on CPU"

    try:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        params = sum(p.numel() for p in model.model.parameters())
        report["yolo26n"] = {"loaded": True, "task": model.task,
                             "parameters": params, "pretrained_classes": len(model.names)}
        checks["yolo26n_load"] = True
    except Exception as exc:  # noqa: BLE001
        report["yolo26n"] = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
        checks["yolo26n_load"] = False

    report["checks"] = checks
    (REPORTS / "environment_report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# YOLO environment report", "",
        "Environment: **BinSight_YOLO** (separate from the TensorFlow baseline's",
        "`binsight-trashnet-metal`; no TensorFlow is installed here).", "",
        "| | |", "|---|---|",
        f"| Python | {report['python']} |",
        f"| Architecture | {report['architecture']} |",
        f"| PyTorch | {report.get('torch')} |",
        f"| torchvision | {report.get('torchvision')} |",
        f"| Ultralytics | {report.get('ultralytics')} |",
        f"| MPS built | {report.get('mps_built')} |",
        f"| MPS available | {report.get('mps_available')} |",
        "", "## MPS verification", "",
        "`is_available()` reports capability, not correctness, so both of these run",
        "real computation.", "",
    ]
    if "mps_matmul" in report:
        m = report["mps_matmul"]
        lines += [f"**Matmul** 2048×2048: shape {tuple(m['shape'])}, mean {m['mean']}, "
                  f"all finite {m['all_finite']}, {m['ms']} ms.", ""]
    if "mps_conv_backward" in report:
        c = report["mps_conv_backward"]
        lines += [f"**Conv2d forward/backward**: output {tuple(c['output_shape'])}, "
                  f"loss {c['loss']}, gradients finite {c['grad_finite']}, "
                  f"grad norm {c['grad_norm']}.", ""]
    y = report.get("yolo26n", {})
    lines += ["## YOLO26n", ""]
    if y.get("loaded"):
        lines += [f"Loaded successfully: task `{y['task']}`, **{y['parameters']:,} parameters**, "
                  f"{y['pretrained_classes']} pretrained COCO classes.", "",
                  "Weights downloaded for a load smoke test only. No training was run.", ""]
    else:
        lines += [f"Load FAILED: {y.get('error')}", ""]

    (REPORTS / "environment_report.md").write_text("\n".join(lines))
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.get(k, False) for k in ("imports", "yolo26n_load")) else 1


if __name__ == "__main__":
    sys.exit(main())
