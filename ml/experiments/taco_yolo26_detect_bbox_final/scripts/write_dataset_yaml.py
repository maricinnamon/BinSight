#!/usr/bin/env python
"""Materialises configs/dataset.yaml with an absolute `path` for this machine.

Ultralytics resolves a relative `path:` against its own DATASETS_DIR setting, not
against the YAML's location — which is why the committed template carries a
placeholder and this script fills it in. Keeping the machine-specific value out
of Git also keeps the repository portable.
"""
from __future__ import annotations
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
template = (HERE / "configs" / "dataset.yaml").read_text()
resolved = template.replace("path: PLACEHOLDER", f"path: {HERE / 'data'}")
out = HERE / "configs" / "dataset.resolved.yaml"
out.write_text(resolved)
print(f"wrote {out.relative_to(HERE)} -> {HERE / 'data'}")
sys.exit(0)
