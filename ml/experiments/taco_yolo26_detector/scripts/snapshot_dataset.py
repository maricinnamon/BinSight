#!/usr/bin/env python
"""Freezes an immutable description of the dataset a training run used."""
from __future__ import annotations
import collections, csv, hashlib, json, pathlib, subprocess, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS, CONFIGS = ROOT / "reports", ROOT / "configs"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    classes = (CONFIGS / "classes.txt").read_text().split()
    mapping = yaml.safe_load((CONFIGS / "class_mapping.yaml").read_text())
    excluded = {n: e.get("reason", "") for n, e in mapping["mapping"].items()
                if not e.get("include")}

    per_split, imgs = {}, collections.defaultdict(lambda: collections.defaultdict(set))
    counts = collections.defaultdict(collections.Counter)
    for split in ("train", "val", "test"):
        label_dir = ROOT / "data" / "prepared" / "labels" / split
        boxes = 0
        for label in label_dir.glob("*.txt"):
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    counts[split][cid] += 1
                    imgs[split][cid].add(label.stem)
                    boxes += 1
        per_split[split] = {
            "images": len(list((ROOT / "data" / "prepared" / "images" / split).iterdir())),
            "annotations": boxes,
        }

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT / "data" / "source" / "TACO").stdout.strip()
    except Exception:
        sha = None

    download = {}
    dl_path = REPORTS / "download_report.json"
    if dl_path.exists():
        d = json.loads(dl_path.read_text())
        download = {k: d[k] for k in ("images_referenced", "present_on_disk",
                                      "missing_locally", "failed", "error_breakdown")}

    snapshot = {
        "run": "yolo26n_run_001",
        "target_classes": classes,
        "class_ids": {name: i for i, name in enumerate(classes)},
        "splits": per_split,
        "per_class_annotations": {
            split: {classes[c]: counts[split][c] for c in range(len(classes))}
            for split in ("train", "val", "test")},
        "per_class_images": {
            split: {classes[c]: len(imgs[split][c]) for c in range(len(classes))}
            for split in ("train", "val", "test")},
        "excluded_taco_categories": excluded,
        "excluded_category_count": len(excluded),
        "taco_commit_sha": sha,
        "annotation_file": "data/annotations.json (official)",
        "class_mapping_sha256": sha256(CONFIGS / "class_mapping.yaml"),
        "dataset_yaml_sha256": sha256(CONFIGS / "dataset.yaml"),
        "classes_txt_sha256": sha256(CONFIGS / "classes.txt"),
        "split_seed": 42,
        "source_images": download,
    }
    (REPORTS / "training_dataset_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    total_imgs = sum(v["images"] for v in per_split.values())
    total_anns = sum(v["annotations"] for v in per_split.values())
    md = ["# Training dataset snapshot — Run 001", "",
          "The immutable description of what Run 001 was trained on.", "",
          "| | |", "|---|---|",
          f"| TACO commit | `{sha}` |",
          f"| Annotations file | official `data/annotations.json` |",
          f"| Split seed | 42 |",
          f"| `class_mapping.yaml` SHA-256 | `{snapshot['class_mapping_sha256'][:16]}…` |",
          f"| `dataset.yaml` SHA-256 | `{snapshot['dataset_yaml_sha256'][:16]}…` |",
          "", "## Splits", "", "| Split | Images | Annotations |", "|---|---:|---:|"]
    for split in ("train", "val", "test"):
        md.append(f"| {split} | {per_split[split]['images']} | {per_split[split]['annotations']} |")
    md += [f"| **total** | **{total_imgs}** | **{total_anns}** |", "",
           "## Per class", "",
           "| ID | Class | Train | Val | Test | Total |", "|---:|---|---:|---:|---:|---:|"]
    for cid, name in enumerate(classes):
        t = sum(counts[s][cid] for s in ("train", "val", "test"))
        md.append(f"| {cid} | {name} | {counts['train'][cid]} | {counts['val'][cid]} "
                  f"| {counts['test'][cid]} | {t} |")
    if download:
        md += ["", "## Source images", "",
               f"- Referenced by annotations: **{download.get('images_referenced')}**",
               f"- Present on disk: **{download.get('present_on_disk')}**",
               f"- Missing: **{download.get('missing_locally')}** "
               f"(`{download.get('error_breakdown')}`)", "",
               "> Flickr rate-limiting (HTTP 429) left part of the dataset undownloaded.",
               "> Run 001 therefore trains on a subset. This is recorded rather than", 
               "> worked around, because it materially limits what the run can show.", ""]
    md += ["", f"## Excluded TACO categories ({len(excluded)})", "",
           "| Category | Reason |", "|---|---|"]
    for name, reason in sorted(excluded.items()):
        md.append(f"| {name} | {reason[:100]} |")
    (REPORTS / "training_dataset_snapshot.md").write_text("\n".join(md) + "\n")
    print(f"snapshot: images={total_imgs} annotations={total_anns} classes={len(classes)}")
    print(f"  taco commit {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
