# `data/partial_run_001/` — the dataset Run 001 actually trained on

Run 001 trained on 509 images: everything that survived a per-image Flickr
download in which 898 of 1500 URLs returned HTTP 429. This directory names that
dataset so it is not confused with the recovered one.

The three entries here are **symlinks**, not copies:

| Entry | Points at | What it is |
|---|---|---|
| `raw/` | `../raw/` | the images Flickr actually served |
| `prepared/` | `../prepared/` | Run 001's YOLO images and labels |
| `splits/` | `../splits/` | Run 001's train/val/test manifests |

They are symlinks because the physical files must not move. Run 001's prepared
images are themselves symlinks into `data/raw/` by absolute path, and
`configs/dataset.yaml` — the file recorded in `runs/yolo26n_run_001/args.yaml` —
resolves to `data/prepared`. Relocating either directory would break the frozen
run's dataset out from under it, which the freeze policy in
`reports/run_001_frozen_manifest.json` forbids.

Run 002's dataset is separate and additive: `data/prepared_run_002/`, built from
`data/official_complete/` plus `data/raw/`, with its own splits under
`data/splits/run_002/`. Nothing in this directory is read by it.
