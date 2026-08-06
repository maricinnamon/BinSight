# Model ablation table

Every row is a real trained run, evaluated on its own untouched test split with `best.pt`. Latency is Mac/MPS at batch 1 — **not** iPhone latency.

| Run | Dataset | imgsz | P | R | mAP50 | mAP50-95 | Tiny R | Small R | MPS p50 | Winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolo26n_run_001 | partial TACO (509 img) | 640 | 0.2696 | 0.1853 | 0.2007 | 0.1586 | 0.0511 | 0.1739 | 48.488 ms |  |
| yolo26n_run_002_full_taco_640 | recovered TACO (1082 img) | 640 | 0.3158 | 0.2470 | 0.2105 | 0.1453 | 0.0949 | 0.2125 | — |  |

**Rows are not scored on the same test set.** Run 001's test split came from a 509-image pool; later runs from the recovered 1082-image pool, whose small-object share is 9.3 pp higher. Compare across rows with that in mind: an equal number on a harder split is an improvement.

