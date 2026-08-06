# Training Run 001 — summary

| | |
|---|---|
| Model | YOLO26n (pretrained `yolo26n.pt`) |
| Device | **mps** (Apple M1, PyTorch MPS) |
| Image size | 640 |  
| Batch / workers | 8 / 4 |
| Seed / deterministic | 42 / True |
| Epochs configured | 100 |
| **Epochs completed** | **64** (early stopping, patience 20) |
| **Best epoch** | **44** |
| Training duration | 1.59 hours |
| Class weighting | none — see below |

## Metrics

| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.4483 | 0.2619 | 0.3306 | 0.2861 | 0.2357 | 0.2120 |
| **test** | **0.2696** | **0.1853** | **0.2197** | **0.2007** | **0.1696** | **0.1586** |

## Class weighting

Not used. Imbalance after mapping is 5.18x, which is moderate, and `cls_pw` does
not exist in Ultralytics 8.4.9 — this was checked against `DEFAULT_CFG_DICT`
rather than assumed. Default loss gains were left alone.

## Augmentation

Ultralytics detection defaults, unchanged:

| Parameter | Value |
|---|---:|
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| degrees | 0.0 |
| translate | 0.1 |
| scale | 0.5 |
| shear | 0.0 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| mosaic | 1.0 |
| mixup | 0.0 |
| copy_paste | 0.0 |
| erasing | 0.4 |
| auto_augment | randaugment |

## Error counts (test, IoU >= 0.50, conf >= 0.25)

| | |
|---|---:|
| True positives | 24 |
| False positives | 18 |
| False negatives | 171 |
| Class confusions | 16 |
| Poor localisation | 2 |

Recall is the binding constraint: the model misses far more than it gets wrong.

