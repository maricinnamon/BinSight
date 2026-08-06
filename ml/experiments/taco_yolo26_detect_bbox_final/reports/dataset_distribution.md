# Dataset distribution

Official reviewed TACO, converted with `annotation["bbox"]` only. Splits are reused unchanged from the earlier verified experiment so the numbers stay comparable.

| Split | Images | Boxes |
|---|---:|---:|
| train | 748 | 1861 |
| val | 168 | 405 |
| test | 166 | 405 |
| **total** | **1082** | **2671** |

## Boxes per class

| ID | Class | Train | Val | Test | Total | Share |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `bag_wrapper` | 532 | 115 | 115 | 762 | 28.5% |
| 1 | `bottle` | 241 | 56 | 52 | 349 | 13.1% |
| 2 | `bottle_cap` | 166 | 36 | 36 | 238 | 8.9% |
| 3 | `can` | 164 | 35 | 36 | 235 | 8.8% |
| 4 | `carton` | 155 | 33 | 33 | 221 | 8.3% |
| 5 | `cigarette` | 374 | 80 | 83 | 537 | 20.1% |
| 6 | `cup` | 123 | 26 | 27 | 176 | 6.6% |
| 7 | `straw` | 106 | 24 | 23 | 153 | 5.7% |
| | **total** | **1861** | **405** | **405** | **2671** | |

Imbalance: **4.98:1** (`bag_wrapper` vs `straw`). No reweighting or resampling is applied — this run measures the dataset as it is.

