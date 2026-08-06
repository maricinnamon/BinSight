# What recovery added, per class

Counted from the two prepared label trees — `data/prepared/labels/` for Run 001 and `data/prepared_run_002/labels/` — so these are the boxes the model is actually handed, not the raw COCO totals.

| ID | Class | Ann. 001 | Ann. 002 | Δ | × | Img 001 | Img 002 | Δ img |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `bag_wrapper` | 430 | 762 | +332 | 1.77× | 221 | 454 | +233 |
| 1 | `bottle` | 152 | 349 | +197 | 2.3× | 126 | 261 | +135 |
| 2 | `bottle_cap` | 130 | 238 | +108 | 1.83× | 108 | 198 | +90 |
| 3 | `can` | 129 | 235 | +106 | 1.82× | 83 | 156 | +73 |
| 4 | `carton` | 85 | 221 | +136 | 2.6× | 62 | 160 | +98 |
| 5 | `cigarette` | 186 | 537 | +351 | 2.89× | 65 | 173 | +108 |
| 6 | `cup` | 83 | 176 | +93 | 2.12× | 68 | 147 | +79 |
| 7 | `straw` | 95 | 153 | +58 | 1.61× | 55 | 106 | +51 |
| | **TOTAL** | **1290** | **2671** | **+1381** | **2.07×** | | | |

## Reading it

- Largest absolute gain: **`cigarette`**, +351 annotations (186 → 537).
- Still the scarcest class: **`straw`** at 153 annotations — recovery narrowed the imbalance but did not remove it.

Class IDs are unchanged from Run 001, so Run 002's numbers are directly comparable to Run 001's per-class metrics.

