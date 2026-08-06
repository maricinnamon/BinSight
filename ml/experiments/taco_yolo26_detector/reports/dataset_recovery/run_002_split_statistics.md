# Run 002 split statistics

Deterministic, seed **42**, target 70/15/15, split at **image** level — the same algorithm as Run 001, so differences between the runs come from the data rather than the rule.

| Split | Images | Annotations |
|---|---:|---:|
| train | 748 | 1861 |
| val | 168 | 405 |
| test | 166 | 405 |
| **total** | **1082** | **2671** |

## Per class

| ID | Class | Total | Train | Val | Test | Train % | Val % | Test % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 762 | 532 | 115 | 115 | 69.8% | 15.1% | 15.1% |
| 1 | bottle | 349 | 241 | 56 | 52 | 69.1% | 16.0% | 14.9% |
| 2 | bottle_cap | 238 | 166 | 36 | 36 | 69.7% | 15.1% | 15.1% |
| 3 | can | 235 | 164 | 35 | 36 | 69.8% | 14.9% | 15.3% |
| 4 | carton | 221 | 155 | 33 | 33 | 70.1% | 14.9% | 14.9% |
| 5 | cigarette | 537 | 374 | 80 | 83 | 69.6% | 14.9% | 15.5% |
| 6 | cup | 176 | 123 | 26 | 27 | 69.9% | 14.8% | 15.3% |
| 7 | straw | 153 | 106 | 24 | 23 | 69.3% | 15.7% | 15.0% |

## Integrity

- Content-hash groups: **1082** over 1082 images (0 groups hold more than one image record).
- No image appears in two splits.
- No content hash appears in two splits — byte-identical pictures travel together, so a duplicate cannot straddle the train/val boundary.
- Largest deviation from the 70% train target: **`bottle` at 69.1%**.

`create_splits_run_002.py` exits non-zero if any of the first three fail.

