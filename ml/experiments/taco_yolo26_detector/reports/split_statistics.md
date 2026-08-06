# Split statistics

Deterministic, seed **42**, split at **image** level so no image's
annotations straddle a boundary. Target 70/15/15.

| Split | Images | Annotations |
|---|---:|---:|
| train | 368 | 878 |
| val | 84 | 201 |
| test | 57 | 211 |
| **total** | **509** | **1290** |

## Per class

| ID | Class | Total | Train | Val | Test | Train % | Val % | Test % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 430 | 299 | 64 | 67 | 69.5% | 14.9% | 15.6% |
| 1 | bottle | 152 | 105 | 23 | 24 | 69.1% | 15.1% | 15.8% |
| 2 | bottle_cap | 130 | 87 | 24 | 19 | 66.9% | 18.5% | 14.6% |
| 3 | can | 129 | 88 | 21 | 20 | 68.2% | 16.3% | 15.5% |
| 4 | carton | 85 | 58 | 14 | 13 | 68.2% | 16.5% | 15.3% |
| 5 | cigarette | 186 | 128 | 30 | 28 | 68.8% | 16.1% | 15.1% |
| 6 | cup | 83 | 58 | 13 | 12 | 69.9% | 15.7% | 14.5% |
| 7 | straw | 95 | 55 | 12 | 28 | 57.9% | 12.6% | 29.5% |

## Stratification quality

Images carry multiple classes, so exact stratification is impossible without
splitting an image — which would leak. The assignment is greedy over class
deficits, rarest class first.

Largest deviation from the 70% train target: **`straw` at 57.9%**.

Leakage check: no image appears in more than one split (verified by
`create_splits.py`, which exits non-zero otherwise).

