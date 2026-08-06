# Mapped class statistics

After mapping and splitting: **1290 annotations** across 8 classes.

| ID | Class | Anns | Images | Train | Val | Test | % | Median rel. bbox |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 430 | 221 | 299 | 64 | 67 | 33.33% | 1.3190% |
| 1 | bottle | 152 | 126 | 105 | 23 | 24 | 11.78% | 3.4434% |
| 2 | bottle_cap | 130 | 108 | 87 | 24 | 19 | 10.08% | 0.5362% |
| 3 | can | 129 | 83 | 88 | 21 | 20 | 10.0% | 2.9951% |
| 4 | carton | 85 | 62 | 58 | 14 | 13 | 6.59% | 2.8058% |
| 5 | cigarette | 186 | 65 | 128 | 30 | 28 | 14.42% | 0.0338% |
| 6 | cup | 83 | 68 | 58 | 13 | 12 | 6.43% | 2.1124% |
| 7 | straw | 95 | 55 | 55 | 12 | 28 | 7.36% | 0.8110% |

## Imbalance

Largest/smallest = **5.18×** (430 vs 83).

**Flagged as too thin for reliable evaluation** (fewer than 15 validation instances): `carton`, `cup`, `straw`.

Not addressed here — this step only measures it. Options for the training step include class-balanced sampling, loss weighting, or merging the thinnest classes.
