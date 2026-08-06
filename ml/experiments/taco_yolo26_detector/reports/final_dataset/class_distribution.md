# Final training class distribution

Val and test are unchanged and are shown for reference only — no crops, no oversampling, no synthetic images.

| Class | Train before | Train after | Δ | × | Val | Test |
|---|---:|---:|---:|---:|---:|---:|
| `bag_wrapper` | 532 | 894 | +362 | 1.68× | 115 | 115 |
| `bottle` | 241 | 465 | +224 | 1.93× | 56 | 52 |
| `bottle_cap` | 166 | 372 | +206 | 2.24× | 36 | 36 |
| `can` | 164 | 318 | +154 | 1.94× | 35 | 36 |
| `carton` | 155 | 278 | +123 | 1.79× | 33 | 33 |
| `cigarette` | 374 | 630 | +256 | 1.68× | 80 | 83 |
| `cup` | 123 | 243 | +120 | 1.98× | 26 | 27 |
| `straw` | 106 | 197 | +91 | 1.86× | 24 | 23 |
| **total** | **1861** | **3397** | **+1536** | **1.83×** | **405** | **405** |

## Imbalance

Before: 5.02:1 (`bag_wrapper` vs `straw`)
After: **4.54:1** (`bag_wrapper` vs `straw`)

The ratio remains above the 4:1 target. It was not forced down by duplicating the rare classes further: the caps (2 crops per image per class, 4 per image) exist precisely to avoid manufacturing dozens of near-identical samples, and multi-object scenes carry several classes at once so the counts cannot be tuned independently without corrupting the semantics of those images. Cropping narrowed the gap; it could not close it without doing harm.

