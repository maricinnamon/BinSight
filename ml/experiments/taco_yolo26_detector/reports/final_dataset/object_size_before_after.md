# Object size distribution — train, before and after crop augmentation

Buckets are the historical definitions used by Run 001 and Run 002: tiny < 1% of frame area, small < 5%, medium < 20%, large above.

| Bucket | Before | share | After | share | Δ share |
|---|---:|---:|---:|---:|---:|
| tiny | 1099 | 59.1% | 1236 | 36.4% | -22.7 pp |
| small | 454 | 24.4% | 647 | 19.0% | -5.3 pp |
| medium | 216 | 11.6% | 1036 | 30.5% | +18.9 pp |
| large | 92 | 4.9% | 478 | 14.1% | +9.1 pp |
| **total** | **1861** | | **3397** | | |

## What this means

Cropping added **1536** training annotations, and — the point of the exercise — moved the small-object share down from 83.4% to 55.4%: the same objects now also appear at a scale the network can resolve.

**Val and test are untouched**, so their size distribution still reflects real photographs and the final metrics still describe real contextual detection rather than the augmentation.

