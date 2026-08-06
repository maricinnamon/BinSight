# Object size distribution, recovered dataset

Areas are measured at **640 px**, the training size — unchanged from Run 001. Buckets follow the COCO convention: small < 32² px, medium < 96² px, large above.

| ID | Class | Ann. | Small | Medium | Large | Small % | p10 | Median | p90 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `bag_wrapper` | 762 | 168 | 330 | 264 | 22.0% | 459.2 | 4303.8 | 61416.7 |
| 1 | `bottle` | 349 | 56 | 163 | 130 | 16.0% | 531.2 | 5331.4 | 58954.1 |
| 2 | `bottle_cap` | 238 | 143 | 78 | 17 | 60.1% | 47.6 | 507.7 | 6651.9 |
| 3 | `can` | 235 | 33 | 115 | 87 | 14.0% | 829.5 | 4596.3 | 40546.8 |
| 4 | `carton` | 221 | 26 | 72 | 123 | 11.8% | 771.1 | 11492.6 | 86532.8 |
| 5 | `cigarette` | 537 | 518 | 16 | 3 | 96.5% | 20.3 | 77.9 | 317.4 |
| 6 | `cup` | 176 | 23 | 97 | 56 | 13.1% | 618.3 | 4002.3 | 45972.4 |
| 7 | `straw` | 153 | 58 | 59 | 36 | 37.9% | 111.9 | 2395.9 | 23584.2 |

Overall **38.4%** of objects fall in the small bucket. The extreme is **`cigarette` at 96.5%** small, median area 77.9 px² — a few pixels across at 640. That is a resolution problem, not a data-volume problem, and no amount of extra images removes it at this input size.

