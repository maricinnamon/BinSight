# Bounding-box annotation audit

Every target came from `annotation["bbox"]`. Segmentation polygons were stripped at load time and never consulted.

## Source

| | |
|---|---:|
| Official image records | 1500 |
| Official annotations | 4784 |
| Images resolved on disk | 1300 |
| **Images used** | **1082** |
| **Valid mapped boxes** | **2671** |
| Excluded annotations | 2113 |
| Boxes clipped to frame | 1 |

## Why annotations were excluded

| Reason | Count |
|---|---:|
| `category_not_mapped` | 1640 |
| `image_file_unavailable` | 473 |

**0 annotations were excluded for an invalid or missing `bbox`.** None were reconstructed from a segmentation polygon — a mask-derived box is a different annotation, and substituting one would make the dataset a silent mixture of two sources.

`category_not_mapped` is the deliberate taxonomy decision carried over from the earlier experiment: TACO's 60 categories collapse to 8 BinSight classes with no catch-all, and the remainder becomes background.

## Object size distribution

Historical BinSight buckets, by fraction of frame area — unchanged since Run 001.

| Bucket | Range | Train | Val | Test | Total | Share |
|---|---|---:|---:|---:|---:|---:|
| tiny | 0.0–0.01 | 1099 | 263 | 253 | 1615 | 60.5% |
| small | 0.01–0.05 | 454 | 80 | 80 | 614 | 23.0% |
| medium | 0.05–0.2 | 216 | 47 | 55 | 318 | 11.9% |
| large | 0.2–1.01 | 92 | 15 | 17 | 124 | 4.6% |

**83.5% of boxes are tiny or small.** That is the defining difficulty of this dataset and the reason earlier BinSight runs reported low recall: TACO is street litter photographed from standing height.

## Size composition per class

| Class | tiny | small | medium | large | tiny+small |
|---|---:|---:|---:|---:|---:|
| `bag_wrapper` | 373 | 227 | 107 | 55 | 79% |
| `bottle` | 157 | 115 | 52 | 25 | 78% |
| `bottle_cap` | 191 | 38 | 9 | 0 | 96% |
| `can` | 112 | 72 | 43 | 8 | 78% |
| `carton` | 67 | 63 | 66 | 25 | 59% |
| `cigarette` | 529 | 7 | 1 | 0 | 100% |
| `cup` | 89 | 56 | 22 | 9 | 82% |
| `straw` | 97 | 36 | 18 | 2 | 87% |

## Split leakage

**PASS** — checked three independent ways:

- by source image id: 0 violations
- by relative source path: 0 violations
- by image content SHA-256: 0 violations

1082 unique image hashes across 1082 images (0 duplicate-content groups).

