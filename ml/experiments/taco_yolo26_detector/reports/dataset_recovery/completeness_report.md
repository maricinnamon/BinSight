# Dataset completeness after recovery

**Status: maximally recovered official subset**

| | |
|---|---:|
| Official image records | 1500 |
| Official annotations | 4784 |
| **Resolved and validated** | **1300** (86.67%) |
| Missing | 200 |
| Unreadable | 0 |
| Dimension mismatches | 0 |
| Annotations on available images | 4040 |
| Annotations lost to missing images | 744 |
| Duplicate content hashes | 0 |

## By source

| Source | Images |
|---|---:|
| `zenodo_archive` | 715 |
| `flickr_partial` | 585 |

Every file was opened and fully decoded, and its **EXIF-corrected** size compared against the COCO record — the raw JPEG size disagrees for about a quarter of TACO, which is what misplaced a quarter of Run 001's boxes.

## Why this is not the complete dataset

200 image records could not be resolved from either the archived Zenodo release or the Flickr download. The archive predates the current annotation revision (it holds 715 images against the annotations' 1500), and the remaining Flickr URLs are rate-limited or gone.

It is therefore labelled a **maximally recovered official subset**, not the complete TACO dataset.

