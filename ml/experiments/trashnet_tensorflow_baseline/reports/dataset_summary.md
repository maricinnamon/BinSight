# TrashNet dataset summary

- **Source:** <https://github.com/garythung/trashnet> (`dataset-resized.zip`)
- **Licence:** MIT; the repository asks to be cited (see `THIRD_PARTY_NOTICES.md`)
- **Readable images:** 2527
- **Corrupt / unreadable:** 0
- **Duplicate groups:** 3 (3 redundant files)
- **Imbalance ratio:** 4.34 (paper vs trash)

## Class distribution and splits

| class | total | train | validation | test |
|---|---:|---:|---:|---:|
| cardboard | 403 | 282 | 60 | 61 |
| glass | 501 | 351 | 75 | 75 |
| metal | 410 | 287 | 61 | 62 |
| paper | 594 | 416 | 89 | 89 |
| plastic | 482 | 338 | 72 | 72 |
| trash | 137 | 96 | 21 | 20 |
| **total** | **2527** | **1770** | **378** | **379** |

## Image geometry

- width: {'min': 512.0, 'max': 512.0, 'mean': 512.0, 'median': 512.0}
- height: {'min': 384.0, 'max': 384.0, 'mean': 384.0, 'median': 384.0}
- aspect ratio: {'min': 1.3333, 'max': 1.3333, 'mean': 1.3333, 'median': 1.3333}

## Split method

Stratified 70/15/15, `seed=42`, split over **content hash groups** rather
than individual files so byte-identical duplicates cannot straddle a
boundary and inflate the test score. Verified disjoint by path and by hash.
