# Label validation — **PASS**

2671 boxes across 1082 label files, 0 problems.

| Split | Images | Labels | Boxes |
|---|---:|---:|---:|
| train | 748 | 748 | 1861 |
| val | 168 | 168 | 405 |
| test | 166 | 166 | 405 |

## Field-count histogram

A YOLO **Detect** label has exactly five values. A YOLO-seg polygon label has a variable number, so anything other than 5 below would mean segmentation data reached the dataset.

| Values per line | Lines |
|---:|---:|
| 5 | 2671 |

## Round trip: COCO → YOLO → COCO

Every normalised box was converted back to pixels and compared with the original `annotation["bbox"]`.

- checked: **2671**
- skipped (legitimately clipped to the frame): 1
- tolerance: 1.5 px
- worst error: **1.000912 px**
- failures: **0**

## Checks applied

- exactly 5 fields per label line
- class_id within 0..7
- no NaN/Inf
- width > 0
- height > 0
- centre within [0,1]
- box inside frame
- image/label pairing
- no orphan labels
- no empty label files
- no dangling symlinks
- no polygon/segmentation-style lines
- no mask artefacts on disk
- COCO -> YOLO -> COCO round trip within tolerance

No problems found. Numbers being well-formed is necessary but not sufficient — see `bbox_visual_validation.jpg`, which renders these labels back onto the images.

