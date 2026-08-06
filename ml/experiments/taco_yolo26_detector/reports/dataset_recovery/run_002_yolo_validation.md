# YOLO label validation

Dataset: `data/prepared_run_002`

**Result: PASS** — 2671 boxes across 1082 label files, 0 problems.

| Split | Images | Label files | Boxes |
|---|---:|---:|---:|
| train | 748 | 748 | 1861 |
| val | 168 | 168 | 405 |
| test | 166 | 166 | 405 |

## Boxes per class

| ID | Class | Boxes |
|---:|---|---:|
| 0 | bag_wrapper | 762 |
| 1 | bottle | 349 |
| 2 | bottle_cap | 238 |
| 3 | can | 235 |
| 4 | carton | 221 |
| 5 | cigarette | 537 |
| 6 | cup | 176 |
| 7 | straw | 153 |

## Checks applied

- 5 values per row
- class id within range
- finite coordinates
- x_center/y_center in [0,1]
- 0 < width,height <= 1
- reconstructed box inside image
- no orphan labels
- no image without a label
- no dangling image symlink
- no image shared between splits

No problems found.


Programmatic validation proves the numbers are well-formed. It does not prove
the boxes are in the right *place* — see `reports/samples/`, which renders
these YOLO labels back onto the images.
