# YOLO label validation

**Result: PASS** — 1290 boxes across 509 label files, 0 problems.

| Split | Images | Label files | Boxes |
|---|---:|---:|---:|
| train | 368 | 368 | 878 |
| val | 84 | 84 | 201 |
| test | 57 | 57 | 211 |

## Boxes per class

| ID | Class | Boxes |
|---:|---|---:|
| 0 | bag_wrapper | 430 |
| 1 | bottle | 152 |
| 2 | bottle_cap | 130 |
| 3 | can | 129 |
| 4 | carton | 85 |
| 5 | cigarette | 186 |
| 6 | cup | 83 |
| 7 | straw | 95 |

## Checks applied

- 5 values per row
- class id within range
- finite coordinates
- x_center/y_center in [0,1]
- 0 < width,height <= 1
- reconstructed box inside image
- no orphan labels
- no image without a label

No problems found.


Programmatic validation proves the numbers are well-formed. It does not prove
the boxes are in the right *place* — see `reports/samples/`, which renders
these YOLO labels back onto the images.
