# Final dataset validation

**Result: PASS** — 0 problems.

| Split | Images | Labels | Boxes |
|---|---:|---:|---:|
| train | 1804 | 1804 | 3397 |
| val | 168 | 168 | 405 |
| test | 166 | 166 | 405 |

## Leakage

- Images shared between splits: **0**
- Identical image content across splits: **0**
- Crops whose source is not a train image: **0**
- Crops derived from a val/test image: **0**
- Generated crops present in val/test: **0**

Crops inherit their source image's split, and only train images were cropped, so leakage is structurally impossible — but it is checked rather than assumed, because the cost of being wrong is every final metric.

## Crop geometry

- Crops: **1056**
- Target area gain: median **29.1x**, mean 78.6x, min 1.8x
- Targets still tiny/small after cropping: **107** of 1056

| Bucket move | Crops |
|---|---:|
| tiny->medium | 395 |
| small->medium | 233 |
| tiny->large | 170 |
| small->large | 151 |
| tiny->small | 96 |
| tiny->tiny | 7 |
| small->small | 4 |

## Checks applied

- class ids in range
- finite coordinates
- no zero-area boxes
- centres in [0,1]
- boxes inside image
- image/label pairing
- no orphan labels
- no empty label files
- no dangling symlinks
- no image in two splits
- no identical image content across splits
- crops derive only from train images
- no crop in val/test
- every crop enlarges its target

No problems found.

Programmatic validation proves the numbers are well-formed. It does not prove the boxes are in the right *place* — see the rendered crops in this directory, which were inspected.

