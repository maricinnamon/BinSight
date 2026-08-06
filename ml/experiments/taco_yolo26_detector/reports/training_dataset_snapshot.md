# Training dataset snapshot — Run 001

The immutable description of what Run 001 was trained on.

| | |
|---|---|
| TACO commit | `29de1a9ba05a647b83a90f18d7772e20bb23d846` |
| Annotations file | official `data/annotations.json` |
| Split seed | 42 |
| `class_mapping.yaml` SHA-256 | `b9bd9a952bb763bf…` |
| `dataset.yaml` SHA-256 | `46d2b27771cf27eb…` |

## Splits

| Split | Images | Annotations |
|---|---:|---:|
| train | 368 | 878 |
| val | 84 | 201 |
| test | 57 | 211 |
| **total** | **509** | **1290** |

## Per class

| ID | Class | Train | Val | Test | Total |
|---:|---|---:|---:|---:|---:|
| 0 | bag_wrapper | 299 | 64 | 67 | 430 |
| 1 | bottle | 105 | 23 | 24 | 152 |
| 2 | bottle_cap | 87 | 24 | 19 | 130 |
| 3 | can | 88 | 21 | 20 | 129 |
| 4 | carton | 58 | 14 | 13 | 85 |
| 5 | cigarette | 128 | 30 | 28 | 186 |
| 6 | cup | 58 | 13 | 12 | 83 |
| 7 | straw | 55 | 12 | 28 | 95 |

## Source images

- Referenced by annotations: **1500**
- Present on disk: **602**
- Missing: **898** (`{'HTTP 429': 898}`)

> Flickr rate-limiting (HTTP 429) left part of the dataset undownloaded.
> Run 001 therefore trains on a subset. This is recorded rather than
> worked around, because it materially limits what the run can show.


## Excluded TACO categories (29)

| Category | Reason |
|---|---|
| Aluminium blister pack | aluminium-backed blister; mixed with polymer but the label names aluminium |
| Aluminium foil | aluminium |
| Battery | hazardous waste requiring separate handling; 2 annotations. Misclassifying a battery is worse than n |
| Broken glass | explicitly glass |
| Carded blister pack | board + polymer laminate, 1 annotation |
| Disposable food container | polymer container |
| Foam food container | expanded polystyrene |
| Food waste | organic; outside BinSight's six-material scope and only 8 annotations |
| Glass jar | glass jar |
| Magazine paper | printed paper |
| Metal lid | explicitly metal |
| Normal paper | paper |
| Other plastic | explicitly plastic, object unspecified |
| Other plastic container | explicitly plastic |
| Plastic glooves | polymer gloves (TACO's spelling) |
| Plastic lid | polymer lid |
| Plastic utensils | polymer cutlery |
| Pop tab | aluminium tab |
| Rope & strings | natural or synthetic fibre, unresolvable from the label |
| Scrap metal | explicitly metal |
| Shoe | multi-material; not packaging waste |
| Six pack rings | polymer rings |
| Spread tub | polymer tub |
| Squeezable tube | usually polymer, occasionally aluminium-lined |
| Styrofoam piece | expanded polystyrene is a plastic |
| Tissues | paper tissue |
| Tupperware | reusable polymer container |
| Unlabeled litter | material unknown by definition; 517 annotations of 'something' would teach the detector nothing abou |
| Wrapping paper | paper |
