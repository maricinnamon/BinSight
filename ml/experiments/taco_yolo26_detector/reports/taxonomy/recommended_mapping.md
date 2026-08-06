# Recommended mapping for the first YOLO26 experiment

**Recommendation: the object-oriented mapping with a narrowed residual (candidate C), 8 classes.**

## The three candidates, measured

| | Classes | Annotations used | Largest class | Smallest | Imbalance | Catch-all share |
|---|---:|---:|---:|---:|---:|---:|
| A — material | 5 | 3440 (72%) | 2251 | 179 | 12.58x | none |
| B — object | 7 | 4783 (100%) | 2090 | 192 | 10.89x | 44% |
| **C — object, narrowed** | **8** | **3143 (66%)** | **872** | **161** | **5.42x** | **none** |

## Why not the material mapping

It is the one that preserves BinSight's existing UX, so it deserved a fair
hearing. It does not survive the numbers:

* it discards **1231 annotations** (26%) as materially ambiguous, plus 112 composite ones needing review;
* `plastic` still ends up with 2251 of 3440 included annotations (65%);
* `paper` (179) and `cardboard` (206) are thin before a 70/15/15 split, leaving roughly 27 and 31 validation instances;
* most importantly, it asks the detector to infer **material from pixels** —
  the same inference the TrashNet baseline already showed does not transfer to
  uncontrolled scenes. Repeating it on harder data is not a plan.

## Why not the wide object mapping

Candidate B keeps every annotation, but `other_litter` absorbs 2090 of them — 44% of the dataset. A class that large and that heterogeneous teaches the model to fire on almost
any litter-shaped region without saying anything useful. TACO's own official
maps show the same pattern (`map_4` 67% `Other`, `map_10` 36%, `map_17` 36%),
which is a property of the taxonomy rather than a flaw in any one mapping.

## The recommendation

Eight classes, every one of them a thing a person can point a phone at and
name, with no residual class at all:

| ID | Class | Annotations | Source categories |
|---:|---|---:|---|
| 0 | `bag_wrapper` | 872 | Plastic film, Other plastic wrapper, Single-use carrier bag, Crisp packet, Garbage bag, Paper bag, Polypropylene bag, Plastified paper bag |
| 1 | `bottle` | 438 | Clear plastic bottle, Other plastic bottle, Glass bottle |
| 2 | `bottle_cap` | 289 | Plastic bottle cap, Metal bottle cap |
| 3 | `can` | 273 | Drink can, Food Can, Aerosol |
| 4 | `carton` | 251 | Other carton, Corrugated carton, Drink carton, Meal carton, Egg carton, Pizza box, Toilet tube |
| 5 | `cigarette` | 667 | Cigarette |
| 6 | `cup` | 192 | Disposable plastic cup, Paper cup, Foam cup, Glass cup, Other plastic cup |
| 7 | `straw` | 161 | Plastic straw, Paper straw |

Total: **3143 annotations** across 8 classes, imbalance **5.42x** — less than half of either alternative.

## What this costs, stated plainly

**1640 annotations (34%) become unlabelled background.** That is the real price, and it has a specific
consequence: where an image contains both a mapped object and a dropped one,
the detector is trained to treat the dropped object as background. For
BinSight that is arguably correct — we would rather the app stay silent than
announce 'unidentified litter' — but it is a deliberate trade, not a free win.

The largest dropped categories:

| Category | Annotations | Why dropped |
|---|---:|---|
| Unlabeled litter | 517 | material unknown by definition; 517 annotations of 'something' would teach the detector no |
| Other plastic | 273 | not a distinct visual class at this scale |
| Broken glass | 138 | not a distinct visual class at this scale |
| Styrofoam piece | 112 | not a distinct visual class at this scale |
| Pop tab | 99 | not a distinct visual class at this scale |
| Normal paper | 82 | not a distinct visual class at this scale |
| Plastic lid | 77 | not a distinct visual class at this scale |
| Aluminium foil | 62 | not a distinct visual class at this scale |
| Tissues | 42 | not a distinct visual class at this scale |
| Disposable food container | 38 | not a distinct visual class at this scale |

## Expected limitations

* **Object classes are not disposal classes.** `bottle` spans PET and glass;
  `cup` spans polymer, board and foam. If BinSight later needs a bin
  recommendation, it will need either a material head or a second stage. This
  mapping is chosen to get a detector that *works* first.
* **`cigarette` is the second-largest class and is 0.02% of frame at the
  median** — the smallest objects in the dataset. It will likely need a higher
  input resolution than the rest, and may be worth dropping if it degrades the
  others.
* **67% of all TACO objects occupy under 1% of the frame.** TACO is street
  litter photographed from standing height; BinSight users hold a phone close
  to one item. The domain gap has moved, not disappeared.
* **`straw` (161) and `cup` (192) are thin** and will have roughly 24 and 29
  validation instances after splitting.

## Categories needing later data

`Broken glass` (138), `Styrofoam piece` (112), `Pop tab` (99) and
`Normal paper` (82) are visually distinct and plausibly useful, but each is
too thin to add as a ninth or tenth class now. They are the first candidates
if TACO is supplemented.

