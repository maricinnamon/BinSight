# Official TACO class maps — analysis

TACO ships example class maps under `detector/taco_config/`. They are read
here rather than copied, because each encodes a different answer to the same
question BinSight has to answer: *how do you collapse 60 sparse categories
into something a detector can learn?*

| Map | Target classes | Largest class | Smallest class | Notes |
|---|---:|---:|---:|---|
| `map_1` | 1 | 4783 | 4783 |  |
| `map_2` | 3 | 4072 | 273 | catch-all holds 85% of annotations |
| `map_3` | 4 | 3222 | 273 | catch-all holds 67% of annotations |
| `map_4` | 4 | 3222 | 273 | catch-all holds 67% of annotations |
| `map_10` | 10 | 1727 | 87 | catch-all holds 36% of annotations |
| `map_17` | 17 | 1732 | 62 | catch-all holds 36% of annotations |

## What the official maps show

* **Every official map needs a catch-all.** `map_4` puts 67% of annotations
  into `Other`; `map_10` still puts 36% there; even `map_17` — the most
  granular — leaves 36% in `Other`. No mapping of this taxonomy avoids a
  large residual class, because the residual is real: `Cigarette` (667) and
  `Unlabeled litter` (517) alone are 25% of the dataset.
* **The official maps are object-oriented, not material-oriented.** `Bottle`,
  `Can`, `Cup`, `Straw`, `Pop tab` group by silhouette. None of them attempts
  a `plastic` / `metal` / `glass` split. That is a signal worth taking
  seriously: the people who built the taxonomy chose visual groupings when
  they needed a detector to work.
* **`map_17` is close to the limit of the data.** Its smallest class has 62
  annotations across the whole dataset — before splitting 70/15/15, which
  leaves roughly 9 validation instances.

## Per-map class counts

**`map_1`** — `Litter` 4783

**`map_2`** — `Background` 4072, `Bottle` 438, `Can` 273

**`map_3`** — `Background` 3222, `Plastic bag + wrapper` 850, `Bottle` 438, `Can` 273

**`map_4`** — `Other` 3222, `Plastic bag + wrapper` 850, `Bottle` 438, `Can` 273

**`map_10`** — `Other` 1727, `Plastic bag + wrapper` 850, `Cigarette` 667, `Bottle` 438, `Bottle cap` 289, `Can` 273, `Cup` 192, `Straw` 161, `Pop tab` 99, `Lid` 87

**`map_17`** — `Other` 1732, `Plastic film` 551, `Plastic bottle` 334, `Wrapper` 299, `Can` 273, `Carton` 251, `Plastic bottle cap` 209, `Cup` 192, `Paper` 175, `Straw` 161, `Styrofoam piece` 112, `Glass bottle` 104, `Pop tab` 99, `Metal bottle cap` 80, `Plastic lid` 77, `Plastic container` 72, `Aluminium foil` 62

## Categories that dominate

| Category | Annotations | Share |
|---|---:|---:|
| Cigarette | 667 | 13.9% |
| Unlabeled litter | 517 | 10.8% |
| Plastic film | 451 | 9.4% |
| Clear plastic bottle | 284 | 5.9% |
| Other plastic | 273 | 5.7% |
| Other plastic wrapper | 260 | 5.4% |
| Drink can | 229 | 4.8% |
| Plastic bottle cap | 209 | 4.4% |

## The long tail

**26 of 60 categories have fewer than 20 annotations**, and 1 has none at all (Plastified paper bag).

Those categories cannot support a detector class individually. They must be
merged, excluded, or accepted as a residual class — there is no fourth option.

## Materials that cannot be inferred reliably from TACO labels

| Category | Annotations | Why the material is unclear |
|---|---:|---|
| Cigarette | 667 | cellulose-acetate filter + paper + tobacco. No single material applies, and it is not an item BinSight can usefully advise on. Largest category (667) — including it as 'other' would let one ambiguous class dominate |
| Unlabeled litter | 517 | material unknown by definition; 517 annotations of 'something' would teach the detector nothing about material |
| Drink carton | 45 | Tetra Pak-style laminate: board + polymer + foil. Disposal differs by locality and it is not honestly 'cardboard' |
| Paper cup | 67 | board with a polymer lining; widely NOT recyclable as paper, so labelling it 'paper' would mislead |
| Crisp packet | 39 | metallised polymer film; the metal layer is too thin to be a metal item, disposal is with plastics |
| Rope & strings | 29 | natural or synthetic fibre, unresolvable from the label |
| Aluminium blister pack | 6 | aluminium-backed blister; mixed with polymer but the label names aluminium |
| Squeezable tube | 7 | usually polymer, occasionally aluminium-lined |
| Egg carton | 11 | moulded board or foam pulp; usually board |

This is the crux of the mapping decision: **25% of TACO's annotations name an
object whose material the label does not establish**. A material-oriented
detector must either exclude them or invent a label for them.
