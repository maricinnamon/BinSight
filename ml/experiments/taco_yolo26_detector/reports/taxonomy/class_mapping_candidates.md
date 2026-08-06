# Class mapping candidates

Every one of TACO's 60 categories appears below with an explicit decision in
both candidate mappings. No category is silently unmapped.

Values: a target class name, `EXCLUDE` (deliberately dropped), or `REVIEW`
(composite material — needs a decision the label cannot make for us).

## Candidate A — material-oriented

Target classes: `plastic`, `metal`, `glass`, `paper`, `cardboard`

| Class | Annotations |
|---|---:|
| plastic | 2251 |
| metal | 550 |
| glass | 254 |
| cardboard | 206 |
| paper | 179 |
| *(excluded)* | 1231 |
| *(review — composite)* | 112 |

Included: **3440** of 4783 annotations (71.9%). Imbalance ratio 12.58×.

## Candidate B — object-oriented

Target classes: `bottle`, `can`, `cup`, `bag_wrapper`, `carton`, `cigarette`, `other_litter`

| Class | Annotations |
|---|---:|
| other_litter | 2090 |
| bag_wrapper | 872 |
| cigarette | 667 |
| bottle | 438 |
| can | 273 |
| carton | 251 |
| cup | 192 |

Included: **4783** annotations (100.0%). Imbalance ratio 10.89×.

## Full decision table

| ID | Original category | Anns | Material | Object | Rationale (material) | Ambiguity |
|---:|---|---:|---|---|---|---|
| 59 | Cigarette | 667 | `EXCLUDE` | `cigarette` | cellulose-acetate filter + paper + tobacco. No single material applies, and it is not an item BinSight can usefully advise on. Largest category (667) — including it as 'other' would let one ambiguous class dominate | high |
| 58 | Unlabeled litter | 517 | `EXCLUDE` | `other_litter` | material unknown by definition; 517 annotations of 'something' would teach the detector nothing about material | high |
| 36 | Plastic film | 451 | `plastic` | `bag_wrapper` | polymer film, material named in the label | low |
| 5 | Clear plastic bottle | 284 | `plastic` | `bottle` | PET bottle | low |
| 29 | Other plastic | 273 | `plastic` | `other_litter` | explicitly plastic, object unspecified | low |
| 39 | Other plastic wrapper | 260 | `plastic` | `bag_wrapper` | polymer wrapper | low |
| 12 | Drink can | 229 | `metal` | `can` | aluminium beverage can | low |
| 7 | Plastic bottle cap | 209 | `plastic` | `other_litter` | HDPE/PP closure | low |
| 55 | Plastic straw | 157 | `plastic` | `other_litter` | polymer straw | low |
| 9 | Broken glass | 138 | `glass` | `other_litter` | explicitly glass | low |
| 57 | Styrofoam piece | 112 | `plastic` | `other_litter` | expanded polystyrene is a plastic | low |
| 21 | Disposable plastic cup | 104 | `plastic` | `cup` | polymer cup | low |
| 6 | Glass bottle | 104 | `glass` | `bottle` | glass bottle | low |
| 50 | Pop tab | 99 | `metal` | `other_litter` | aluminium tab | low |
| 14 | Other carton | 93 | `cardboard` | `carton` | carton board | low |
| 33 | Normal paper | 82 | `paper` | `other_litter` | paper | low |
| 8 | Metal bottle cap | 80 | `metal` | `other_litter` | steel/aluminium crown cap | low |
| 27 | Plastic lid | 77 | `plastic` | `other_litter` | polymer lid | low |
| 20 | Paper cup | 67 | `REVIEW` | `cup` | board with a polymer lining; widely NOT recyclable as paper, so labelling it 'paper' would mislead | high |
| 17 | Corrugated carton | 64 | `cardboard` | `carton` | corrugated board | low |
| 0 | Aluminium foil | 62 | `metal` | `other_litter` | aluminium | low |
| 40 | Single-use carrier bag | 61 | `plastic` | `bag_wrapper` | polymer carrier bag | low |
| 4 | Other plastic bottle | 50 | `plastic` | `bottle` | non-clear polymer bottle | low |
| 16 | Drink carton | 45 | `REVIEW` | `carton` | Tetra Pak-style laminate: board + polymer + foil. Disposal differs by locality and it is not honestly 'cardboard' | high |
| 31 | Tissues | 42 | `paper` | `other_litter` | paper tissue | low |
| 42 | Crisp packet | 39 | `plastic` | `bag_wrapper` | metallised polymer film; the metal layer is too thin to be a metal item, disposal is with plastics | medium |
| 45 | Disposable food container | 38 | `plastic` | `other_litter` | polymer container | low |
| 49 | Plastic utensils | 37 | `plastic` | `other_litter` | polymer cutlery | low |
| 10 | Food Can | 34 | `metal` | `can` | steel food can | low |
| 38 | Garbage bag | 31 | `plastic` | `bag_wrapper` | polymer sack | low |
| 18 | Meal carton | 30 | `cardboard` | `carton` | carton board | low |
| 51 | Rope & strings | 29 | `EXCLUDE` | `other_litter` | natural or synthetic fibre, unresolvable from the label | high |
| 34 | Paper bag | 27 | `paper` | `bag_wrapper` | paper bag | low |
| 52 | Scrap metal | 20 | `metal` | `other_litter` | explicitly metal | low |
| 46 | Foam food container | 15 | `plastic` | `other_litter` | expanded polystyrene | low |
| 22 | Foam cup | 13 | `plastic` | `cup` | expanded polystyrene | low |
| 30 | Magazine paper | 12 | `paper` | `other_litter` | printed paper | low |
| 32 | Wrapping paper | 12 | `paper` | `other_litter` | paper | low |
| 15 | Egg carton | 11 | `cardboard` | `carton` | moulded board or foam pulp; usually board | medium |
| 11 | Aerosol | 10 | `metal` | `can` | steel/aluminium pressurised can | low |
| 28 | Metal lid | 10 | `metal` | `other_litter` | explicitly metal | low |
| 43 | Spread tub | 9 | `plastic` | `other_litter` | polymer tub | low |
| 25 | Food waste | 8 | `EXCLUDE` | `other_litter` | organic; outside BinSight's six-material scope and only 8 annotations | medium |
| 53 | Shoe | 7 | `EXCLUDE` | `other_litter` | multi-material; not packaging waste | high |
| 54 | Squeezable tube | 7 | `plastic` | `other_litter` | usually polymer, occasionally aluminium-lined | medium |
| 2 | Aluminium blister pack | 6 | `metal` | `other_litter` | aluminium-backed blister; mixed with polymer but the label names aluminium | medium |
| 23 | Glass cup | 6 | `glass` | `cup` | glass drinkware | low |
| 26 | Glass jar | 6 | `glass` | `other_litter` | glass jar | low |
| 47 | Other plastic container | 6 | `plastic` | `other_litter` | explicitly plastic | low |
| 37 | Six pack rings | 5 | `plastic` | `other_litter` | polymer rings | low |
| 13 | Toilet tube | 5 | `cardboard` | `carton` | board tube | low |
| 56 | Paper straw | 4 | `paper` | `other_litter` | paper straw | low |
| 48 | Plastic glooves | 4 | `plastic` | `other_litter` | polymer gloves (TACO's spelling) | low |
| 44 | Tupperware | 4 | `plastic` | `other_litter` | reusable polymer container | low |
| 19 | Pizza box | 3 | `cardboard` | `carton` | corrugated board | low |
| 41 | Polypropylene bag | 3 | `plastic` | `bag_wrapper` | named polymer | low |
| 1 | Battery | 2 | `EXCLUDE` | `other_litter` | hazardous waste requiring separate handling; 2 annotations. Misclassifying a battery is worse than not detecting it | high |
| 24 | Other plastic cup | 2 | `plastic` | `cup` | polymer cup | low |
| 3 | Carded blister pack | 1 | `EXCLUDE` | `other_litter` | board + polymer laminate, 1 annotation | high |
| 35 | Plastified paper bag | 0 | `EXCLUDE` | `bag_wrapper` | board + polymer laminate, 0 annotations | high |
