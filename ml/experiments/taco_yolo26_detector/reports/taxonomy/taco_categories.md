# TACO taxonomy audit

Source: <https://github.com/pedropro/TACO> · `data/annotations.json` (official, reviewed — `annotations_unofficial.json` is deliberately not used).

## Dataset

| | |
|---|---:|
| Images | 1500 |
| Annotations | 4784 |
| Valid annotations | 4783 |
| Invalid annotations | 1 |
| Categories | 60 |
| Supercategories | 28 |
| Images with ≥1 annotation | 1500 |
| Images with no annotation | 0 |
| Annotations per image (mean / median / max) | 3.189 / 2.0 / 90 |

## Object size

`relative_bbox_area = (bbox_w × bbox_h) / (image_w × image_h)`

| Bucket | Range | Annotations | Share |
|---|---|---:|---:|
| tiny | [0%, 1%) | 3215 | 67.22% |
| small | [1%, 5%) | 968 | 20.24% |
| medium | [5%, 20%) | 436 | 9.12% |
| large | >= 20% | 164 | 3.43% |

Median object occupies **0.35%** of the frame.

tiny <1% of frame area, small 1-5%, medium 5-20%, large >20%. Chosen for a phone-camera use case: below 1% an object is a few dozen pixels at a 640 px detector input.

## Categories

All 60 original categories, sorted by annotation count.

| # | ID | Category | Supercategory | Anns | Images | % | Median rel. area |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 59 | Cigarette | Cigarette | 667 | 227 | 13.945% | 0.0207% |
| 2 | 58 | Unlabeled litter | Unlabeled litter | 517 | 269 | 10.809% | 0.0788% |
| 3 | 36 | Plastic film | Plastic bag & wrapper | 451 | 310 | 9.429% | 0.7977% |
| 4 | 5 | Clear plastic bottle | Bottle | 284 | 224 | 5.938% | 0.9708% |
| 5 | 29 | Other plastic | Other plastic | 273 | 171 | 5.708% | 0.1531% |
| 6 | 39 | Other plastic wrapper | Plastic bag & wrapper | 260 | 184 | 5.436% | 0.5586% |
| 7 | 12 | Drink can | Can | 229 | 151 | 4.788% | 0.8297% |
| 8 | 7 | Plastic bottle cap | Bottle cap | 209 | 185 | 4.37% | 0.0866% |
| 9 | 55 | Plastic straw | Straw | 157 | 110 | 3.282% | 0.4529% |
| 10 | 9 | Broken glass | Broken glass | 138 | 15 | 2.885% | 0.0256% |
| 11 | 57 | Styrofoam piece | Styrofoam piece | 112 | 78 | 2.342% | 0.7841% |
| 12 | 21 | Disposable plastic cup | Cup | 104 | 91 | 2.174% | 0.8174% |
| 13 | 6 | Glass bottle | Bottle | 104 | 74 | 2.174% | 1.0513% |
| 14 | 50 | Pop tab | Pop tab | 99 | 71 | 2.07% | 0.0437% |
| 15 | 14 | Other carton | Carton | 93 | 79 | 1.944% | 1.9740% |
| 16 | 33 | Normal paper | Paper | 82 | 68 | 1.714% | 0.6286% |
| 17 | 8 | Metal bottle cap | Bottle cap | 80 | 55 | 1.673% | 0.1612% |
| 18 | 27 | Plastic lid | Lid | 77 | 69 | 1.61% | 0.7812% |
| 19 | 20 | Paper cup | Cup | 67 | 62 | 1.401% | 1.0427% |
| 20 | 17 | Corrugated carton | Carton | 64 | 41 | 1.338% | 7.2988% |
| 21 | 0 | Aluminium foil | Aluminium foil | 62 | 43 | 1.296% | 0.7584% |
| 22 | 40 | Single-use carrier bag | Plastic bag & wrapper | 61 | 49 | 1.275% | 2.5929% |
| 23 | 4 | Other plastic bottle | Bottle | 50 | 45 | 1.045% | 3.3901% |
| 24 | 16 | Drink carton | Carton | 45 | 41 | 0.941% | 0.6568% |
| 25 | 31 | Tissues | Paper | 42 | 36 | 0.878% | 0.9641% |
| 26 | 42 | Crisp packet | Plastic bag & wrapper | 39 | 35 | 0.815% | 1.5920% |
| 27 | 45 | Disposable food container | Plastic container | 38 | 37 | 0.794% | 2.3618% |
| 28 | 49 | Plastic utensils | Plastic utensils | 37 | 29 | 0.774% | 0.6871% |
| 29 | 10 | Food Can | Can | 34 | 20 | 0.711% | 5.7050% |
| 30 | 38 | Garbage bag | Plastic bag & wrapper | 31 | 20 | 0.648% | 6.5342% |
| 31 | 18 | Meal carton | Carton | 30 | 27 | 0.627% | 2.1131% |
| 32 | 51 | Rope & strings | Rope & strings | 29 | 28 | 0.606% | 1.4401% |
| 33 | 34 | Paper bag | Paper bag | 27 | 23 | 0.564% | 3.7691% |
| 34 | 52 | Scrap metal | Scrap metal | 20 | 12 | 0.418% | 0.5771% |
| 35 | 46 | Foam food container | Plastic container | 15 | 13 | 0.314% | 2.2715% |
| 36 | 22 | Foam cup | Cup | 13 | 11 | 0.272% | 2.3013% |
| 37 | 30 | Magazine paper | Paper | 12 | 5 | 0.251% | 3.5909% |
| 38 | 32 | Wrapping paper | Paper | 12 | 11 | 0.251% | 5.4928% |
| 39 | 15 | Egg carton | Carton | 11 | 9 | 0.23% | 3.0368% |
| 40 | 11 | Aerosol | Can | 10 | 10 | 0.209% | 11.3907% |
| 41 | 28 | Metal lid | Lid | 10 | 6 | 0.209% | 2.4200% |
| 42 | 43 | Spread tub | Plastic container | 9 | 9 | 0.188% | 1.0465% |
| 43 | 25 | Food waste | Food waste | 8 | 7 | 0.167% | 1.1089% |
| 44 | 53 | Shoe | Shoe | 7 | 6 | 0.146% | 1.8390% |
| 45 | 54 | Squeezable tube | Squeezable tube | 7 | 6 | 0.146% | 1.6794% |
| 46 | 2 | Aluminium blister pack | Blister pack | 6 | 6 | 0.125% | 2.2641% |
| 47 | 23 | Glass cup | Cup | 6 | 4 | 0.125% | 1.6201% |
| 48 | 26 | Glass jar | Glass jar | 6 | 4 | 0.125% | 10.1519% |
| 49 | 47 | Other plastic container | Plastic container | 6 | 6 | 0.125% | 0.3272% |
| 50 | 37 | Six pack rings | Plastic bag & wrapper | 5 | 4 | 0.105% | 3.2451% |
| 51 | 13 | Toilet tube | Carton | 5 | 3 | 0.105% | 1.5791% |
| 52 | 56 | Paper straw | Straw | 4 | 4 | 0.084% | 2.6241% |
| 53 | 48 | Plastic glooves | Plastic glooves | 4 | 4 | 0.084% | 1.7017% |
| 54 | 44 | Tupperware | Plastic container | 4 | 4 | 0.084% | 10.8129% |
| 55 | 19 | Pizza box | Carton | 3 | 3 | 0.063% | 13.4095% |
| 56 | 41 | Polypropylene bag | Plastic bag & wrapper | 3 | 3 | 0.063% | 12.7462% |
| 57 | 1 | Battery | Battery | 2 | 2 | 0.042% | 3.3805% |
| 58 | 24 | Other plastic cup | Cup | 2 | 2 | 0.042% | 1.1318% |
| 59 | 3 | Carded blister pack | Blister pack | 1 | 1 | 0.021% | 6.8539% |
| 60 | 35 | Plastified paper bag | Paper bag | 0 | 0 | 0.0% | 0.0000% |

**26 of 60 categories have fewer than 20 annotations.** That long tail is the central problem for the class mapping: most original categories cannot support a detector class on their own.
