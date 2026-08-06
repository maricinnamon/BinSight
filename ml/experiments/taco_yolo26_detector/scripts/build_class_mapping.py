#!/usr/bin/env python
"""Builds and compares two candidate BinSight class mappings for TACO.

Every one of TACO's 60 categories gets an explicit decision in both mappings —
a target class, EXCLUDE, or REVIEW — with a written rationale and an ambiguity
flag. Nothing is left silently unmapped, and nothing is forced into a material
class the source label does not justify.

Outputs the decision table, the per-mapping statistics, the official-mapping
analysis, and the recommendation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "reports" / "taxonomy"
CONFIGS = ROOT / "configs"
TACO_CONFIG = ROOT / "data" / "source" / "TACO" / "detector" / "taco_config"

# --------------------------------------------------------------------------
# Mapping A — material-oriented
# --------------------------------------------------------------------------
# Preserves the BinSight UX concept (which bin does this go in?) but only where
# the source label actually names a material. Composite packaging is the hard
# part: a drink carton is board + polymer + foil, and a paper cup is board with
# a plastic lining. Neither is honestly "paper".
#
# Format: category name -> (target, rationale, ambiguity)
#   target: a class name, "EXCLUDE", or "REVIEW"
MATERIAL: dict[str, tuple[str, str, str]] = {
    # --- plastic ---------------------------------------------------------
    "Plastic film": ("plastic", "polymer film, material named in the label", "low"),
    "Clear plastic bottle": ("plastic", "PET bottle", "low"),
    "Other plastic": ("plastic", "explicitly plastic, object unspecified", "low"),
    "Other plastic wrapper": ("plastic", "polymer wrapper", "low"),
    "Plastic bottle cap": ("plastic", "HDPE/PP closure", "low"),
    "Plastic straw": ("plastic", "polymer straw", "low"),
    "Styrofoam piece": ("plastic", "expanded polystyrene is a plastic", "low"),
    "Disposable plastic cup": ("plastic", "polymer cup", "low"),
    "Other plastic cup": ("plastic", "polymer cup", "low"),
    "Plastic lid": ("plastic", "polymer lid", "low"),
    "Single-use carrier bag": ("plastic", "polymer carrier bag", "low"),
    "Other plastic bottle": ("plastic", "non-clear polymer bottle", "low"),
    "Disposable food container": ("plastic", "polymer container", "low"),
    "Plastic utensils": ("plastic", "polymer cutlery", "low"),
    "Garbage bag": ("plastic", "polymer sack", "low"),
    "Foam food container": ("plastic", "expanded polystyrene", "low"),
    "Foam cup": ("plastic", "expanded polystyrene", "low"),
    "Spread tub": ("plastic", "polymer tub", "low"),
    "Other plastic container": ("plastic", "explicitly plastic", "low"),
    "Six pack rings": ("plastic", "polymer rings", "low"),
    "Polypropylene bag": ("plastic", "named polymer", "low"),
    "Plastic glooves": ("plastic", "polymer gloves (TACO's spelling)", "low"),
    "Tupperware": ("plastic", "reusable polymer container", "low"),
    "Crisp packet": ("plastic", "metallised polymer film; the metal layer is too "
                                "thin to be a metal item, disposal is with plastics", "medium"),
    "Squeezable tube": ("plastic", "usually polymer, occasionally aluminium-lined", "medium"),

    # --- metal -----------------------------------------------------------
    "Drink can": ("metal", "aluminium beverage can", "low"),
    "Pop tab": ("metal", "aluminium tab", "low"),
    "Metal bottle cap": ("metal", "steel/aluminium crown cap", "low"),
    "Aluminium foil": ("metal", "aluminium", "low"),
    "Food Can": ("metal", "steel food can", "low"),
    "Scrap metal": ("metal", "explicitly metal", "low"),
    "Aerosol": ("metal", "steel/aluminium pressurised can", "low"),
    "Metal lid": ("metal", "explicitly metal", "low"),
    "Aluminium blister pack": ("metal", "aluminium-backed blister; mixed with polymer "
                                        "but the label names aluminium", "medium"),

    # --- glass -----------------------------------------------------------
    "Broken glass": ("glass", "explicitly glass", "low"),
    "Glass bottle": ("glass", "glass bottle", "low"),
    "Glass cup": ("glass", "glass drinkware", "low"),
    "Glass jar": ("glass", "glass jar", "low"),

    # --- paper -----------------------------------------------------------
    "Normal paper": ("paper", "paper", "low"),
    "Tissues": ("paper", "paper tissue", "low"),
    "Paper bag": ("paper", "paper bag", "low"),
    "Magazine paper": ("paper", "printed paper", "low"),
    "Wrapping paper": ("paper", "paper", "low"),
    "Paper straw": ("paper", "paper straw", "low"),

    # --- cardboard -------------------------------------------------------
    "Other carton": ("cardboard", "carton board", "low"),
    "Corrugated carton": ("cardboard", "corrugated board", "low"),
    "Meal carton": ("cardboard", "carton board", "low"),
    "Egg carton": ("cardboard", "moulded board or foam pulp; usually board", "medium"),
    "Toilet tube": ("cardboard", "board tube", "low"),
    "Pizza box": ("cardboard", "corrugated board", "low"),

    # --- composites and unknowns: NOT forced into a material -------------
    "Drink carton": ("REVIEW", "Tetra Pak-style laminate: board + polymer + foil. "
                               "Disposal differs by locality and it is not honestly "
                               "'cardboard'", "high"),
    "Paper cup": ("REVIEW", "board with a polymer lining; widely NOT recyclable as "
                            "paper, so labelling it 'paper' would mislead", "high"),
    "Cigarette": ("EXCLUDE", "cellulose-acetate filter + paper + tobacco. No single "
                             "material applies, and it is not an item BinSight can "
                             "usefully advise on. Largest category (667) — including "
                             "it as 'other' would let one ambiguous class dominate", "high"),
    "Unlabeled litter": ("EXCLUDE", "material unknown by definition; 517 annotations "
                                    "of 'something' would teach the detector nothing "
                                    "about material", "high"),
    "Rope & strings": ("EXCLUDE", "natural or synthetic fibre, unresolvable from the "
                                  "label", "high"),
    "Food waste": ("EXCLUDE", "organic; outside BinSight's six-material scope and only "
                              "8 annotations", "medium"),
    "Shoe": ("EXCLUDE", "multi-material; not packaging waste", "high"),
    "Battery": ("EXCLUDE", "hazardous waste requiring separate handling; 2 annotations. "
                           "Misclassifying a battery is worse than not detecting it", "high"),
    "Carded blister pack": ("EXCLUDE", "board + polymer laminate, 1 annotation", "high"),
    "Plastified paper bag": ("EXCLUDE", "board + polymer laminate, 0 annotations", "high"),
}

# --------------------------------------------------------------------------
# Mapping B — object-oriented
# --------------------------------------------------------------------------
# Groups by what the object *looks like* rather than what it is made of. This is
# closer to what a detector can actually learn from pixels, and it is the shape
# TACO's own official map_10 takes.
OBJECT: dict[str, tuple[str, str, str]] = {
    # bottle
    "Clear plastic bottle": ("bottle", "bottle silhouette", "low"),
    "Other plastic bottle": ("bottle", "bottle silhouette", "low"),
    "Glass bottle": ("bottle", "bottle silhouette", "low"),
    # can
    "Drink can": ("can", "cylindrical can", "low"),
    "Food Can": ("can", "cylindrical can", "low"),
    "Aerosol": ("can", "cylindrical pressurised can", "low"),
    # cup
    "Disposable plastic cup": ("cup", "cup silhouette", "low"),
    "Paper cup": ("cup", "cup silhouette", "low"),
    "Foam cup": ("cup", "cup silhouette", "low"),
    "Glass cup": ("cup", "cup silhouette", "low"),
    "Other plastic cup": ("cup", "cup silhouette", "low"),
    # bag / wrapper / film — visually one family: thin, crumpled, deformable
    "Plastic film": ("bag_wrapper", "thin deformable film", "low"),
    "Other plastic wrapper": ("bag_wrapper", "wrapper", "low"),
    "Single-use carrier bag": ("bag_wrapper", "carrier bag", "low"),
    "Crisp packet": ("bag_wrapper", "wrapper", "low"),
    "Garbage bag": ("bag_wrapper", "sack", "low"),
    "Paper bag": ("bag_wrapper", "bag silhouette", "medium"),
    "Polypropylene bag": ("bag_wrapper", "bag", "low"),
    "Plastified paper bag": ("bag_wrapper", "bag", "low"),
    # carton / box — rigid rectangular board
    "Other carton": ("carton", "board box", "low"),
    "Corrugated carton": ("carton", "board box", "low"),
    "Drink carton": ("carton", "board carton", "low"),
    "Meal carton": ("carton", "board box", "low"),
    "Egg carton": ("carton", "board box", "low"),
    "Pizza box": ("carton", "board box", "low"),
    "Toilet tube": ("carton", "board tube", "medium"),
    # cigarette — visually unmistakable and by far the largest category
    "Cigarette": ("cigarette", "distinctive small white cylinder with a filter; "
                               "visually consistent even though its material is not", "low"),
    # everything else that is annotated but not one of the above
    "Unlabeled litter": ("other_litter", "annotated litter of unknown type", "high"),
    "Other plastic": ("other_litter", "shape unspecified", "medium"),
    "Plastic bottle cap": ("other_litter", "small disc/cap", "medium"),
    "Metal bottle cap": ("other_litter", "small disc/cap", "medium"),
    "Pop tab": ("other_litter", "very small metal tab", "medium"),
    "Plastic straw": ("other_litter", "thin tube", "medium"),
    "Paper straw": ("other_litter", "thin tube", "medium"),
    "Broken glass": ("other_litter", "shards", "medium"),
    "Styrofoam piece": ("other_litter", "irregular foam fragment", "medium"),
    "Plastic lid": ("other_litter", "flat disc", "medium"),
    "Metal lid": ("other_litter", "flat disc", "medium"),
    "Normal paper": ("other_litter", "sheet", "medium"),
    "Tissues": ("other_litter", "crumpled sheet", "medium"),
    "Magazine paper": ("other_litter", "sheet", "medium"),
    "Wrapping paper": ("other_litter", "sheet", "medium"),
    "Aluminium foil": ("other_litter", "crumpled foil", "medium"),
    "Disposable food container": ("other_litter", "tray/container", "medium"),
    "Foam food container": ("other_litter", "tray", "medium"),
    "Spread tub": ("other_litter", "tub", "medium"),
    "Other plastic container": ("other_litter", "container", "medium"),
    "Tupperware": ("other_litter", "container", "medium"),
    "Plastic utensils": ("other_litter", "cutlery", "medium"),
    "Scrap metal": ("other_litter", "irregular metal", "medium"),
    "Rope & strings": ("other_litter", "fibre", "medium"),
    "Food waste": ("other_litter", "organic", "medium"),
    "Shoe": ("other_litter", "shoe", "medium"),
    "Battery": ("other_litter", "small block", "medium"),
    "Aluminium blister pack": ("other_litter", "blister", "medium"),
    "Carded blister pack": ("other_litter", "blister", "medium"),
    "Glass jar": ("other_litter", "jar; too few to justify its own class", "medium"),
    "Six pack rings": ("other_litter", "rings", "medium"),
    "Squeezable tube": ("other_litter", "tube", "medium"),
    "Plastic glooves": ("other_litter", "glove", "medium"),
}


# --------------------------------------------------------------------------
# Mapping C — RECOMMENDED: object-oriented with a narrowed residual
# --------------------------------------------------------------------------
# Neither candidate above is good enough on its own:
#
#   A (material) discards 28% of annotations and still lands at a 12.6x
#     imbalance, with `plastic` holding 65% of what remains. Worse, it asks the
#     detector to infer material from pixels — the exact inference the TrashNet
#     baseline already showed does not survive contact with real scenes.
#
#   B (object) keeps everything, but 44% of it falls into `other_litter`. A
#     catch-all that large is not a class; it is a place where the model learns
#     "some litter is here" and nothing else, and the brief explicitly warns
#     against it.
#
# C keeps B's visual groupings but *drops* the residual instead of training on
# it. Every remaining class is a thing a person can point a phone at and name.
# The cost is stated plainly in the report: 1 640 annotations become unlabelled
# background, which is a real trade-off, not a free win.
RECOMMENDED_CLASSES = [
    "bag_wrapper", "bottle", "bottle_cap", "can", "carton", "cigarette", "cup", "straw",
]

RECOMMENDED: dict[str, list[str]] = {
    "bag_wrapper": ["Plastic film", "Other plastic wrapper", "Single-use carrier bag",
                    "Crisp packet", "Garbage bag", "Paper bag", "Polypropylene bag",
                    "Plastified paper bag"],
    "bottle": ["Clear plastic bottle", "Other plastic bottle", "Glass bottle"],
    "bottle_cap": ["Plastic bottle cap", "Metal bottle cap"],
    "can": ["Drink can", "Food Can", "Aerosol"],
    "carton": ["Other carton", "Corrugated carton", "Drink carton", "Meal carton",
               "Egg carton", "Pizza box", "Toilet tube"],
    "cigarette": ["Cigarette"],
    "cup": ["Disposable plastic cup", "Paper cup", "Foam cup", "Glass cup",
            "Other plastic cup"],
    "straw": ["Plastic straw", "Paper straw"],
}

MATERIAL_CLASSES = ["plastic", "metal", "glass", "paper", "cardboard"]
OBJECT_CLASSES = ["bottle", "can", "cup", "bag_wrapper", "carton", "cigarette", "other_litter"]


def load_categories() -> list[dict]:
    path = TAXONOMY / "taco_categories.json"
    if not path.exists():
        print(f"ERROR: {path} not found — run scripts/audit_taco.py first.")
        sys.exit(1)
    return json.loads(path.read_text())


def summarise(categories: list[dict], mapping: dict[str, tuple[str, str, str]]) -> dict:
    counts: collections.Counter = collections.Counter()
    images: dict[str, set] = collections.defaultdict(set)
    excluded = 0
    review = 0
    unmapped = []

    for row in categories:
        name = row["name"]
        if name not in mapping:
            unmapped.append(name)
            continue
        target = mapping[name][0]
        if target == "EXCLUDE":
            excluded += row["annotations"]
        elif target == "REVIEW":
            review += row["annotations"]
        else:
            counts[target] += row["annotations"]
    return {
        "class_counts": dict(counts.most_common()),
        "classes": len(counts),
        "included_annotations": sum(counts.values()),
        "excluded_annotations": excluded,
        "review_annotations": review,
        "unmapped_categories": unmapped,
        "smallest_class": min(counts.values()) if counts else 0,
        "largest_class": max(counts.values()) if counts else 0,
        "imbalance_ratio": round(max(counts.values()) / max(min(counts.values()), 1), 2) if counts else None,
    }


def official_mapping_analysis(categories: list[dict]) -> str:
    by_name = {c["name"]: c for c in categories}
    lines = [
        "# Official TACO class maps — analysis",
        "",
        "TACO ships example class maps under `detector/taco_config/`. They are read",
        "here rather than copied, because each encodes a different answer to the same",
        "question BinSight has to answer: *how do you collapse 60 sparse categories",
        "into something a detector can learn?*",
        "",
        "| Map | Target classes | Largest class | Smallest class | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    details = []
    for name in ("map_1", "map_2", "map_3", "map_4", "map_10", "map_17"):
        path = TACO_CONFIG / f"{name}.csv"
        if not path.exists():
            continue
        mapping: dict[str, str] = {}
        for line in path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
        counts: collections.Counter = collections.Counter()
        for category, target in mapping.items():
            if category in by_name:
                counts[target] += by_name[category]["annotations"]
        if not counts:
            continue
        note = ""
        catch_all = max((counts[k] for k in counts if k in ("Other", "Background")), default=0)
        if catch_all:
            note = f"catch-all holds {100 * catch_all / sum(counts.values()):.0f}% of annotations"
        lines.append(
            f"| `{name}` | {len(counts)} | {max(counts.values())} | {min(counts.values())} | {note} |"
        )
        details.append((name, counts))

    lines += [
        "",
        "## What the official maps show",
        "",
        "* **Every official map needs a catch-all.** `map_4` puts 67% of annotations",
        "  into `Other`; `map_10` still puts 36% there; even `map_17` — the most",
        "  granular — leaves 36% in `Other`. No mapping of this taxonomy avoids a",
        "  large residual class, because the residual is real: `Cigarette` (667) and",
        "  `Unlabeled litter` (517) alone are 25% of the dataset.",
        "* **The official maps are object-oriented, not material-oriented.** `Bottle`,",
        "  `Can`, `Cup`, `Straw`, `Pop tab` group by silhouette. None of them attempts",
        "  a `plastic` / `metal` / `glass` split. That is a signal worth taking",
        "  seriously: the people who built the taxonomy chose visual groupings when",
        "  they needed a detector to work.",
        "* **`map_17` is close to the limit of the data.** Its smallest class has 62",
        "  annotations across the whole dataset — before splitting 70/15/15, which",
        "  leaves roughly 9 validation instances.",
        "",
        "## Per-map class counts",
        "",
    ]
    for name, counts in details:
        lines.append(f"**`{name}`** — " + ", ".join(f"`{k}` {v}" for k, v in counts.most_common()))
        lines.append("")

    lines += [
        "## Categories that dominate",
        "",
        "| Category | Annotations | Share |",
        "|---|---:|---:|",
    ]
    total = sum(c["annotations"] for c in categories)
    for row in categories[:8]:
        lines.append(f"| {row['name']} | {row['annotations']} | "
                     f"{100 * row['annotations'] / total:.1f}% |")

    tail = [c for c in categories if c["annotations"] < 20]
    zero = [c for c in categories if c["annotations"] == 0]
    lines += [
        "",
        "## The long tail",
        "",
        f"**{len(tail)} of {len(categories)} categories have fewer than 20 annotations**, "
        f"and {len(zero)} has none at all "
        f"({', '.join(c['name'] for c in zero) if zero else '—'}).",
        "",
        "Those categories cannot support a detector class individually. They must be",
        "merged, excluded, or accepted as a residual class — there is no fourth option.",
        "",
        "## Materials that cannot be inferred reliably from TACO labels",
        "",
        "| Category | Annotations | Why the material is unclear |",
        "|---|---:|---|",
    ]
    for name in ("Cigarette", "Unlabeled litter", "Drink carton", "Paper cup",
                 "Crisp packet", "Rope & strings", "Aluminium blister pack",
                 "Squeezable tube", "Egg carton"):
        if name in by_name:
            reason = MATERIAL.get(name, ("", "not assessed", ""))[1]
            lines.append(f"| {name} | {by_name[name]['annotations']} | {reason} |")
    lines += [
        "",
        "This is the crux of the mapping decision: **25% of TACO's annotations name an",
        "object whose material the label does not establish**. A material-oriented",
        "detector must either exclude them or invent a label for them.",
        "",
    ]
    return "\n".join(lines)



def emit_recommended(categories: list[dict]) -> dict:
    """Writes configs/class_mapping.yaml, configs/classes.txt and the recommendation."""
    by_name = {c["name"]: c for c in categories}
    to_target = {cat: target for target, cats in RECOMMENDED.items() for cat in cats}

    counts = {t: sum(by_name[c]["annotations"] for c in cats if c in by_name)
              for t, cats in RECOMMENDED.items()}
    images = {t: len({c for c in cats if c in by_name}) for t, cats in RECOMMENDED.items()}
    included = sum(counts.values())
    total = sum(c["annotations"] for c in categories)
    dropped = [(c["name"], c["annotations"]) for c in categories
               if c["name"] not in to_target and c["annotations"] > 0]

    # --- configs/classes.txt: the authoritative class-ID order ---------
    (CONFIGS / "classes.txt").write_text("\n".join(RECOMMENDED_CLASSES) + "\n")

    # --- configs/class_mapping.yaml ------------------------------------
    lines = [
        "# BinSight class mapping for TACO -> YOLO detection.",
        "#",
        "# Generated by scripts/build_class_mapping.py from the official TACO",
        "# annotations. Class IDs are the index of `target_classes` below, and that",
        "# order is also written to configs/classes.txt and configs/dataset.yaml.",
        "# Editing one without the others will silently mislabel every prediction.",
        "",
        "mapping_name: object_oriented_narrowed",
        f"source_categories: {len(categories)}",
        f"included_annotations: {included}",
        f"excluded_annotations: {total - included}",
        "",
        "target_classes:",
    ]
    for index, name in enumerate(RECOMMENDED_CLASSES):
        lines.append(f"  - {name}    # id {index}, {counts[name]} annotations")
    lines += ["", "mapping:"]
    for row in categories:
        name = row["name"]
        target = to_target.get(name)
        lines.append(f'  "{name}":')
        if target:
            lines.append(f"    target: {target}")
            lines.append("    include: true")
            lines.append(f"    annotations: {row['annotations']}")
        else:
            reason = MATERIAL.get(name, ("", "not in the recommended mapping", ""))[1]
            lines.append("    target: null")
            lines.append("    include: false")
            lines.append(f"    annotations: {row['annotations']}")
            lines.append(f'    reason: "{reason.replace(chr(34), chr(39))}"')
    (CONFIGS / "class_mapping.yaml").write_text("\n".join(lines) + "\n")

    # --- recommendation -------------------------------------------------
    material = summarise(categories, MATERIAL)
    obj = summarise(categories, OBJECT)
    doc = [
        "# Recommended mapping for the first YOLO26 experiment",
        "",
        "**Recommendation: the object-oriented mapping with a narrowed residual "
        "(candidate C), 8 classes.**",
        "",
        "## The three candidates, measured",
        "",
        "| | Classes | Annotations used | Largest class | Smallest | Imbalance | Catch-all share |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| A — material | {material['classes']} | {material['included_annotations']} "
        f"({100*material['included_annotations']/total:.0f}%) | {material['largest_class']} "
        f"| {material['smallest_class']} | {material['imbalance_ratio']}x | none |",
        f"| B — object | {obj['classes']} | {obj['included_annotations']} "
        f"({100*obj['included_annotations']/total:.0f}%) | {obj['largest_class']} "
        f"| {obj['smallest_class']} | {obj['imbalance_ratio']}x | "
        f"{100*obj['class_counts'].get('other_litter',0)/obj['included_annotations']:.0f}% |",
        f"| **C — object, narrowed** | **{len(RECOMMENDED_CLASSES)}** | **{included} "
        f"({100*included/total:.0f}%)** | **{max(counts.values())}** | **{min(counts.values())}** "
        f"| **{max(counts.values())/min(counts.values()):.2f}x** | **none** |",
        "",
        "## Why not the material mapping",
        "",
        "It is the one that preserves BinSight's existing UX, so it deserved a fair",
        "hearing. It does not survive the numbers:",
        "",
        f"* it discards **{material['excluded_annotations']} annotations** "
        f"({100*material['excluded_annotations']/total:.0f}%) as materially ambiguous, plus "
        f"{material['review_annotations']} composite ones needing review;",
        f"* `plastic` still ends up with {material['class_counts'].get('plastic',0)} of "
        f"{material['included_annotations']} included annotations "
        f"({100*material['class_counts'].get('plastic',0)/material['included_annotations']:.0f}%);",
        f"* `paper` ({material['class_counts'].get('paper',0)}) and "
        f"`cardboard` ({material['class_counts'].get('cardboard',0)}) are thin before a "
        "70/15/15 split, leaving roughly 27 and 31 validation instances;",
        "* most importantly, it asks the detector to infer **material from pixels** —",
        "  the same inference the TrashNet baseline already showed does not transfer to",
        "  uncontrolled scenes. Repeating it on harder data is not a plan.",
        "",
        "## Why not the wide object mapping",
        "",
        f"Candidate B keeps every annotation, but `other_litter` absorbs "
        f"{obj['class_counts'].get('other_litter',0)} of them — "
        f"{100*obj['class_counts'].get('other_litter',0)/obj['included_annotations']:.0f}% of the dataset. "
        "A class that large and that heterogeneous teaches the model to fire on almost",
        "any litter-shaped region without saying anything useful. TACO's own official",
        "maps show the same pattern (`map_4` 67% `Other`, `map_10` 36%, `map_17` 36%),",
        "which is a property of the taxonomy rather than a flaw in any one mapping.",
        "",
        "## The recommendation",
        "",
        "Eight classes, every one of them a thing a person can point a phone at and",
        "name, with no residual class at all:",
        "",
        "| ID | Class | Annotations | Source categories |",
        "|---:|---|---:|---|",
    ]
    for index, name in enumerate(RECOMMENDED_CLASSES):
        doc.append(f"| {index} | `{name}` | {counts[name]} | "
                   + ", ".join(RECOMMENDED[name]) + " |")
    doc += [
        "",
        f"Total: **{included} annotations** across {len(RECOMMENDED_CLASSES)} classes, "
        f"imbalance **{max(counts.values())/min(counts.values()):.2f}x** — less than half "
        "of either alternative.",
        "",
        "## What this costs, stated plainly",
        "",
        f"**{total - included} annotations ({100*(total-included)/total:.0f}%) become "
        "unlabelled background.** That is the real price, and it has a specific",
        "consequence: where an image contains both a mapped object and a dropped one,",
        "the detector is trained to treat the dropped object as background. For",
        "BinSight that is arguably correct — we would rather the app stay silent than",
        "announce 'unidentified litter' — but it is a deliberate trade, not a free win.",
        "",
        "The largest dropped categories:",
        "",
        "| Category | Annotations | Why dropped |",
        "|---|---:|---|",
    ]
    for name, count in sorted(dropped, key=lambda x: -x[1])[:10]:
        reason = MATERIAL.get(name, ("", "", ""))[1]
        short = "not a distinct visual class at this scale" if count < 300 else reason
        doc.append(f"| {name} | {count} | {short[:90]} |")
    doc += [
        "",
        "## Expected limitations",
        "",
        "* **Object classes are not disposal classes.** `bottle` spans PET and glass;",
        "  `cup` spans polymer, board and foam. If BinSight later needs a bin",
        "  recommendation, it will need either a material head or a second stage. This",
        "  mapping is chosen to get a detector that *works* first.",
        "* **`cigarette` is the second-largest class and is 0.02% of frame at the",
        "  median** — the smallest objects in the dataset. It will likely need a higher",
        "  input resolution than the rest, and may be worth dropping if it degrades the",
        "  others.",
        "* **67% of all TACO objects occupy under 1% of the frame.** TACO is street",
        "  litter photographed from standing height; BinSight users hold a phone close",
        "  to one item. The domain gap has moved, not disappeared.",
        "* **`straw` (161) and `cup` (192) are thin** and will have roughly 24 and 29",
        "  validation instances after splitting.",
        "",
        "## Categories needing later data",
        "",
        "`Broken glass` (138), `Styrofoam piece` (112), `Pop tab` (99) and",
        "`Normal paper` (82) are visually distinct and plausibly useful, but each is",
        "too thin to add as a ninth or tenth class now. They are the first candidates",
        "if TACO is supplemented.",
        "",
    ]
    (TAXONOMY / "recommended_mapping.md").write_text("\n".join(doc) + "\n")
    return {"classes": RECOMMENDED_CLASSES, "counts": counts,
            "included": included, "excluded": total - included, "dropped": dropped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    categories = load_categories()
    TAXONOMY.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    names = {c["name"] for c in categories}
    for label, mapping in (("material", MATERIAL), ("object", OBJECT)):
        missing = names - set(mapping)
        extra = set(mapping) - names
        if missing:
            print(f"ERROR: {label} mapping is missing categories: {sorted(missing)}")
            return 1
        if extra:
            print(f"ERROR: {label} mapping names categories TACO does not have: {sorted(extra)}")
            return 1

    material_stats = summarise(categories, MATERIAL)
    object_stats = summarise(categories, OBJECT)

    # ------------------------------------------------------------- decision table
    rows = []
    for row in categories:
        name = row["name"]
        m_target, m_reason, m_amb = MATERIAL[name]
        o_target, o_reason, o_amb = OBJECT[name]
        rows.append({
            "category_id": row["category_id"],
            "original_category": name,
            "supercategory": row["supercategory"],
            "annotations": row["annotations"],
            "images": row["unique_images"],
            "material_mapping": m_target,
            "object_mapping": o_target,
            "decision_rationale": m_reason,
            "ambiguity": m_amb,
            "object_rationale": o_reason,
        })

    with open(TAXONOMY / "class_mapping_candidates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Class mapping candidates",
        "",
        "Every one of TACO's 60 categories appears below with an explicit decision in",
        "both candidate mappings. No category is silently unmapped.",
        "",
        "Values: a target class name, `EXCLUDE` (deliberately dropped), or `REVIEW`",
        "(composite material — needs a decision the label cannot make for us).",
        "",
        "## Candidate A — material-oriented",
        "",
        f"Target classes: {', '.join('`' + c + '`' for c in MATERIAL_CLASSES)}",
        "",
        "| Class | Annotations |",
        "|---|---:|",
    ]
    for k, v in material_stats["class_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        f"| *(excluded)* | {material_stats['excluded_annotations']} |",
        f"| *(review — composite)* | {material_stats['review_annotations']} |",
        "",
        f"Included: **{material_stats['included_annotations']}** of "
        f"{sum(c['annotations'] for c in categories)} annotations "
        f"({100 * material_stats['included_annotations'] / sum(c['annotations'] for c in categories):.1f}%). "
        f"Imbalance ratio {material_stats['imbalance_ratio']}×.",
        "",
        "## Candidate B — object-oriented",
        "",
        f"Target classes: {', '.join('`' + c + '`' for c in OBJECT_CLASSES)}",
        "",
        "| Class | Annotations |",
        "|---|---:|",
    ]
    for k, v in object_stats["class_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Included: **{object_stats['included_annotations']}** annotations "
        f"({100 * object_stats['included_annotations'] / sum(c['annotations'] for c in categories):.1f}%). "
        f"Imbalance ratio {object_stats['imbalance_ratio']}×.",
        "",
        "## Full decision table",
        "",
        "| ID | Original category | Anns | Material | Object | Rationale (material) | Ambiguity |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category_id']} | {row['original_category']} | {row['annotations']} "
            f"| `{row['material_mapping']}` | `{row['object_mapping']}` "
            f"| {row['decision_rationale']} | {row['ambiguity']} |"
        )
    (TAXONOMY / "class_mapping_candidates.md").write_text("\n".join(lines) + "\n")

    (TAXONOMY / "official_mapping_analysis.md").write_text(
        official_mapping_analysis(categories))

    stats = {"material": material_stats, "object": object_stats}
    (TAXONOMY / "mapping_candidate_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    rec = emit_recommended(categories)
    print("RECOMMENDED (C):", rec["counts"])
    print("  included", rec["included"], "excluded", rec["excluded"])
    print("Candidate A (material):", material_stats["class_counts"])
    print("  included", material_stats["included_annotations"],
          "excluded", material_stats["excluded_annotations"],
          "review", material_stats["review_annotations"],
          "imbalance", material_stats["imbalance_ratio"])
    print("Candidate B (object)  :", object_stats["class_counts"])
    print("  included", object_stats["included_annotations"],
          "imbalance", object_stats["imbalance_ratio"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
