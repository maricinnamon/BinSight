"""COCO → YOLO Detect conversion that reads bounding boxes and nothing else.

TACO ships instance-segmentation polygons alongside every box. BinSight does not
do segmentation: the iOS app needs `x1, y1, x2, y2, class_id, confidence` and
nothing more, and a mask head would cost parameters and latency on a phone for
output the product discards.

So this module reads `annotation["bbox"]` and never touches
`annotation["segmentation"]`. That is not merely a convention here — the loader
below refuses to return the segmentation field at all, so a later edit cannot
quietly start training on polygons. `assert_no_segmentation_keys` makes the same
guarantee testable from outside.

COCO bbox is `[x, y, width, height]` in absolute pixels with the origin at the
top-left. YOLO Detect wants `class_id x_center y_center width height`, every
coordinate normalised to [0, 1], exactly five values per line.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

# The only COCO annotation keys this pipeline is permitted to read.
ALLOWED_ANNOTATION_KEYS = frozenset({"id", "image_id", "category_id", "bbox", "area"})
FORBIDDEN_ANNOTATION_KEYS = frozenset({"segmentation", "counts", "mask", "rle"})

# Boxes may exceed the frame by this many pixels before clipping is recorded.
CLIP_TOLERANCE_PX = 1.0
# Below this, a box is not a learnable target.
MIN_BOX_PX = 1.0

# Historical BinSight buckets — relative area of the frame. Unchanged since
# Run 001 so that numbers stay comparable across every experiment.
SIZE_BUCKETS = (("tiny", 0.0, 0.01), ("small", 0.01, 0.05),
                ("medium", 0.05, 0.20), ("large", 0.20, 1.01))


def bucket_for(relative_area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= relative_area < high:
            return name
    return "large"


@dataclass(frozen=True)
class BoxRecord:
    """One converted annotation. Deliberately has no mask/polygon field."""
    annotation_id: int
    image_id: int
    class_id: int
    class_name: str
    # normalised YOLO
    x_center: float
    y_center: float
    width: float
    height: float
    # the original COCO pixels, kept so the conversion can be reversed and checked
    coco_bbox: tuple[float, float, float, float]
    image_size: tuple[int, int]
    clipped: bool

    def to_label_line(self) -> str:
        """Exactly five values. Never a sixth, never polygon coordinates."""
        return (f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
                f"{self.width:.6f} {self.height:.6f}")

    @property
    def relative_area(self) -> float:
        return self.width * self.height

    @property
    def size_bucket(self) -> str:
        return bucket_for(self.relative_area)


def load_annotations_bbox_only(path: pathlib.Path) -> tuple[dict, list[dict], dict]:
    """Loads COCO, returning annotations stripped of every mask-bearing key.

    The stripping is the point. Callers physically cannot reach `segmentation`
    through this function, so "we ignore polygons" is enforced rather than
    promised.
    """
    raw = json.loads(path.read_text())
    images = {img["id"]: img for img in raw["images"]}
    categories = {c["id"]: c["name"] for c in raw["categories"]}
    annotations = [
        {k: v for k, v in ann.items() if k in ALLOWED_ANNOTATION_KEYS}
        for ann in raw["annotations"]
    ]
    return images, annotations, categories


def assert_no_segmentation_keys(annotations: list[dict]) -> None:
    """Fails loudly if any mask-bearing key survived the load."""
    for ann in annotations:
        leaked = FORBIDDEN_ANNOTATION_KEYS & set(ann)
        if leaked:
            raise AssertionError(
                f"annotation {ann.get('id')} still carries {sorted(leaked)} — "
                "this pipeline must never see segmentation data")


def convert_bbox(ann: dict, image: dict, class_id: int, class_name: str
                 ) -> tuple[BoxRecord | None, str | None]:
    """COCO bbox → YOLO Detect. Returns (record, None) or (None, reason).

    An invalid box is excluded and the reason recorded. It is never repaired
    from the segmentation polygon: a box derived from a mask is a different
    annotation, and silently substituting one would make the dataset a mixture
    of two sources that the audit could not distinguish.
    """
    bbox = ann.get("bbox")
    if bbox is None:
        return None, "bbox_missing"
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, "bbox_malformed"

    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None, "bbox_non_numeric"
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None, "bbox_non_finite"
    if w <= 0 or h <= 0:
        return None, "bbox_non_positive_size"

    image_w, image_h = float(image["width"]), float(image["height"])
    if image_w <= 0 or image_h <= 0:
        return None, "image_size_invalid"

    x0, y0, x1, y1 = x, y, x + w, y + h
    needs_clip = (x0 < -CLIP_TOLERANCE_PX or y0 < -CLIP_TOLERANCE_PX
                  or x1 > image_w + CLIP_TOLERANCE_PX or y1 > image_h + CLIP_TOLERANCE_PX)
    cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
    cx1, cy1 = min(x1, image_w), min(y1, image_h)
    if (cx1 - cx0) < MIN_BOX_PX or (cy1 - cy0) < MIN_BOX_PX:
        return None, "bbox_empty_after_clip"

    xc = ((cx0 + cx1) / 2) / image_w
    yc = ((cy0 + cy1) / 2) / image_h
    nw = (cx1 - cx0) / image_w
    nh = (cy1 - cy0) / image_h
    # Float drift can push a value a hair outside range; clamp rather than reject.
    xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
    nw, nh = min(nw, 1.0), min(nh, 1.0)
    if nw <= 0 or nh <= 0:
        return None, "normalised_size_non_positive"

    return BoxRecord(
        annotation_id=int(ann["id"]), image_id=int(ann["image_id"]),
        class_id=class_id, class_name=class_name,
        x_center=xc, y_center=yc, width=nw, height=nh,
        coco_bbox=(x, y, w, h), image_size=(int(image_w), int(image_h)),
        clipped=needs_clip,
    ), None


def yolo_to_pixels(record: BoxRecord) -> tuple[float, float, float, float]:
    """Reverses the conversion, for round-trip verification. Returns COCO xywh."""
    image_w, image_h = record.image_size
    w = record.width * image_w
    h = record.height * image_h
    x = record.x_center * image_w - w / 2
    y = record.y_center * image_h - h / 2
    return x, y, w, h


def parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    """Parses one YOLO Detect line, rejecting anything that is not five values.

    A polygon-style YOLO-seg line (`class x1 y1 x2 y2 ...`) has an odd number of
    trailing floats and will fail here, which is exactly the intent.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected exactly 5 values, got {len(parts)}: {line!r}")
    class_id = int(parts[0])
    xc, yc, w, h = (float(v) for v in parts[1:])
    return class_id, xc, yc, w, h
