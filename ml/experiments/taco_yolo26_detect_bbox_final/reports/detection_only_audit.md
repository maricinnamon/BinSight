# Detection-only audit — **PASS**

Every line below was verified at runtime, not asserted in prose.

| Claim | Verified value |
|---|---|
| model | `yolo26n.pt` |
| model class | `DetectionModel` |
| is a DetectionModel | **True** |
| head class | `Detect` |
| head is Detect | **True** |
| head is Segment | **False** |
| task | **`detect`** |
| loss function | `E2ELoss` |
| mask prototype branch present | **False** |
| COCO field used for targets | **`bbox`** |
| COCO `segmentation` used for targets | **false** |
| YOLO label fields per line | **[5]** |
| label files | 1082 |
| mask labels generated | **False** |
| segmentation model loaded | **False** |
| segmentation refs in pipeline code | **0** |

## How the bbox-only rule is enforced

`src/bbox_only.py` whitelists the COCO annotation keys it will return (`id`, `image_id`, `category_id`, `bbox`, `area`) and drops the rest at load time. `segmentation` is therefore not merely unused — it is unreachable, and `assert_no_segmentation_keys()` fails the build if one ever survives.

An invalid or missing `bbox` excludes the annotation and is recorded in `manifests/excluded_annotations.csv`. It is never reconstructed from the polygon: a mask-derived box is a different annotation, and substituting one would make the dataset a silent mixture of two sources.

## On the source dataset

TACO's annotations.json does contain a 'segmentation' field — it is a segmentation dataset. That is not a failure. src/bbox_only.py strips every mask-bearing key at load time, so the field is unreachable from this pipeline and cannot become training data.

