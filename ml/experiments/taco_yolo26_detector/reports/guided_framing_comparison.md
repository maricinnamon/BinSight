# Guided-framing diagnostic — does BinSight's UX solve the small-object problem?

> **PRODUCT DIAGNOSTIC, not test performance.** Only the full-frame row is a
> test-set result. The ROI rows answer a different question — how the same
> weights behave when the camera frames one item — and must never be quoted as
> standard model metrics. No weights were changed and nothing was trained.

Model: `runs/yolo26n_run_002_full_taco_640/weights/best.pt`, imgsz 640, conf 0.25.
Test set: 166 untouched images, 405 objects (253 tiny, 80 small, 55 medium, 17 large).

## Results

| Mode | Recall | Precision | mAP50 | tiny recall | small recall |
|---|---:|---:|---:|---:|---:|
| Full frame | 0.1531 | 0.4697 | 0.1233 | 0.0988 | 0.2250 |
| Object-guided ROI | 0.1481 | 0.3581 | 0.1847 | 0.1067 | 0.2250 |
| Central 90% | 0.1679 | 0.4416 | 0.1337 | 0.1067 | 0.2875 |
| **Central 75%** | **0.1778** | 0.4235 | 0.1332 | 0.1067 | **0.3125** |
| Central 60% | 0.1432 | 0.4234 | 0.1219 | 0.1067 | 0.2000 |

Absolute change against full frame:

| Mode | Δ Recall | Δ Precision | Δ tiny recall | Δ small recall |
|---|---:|---:|---:|---:|
| Object-guided ROI | -0.0049 | -0.1116 | +0.0079 | +0.0000 |
| Central 90% | +0.0148 | -0.0281 | +0.0079 | +0.0625 |
| **Central 75%** | **+0.0247** | -0.0462 | +0.0079 | **+0.0875** |
| Central 60% | -0.0099 | -0.0463 | +0.0079 | -0.0250 |

## How these were measured, and one correction

Recall is scored **per ground-truth object over all 405 objects** in every row, so
the denominators are identical and the rows are comparable.

That correction matters. The diagnostic's first pass reported object-guided recall
of 0.2741, because it counted true positives against *any* object inside each ROI
while dividing by the target count alone. Recomputed per target — one row per
object, "was this object found in this framing" — guided ROI recall is
**0.1481**, slightly *below* full frame. The inflated figure has been
discarded; `corrected_metrics.json` holds the consistent numbers.

For the central ROIs, objects falling outside the crop are counted as **misses**,
not excluded. The app would never see them, so excluding them would flatter the
result: at 75% that is 53 of 405 objects, at 60% it is 90.

## Object-guided ROI does not work

Following each object through both framings — same object, same weights:

| | objects |
|---|---:|
| found in both | 22 |
| **rescued** by guided ROI | 38 |
| **lost** by guided ROI | 40 |
| missed in both | 305 |

Tight framing rescues 38 objects and loses 40, a net change of
-2. It is a reshuffle, not an improvement — and this is the *best
case*, an oracle ROI centred on a known object with a median 22x area gain. A real
user cannot aim better than this.

Why losses happen at all: the model was trained on wide litter scenes, so cropping
away context moves the input off the distribution it learned. Enlarging the object
and destroying its surroundings roughly cancel out.

## Tiny objects are unmoved by any framing

This is the finding that decides the question.

| Mode | tiny recall |
|---|---:|
| Full frame | 0.0988 |
| Object-guided ROI | 0.1067 |
| Central 90% / 75% / 60% | 0.1067 / 0.1067 / 0.1067 |

Every framing lands between 0.099 and 0.107. Tiny objects are **62% of the
test set** (253 of 405), and roughly nine in ten remain undetected no matter how
the frame is composed. Cropping cannot manufacture detail that was never captured:
a cigarette occupying 78 px² in the original photograph is still a smear of pixels
after any crop.

## Recommendation for the iOS app

**Central 75% ROI.** It is the best of the fixed options and the only one whose
gain is real rather than noise:

- recall 0.1531 -> 0.1778 (+16% relative)
- small-object recall 0.2250 -> 0.3125 (+39% relative)
- precision 0.4697 -> 0.4235, a modest and acceptable cost
- 53 of 405 objects fall outside it, against 90 at 60%

90% is too timid to change anything; 60% discards 90 objects and recall falls
below full frame. 75% sits at the turning point.

**Do not build an object-guided crop pipeline.** The oracle version of it does not
beat full frame, so an implementation driven by a user's aim will not either.

## Verdict

**Guided framing does not solve the small-object problem.** It delivers a modest,
genuine improvement for *small* objects — small recall 0.2250 -> 0.3125 at central
75% — and essentially nothing for *tiny* ones, which are the majority of the data
and the whole difficulty.

The limiting factor is **training data and domain mismatch**, not framing and not
hyperparameters. TACO is litter photographed from standing height across whole
scenes; BinSight users hold a phone close to a single item. Cropping a TACO image
does not turn it into a BinSight photograph — it produces an off-distribution
image of an object that was never captured in enough detail.

Further improvement requires **real phone-camera photographs of single items in
the intended use case**. A few hundred such images would very likely be worth more
than any further training on TACO. No additional hyperparameter run is proposed.
