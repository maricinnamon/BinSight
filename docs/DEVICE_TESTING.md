# BinSight — physical device testing

> ## Status: FUNCTIONAL CHECK PASSED — quantitative measurements still pending
>
> A signed Debug build was run on an **iPhone 15** and the detection path was
> confirmed working by direct observation. **No timing or accuracy has been
> measured on device**, so every number below is still a blank.
>
> The two blockers that previously prevented a session are both resolved:
>
> 1. ~~There is no model.~~ **Resolved.** `BinSightYOLO26n.mlpackage` is bundled
>    in the app target and compiles to `BinSightYOLO26n.mlmodelc`. The detection
>    path is `LiveDetectionEngine` → `YOLODetectionService`; the old LiteRT
>    classifier is retired and no TFLite runtime is linked.
> 2. ~~Device signing is not configured.~~ **Resolved.** `DEVELOPMENT_TEAM` is
>    set and automatic provisioning signs the target.
>
> Latency in this file must be measured on the device. The Simulator figures in
> the README (58–136 ms model, 116–191 ms end-to-end) are Simulator figures and
> may not be copied here.
>
> ### Functional check — iPhone 15, signed Debug build
>
> | Check | Result |
> |---|---|
> | Live preview opens, camera permission granted | **PASS** |
> | Detections appear | **PASS** |
> | Boxes align with the objects | **PASS** |
> | `paper` detected and labelled correctly | **PASS** |
> | `plastic` detected and labelled correctly | **PASS** |
> | `metal` detected and labelled correctly | **PASS** |
> | Multiple objects shown with separate boxes | **PASS** |
> | Boxes stay reasonably aligned near the preview edges | **PASS** |
>
> This is a **qualitative** result, reported by the person holding the phone. It
> establishes that the pipeline is correct end to end — letterbox, decode,
> un-letterbox and preview projection all agree with what the camera sees — but
> it is not a measurement. Nothing here should be quoted as device accuracy.
>
> ### Still outstanding
>
> - on-device inference latency and sustained frame rate
> - thermal behaviour over a long session
> - battery drain
> - behaviour in poor light and at distance

---

## Session record

Fill one block per session. Do not merge sessions — thermal state and lighting
differ, and averaging across them hides both.

| Field | Value |
|---|---|
| Date | `PENDING` |
| Device model | `PENDING` |
| iOS version | `PENDING` |
| Build configuration | `PENDING` (Release for latency; Debug distorts it) |
| Model file | `PENDING` |
| Model SHA256 | `PENDING` (from `ml/artifacts/checksums.sha256`) |
| Model size | `PENDING` |
| Quantisation | `PENDING` |
| LiteRT runtime version | `PENDING` |
| `ScannerConfiguration` used | `PENDING` (record any deviation from defaults) |
| Session length | `PENDING` |
| Ambient conditions | `PENDING` |

## Latency — end-to-end, on device

Read from the DEBUG performance overlay, or from `os_signpost` traces. Record
preprocessing and interpreter invocation separately: if the total is
disappointing, which half is at fault changes what to do about it.

| Metric | Value |
|---|---|
| Preprocessing, median | `PENDING` |
| Interpreter invocation, median | `PENDING` |
| End-to-end, average | `PENDING` |
| End-to-end, median | `PENDING` |
| End-to-end, P95 | `PENDING` |
| Processed classifications/s | `PENDING` |
| Dropped frames | `PENDING` |
| Thermal state at end | `PENDING` |

CPU only — no Core ML or GPU delegate. That is deliberate: establish a correct
CPU baseline first, then measure whether a delegate is worth its complexity.

## Per-class behaviour

One row per class actually tested. **Leave a row blank rather than inventing a
sample** — a class with no representative item to hand is untested, not passing.

| Class | Item used | Recognised? | Typical smoothed confidence | Stability | Notes |
|---|---|---|---|---|---|
| cardboard | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |
| glass | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |
| metal | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |
| paper | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |
| plastic | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |
| general_waste | `PENDING` | `PENDING` | `PENDING` | `PENDING` | |

## Hard scenes

These are where the TrashNet domain shift shows up. Expect worse results than
the test-split accuracy, and record what actually happens rather than what the
metrics predicted.

| Scenario | Observed | Notes |
|---|---|---|
| Cluttered background (several items in frame) | `PENDING` | |
| Low light / indoor evening | `PENDING` | |
| Strong backlight or glare | `PENDING` | |
| Partial object (cropped by the reticle) | `PENDING` | |
| Ambiguous composite (coffee cup, crisp packet) | `PENDING` | |
| Fast pan between two different items | `PENDING` | |
| Empty scene / no object | `PENDING` | should stay "not sure" |

## Stability and smoothing

The provisional thresholds live in `ScannerConfiguration` and are **not
calibrated**. This section is the evidence for changing them.

| Question | Observed |
|---|---|
| Does the label flicker between categories? | `PENDING` |
| Time from pointing to a stable label | `PENDING` |
| Does it hold a stale label after moving to a new item? | `PENDING` |
| False confident readings on ambiguous items | `PENDING` |
| Does "not sure" appear when it should? | `PENDING` |
| Proposed threshold changes, with reasoning | `PENDING` |

## Lifecycle and privacy

Carried over from the Phase 4 camera checklist — re-verify with inference running.

| Check | Result |
|---|---|
| First launch shows the camera permission prompt once | `PENDING` |
| Live preview fills the screen, no distortion | `PENDING` |
| Deny → "Camera access needed" + Open Settings | `PENDING` |
| Settings recovery restores the preview | `PENDING` |
| Background/foreground: inference stops and restarts | `PENDING` |
| Phone call / Control Centre: "Camera paused", then resumes | `PENDING` |
| Rotation keeps the image upright | `PENDING` |
| Sustained use: thermal state and battery drain | `PENDING` |
| **Photos gains no images** | `PENDING` |
| **No files written** | `PENDING` |
| **Settings ▸ Cellular shows no data for BinSight** | `PENDING` |

The last three back the on-screen claim "Frames stay on this iPhone." The app
has no networking code, no analytics and no image persistence; this is the
check that keeps that true.

## How to run a session

1. Build **Release** to a signed device — Debug's unoptimised build and the
   performance overlay both distort latency.
2. Open the scanner and let it run ~60 s per scenario so the latency ring
   (60 samples) fills.
3. Read the overlay in a Debug build for the numbers; use Release for the
   qualitative stability and thermal observations.
4. Record everything above, then update `docs/MODEL_CARD.md` if real-world
   behaviour contradicts the test-split metrics.
