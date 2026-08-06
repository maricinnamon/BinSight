# BinSight — physical device testing

> ## Status: ALL RESULTS PENDING
>
> **No physical-device session has been run.** Every measurement below is a
> blank to be filled in, not a result. Nothing in this file has been observed.
>
> Two things block a session, both carried over from earlier phases:
>
> 1. **There is no model.** Phase 3's notebook has never been executed, so
>    `ml/artifacts/BinSightWasteClassifier.tflite` does not exist and no LiteRT
>    runtime is linked. The app currently reports "Classifier unavailable" by
>    design — see `LiveScanEngine`.
> 2. **Device signing is not configured.** `DEVELOPMENT_TEAM` is unset, so
>    `xcodebuild -destination 'generic/platform=iOS'` fails with *"Signing for
>    'BinSight' requires a development team."*
>
> Until both are resolved, latency and stability cannot be measured, and none
> may be estimated from Simulator behaviour.

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
