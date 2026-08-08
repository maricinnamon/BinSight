## Status

**The app ships without a model.** `ScannerView.makeClassifier()` returns `nil`,
so the camera runs and the scanner reports its `.modelUnavailable` state rather
than inventing labels. Nothing is recognised — that is the intended current
behaviour, not a bug.

The ML experiments that previously lived in `ml/` (a TrashNet TensorFlow
classifier and three TACO YOLO26 detector runs) were removed from the working
tree to reclaim disk. They remain in Git history and on GitHub; nothing is lost.
The last commit containing them is `05eb8ed`.

To restore recognition: implement `WasteClassifying`, return it from
`makeClassifier()`, and add the runtime back to the `Podfile`. The engine, the
prediction smoother and every test above that seam are written against the
protocol and need no changes.

# BinSight

An iOS app that points the camera at a single piece of waste and tells you which bin it
belongs in — entirely on the device.

## Goal

Make the everyday "which bin does this go in?" question answerable in a second, without
sending anything anywhere. BinSight classifies one item at a time from the camera and
shows the predicted category.

## Planned waste classes

BinSight classifies an item into exactly one of six classes:

| Class | |
| --- | --- |
| Cardboard | Boxes, corrugated packaging |
| Glass | Bottles, jars |
| Metal | Cans, foil, tins |
| Paper | Newspaper, office paper, envelopes |
| Plastic | Bottles, tubs, film |
| General waste | Anything that fits none of the above |

## Planned inference

Classification is planned to run on-device with **LiteRT** (formerly TensorFlow Lite),
using an image classification model trained separately under [`ml/`](ml/). No model is
integrated yet — see *Current status*.

## Privacy

Camera frames stay on the device. Because inference runs locally, there is no image
upload, no backend, and no account: the app needs camera access and nothing else. The
current build has no networking code at all.

## Current status

**UI prototype with mock states.** What exists today:

- An Xcode project with a SwiftUI app target and a unit test target.
- The full single-screen experience: mock camera background, scanning reticle, and a
  state-aware result card.
- Five scanner states — ready, scanning, confident result, not sure, and camera
  unavailable — driven by fixed sample values, switchable in DEBUG builds only.
- The six waste classes with their accent colour, SF Symbol and generic guidance.
- A design-system glass surface that uses iOS 26 Liquid Glass where available and falls
  back to a standard material on iOS 17–25.

Plus a complete training pipeline under [`ml/`](ml/README.md) — written and statically
validated, but **not yet run**.

**Every classification shown in the app right now is a hardcoded sample.** There is no
camera capture, no model file, and no LiteRT dependency. No accuracy figures,
benchmarks, or dataset results exist yet, so none are reported here or in the
[model card](docs/MODEL_CARD.md).

## Roadmap

1. ~~**UI prototype** — build out the scanner screen and its states.~~ Done.
2. **Model training** — pipeline written ([`ml/`](ml/README.md)); needs a Colab run to
   produce the model and metrics.
3. **Camera** — AVFoundation capture and live preview.
4. **LiteRT integration** — bundle the converted model and run inference on frames.
5. **Testing** — unit tests for the classification mapping, UI tests for the scanner flow.
6. **Portfolio documentation** — write-up, screenshots, and results under `docs/`.

## Requirements

- Xcode 26 or later
- iOS 17.0 or later

## Build and test

```bash
xcodebuild -project BinSight.xcodeproj -scheme BinSight -destination 'platform=iOS Simulator,name=iPhone 17' build
```

```bash
xcodebuild -project BinSight.xcodeproj -scheme BinSight -destination 'platform=iOS Simulator,name=iPhone 17' test
```

## Repository layout

```
BinSight/            App sources
  App/               Entry point and root view
  Features/Scanner/  The single primary screen
  Core/              Shared model types and design system
  Resources/         Asset catalog
BinSightTests/       Unit tests
ml/                  Machine learning — see ml/README.md
  experiments/       One directory per experiment
    trashnet_tensorflow_baseline/   completed baseline (83.38% accuracy)
docs/                MODEL_CARD.md
```

## Licences

The app target uses only Apple frameworks. Dataset and pretrained-weight
attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
