# Licensing

Short version: **BinSight is AGPL-3.0, and it did not choose to be.** The licence
is inherited from the detection model's weights, and it has a real consequence
for App Store distribution that is spelled out below rather than buried.

## Why AGPL-3.0

The detector is a fine-tune of Ultralytics' COCO-pretrained `yolo26n.pt`.
Ultralytics distributes both its code and those weights under **AGPL-3.0**, and
the obligation follows anything derived from them:

| Evidence | Value |
|---|---|
| `ultralytics` package metadata | `License: AGPL-3.0` |
| Exported CoreML package metadata | `AGPL-3.0 License (https://ultralytics.com/license)`, `author: Ultralytics` |
| Our release model | `models/pytorch/BinSightYOLO26n.pt`, fine-tuned from `yolo26n.pt` |

This was verified from the installed package, not assumed.

Because the bundled model is a derived work, the app that ships it is a derived
work too. `LICENSE` in this repository is therefore the AGPL-3.0 text, copied
verbatim from the Ultralytics distribution.

## What that means in practice

**For this repository — nothing awkward.** The source is public, the licence
text accompanies it, and AGPL is satisfied. This is a portfolio project on
GitHub and that is exactly the distribution AGPL was written for.

**For the App Store — a genuine obstacle.** GPL-family licences forbid a
distributor from adding restrictions beyond the licence (AGPL-3.0 §10). Apple's
App Store terms impose such restrictions — device limits and DRM among them —
which is why GPL-licensed apps have been removed from the store before. Being
the app's author does not resolve it: the constraint comes from Ultralytics'
copyright in the base weights, not from ours.

## Decision taken: route 3

**BinSight is not published to the App Store.** It is an open-source portfolio
project, installed from source via Xcode. Under that distribution AGPL-3.0 is
fully satisfied — public source accompanied by the licence text is exactly what
the licence asks for — and the incompatibility described above never arises.

Nothing is outstanding. The alternatives below are recorded for the future, not
as pending work.

**1 — Buy an Ultralytics Enterprise Licence.** This is precisely what Ultralytics
sells it for: it lifts the AGPL obligation for commercial and closed
distribution. Cost is a commercial matter between the developer and Ultralytics.
Nothing in this repository changes.

**2 — Retrain on a permissively licensed architecture.** The *dataset* is CC BY
4.0 and poses no problem at all; only the pretrained weights are encumbered.
Replacing the backbone with an Apache-2.0 or BSD detector and re-training would
produce an unencumbered model. This is real work — retraining, re-export,
re-verification of parity — and accuracy would have to be re-measured, not
assumed to carry over. Every script needed to do it is in `scripts/`.

**3 — Do not publish to the App Store.** ← **chosen.** Keep BinSight as an
open-source portfolio project, installed via Xcode or TestFlight. The AGPL
question does not arise, and nothing else about the app changes.

## Where this leaves the project

Compliant and complete. The engineering for App Store submission was finished
before the decision was taken, and `docs/app-store/` retains the full
preparation — metadata, privacy answers, age rating, release checklist — so the
route stays open if the licensing question is ever resolved differently.

Keeping that work is deliberate: it is part of what the project demonstrates.

## Third-party attribution

The dataset, the model and the tooling each carry their own terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
