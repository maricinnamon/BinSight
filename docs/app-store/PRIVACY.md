# BinSight — Privacy Policy

*Last updated: this document tracks the app source; see the repository history
for the exact revision.*

## The whole policy in one paragraph

BinSight looks at what your camera sees, decides whether it is paper, plastic or
metal, and draws a box around it. All of that happens on your iPhone. BinSight
has no servers, makes no network requests, contains no analytics or advertising
code, and stores no photographs. Nothing you point it at is recorded, uploaded,
or shared with anyone — including us.

## What BinSight accesses

**The camera.** BinSight asks for camera access the first time you open it, and
uses the live video feed to recognise materials. Each frame is held in memory
just long enough to run the recognition model, then discarded. No frame is
written to storage, added to your photo library, or transmitted.

If you decline camera access, BinSight will tell you so and offer to open
Settings. It will not ask repeatedly.

## What BinSight stores

One setting: your chosen interface language, if you pick one instead of
following your device. It is stored on your device only.

That is the complete list. There is no account, no profile, no history of what
you scanned, no usage statistics.

## What BinSight sends

Nothing. The app contains no networking code of any kind. This is verifiable —
the source is public, and the app declares no network entitlement.

## Third parties

None. BinSight has no third-party SDKs, no advertising, no analytics, and no
crash reporting. Nobody else receives data from this app because the app does
not send data anywhere.

## Tracking

BinSight does not track you across apps or websites, and does not ask for
permission to do so. Its privacy manifest declares `NSPrivacyTracking = false`
with no tracking domains.

## Children

BinSight collects nothing from anyone, which includes children. There is no
account creation, no messaging, and no user-generated content.

## Your rights

Because BinSight holds no data about you, there is nothing to request, correct,
or delete. Deleting the app removes the one stored preference along with it.

## Machine learning, stated plainly

The recognition model was trained before the app shipped, on a public dataset of
waste photographs. It does **not** learn from what you scan, and your camera
frames are never used to improve it. The model in the app is fixed until the app
is updated.

Accuracy is limited and BinSight can be wrong. On its held-out test set it
correctly identifies about 57% of annotated objects, and about 75% of the boxes
it draws are correct. It recognises only paper, plastic and metal. Treat it as a
hint, not as an authority — and never as a substitute for your local recycling
rules.

## Contact

Raise an issue on the project's GitHub repository.
