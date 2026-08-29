# Screenshot plan

## Required sizes

Apple accepts a single set for modern iPhones and scales down, but two are still
worth capturing:

| Display | Portrait resolution | Devices |
|---|---|---|
| 6.9" | 1320 × 2868 | iPhone 17 Pro Max, 16 Pro Max |
| 6.5" | 1242 × 2688 | fallback for older listings |

Portrait only — the app is portrait-locked.

iPad screenshots are required **only if** the app remains iPad-compatible
(`TARGETED_DEVICE_FAMILY = 1,2`). If you would rather not photograph waste on an
iPad, set the family to iPhone-only before submitting; that is a one-line change
and removes the requirement.

## The blocker

**Screenshots cannot be captured on a Simulator.** It has no camera, so every
screen shows the "Camera unavailable" state. Every shot below needs a physical
iPhone.

## Five shots, in order

Apple shows the first two in search results, so they carry the most weight.

**1 — The core moment.** A plastic bottle on a plain surface, box drawn, label
reading `Plastic 92%`. This is the whole product in one image. Shoot in daylight
against a matte, uncluttered background; the model was trained on studio-like
photography and behaves best there.

**2 — A second material.** An aluminium can, `Metal 87%`. Establishes that it is
not a one-trick app and shows the accent colour changing.

**3 — Paper.** A sheet or flattened card, `Paper 78%`. Deliberately the weakest
class — do not stage an unrealistically perfect example.

**4 — The result surface.** Close on the bottom card showing the material,
confidence, and the handling advice with its "Generic guidance · local rules
vary" caveat visible. This is where the app's honesty is legible.

**5 — Privacy.** The full screen with the "ON-DEVICE AI" chip and "Frames stay on
this iPhone." both in frame. For a camera app, this is the objection being
answered.

## Localized sets

Capture all five twice — English and Ukrainian — using the in-app language
switch. Do not translate a screenshot in Photoshop; take the real one.

## Rules

- Never fake a detection or edit a confidence number. A screenshot that shows
  `98%` on an object the model would rate `61%` is a misrepresentation, and it
  sets an expectation the app cannot meet.
- Leave the status bar as it is. Do not clean it up in an editor.
- No text overlays or device frames for the first release; the screens are
  legible on their own.
- Capture with the debug overlay off — build in Release, or use a Debug build
  and confirm the overlay is not on screen.

## Capture command

With the device connected:

```bash
xcrun devicectl device screenshot --device <UDID> shot.png
```

Or use Xcode → Window → Devices and Simulators → Take Screenshot.
