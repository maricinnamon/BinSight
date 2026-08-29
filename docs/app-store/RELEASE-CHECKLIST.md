# Release checklist

> ## Status: prepared, not being submitted
>
> BinSight ships as an **open-source portfolio project**, not through the App
> Store. That decision resolves the licensing question — see
> [`LICENSING.md`](../../LICENSING.md) — because AGPL-3.0 is satisfied outright
> by public source plus the licence text.
>
> Everything below was completed and verified before the decision was taken, and
> is kept so the route remains open. Nothing here is outstanding work.

State as of the last verification run. Everything marked ✅ was checked, not
assumed; the command or evidence is given.

## ⚠️ Read this first — the licensing position

**BinSight is AGPL-3.0, inherited from the model's weights**, and that has a real
consequence for the App Store. The repository now carries the full licence text
in `LICENSE`, and `LICENSING.md` sets out the position in detail. Verified from
the installed package, not assumed: `ultralytics` declares `License: AGPL-3.0`,
and the exporter stamps `AGPL-3.0 License` with `author: Ultralytics` into the
CoreML package metadata.

AGPL-3.0 §10 forbids a distributor adding restrictions beyond the licence, and
Apple's App Store terms impose exactly such restrictions. GPL-licensed apps have
been removed from the store on those grounds. Being the app's author does not
resolve it — the constraint comes from Ultralytics' copyright in the base
weights.

**This is a purchase or a decision, not an engineering task.** There is no code
change that removes it. The three routes are in `LICENSING.md`:

1. Buy an Ultralytics Enterprise Licence and ship as-is
2. Retrain on a permissively licensed architecture (the dataset is CC BY 4.0 and
   is not the problem; every script needed is in `scripts/`)
3. Do not publish to the App Store — keep BinSight open-source and installed via
   Xcode or TestFlight

Route 1 puts this in the same category as the Apple Developer membership: a paid
account-level action outside the codebase. Route 2 is genuine ML work and would
require re-measuring accuracy rather than assuming it carries over.

---

## Technical readiness

| Item | State | Evidence |
|---|---|---|
| Debug build | ✅ | `** BUILD SUCCEEDED **` |
| Release build | ✅ | `** BUILD SUCCEEDED **`, 0 errors, 0 app-code warnings |
| Test suite | ✅ | 188 tests / 21 suites passing |
| No debug code in Release | ✅ | `DEBUG` undefined in Release; 0 debug strings found in the Release binary |
| Release optimisation | ✅ | `SWIFT_COMPILATION_MODE = wholemodule`, `DEAD_CODE_STRIPPING = YES`, `STRIP_INSTALLED_PRODUCT = YES`, `VALIDATE_PRODUCT = YES` |
| dSYM generation | ✅ | `DEBUG_INFORMATION_FORMAT = dwarf-with-dsym` |
| App icon | ✅ | 1024×1024 RGB, no alpha, no manual rounding; verified on the Home Screen |
| CoreML model bundled | ✅ | `BinSightYOLO26n.mlmodelc`, 4.8 MB, present in both simulator and signed device builds |
| Privacy manifest | ✅ | `PrivacyInfo.xcprivacy` in the bundle, declaring `SystemBootTime` (35F9.1) and `UserDefaults` (CA92.1) |
| Localization | ✅ | `en.lproj` and `uk.lproj` built, 57 keys each, plurals compiled to `.stringsdict` |
| Camera usage string | ✅ | Localized via `InfoPlist.xcstrings` in both languages |
| Orientation | ✅ | Locked to portrait, matching the camera pipeline's actual assumption |
| Encryption declaration | ⬜ | Set `ITSAppUsesNonExemptEncryption = NO` in Info.plist or answer in App Store Connect |
| Bundle identifier | ✅ | `com.marynaantonevych.BinSight` — unchanged |
| Licence text present | ✅ | `LICENSE` (AGPL-3.0, 661 lines, verbatim from the Ultralytics distribution) and `LICENSING.md` |
| Version / build | ⬜ | Currently `1.0 (1)`. See the versioning note below. |

## Deferred — needs a paid Apple Developer membership

None of these can be done or tested without the paid programme. They are the
*only* steps in that category.

| Step | Note |
|---|---|
| Distribution signing | Only Apple Development signing exists. An Apple Distribution certificate requires the paid programme. |
| App Store Connect record | Cannot be created without membership. |
| TestFlight | Same. |
| Archive + upload validation | `xcodebuild archive -exportOptionsPlist` with `method: app-store` needs a distribution profile. |
| App Store screenshots | Need a **physical iPhone pointed at real objects** — the Simulator has no camera, so every screen there shows "Camera unavailable". Faking them with the DEBUG pinned states would misrepresent the app. Plan in `SCREENSHOT-PLAN.md`. |

## Before the first submission

- [ ] Choose a licensing route (see `LICENSING.md`) — this gates App Store distribution
- [ ] Publish `PRIVACY.md` at an HTTPS URL and put it in App Store Connect
- [ ] Provide a Support URL
- [ ] Capture screenshots for 6.9" and 6.5" iPhone (see `SCREENSHOT-PLAN.md`)
- [ ] Answer the Privacy questionnaire as "no data collected" (`DATA-COLLECTION.md`)
- [ ] Answer the age-rating questionnaire (`AGE-RATING.md`) → expect 4+
- [ ] Set `ITSAppUsesNonExemptEncryption = NO`
- [ ] Write review notes explaining that the app needs a physical device: the
      Simulator has no camera, so a reviewer on a Simulator sees only the
      "Camera unavailable" state

## Versioning

`MARKETING_VERSION` 1.0, `CURRENT_PROJECT_VERSION` 1.

Suggested strategy, and the reason for it:

- `MARKETING_VERSION` — semantic and user-facing. Bump the minor for a new
  material class or a visible feature, the patch for fixes. A model retrain that
  changes accuracy is a minor bump: it changes what the app does.
- `CURRENT_PROJECT_VERSION` — a monotonically increasing integer, incremented on
  every upload to App Store Connect. It never resets, including across marketing
  versions. Apple rejects a build whose number is not higher than the last.

Record the model's SHA-256 in the release notes for any build where it changed,
so a regression can be tied to a specific checkpoint.
