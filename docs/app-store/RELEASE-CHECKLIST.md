# Release checklist

State as of the last verification run. Everything marked ✅ was checked, not
assumed; the command or evidence is given.

## ⚠️ Read this first — the one genuine blocker

**The bundled model inherits AGPL-3.0 from Ultralytics, and that is a real
obstacle to App Store distribution — not a paperwork detail.**

`BinSightYOLO26n.mlpackage` is a fine-tune of Ultralytics' COCO-pretrained
`yolo26n.pt`. The exported package records the licence in its own metadata:
`AGPL-3.0 License (https://ultralytics.com/license)`. AGPL-3.0 obligations follow
the derived weights.

Two consequences:

1. **AGPL and the App Store are widely held to be incompatible.** The App Store
   terms impose usage restrictions that GPL-family licences forbid an
   distributor from adding. Apple has removed GPL-licensed apps before on those
   grounds.
2. Ultralytics sells an **Enterprise Licence** precisely to remove this
   obligation for commercial and closed distribution.

Resolving this is a **decision, not a task**, and there are three honest routes:

- Buy an Ultralytics Enterprise Licence, and ship as-is.
- Retrain the detector from a permissively licensed architecture (e.g. an
  Apache-2.0 or BSD detector) and re-export. The dataset itself is CC BY 4.0 and
  poses no problem.
- Do not publish to the App Store; keep BinSight as a portfolio and TestFlight
  project, where the AGPL question is far less fraught.

Nothing else in this checklist is blocked on money or on Apple. This item is.

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
| App Store screenshots | Can be captured now (see `SCREENSHOT-PLAN.md`) but cannot be uploaded. |

## Before the first submission

- [ ] Resolve the AGPL question above — this gates everything else
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
