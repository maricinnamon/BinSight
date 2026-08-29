# Data collection — App Store Connect answers

Every answer below was verified against the source, not assumed. The audit that
produced them searched the whole app target for `URLSession`, `URLRequest`,
`dataTask`, `NWConnection`, analytics and crash-reporting SDKs, `FileManager`,
`NSTemporaryDirectory`, `PHPhotoLibrary` and `UIImageWriteToSavedPhotosAlbum`.

## The short answer

**BinSight collects nothing.** In App Store Connect's Privacy questionnaire, the
correct response to *"Do you or your third-party partners collect data from this
app?"* is **No**.

## Why that answer is defensible

| Question | Answer | Evidence |
|---|---|---|
| Does the app make network requests? | **No** | Zero hits for any networking API across the target. No `.entitlements` file exists, so there is no network entitlement. |
| Third-party SDKs? | **None** | `Podfile` declares the `BinSight` target with no pods. `Podfile.lock` lists none. No Swift Package dependencies (`XCRemoteSwiftPackageReference` count: 0). |
| Analytics / crash reporting? | **None** | No Firebase, Crashlytics, Sentry, or Apple analytics opt-in. |
| Are camera frames stored? | **No** | Frames are read into memory, letterboxed, passed to CoreML, and released. Nothing is written to disk or added to Photos. The Photos framework is never imported. |
| Does the app read the photo library? | **No** | No `NSPhotoLibraryUsageDescription` key; the framework is not linked. |
| Is anything logged that identifies a person? | **No** | Five `Logger` instances, all interpolations marked `privacy: .public`, all values are shapes, feature names or error kinds. No image bytes, no user content. |
| Tracking across apps/websites? | **No** | `NSPrivacyTracking` is `false` and `NSPrivacyTrackingDomains` is empty. |

## What the app *does* store

One value, in `UserDefaults`:

| Key | Value | Purpose |
|---|---|---|
| `binsight.language` | `"en"` or `"uk"` | Remembers the in-app language choice. Absent while following the device language. |

This is a preference, not personal data. It never leaves the device and is not
used to identify anyone. It is declared in `PrivacyInfo.xcprivacy` under
`NSPrivacyAccessedAPICategoryUserDefaults`, reason `CA92.1`.

## Required-reason APIs

A full sweep found exactly two:

| API | Where | Category | Reason |
|---|---|---|---|
| `ProcessInfo.processInfo.systemUptime` | `ScannerView.announceDetections` | `SystemBootTime` | `35F9.1` — measuring elapsed time inside the app, to throttle VoiceOver announcements |
| `UserDefaults` | `LocalizationStore` | `UserDefaults` | `CA92.1` — app-scoped preference |

Not used anywhere: file timestamps, disk space, active keyboards.

## Permissions requested

One. `NSCameraUsageDescription`, localized into English and Ukrainian.

> BinSight uses the camera to recognise paper, plastic and metal in front of you.
> Everything runs on this iPhone — no frame is stored or uploaded.

No microphone, no location, no contacts, no photo library, no notifications, no
`NSUserTrackingUsageDescription`.
