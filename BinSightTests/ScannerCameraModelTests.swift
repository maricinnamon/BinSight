import Testing
@testable import BinSight

/// Lifecycle logic that can be exercised without hardware.
///
/// These use `ScannerCameraModel.preview(_:)`, which never constructs a capture
/// session and never asks for permission — so running the suite cannot trigger
/// a system prompt or touch the camera.
@MainActor
struct ScannerCameraModelTests {

    @Test("A preview model never drives hardware")
    func previewModelIsNotLive() {
        let model = ScannerCameraModel.preview(.running)
        #expect(!model.usesLiveCapture)
        #expect(model.captureSession == nil)
    }

    @Test("Activating a preview model is a no-op, so tests never prompt")
    func activateDoesNothingWhenNotLive() async {
        let model = ScannerCameraModel.preview(.idle)
        await model.activate()
        // Still idle: no permission request, no session start.
        #expect(model.state == .idle)
    }

    @Test("Deactivating a preview model leaves its pinned state alone")
    func deactivateDoesNothingWhenNotLive() {
        let model = ScannerCameraModel.preview(.running)
        model.deactivate()
        #expect(model.state == .running)
    }

    @Test("Retrying a preview model is inert")
    func retryDoesNothingWhenNotLive() async {
        let model = ScannerCameraModel.preview(.interrupted)
        await model.retry()
        #expect(model.state == .interrupted)
    }

    @Test("Each pinned camera state surfaces the matching scanner reason")
    func pinnedStatesSurfaceTheRightReason() {
        #expect(ScannerCameraModel.preview(.denied).unavailableReason == .permissionDenied)
        #expect(ScannerCameraModel.preview(.restricted).unavailableReason == .permissionRestricted)
        #expect(ScannerCameraModel.preview(.unavailable).unavailableReason == .cameraUnavailable)
        #expect(ScannerCameraModel.preview(.interrupted).unavailableReason == .interrupted)
        #expect(ScannerCameraModel.preview(.failed).unavailableReason == .captureFailed)
        #expect(ScannerCameraModel.preview(.running).unavailableReason == nil)
        #expect(ScannerCameraModel.preview(.idle).unavailableReason == nil)
    }
}

/// Recovery affordances and the UI-test hooks that expose them.
@MainActor
struct ScannerUnavailableReasonTests {

    @Test("Every reason has a title, a message, and a UI-test identifier")
    func everyReasonIsFullyDescribed() {
        for reason in ScannerUnavailableReason.allCases {
            #expect(!reason.title.isEmpty)
            #expect(!reason.message.isEmpty)
            #expect(!reason.accessibilityIdentifier.isEmpty)
        }
    }

    @Test("Identifiers are unique, so a UI test can target one state exactly")
    func identifiersAreUnique() {
        let identifiers = ScannerUnavailableReason.allCases.map(\.accessibilityIdentifier)
        #expect(Set(identifiers).count == identifiers.count)
    }

    @Test("The UI-test hooks named in the phase brief all exist")
    func requiredIdentifiersExist() {
        #expect(ScannerUnavailableReason.permissionDenied.accessibilityIdentifier == "camera.denied")
        #expect(ScannerUnavailableReason.cameraUnavailable.accessibilityIdentifier == "camera.unavailable")
    }

    @Test("Recovery is offered only where it can actually achieve something")
    func recoveryOfferedOnlyWhereUseful() {
        // Denied: iOS will not prompt again, so the only route is Settings.
        #expect(ScannerUnavailableReason.permissionDenied.recovery == .openSettings)
        // Transient: retrying is exactly right.
        #expect(ScannerUnavailableReason.interrupted.recovery == .retry)
        #expect(ScannerUnavailableReason.captureFailed.recovery == .retry)
        // Nothing the person can do from inside the app.
        #expect(ScannerUnavailableReason.cameraUnavailable.recovery == nil)
        #expect(ScannerUnavailableReason.permissionRestricted.recovery == nil)
    }

    @Test("Recovery actions have button titles")
    func recoveryTitles() {
        #expect(!ScannerUnavailableReason.Recovery.retry.title.isEmpty)
        #expect(!ScannerUnavailableReason.Recovery.openSettings.title.isEmpty)
    }
}
