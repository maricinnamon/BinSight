import Foundation

/// Why the scanner cannot show a preview.
enum ScannerUnavailableReason: String, CaseIterable, Equatable, Sendable {
    case permissionDenied
    case permissionRestricted
    case cameraUnavailable
    case interrupted
    case captureFailed
    /// The classifier could not be loaded — no bundled model, bad labels, or a
    /// tensor contract that does not match. A build problem, not a user one.
    case modelUnavailable

    /// What the person can do about it, if anything.
    enum Recovery: Equatable, Sendable {
        case retry
        case openSettings

        var title: String {
            switch self {
            case .retry: L("recovery.retry.title")
            case .openSettings: L("recovery.openSettings.title")
            }
        }
    }

    var title: String {
        switch self {
        case .permissionDenied: L("unavailable.permissionDenied.title")
        case .permissionRestricted: L("unavailable.permissionRestricted.title")
        case .cameraUnavailable: L("unavailable.cameraUnavailable.title")
        case .interrupted: L("unavailable.interrupted.title")
        case .captureFailed: L("unavailable.captureFailed.title")
        case .modelUnavailable: L("unavailable.modelUnavailable.title")
        }
    }

    var message: String {
        switch self {
        case .permissionDenied: L("unavailable.permissionDenied.message")
        case .permissionRestricted: L("unavailable.permissionRestricted.message")
        case .cameraUnavailable: L("unavailable.cameraUnavailable.message")
        case .interrupted: L("unavailable.interrupted.message")
        case .captureFailed: L("unavailable.captureFailed.message")
        case .modelUnavailable: L("unavailable.modelUnavailable.message")
        }
    }

    var recovery: Recovery? {
        switch self {
        // Re-asking is pointless once denied — iOS will not prompt twice.
        case .permissionDenied: .openSettings
        // A restriction is not the user's to lift from here.
        case .permissionRestricted: nil
        // No hardware; retrying will not conjure any.
        case .cameraUnavailable: nil
        case .interrupted, .captureFailed: .retry
        // Retrying will not put a model in the bundle.
        case .modelUnavailable: nil
        }
    }

    /// Stable hook for UI tests.
    var accessibilityIdentifier: String {
        switch self {
        case .permissionDenied: "camera.denied"
        case .permissionRestricted: "camera.restricted"
        case .cameraUnavailable: "camera.unavailable"
        case .interrupted: "camera.interrupted"
        case .captureFailed: "camera.failed"
        case .modelUnavailable: "model.unavailable"
        }
    }
}

/// Everything the scanner screen can be showing.
///
/// The copy lives on the state rather than in the view so each state reads as
/// one thing, and so the wording is testable without rendering anything.
enum ScannerDisplayState: Equatable, Sendable {
    case ready
    /// The classifier is being loaded and allocated. Distinct from `scanning`:
    /// nothing is being looked at yet.
    case modelLoading
    case scanning
    /// A prediction the app is willing to name.
    case result(ClassificationResult)
    /// A prediction too weak to name.
    case notSure
    case unavailable(ScannerUnavailableReason)

    /// Routes a raw prediction to `.result` or `.notSure`.
    ///
    /// The threshold is applied here, once, so no view ever has to decide
    /// whether a number is good enough to show.
    static func classified(_ result: ClassificationResult) -> ScannerDisplayState {
        result.isConfident ? .result(result) : .notSure
    }

    /// Shown verbatim whenever confidence falls under the threshold.
    ///
    /// Computed rather than `static let`: the language can change while the app
    /// is running, and a stored constant would freeze the string at whatever
    /// language was active when the type was first touched.
    static var notSureMessage: String { L("state.notSure.message") }

    var title: String {
        switch self {
        case .ready: L("state.ready.title")
        case .modelLoading: L("state.modelLoading.title")
        case .scanning: L("state.scanning.title")
        case .result: L("state.result.title")
        case .notSure: L("state.notSure.title")
        case .unavailable(let reason): reason.title
        }
    }

    /// Secondary line under the title. `nil` where the result itself carries
    /// the detail.
    var supportingText: String? {
        switch self {
        case .ready: L("state.ready.supporting")
        case .modelLoading: L("state.modelLoading.supporting")
        case .scanning: L("state.scanning.supporting")
        case .result: nil
        // Actionable, and deliberately generic - the three things that most
        // often fix a low-confidence reading on a handheld phone.
        case .notSure: L("state.notSure.supporting")
        case .unavailable(let reason): reason.message
        }
    }

    var result: ClassificationResult? {
        if case .result(let result) = self { return result }
        return nil
    }

    var isScanning: Bool {
        self == .scanning
    }

    /// A percentage is only ever shown when there is a result behind it.
    var showsConfidence: Bool {
        result != nil
    }
}
