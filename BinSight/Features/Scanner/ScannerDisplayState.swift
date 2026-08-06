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
            case .retry: "Try again"
            case .openSettings: "Open Settings"
            }
        }
    }

    var title: String {
        switch self {
        case .permissionDenied: "Camera access needed"
        case .permissionRestricted: "Camera not allowed"
        case .cameraUnavailable: "Camera unavailable"
        case .interrupted: "Camera paused"
        case .captureFailed: "Camera stopped"
        case .modelUnavailable: "Classifier unavailable"
        }
    }

    var message: String {
        switch self {
        case .permissionDenied:
            "BinSight needs the camera to look at an item. Nothing is recorded or uploaded."
        case .permissionRestricted:
            "Camera access is turned off by a restriction on this device, such as Screen Time."
        case .cameraUnavailable:
            "No camera is available on this device right now."
        case .interrupted:
            "Something else is using the camera. It will resume on its own when that finishes."
        case .captureFailed:
            "The camera stopped unexpectedly."
        case .modelUnavailable:
            "The on-device model could not be loaded, so nothing can be classified yet."
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
    static let notSureMessage = "Not sure — try a clearer angle"

    var title: String {
        switch self {
        case .ready: "Ready when you are"
        case .modelLoading: "Getting the model ready…"
        case .scanning: "Reading the item…"
        case .result: "Best match"
        case .notSure: "No confident match"
        case .unavailable(let reason): reason.title
        }
    }

    /// Secondary line under the title. `nil` where the result itself carries
    /// the detail.
    var supportingText: String? {
        switch self {
        case .ready: "Hold steady and let one item fill the frame."
        case .modelLoading: "This happens once, on this device."
        case .scanning: "Working it out on this iPhone."
        case .result: nil
        // Actionable, and deliberately generic - the three things that most
        // often fix a low-confidence reading on a handheld phone.
        case .notSure: "More light, move closer, or use a plainer background."
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
