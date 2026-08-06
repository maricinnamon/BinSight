import AVFoundation

/// What the camera pipeline is doing, as far as the UI needs to care.
///
/// Deliberately coarse: the view never sees an `AVCaptureSession`, an
/// authorization status or an `NSError`. It sees one of these.
enum CameraState: Equatable, Sendable {
    /// Nothing has been attempted yet — the scanner screen is not active.
    case idle
    /// Permission is being requested, or the session is being configured.
    case preparing
    /// Frames are flowing.
    case running
    /// The person said no, or Screen Time / MDM said no on their behalf.
    case denied
    case restricted
    /// No usable capture device. Every Simulator lands here.
    case unavailable
    /// A call, Control Centre, Split View, or another app took the camera.
    case interrupted
    /// `AVCaptureSessionRuntimeError`, or configuration failed.
    case failed

    /// How this maps onto the screen's existing unavailable states.
    ///
    /// `nil` means "nothing is wrong" — the scanner shows its normal content.
    /// `idle` and `preparing` deliberately return `nil` too: flashing an error
    /// card during the half second before the session starts would be worse
    /// than showing nothing.
    var unavailableReason: ScannerUnavailableReason? {
        switch self {
        case .idle, .preparing, .running: nil
        case .denied: .permissionDenied
        case .restricted: .permissionRestricted
        case .unavailable: .cameraUnavailable
        case .interrupted: .interrupted
        case .failed: .captureFailed
        }
    }

    var isRunning: Bool { self == .running }
}

/// Translates `AVAuthorizationStatus` into camera state.
///
/// Split out as a plain function with no AVFoundation side effects so the
/// mapping can be unit tested without a camera, a simulator, or a permission
/// prompt.
enum CameraAuthorization {
    /// The state to show for a known authorization status.
    ///
    /// `.notDetermined` maps to `.preparing`, not to an error: the system
    /// prompt is about to appear and the screen should look like it is getting
    /// ready, not like something failed.
    static func state(for status: AVAuthorizationStatus) -> CameraState {
        switch status {
        case .authorized: .running
        case .notDetermined: .preparing
        case .denied: .denied
        case .restricted: .restricted
        @unknown default: .failed
        }
    }

    /// Whether a status warrants showing the system permission prompt.
    /// Only `.notDetermined` does — asking again after a denial does nothing
    /// except waste a round trip, because iOS will not re-prompt.
    static func shouldRequestPermission(for status: AVAuthorizationStatus) -> Bool {
        status == .notDetermined
    }

    /// The state after the user answers the prompt.
    static func state(afterPromptGranted granted: Bool) -> CameraState {
        granted ? .running : .denied
    }
}
