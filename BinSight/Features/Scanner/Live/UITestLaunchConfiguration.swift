#if DEBUG
import Foundation

/// Lets a UI test pin the scanner to a known state via launch arguments,
/// so screens can be asserted without a camera, a permission prompt or a model.
///
/// DEBUG-only, and it does nothing unless a UI test passes the flag — a normal
/// Debug run is unaffected.
///
/// Usage from a UI test:
/// ```swift
/// app.launchArguments += ["-BinSightUITestScenario", "confidentResult"]
/// ```
enum UITestLaunchConfiguration {

    /// The screens a UI test can ask for.
    enum Scenario: String, CaseIterable {
        case confidentResult
        case notSure
        case cameraDenied
        case cameraUnavailable
        case modelError
        case ready

        var scannerState: ScannerDisplayState {
            switch self {
            case .confidentResult: .classified(.samplePlastic)
            case .notSure: .notSure
            case .cameraDenied: .unavailable(.permissionDenied)
            case .cameraUnavailable: .unavailable(.cameraUnavailable)
            case .modelError: .unavailable(.modelUnavailable)
            case .ready: .ready
            }
        }

        /// The camera model to pin alongside it. Never live, so the test cannot
        /// trip the permission prompt.
        var cameraState: CameraState {
            switch self {
            case .cameraDenied: .denied
            case .cameraUnavailable: .unavailable
            // Idle, not running: a pinned camera has no session to preview, and
            // idle keeps the designed mock backdrop on screen.
            case .confidentResult, .notSure, .modelError, .ready: .idle
            }
        }
    }

    private static let flag = "-BinSightUITestScenario"

    /// The requested scenario, or `nil` for a normal launch.
    static var scenario: Scenario? {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: flag),
              index + 1 < arguments.count,
              let scenario = Scenario(rawValue: arguments[index + 1])
        else { return nil }
        return scenario
    }

    static var isRunningUITest: Bool { scenario != nil }
}
#endif
