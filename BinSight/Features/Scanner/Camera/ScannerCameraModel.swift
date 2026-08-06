import AVFoundation
import Observation
import SwiftUI

/// Coordinates permission, session lifecycle and frame delivery for the scanner
/// screen. Owned by `ScannerView`; not a singleton, and not shared.
///
/// This is the one place in BinSight that justifies an observable model: it has
/// real asynchronous lifecycle to coordinate. Everything else on the screen is
/// still plain SwiftUI state.
///
/// Everything here runs on the main actor and holds only UI-relevant state. The
/// session itself lives in `CameraSessionController` on its own queues.
@Observable
@MainActor
final class ScannerCameraModel {

    private(set) var state: CameraState = .idle

    /// Whether this model drives real hardware.
    ///
    /// `false` for previews and tests, where it must never touch AVFoundation
    /// or trigger a permission prompt.
    let usesLiveCapture: Bool

    @ObservationIgnored private let controller: CameraSessionController?
    @ObservationIgnored private var isActive = false

    /// A fresh frame stream for `LiveScanEngine` to consume.
    ///
    /// Freshly made per call, never a shared property: `AsyncStream` iterates
    /// once, so reusing one across restarts leaves the second consumer silently
    /// starved.
    func makeFrameStream() -> AsyncStream<CameraFrame>? {
        controller?.makeFrameStream()
    }

    /// The session for the preview layer. `nil` when not driving hardware, so
    /// the view falls back to the mock backdrop.
    var captureSession: AVCaptureSession? {
        controller?.session
    }

    // MARK: - Init

    init(targetFramesPerSecond: Double = 4) {
        // Xcode previews run the real app process. Constructing a capture
        // session there would prompt for permission inside the canvas, so the
        // live path is refused outright.
        let isPreview = ProcessInfo.processInfo.environment["XCODE_RUNNING_FOR_PREVIEWS"] == "1"
        self.usesLiveCapture = !isPreview
        self.controller = isPreview ? nil : CameraSessionController(
            targetFramesPerSecond: targetFramesPerSecond
        )
        self.state = .idle
        wireStatusEvents()
    }

    /// A model that renders a fixed state and never touches the camera.
    /// Used by previews and by tests of the lifecycle logic.
    private init(previewState: CameraState) {
        self.usesLiveCapture = false
        self.controller = nil
        self.state = previewState
    }

    static func preview(_ state: CameraState) -> ScannerCameraModel {
        ScannerCameraModel(previewState: state)
    }

    // MARK: - Lifecycle

    /// Called when the scanner screen becomes active.
    ///
    /// This — and only this — is what may trigger the system permission prompt.
    /// It is never called from previews or tests, because `usesLiveCapture` is
    /// false there and the method returns immediately.
    ///
    /// Idempotent: `.task` and a `scenePhase` change both call it, and calling
    /// it while already running is a no-op.
    func activate() async {
        guard usesLiveCapture, let controller else { return }
        guard !isActive else { return }
        isActive = true

        let status = AVCaptureDevice.authorizationStatus(for: .video)

        if CameraAuthorization.shouldRequestPermission(for: status) {
            state = .preparing
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            state = CameraAuthorization.state(afterPromptGranted: granted)
            guard granted else {
                isActive = false
                return
            }
        } else {
            let mapped = CameraAuthorization.state(for: status)
            guard mapped == .running else {
                state = mapped
                isActive = false
                return
            }
            state = .preparing
        }

        switch await controller.start() {
        case .running:
            state = .running
        case .noCameraAvailable:
            state = .unavailable
            isActive = false
        case .configurationFailed:
            state = .failed
            isActive = false
        }
    }

    /// Called when the screen disappears or the scene stops being active.
    func deactivate() {
        guard usesLiveCapture else { return }
        isActive = false
        controller?.stop()
        if state == .running || state == .preparing {
            state = .idle
        }
    }

    /// Retry affordance for the interrupted and failed states.
    func retry() async {
        guard usesLiveCapture else { return }
        deactivate()
        await activate()
    }

    /// Opens BinSight's page in Settings so a denied permission can be changed.
    func openSystemSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    /// What the scanner screen should show instead of a result, if anything.
    var unavailableReason: ScannerUnavailableReason? {
        state.unavailableReason
    }

    // MARK: - Private

    private func wireStatusEvents() {
        controller?.onStatusEvent = { [weak self] event in
            // Notifications arrive on an arbitrary thread; hop before touching
            // observable state.
            Task { @MainActor [weak self] in
                self?.handle(event)
            }
        }
    }

    private func handle(_ event: CameraSessionController.StatusEvent) {
        switch event {
        case .interrupted:
            state = .interrupted
        case .interruptionEnded:
            // The session resumes itself; reflect that only if we still want it.
            if isActive {
                state = .running
            }
        case .runtimeError:
            state = .failed
        }
    }

}
