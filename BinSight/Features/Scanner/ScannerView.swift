import SwiftUI
import os

/// BinSight's one screen.
///
/// Three pieces of state, kept separate because they fail independently:
///
/// * `camera` — permission and session lifecycle.
/// * `engine` — frames in, a smoothed display state out.
/// * `manualState` — the DEBUG/preview override, so the mock system survives.
///
/// `displayState` ranks them: a camera problem outranks anything the classifier
/// said, because showing a stale result over a dead preview would be a lie.
///
/// Layering, bottom to top: camera (the hero) → scrims → chrome. The scrims are
/// what make the chrome legible: a camera frame can be a white wall or a dark
/// bin, and light text alone survives only one of those.
struct ScannerView: View {
    @State private var camera: ScannerCameraModel
    @State private var engine: LiveScanEngine
    /// Non-nil only when a preview, UI test or the DEBUG menu is driving.
    @State private var manualState: ScannerDisplayState?
    @State private var announcer = ResultAnnouncer()

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Every dependency is injectable so previews and UI tests can pin a screen
    /// without hardware, a permission prompt, or a model.
    @MainActor
    init(
        initialState: ScannerDisplayState? = nil,
        camera: ScannerCameraModel? = nil,
        engine: LiveScanEngine? = nil
    ) {
        _manualState = State(initialValue: initialState)
        // Built here rather than as default arguments: default-argument
        // expressions are evaluated outside the main actor.
        _camera = State(initialValue: camera ?? ScannerCameraModel())
        _engine = State(initialValue: engine ?? LiveScanEngine(classifier: Self.makeClassifier()))
    }

    /// The live classifier, or `nil` when there is none.
    ///
    /// **Placeholder.** No model currently ships with the app, so this returns
    /// `nil` and the engine settles into its honest `.modelUnavailable` state:
    /// the camera runs, the screen says the classifier is unavailable, and
    /// nothing is recognised. That is deliberate — an app that silently shows
    /// invented labels is worse than one that admits it has no model.
    ///
    /// To restore recognition, build a type conforming to `WasteClassifying`
    /// and return it here. Nothing else in the pipeline changes: the engine,
    /// the smoother and every test above this line are written against the
    /// protocol, not against any particular runtime.
    private static func makeClassifier() -> WasteClassifying? {
        Logger(subsystem: "com.marynaantonevych.BinSight", category: "classifier")
            .notice("No classifier bundled — running in placeholder mode.")
        return nil
    }

    private var displayState: ScannerDisplayState {
        if let reason = camera.unavailableReason {
            return .unavailable(reason)
        }
        if let manualState {
            return manualState
        }
        return engine.state
    }

    private var isReady: Bool {
        displayState == .ready && camera.state.isRunning
    }

    /// The class accent currently in play, if any. Used sparingly — the reticle
    /// and the result icon — rather than tinting the whole screen.
    private var activeAccent: Color? {
        displayState.result?.category.accent
    }

    var body: some View {
        ZStack(alignment: .top) {
            background
                .ignoresSafeArea()

            scrims
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: BinSightTheme.spacing) {
                topBar
                scannerArea
                resultSurface
            }
            .padding(.horizontal, BinSightTheme.screenPadding)
            .padding(.top, 4)
            .padding(.bottom, 8)
        }
        .animation(reduceMotion ? nil : .snappy(duration: 0.35), value: displayState)
        .task {
            await camera.activate()
            // Also sync here: relying only on `onChange` loses the transition
            // when the camera settles before the observer is installed.
            syncEngine(with: camera.state)
        }
        .onDisappear {
            engine.stop()
            camera.deactivate()
            announcer.reset()
        }
        .onChange(of: camera.state) { _, state in
            syncEngine(with: state)
        }
        .onChange(of: displayState) { _, state in
            announce(state)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active:
                Task { await camera.activate() }
            case .inactive, .background:
                // Stop inference first: it is the expensive half.
                engine.stop()
                camera.deactivate()
            @unknown default:
                engine.stop()
                camera.deactivate()
            }
        }
    }

    // MARK: - Layers

    /// Live preview once frames are flowing; the mock backdrop otherwise, so a
    /// denied or unavailable camera still lands on a designed screen rather
    /// than a black rectangle.
    @ViewBuilder
    private var background: some View {
        if camera.state.isRunning, let session = camera.captureSession {
            CameraPreviewView(session: session)
                .accessibilityHidden(true)
                .accessibilityIdentifier("camera.preview")
        } else {
            MockCameraBackground()
        }
    }

    /// Top and bottom gradients that guarantee contrast over any frame.
    private var scrims: some View {
        VStack(spacing: 0) {
            LinearGradient(
                colors: [BinSightTheme.scrimStrong, .clear],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 180)

            Spacer(minLength: 0)

            LinearGradient(
                colors: [.clear, BinSightTheme.scrimSoft],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 140)
        }
    }

    // MARK: - Chrome

    private var topBar: some View {
        // Grouped so the system blends the two glass elements as one material
        // instead of two overlapping panes.
        GlassGroup(spacing: 12) {
            HStack(spacing: 10) {
                BinSightWordmark()

                Spacer(minLength: 8)

                PrivacyChip(text: "ON-DEVICE AI", systemImage: "cpu", isCompact: true)

                #if DEBUG
                DebugStateMenu(state: manualStateBinding)
                #endif
            }
        }
    }

    private var scannerArea: some View {
        VStack(spacing: 14) {
            ZStack {
                // The stand-in object is a cue that there is no camera yet —
                // it has no business sitting on top of a live preview.
                if !camera.state.isRunning {
                    SampleObjectSilhouette()
                }
                ScannerReticle(isScanning: displayState.isScanning, accent: activeAccent)
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: 320)
            // Report where the reticle actually is, so the classified region is
            // the region the person is aiming at. See FrameCropPlan.
            .background {
                GeometryReader { proxy in
                    Color.clear
                        .onAppear { reportGeometry(proxy) }
                        .onChange(of: proxy.size) { _, _ in reportGeometry(proxy) }
                }
            }

            Text("Point at one item")
                .font(BinSightTheme.rounded(.subheadline, weight: .semibold))
                .foregroundStyle(BinSightTheme.onCamera)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                // Its own scrim: this sits mid-screen where the top and bottom
                // gradients do not reach, and a camera frame can be any
                // brightness. Light text plus a shadow is not enough.
                .background(Capsule().fill(BinSightTheme.ink.opacity(0.45)))
                // Only a light de-emphasis once a result is up: dimming further
                // fades the scrim along with the text and destroys the contrast
                // the scrim exists to provide.
                .opacity(displayState.showsConfidence ? 0.8 : 1)
                .accessibilityIdentifier(isReady ? "scanner.ready" : "scanner.instruction")

            #if DEBUG
            if camera.state.isRunning {
                DebugPerformanceOverlay(metrics: engine.metrics)
            }
            #endif
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The result panel plus the privacy line, grouped as one glass region.
    private var resultSurface: some View {
        GlassGroup(spacing: 14) {
            VStack(spacing: 10) {
                ResultCard(state: displayState) { recovery in
                    handle(recovery)
                }
                PrivacyChip(text: "Frames stay on this iPhone.", systemImage: "lock.fill")
            }
        }
    }

    #if DEBUG
    /// The DEBUG menu writes into the manual override, which takes precedence
    /// over the engine so mock states keep working with the camera running.
    private var manualStateBinding: Binding<ScannerDisplayState?> {
        Binding(get: { manualState }, set: { manualState = $0 })
    }
    #endif

    // MARK: - Wiring

    private func announce(_ state: ScannerDisplayState) {
        // Monotonic, and only a meaningfully changed category or state gets
        // through — see ResultAnnouncer.
        let now = ProcessInfo.processInfo.systemUptime
        guard let phrase = announcer.announcement(for: state, at: now) else { return }
        AccessibilityNotification.Announcement(phrase).post()
    }

    private func reportGeometry(_ proxy: GeometryProxy) {
        let frame = proxy.frame(in: .global)
        let screen = UIScreen.main.bounds.size
        guard frame.width > 0, screen.width > 0 else { return }
        engine.updateGeometry(previewSize: screen, reticleRect: frame)
    }

    private func syncEngine(with state: CameraState) {
        if state.isRunning, let frames = camera.makeFrameStream() {
            engine.start(frames: frames)
        } else if let reason = state.unavailableReason {
            engine.reportCameraUnavailable(reason)
        } else {
            engine.stop()
        }
    }

    private func handle(_ recovery: ScannerUnavailableReason.Recovery) {
        switch recovery {
        case .retry:
            Task { await camera.retry() }
        case .openSettings:
            camera.openSystemSettings()
        }
    }
}

// MARK: - Previews
//
// Every preview injects a pinned camera and engine, so the canvas never
// constructs a capture session, prompts for permission, or loads a model.

@MainActor
private func previewScanner(
    _ state: ScannerDisplayState,
    camera: CameraState = .idle
) -> some View {
    ScannerView(
        initialState: state,
        camera: .preview(camera),
        engine: .preview(state)
    )
}

#Preview("Ready") { previewScanner(.ready) }
#Preview("Model loading") { previewScanner(.modelLoading) }
#Preview("Scanning") { previewScanner(.scanning) }
#Preview("Confident · Plastic") { previewScanner(.classified(.samplePlastic)) }
#Preview("Confident · Glass") { previewScanner(.classified(.sampleGlass)) }
#Preview("Not sure") { previewScanner(.notSure) }
#Preview("Camera denied") { previewScanner(.unavailable(.permissionDenied), camera: .denied) }
#Preview("Camera unavailable") { previewScanner(.unavailable(.cameraUnavailable), camera: .unavailable) }
#Preview("Model unavailable") { previewScanner(.unavailable(.modelUnavailable)) }

#Preview("Dark") {
    previewScanner(.classified(.samplePlastic))
        .preferredColorScheme(.dark)
}

#Preview("Accessibility text") {
    previewScanner(.classified(.sampleGlass))
        .environment(\.dynamicTypeSize, .accessibility3)
}

// Reduce Transparency has no writable environment key, so it cannot be pinned
// from a preview. Check the opaque path on a Simulator via
// Settings > Accessibility > Display & Text Size > Reduce Transparency.

/// Verifies the iOS 17–25 material path still lays out correctly. The canvas
/// renders on the host OS, so this checks layout rather than the fallback's
/// exact appearance — a 17.x simulator is the real check.
#Preview("Compact phone") {
    previewScanner(.classified(.samplePlastic))
}
