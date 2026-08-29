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
    @State private var engine: LiveDetectionEngine
    /// Non-nil only when a preview, UI test or the DEBUG menu is driving.
    ///
    /// When set, the screen renders the **pinned** presentation — reticle and
    /// `ResultCard` — rather than the live detector. That keeps every existing
    /// preview and UI test meaningful without a camera or a model, and it is the
    /// only path on which `ScannerDisplayState` still drives the UI.
    @State private var manualState: ScannerDisplayState?
    @State private var announcer = ResultAnnouncer()
    /// Last set of class ids announced, so VoiceOver is not spammed per frame.
    @State private var lastAnnouncedMaterials: Set<Int> = []
    /// Monotonic timestamp of that announcement.
    @State private var lastAnnouncedAt: TimeInterval = 0
    @State private var localization = LocalizationStore.shared

    /// Matches `ResultAnnouncer`'s throttle, so both voices pace the same.
    private static let announcementInterval: TimeInterval = 1.5

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Every dependency is injectable so previews and UI tests can pin a screen
    /// without hardware, a permission prompt, or a model.
    @MainActor
    init(
        initialState: ScannerDisplayState? = nil,
        camera: ScannerCameraModel? = nil,
        engine: LiveDetectionEngine? = nil
    ) {
        _manualState = State(initialValue: initialState)
        // Built here rather than as default arguments: default-argument
        // expressions are evaluated outside the main actor.
        _camera = State(initialValue: camera ?? ScannerCameraModel())
        // A pinned screen must not load CoreML: previews and UI tests run
        // without the model and should stay that way.
        _engine = State(initialValue: engine ?? (initialState == nil
            ? LiveDetectionEngine.makeDefault()
            : LiveDetectionEngine.preview()))
    }

    /// True while the live detector is the thing on screen.
    private var isLive: Bool {
        manualState == nil
            && camera.unavailableReason == nil
            && camera.state.isRunning
            && engine.status.isRunning
    }

    /// The pinned/unavailable presentation state.
    ///
    /// Only consulted when the live detector is *not* driving: a camera problem,
    /// a model problem, or a deliberately pinned screen.
    private var displayState: ScannerDisplayState {
        if let reason = camera.unavailableReason {
            return .unavailable(reason)
        }
        if let manualState {
            return manualState
        }
        if case .unavailable(let reason) = engine.status {
            return .unavailable(reason)
        }
        // `.modelLoading` finally has a producer. It was declared, given copy,
        // and never assigned — the screen used to jump straight from launch to
        // scanning while the main thread was blocked loading CoreML.
        if engine.status == .loading {
            return .modelLoading
        }
        return camera.state.isRunning ? .scanning : .ready
    }

    private var isReady: Bool {
        displayState == .ready && camera.state.isRunning
    }

    /// The class accent currently in play, if any. Used sparingly — the reticle
    /// and the result icon — rather than tinting the whole screen.
    private var activeAccent: Color? {
        if let result = displayState.result { return result.category.accent }
        return engine.detections.first?.detectedClass.wasteCategory.accent
    }

    var body: some View {
        ZStack(alignment: .top) {
            background
                .ignoresSafeArea()

            // Above the preview, below the chrome: boxes must sit on the image
            // they describe, but must never cover the result surface.
            if isLive {
                DetectionOverlayView(
                    detections: engine.detections,
                    sourceSize: engine.sourceSize
                )
                .ignoresSafeArea()
            }

            scrims
                .ignoresSafeArea()
                .allowsHitTesting(false)

            // ScrollView, not a bare VStack. At accessibility text sizes on a
            // 375pt screen the unavailable-state card alone exceeds the safe
            // area, and SwiftUI clips the bottom — which is exactly where the
            // "Open Settings" button lives. A person who denies camera access
            // and uses large text would have had no way back.
            //
            // `.scrollBounceBehavior(.basedOnSize)` keeps it feeling like a
            // fixed screen at normal sizes: no rubber-banding unless the
            // content genuinely overflows.
            ScrollView {
                VStack(spacing: BinSightTheme.spacing) {
                    topBar
                    scannerArea
                    resultSurface
                }
                .padding(.horizontal, BinSightTheme.screenPadding)
                .padding(.top, 4)
                .padding(.bottom, 8)
                // Fills the screen at normal text sizes so the layout still
                // reads as a fixed screen; grows past it, and scrolls, only when
                // the content genuinely needs more room.
                .containerRelativeFrame(.vertical, alignment: .top)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
        .animation(reduceMotion ? nil : .snappy(duration: 0.35), value: displayState)
        .task {
            // Model first, then the camera: loading it is what the "Getting the
            // model ready…" state is describing, and starting the session
            // underneath would only compete for the CPU while it happens.
            await engine.loadDetector()
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
        .onChange(of: engine.detections) { _, detections in
            announceDetections(detections)
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

                PrivacyChip(text: L("chip.onDeviceAI"), systemImage: "cpu", isCompact: true)

                LanguageMenu(store: localization)

                #if DEBUG
                DebugStateMenu(state: manualStateBinding)
                #endif
            }
        }
    }

    private var scannerArea: some View {
        VStack(spacing: 14) {
            // The reticle described the square region the old classifier cropped
            // to. The detector reads the whole frame and draws its own boxes, so
            // showing a fixed square over a live preview would be claiming a
            // framing rule that no longer exists. It stays only for the pinned
            // and camera-less screens, where it is still an honest placeholder.
            if !isLive {
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
            } else {
                Spacer(minLength: 0)
            }

            Text(isLive ? L("scanner.instruction.live") : L("scanner.instruction.pinned"))
                .font(BinSightTheme.rounded(.subheadline, weight: .semibold))
                .foregroundStyle(BinSightTheme.onCamera)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                // Its own scrim: this sits mid-screen where the top and bottom
                // gradients do not reach, and a camera frame can be any
                // brightness. Light text plus a shadow is not enough.
                // 0.65, not 0.45. Over a white wall the lighter scrim left
                // cream-on-grey at about 3.7:1, under the 4.5:1 this text needs.
                .background(Capsule().fill(BinSightTheme.ink.opacity(0.65)))
                // Fade it once boxes are on screen: the overlay is the answer,
                // and a standing instruction competes with it.
                .opacity(engine.detections.isEmpty ? 1 : 0.55)
                .accessibilityIdentifier(isReady ? "scanner.ready" : "scanner.instruction")

            #if DEBUG
            if camera.state.isRunning {
                DebugDetectionOverlay(metrics: engine.metrics)
            }
            #endif
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The result panel plus the privacy line, grouped as one glass region.
    ///
    /// Two cards, one slot: the detector's summary while it is running, and the
    /// original `ResultCard` for every pinned or unavailable screen — which is
    /// what keeps the camera-denied, camera-unavailable and model-unavailable
    /// states (and their UI tests) working unchanged.
    private var resultSurface: some View {
        GlassGroup(spacing: 14) {
            VStack(spacing: 10) {
                if isLive {
                    DetectionSummaryCard(
                        detections: engine.detections,
                        isRunning: engine.status.isRunning
                    )
                } else {
                    ResultCard(state: displayState) { recovery in
                        handle(recovery)
                    }
                }
                PrivacyChip(text: L("chip.framesStayOnPhone"), systemImage: "lock.fill")
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

    /// Announces the set of materials in view when it changes.
    ///
    /// Deliberately keyed on the *set*, not on the boxes: a hand-held phone
    /// changes the number and position of boxes several times a second, and
    /// announcing that would make VoiceOver unusable. "Plastic and metal" only
    /// fires when the answer actually changes.
    private func announceDetections(_ detections: [Detection]) {
        let materials = Set(detections.map(\.classID))

        // An empty set is a gap, not an answer. Recording it as "last announced"
        // was the bug: a bottle flickering just under threshold produced
        // {plastic} -> {} -> {plastic}, and because the empty set had been
        // stored, the second {plastic} counted as a change and was announced
        // again. At 4 fps that is VoiceOver talking over itself continuously.
        guard !materials.isEmpty else { return }
        guard materials != lastAnnouncedMaterials else { return }

        // Even a genuine change is throttled. Panning across a cluttered table
        // legitimately changes the answer several times a second, and speaking
        // each one makes the app unusable with VoiceOver on.
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastAnnouncedAt >= Self.announcementInterval else { return }

        lastAnnouncedMaterials = materials
        lastAnnouncedAt = now

        let names = DetectedClass.allCases
            .filter { materials.contains($0.rawValue) }
            .map(\.displayName)
        AccessibilityNotification.Announcement(names.formatted(.list(type: .and))).post()
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
        engine: .preview()
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
