import CoreGraphics
import Foundation
import Observation
import os

/// Drives the live detection loop: camera frames in, boxes out.
///
/// This replaces `LiveScanEngine` as the production pipeline. The two differ in
/// kind, not just in model: the old one produced a single smoothed label from a
/// six-class classifier, this one produces a *set* of located objects from a
/// three-class detector. Smoothing is deliberately absent — averaging a varying
/// number of boxes across frames is a different problem, and a detector that
/// already reports per-object confidence does not need a verdict-stabiliser to
/// avoid naming things it is unsure about.
///
/// ## Backpressure
///
/// Unchanged from the pipeline it replaces, because that design was already
/// right: the loop `await`s each inference before pulling the next frame, and
/// the stream is `bufferingNewest(1)` (see `CameraSessionController`). Together
/// that gives one inference in flight, newest frame only, and at most one
/// retained pixel buffer. Frames arriving mid-inference are dropped, not queued.
///
/// ## Threading
///
/// Main-actor because it publishes observable state, but it does no work here:
/// `detect` is `async` on an actor, so CoreML runs off the main actor. What
/// happens on the main actor per frame is assigning an array of at most ten
/// small structs.
///
/// ## Staleness
///
/// Every run has a generation number. Detections that land after `stop()`, a
/// reset or a restart belong to a previous generation and are discarded rather
/// than being allowed to paint boxes over a stopped preview.
@Observable
@MainActor
final class LiveDetectionEngine {

    /// Current detections, in **camera-buffer pixel coordinates**.
    private(set) var detections: [Detection] = []

    /// The size of the buffer `detections` are expressed in. The overlay needs
    /// both to place a box, and publishing them together means they can never
    /// disagree — a resolution change cannot leave boxes scaled by the old size.
    private(set) var sourceSize: CGSize = .zero

    private(set) var status: Status = .idle

    enum Status: Equatable, Sendable {
        case idle
        case loading
        case running
        case unavailable(ScannerUnavailableReason)

        var isRunning: Bool { self == .running }
    }

    #if DEBUG
    private(set) var metrics = DetectionMetrics()
    #endif

    @ObservationIgnored private let detector: YOLODetectionService?
    @ObservationIgnored private let configuration: DetectionConfiguration
    @ObservationIgnored private let logger = Logger(
        subsystem: "com.marynaantonevych.BinSight",
        category: "detection"
    )
    @ObservationIgnored private var latencies = LatencyTracker()
    @ObservationIgnored private var loop: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    /// - Parameter detector: `nil` means the model could not be loaded. The
    ///   engine then reports `.modelUnavailable` rather than pretending to scan.
    init(detector: YOLODetectionService?, configuration: DetectionConfiguration = .default) {
        self.detector = detector
        self.configuration = configuration
        self.status = detector == nil ? .unavailable(.modelUnavailable) : .idle
        #if DEBUG
        metrics.detectorLoaded = detector != nil
        #endif
    }

    /// Builds the detector, reporting failure as state rather than throwing.
    ///
    /// A missing or malformed model is a build problem, and the screen says so
    /// honestly instead of showing an empty overlay that looks like "nothing is
    /// in view".
    static func makeDefault(configuration: DetectionConfiguration = .default) -> LiveDetectionEngine {
        let logger = Logger(subsystem: "com.marynaantonevych.BinSight", category: "detection")
        do {
            let service = try YOLODetectionService(configuration: configuration)
            logger.notice("YOLO26 detector loaded from the app bundle.")
            return LiveDetectionEngine(detector: service, configuration: configuration)
        } catch {
            logger.error("Detector unavailable: \(String(describing: error), privacy: .public)")
            return LiveDetectionEngine(detector: nil, configuration: configuration)
        }
    }

    /// A pinned engine for previews and UI tests. Runs nothing.
    static func preview(detections: [Detection] = [], sourceSize: CGSize = CGSize(width: 720, height: 1280)) -> LiveDetectionEngine {
        let engine = LiveDetectionEngine(detector: nil)
        engine.detections = detections
        engine.sourceSize = sourceSize
        engine.status = .running
        return engine
    }

    var hasDetector: Bool { detector != nil }

    // MARK: - Lifecycle

    /// Starts consuming `frames`. Idempotent — an existing run is cancelled.
    func start(frames: AsyncStream<CameraFrame>) {
        guard let detector else {
            status = .unavailable(.modelUnavailable)
            return
        }

        stop()
        generation += 1
        let runGeneration = generation
        latencies.reset()
        status = .running
        logger.info("Live detection started (generation \(runGeneration, privacy: .public))")

        loop = Task { [weak self] in
            for await frame in frames {
                guard let self, !Task.isCancelled else { return }
                guard self.isCurrent(runGeneration) else { return }
                await self.process(frame, using: detector, generation: runGeneration)
            }
        }
    }

    /// Stops promptly and clears the overlay.
    ///
    /// Clearing matters: leaving the last frame's boxes on screen over a frozen
    /// or black preview would assert that objects are still there.
    func stop() {
        loop?.cancel()
        loop = nil
        generation += 1
        detections = []
        if status.isRunning {
            status = hasDetector ? .idle : .unavailable(.modelUnavailable)
        }
    }

    /// Surfaces a camera problem, which outranks anything the detector said.
    func reportCameraUnavailable(_ reason: ScannerUnavailableReason) {
        stop()
        status = .unavailable(reason)
    }

    // MARK: - Frame processing

    private func isCurrent(_ candidate: Int) -> Bool { candidate == generation }

    private func process(
        _ frame: CameraFrame,
        using detector: YOLODetectionService,
        generation runGeneration: Int
    ) async {
        let frameSize = frame.pixelSize
        let started = ContinuousClock.now

        do {
            let found = try await detector.detect(frame)
            let elapsed = (ContinuousClock.now - started).seconds

            // The run may have been invalidated while we were awaiting.
            guard isCurrent(runGeneration) else {
                logger.debug("Discarded stale detections from generation \(runGeneration, privacy: .public)")
                return
            }

            sourceSize = frameSize
            detections = found
            latencies.record(elapsed)

            #if DEBUG
            metrics.processedFrames += 1
            metrics.lastDetectionCount = found.count
            metrics.lastEndToEnd = elapsed
            metrics.lastInference = await detector.lastInferenceDuration
            metrics.lastPreprocessing = await detector.lastPreprocessingDuration
            metrics.averageEndToEnd = latencies.average
            metrics.medianEndToEnd = latencies.median
            metrics.p95EndToEnd = latencies.p95
            if let average = latencies.average, average > 0 {
                metrics.effectiveRate = 1.0 / average
            }
            metrics.topLabel = found.first.map(\.label)
            #endif
        } catch let error as DetectionError {
            guard isCurrent(runGeneration) else { return }
            handle(error)
        } catch {
            guard isCurrent(runGeneration) else { return }
            handle(.invocationFailed(String(describing: type(of: error))))
        }
    }

    private func handle(_ error: DetectionError) {
        // Never log image bytes or buffer contents — only the error kind.
        logger.error("Detection failed: \(String(describing: error), privacy: .public)")

        if error.isRecoverable {
            // A single bad frame is not a reason to tear the screen down.
            #if DEBUG
            metrics.droppedFrames += 1
            #endif
            return
        }

        stop()
        status = .unavailable(.modelUnavailable)
    }
}

#if DEBUG
/// Live counters for the debug overlay. DEBUG-only: none of this ships.
struct DetectionMetrics: Equatable, Sendable {
    var detectorLoaded = false
    var processedFrames = 0
    var droppedFrames = 0
    var lastDetectionCount = 0
    var lastPreprocessing: TimeInterval = 0
    var lastInference: TimeInterval = 0
    var lastEndToEnd: TimeInterval = 0
    var averageEndToEnd: TimeInterval?
    var medianEndToEnd: TimeInterval?
    var p95EndToEnd: TimeInterval?
    var effectiveRate: Double = 0
    var topLabel: String?
}
#endif
