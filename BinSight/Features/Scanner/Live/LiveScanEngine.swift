import Foundation
import Observation
import os

/// Drives the live loop: camera frames in, a stable display state out.
///
/// > **Retired from the production runtime.** BinSight now detects rather than
/// > classifies: `ScannerView` drives `LiveDetectionEngine` and the bundled
/// > YOLO26 CoreML detector, and nothing in the shipping path constructs this
/// > type any more. It is kept, along with `WasteClassifying`,
/// > `MockWasteClassifier`, `PredictionSmoother` and `ScoreVector`, because
/// > those types are covered by a working test suite that still documents how
/// > the single-label pipeline behaved — deleting them would remove tests
/// > without replacing what they proved. Nothing here loads a model: the
/// > classifier seam has had no implementation since the LiteRT runtime was
/// > removed.
///
/// ## Backpressure
///
/// There is no queue and no explicit "am I busy" flag. The loop `await`s each
/// classification before pulling the next frame, and the frame stream is
/// `AsyncStream` with `bufferingPolicy: .bufferingNewest(1)`
/// (see `CameraSessionController`). Between them that gives all three
/// properties for free:
///
/// * **one inference at a time** — the loop body is sequential;
/// * **newest frame only** — the stream holds one element and replaces it;
/// * **bounded memory** — at most one pixel buffer is retained while busy.
///
/// Frames arriving during an inference are counted as dropped, not stored.
///
/// ## Threading
///
/// The engine is main-actor because it publishes observable state, but it does
/// no work there: `classify` is `async` on an actor-isolated classifier, so the
/// model runs off the main actor. What happens on the main actor per frame is a
/// smoother update — arithmetic over six doubles.
///
/// ## Staleness
///
/// Every run has a generation number. A result that arrives after `stop()`,
/// a reset, or a restart belongs to a previous generation and is discarded
/// rather than being allowed to overwrite newer state.
@Observable
@MainActor
final class LiveScanEngine {

    private(set) var state: ScannerDisplayState = .ready

    #if DEBUG
    private(set) var metrics = ScanMetrics()
    #endif

    private let configuration: ScannerConfiguration
    @ObservationIgnored private let classifier: WasteClassifying?
    @ObservationIgnored private let logger = Logger(
        subsystem: "com.marynaantonevych.BinSight",
        category: "inference"
    )

    @ObservationIgnored private var smoother: PredictionSmoother
    @ObservationIgnored private var latencies = LatencyTracker()
    @ObservationIgnored private var loop: Task<Void, Never>?

    /// Bumped whenever the run is invalidated. Guards against stale results.
    @ObservationIgnored private var generation = 0

    /// Layout reported by the view, so the crop matches what is on screen.
    /// `nil` until the first layout pass.
    @ObservationIgnored private var previewGeometry: (previewSize: CGSize, reticleRect: CGRect)?

    /// - Parameter classifier: `nil` means no model is available. The engine
    ///   then reports `.modelUnavailable` rather than pretending to scan —
    ///   which is BinSight's current state, since Phase 3 has not produced a
    ///   `.tflite` yet.
    init(
        classifier: WasteClassifying?,
        configuration: ScannerConfiguration = .default
    ) {
        self.classifier = classifier
        self.configuration = configuration
        self.smoother = PredictionSmoother(configuration: configuration)
        self.state = classifier == nil ? .unavailable(.modelUnavailable) : .ready
        #if DEBUG
        metrics.classifierLoaded = classifier != nil
        #endif
    }

    /// A pinned engine for previews and UI tests. Runs nothing.
    static func preview(_ state: ScannerDisplayState) -> LiveScanEngine {
        let engine = LiveScanEngine(classifier: MockWasteClassifier(constant: MockWasteClassifier.ambiguousScores))
        engine.state = state
        return engine
    }

    var hasClassifier: Bool { classifier != nil }

    // MARK: - Geometry

    /// Reports where the preview and reticle actually are, so the classified
    /// region matches the visible one. See `FrameCropPlan`.
    func updateGeometry(previewSize: CGSize, reticleRect: CGRect) {
        previewGeometry = (previewSize, reticleRect)
    }

    // MARK: - Lifecycle

    /// Starts consuming `frames`. Idempotent — an existing run is cancelled.
    func start(frames: AsyncStream<CameraFrame>) {
        guard let classifier else {
            state = .unavailable(.modelUnavailable)
            return
        }

        stop()
        generation += 1
        let runGeneration = generation
        smoother.reset()
        latencies.reset()
        state = .scanning
        logger.info("Live scan started (generation \(runGeneration, privacy: .public))")

        loop = Task { [weak self] in
            for await frame in frames {
                guard let self, !Task.isCancelled else { return }
                guard self.isCurrent(runGeneration) else { return }
                await self.process(frame, using: classifier, generation: runGeneration)
            }
        }
    }

    /// Stops promptly and clears smoothing.
    ///
    /// Called when the scene goes inactive, the camera is interrupted or loses
    /// permission, or the scanner disappears. Bumping the generation means any
    /// inference still in flight cannot publish when it lands.
    func stop() {
        loop?.cancel()
        loop = nil
        generation += 1
        smoother.reset()
        if state.isScanning || state.result != nil || state == .notSure {
            state = hasClassifier ? .ready : .unavailable(.modelUnavailable)
        }
    }

    /// Surfaces a camera problem, which outranks anything the classifier said.
    func reportCameraUnavailable(_ reason: ScannerUnavailableReason) {
        stop()
        state = .unavailable(reason)
    }

    // MARK: - Frame processing

    private func isCurrent(_ candidate: Int) -> Bool { candidate == generation }

    private func process(_ frame: CameraFrame, using classifier: WasteClassifying, generation runGeneration: Int) async {
        let crop = cropPlan(for: frame)
        guard crop.isUsable else {
            #if DEBUG
            metrics.droppedFrames += 1
            #endif
            return
        }

        // Monotonic: ContinuousClock keeps running across system clock changes
        // and cannot go backwards, unlike Date().
        let started = ContinuousClock.now

        do {
            let output = try await classifier.classify(frame, crop: crop)
            let elapsed = ContinuousClock.now - started

            // The run may have been invalidated while we were awaiting.
            guard isCurrent(runGeneration) else {
                logger.debug("Discarded a stale result from generation \(runGeneration, privacy: .public)")
                return
            }

            apply(output, endToEnd: elapsed.seconds, timestamp: frame.timestamp)
        } catch let error as ClassificationError {
            guard isCurrent(runGeneration) else { return }
            handle(error)
        } catch {
            guard isCurrent(runGeneration) else { return }
            handle(.invocationFailed(String(describing: type(of: error))))
        }
    }

    private func apply(_ output: ClassifierOutput, endToEnd: TimeInterval, timestamp: TimeInterval) {
        let outcome = smoother.add(scores: output.scores, at: timestamp)
        latencies.record(endToEnd)

        #if DEBUG
        metrics.processedFrames += 1
        metrics.lastPreprocessing = output.preprocessingDuration
        metrics.lastInvocation = output.invocationDuration
        metrics.lastEndToEnd = endToEnd
        metrics.averageEndToEnd = latencies.average
        metrics.medianEndToEnd = latencies.median
        metrics.p95EndToEnd = latencies.p95
        metrics.rejectedOutputs = smoother.rejectedSampleCount
        // Raw, pre-smoothing top class: proves the model is producing output
        // even while the smoother is still withholding a verdict.
        if let best = zip(WasteCategory.modelOutputOrder, output.scores).max(by: { $0.1 < $1.1 }) {
            metrics.rawTopLabel = best.0.displayName
            metrics.rawTopScore = best.1
        }
        if let average = latencies.average, average > 0 {
            metrics.effectiveRate = min(1.0 / average, configuration.targetClassificationsPerSecond)
        }
        #endif

        switch outcome {
        case .warmingUp:
            state = .scanning
        case .notSure:
            state = .notSure
        case .confident(let result):
            state = .result(result)
        }
    }

    private func handle(_ error: ClassificationError) {
        // Never log image bytes or buffer contents - only the error kind.
        logger.error("Classification failed: \(String(describing: error), privacy: .public)")

        if error.isRecoverable {
            // A single bad frame is not a reason to tear the screen down.
            #if DEBUG
            metrics.droppedFrames += 1
            #endif
            return
        }

        stop()
        state = .unavailable(.modelUnavailable)
    }

    private func cropPlan(for frame: CameraFrame) -> FrameCropPlan {
        let bufferSize = frame.pixelSize
        guard let geometry = previewGeometry, geometry.previewSize.width > 0 else {
            // No layout yet: classify everything the preview would show.
            return .fullVisibleRegion(bufferSize: bufferSize, previewSize: bufferSize)
        }
        return .forReticle(
            reticleRect: geometry.reticleRect,
            previewSize: geometry.previewSize,
            bufferSize: bufferSize
        )
    }
}
