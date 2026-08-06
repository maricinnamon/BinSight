import CoreVideo
import Testing
@testable import BinSight

/// Pipeline state transitions, driven by the mock classifier.
///
/// No camera and no model: frames are synthesised pixel buffers, and the
/// classifier returns a scripted vector.
@MainActor
struct LiveScanEngineTests {

    // MARK: - Helpers

    private func makeFrame(timestamp: TimeInterval, width: Int = 1280, height: Int = 720) -> CameraFrame? {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &buffer
        )
        guard status == kCVReturnSuccess, let buffer else { return nil }
        return CameraFrame(pixelBuffer: buffer, timestamp: timestamp, orientation: .up)
    }

    /// Feeds a fixed number of frames through the engine and waits for the loop
    /// to drain.
    private func run(
        engine: LiveScanEngine,
        frameCount: Int,
        step: TimeInterval = 0.25
    ) async {
        let (stream, continuation) = AsyncStream.makeStream(
            of: CameraFrame.self,
            bufferingPolicy: .unbounded
        )
        engine.start(frames: stream)

        for index in 0..<frameCount {
            guard let frame = makeFrame(timestamp: Double(index) * step) else { continue }
            continuation.yield(frame)
        }
        continuation.finish()

        // Let the consuming task drain.
        for _ in 0..<(frameCount * 4 + 20) {
            await Task.yield()
        }
    }

    // MARK: - No classifier

    @Test("With no classifier the engine says so instead of pretending to scan")
    func noClassifierReportsModelUnavailable() {
        let engine = LiveScanEngine(classifier: nil)
        #expect(engine.state == .unavailable(.modelUnavailable))
        #expect(!engine.hasClassifier)
    }

    @Test("Starting without a classifier cannot enter the scanning state")
    func startWithoutClassifierDoesNothing() {
        let engine = LiveScanEngine(classifier: nil)
        let (stream, continuation) = AsyncStream.makeStream(of: CameraFrame.self)
        engine.start(frames: stream)
        continuation.finish()
        #expect(engine.state == .unavailable(.modelUnavailable))
    }

    // MARK: - Happy path

    @Test("A steady confident stream produces a confident result")
    func steadyStreamProducesResult() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.confidentPlasticScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)

        guard case .result(let result) = engine.state else {
            Issue.record("expected a confident result, got \(engine.state)")
            return
        }
        #expect(result.category == .plastic)
    }

    @Test("An ambiguous stream lands on not-sure, never on a random class")
    func ambiguousStreamIsNotSure() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.ambiguousScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)

        #expect(engine.state == .notSure)
    }

    @Test("A narrow top1/top2 margin does not produce a named class")
    func narrowMarginIsNotSure() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.narrowMarginScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)

        #expect(engine.state == .notSure)
    }

    // MARK: - Errors

    @Test("An unrecoverable classifier error tears the pipeline down")
    func unrecoverableErrorSurfaces() async {
        let classifier = MockWasteClassifier(
            failingWith: .modelMissingFromBundle(resource: "BinSightWasteClassifier.tflite")
        )
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 3)

        #expect(engine.state == .unavailable(.modelUnavailable))
    }

    @Test("A recoverable error does not tear the screen down over one bad frame")
    func recoverableErrorKeepsScanning() async {
        let classifier = MockWasteClassifier(failingWith: .preprocessingFailed("pixel buffer lock"))
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 3)

        // Still scanning, not a hard failure card.
        #expect(engine.state != .unavailable(.modelUnavailable))
    }

    @Test("Error kinds classify themselves as recoverable or not")
    func errorRecoverability() {
        #expect(!ClassificationError.modelMissingFromBundle(resource: "m").isRecoverable)
        #expect(!ClassificationError.labelCountMismatch(expected: 6, found: 5).isRecoverable)
        #expect(!ClassificationError.inputTensorMismatch(expected: "a", found: "b").isRecoverable)
        #expect(ClassificationError.preprocessingFailed("x").isRecoverable)
        #expect(ClassificationError.invocationFailed("x").isRecoverable)
        #expect(ClassificationError.invalidOutput("x").isRecoverable)
    }

    // MARK: - Lifecycle and staleness

    @Test("Stopping clears the result and does not leave a stale card up")
    func stopClearsResult() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.confidentPlasticScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)
        #expect(engine.state.result != nil)

        engine.stop()
        #expect(engine.state.result == nil)
        #expect(engine.state == .ready)
    }

    @Test("A camera problem outranks whatever the classifier last said")
    func cameraProblemOverridesResult() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.confidentPlasticScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)
        #expect(engine.state.result != nil)

        engine.reportCameraUnavailable(.interrupted)
        #expect(engine.state == .unavailable(.interrupted))
    }

    @Test("Results from a stopped run cannot overwrite newer state")
    func staleResultsAreDiscarded() async {
        // A slow classifier, so the run is invalidated while inference is in
        // flight. Without the generation guard the late result would land on
        // top of the newer state.
        let classifier = MockWasteClassifier(
            constant: MockWasteClassifier.confidentPlasticScores,
            simulatedDuration: 0
        )
        let engine = LiveScanEngine(classifier: classifier)

        let (stream, continuation) = AsyncStream.makeStream(
            of: CameraFrame.self, bufferingPolicy: .unbounded
        )
        engine.start(frames: stream)
        for index in 0..<10 {
            if let frame = makeFrame(timestamp: Double(index) * 0.25) {
                continuation.yield(frame)
            }
        }

        // Invalidate mid-flight, then set a state a stale result must not clobber.
        engine.reportCameraUnavailable(.permissionDenied)
        continuation.finish()

        for _ in 0..<60 { await Task.yield() }

        #expect(engine.state == .unavailable(.permissionDenied), "a stale result overwrote newer state")
    }

    @Test("Restarting clears smoothing so the previous item cannot leak through")
    func restartClearsSmoothing() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.confidentPlasticScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 10)
        #expect(engine.state.result != nil)

        let (stream, continuation) = AsyncStream.makeStream(of: CameraFrame.self)
        engine.start(frames: stream)
        // A fresh run starts from scanning, not from the old verdict.
        #expect(engine.state == .scanning)
        continuation.finish()
    }

    #if DEBUG
    @Test("Metrics count processed frames and record latencies")
    func metricsAreRecorded() async {
        let classifier = MockWasteClassifier(constant: MockWasteClassifier.confidentPlasticScores)
        let engine = LiveScanEngine(classifier: classifier)

        await run(engine: engine, frameCount: 6)

        #expect(engine.metrics.processedFrames > 0)
        #expect(engine.metrics.lastEndToEnd != nil)
        #expect(engine.metrics.averageEndToEnd != nil)
        #expect((engine.metrics.lastEndToEnd ?? -1) >= 0)
    }
    #endif
}

/// Latency statistics.
@MainActor
struct LatencyTrackerTests {

    @Test("An empty tracker reports nothing rather than zero")
    func emptyReportsNil() {
        let tracker = LatencyTracker()
        #expect(tracker.average == nil)
        #expect(tracker.median == nil)
        #expect(tracker.p95 == nil)
        #expect(tracker.isEmpty)
    }

    @Test("Average, median and p95 are computed over the samples")
    func statistics() {
        var tracker = LatencyTracker(capacity: 100)
        for value in 1...100 { tracker.record(Double(value) / 1000.0) }

        #expect(abs((tracker.average ?? 0) - 0.0505) < 1e-9)
        #expect(abs((tracker.median ?? 0) - 0.050) < 1e-9)
        #expect(abs((tracker.p95 ?? 0) - 0.095) < 1e-9)
    }

    @Test("The ring is bounded: old samples fall out")
    func ringIsBounded() {
        // An unbounded history is both a slow leak and a way to hide thermal
        // throttling behind a whole-session average.
        var tracker = LatencyTracker(capacity: 10)
        for value in 1...50 { tracker.record(Double(value)) }
        #expect(tracker.count == 10)
        #expect(tracker.samples.first == 41)
        #expect(tracker.samples.last == 50)
    }

    @Test("Non-finite and negative durations are ignored")
    func rejectsNonsense() {
        var tracker = LatencyTracker()
        tracker.record(.nan)
        tracker.record(-1)
        tracker.record(.infinity)
        #expect(tracker.isEmpty)
    }

    @Test("Reset empties the tracker")
    func resetEmpties() {
        var tracker = LatencyTracker()
        tracker.record(0.02)
        tracker.reset()
        #expect(tracker.isEmpty)
    }
}
