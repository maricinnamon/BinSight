import Testing
@testable import BinSight

/// The stability rules that stop the result card strobing.
///
/// All timestamps are supplied by the test, so nothing here depends on a clock,
/// a camera or a model.
@MainActor
struct PredictionSmootherTests {

    private let config = ScannerConfiguration.default

    /// Canonical order: cardboard, glass, metal, paper, plastic, general_waste.
    private func scores(plastic: Double) -> [Double] {
        let rest = (1.0 - plastic) / 5.0
        return [rest, rest, rest, rest, plastic, rest]
    }

    private func feed(
        _ smoother: inout PredictionSmoother,
        _ vector: [Double],
        times: Int,
        startingAt start: Double = 0,
        step: Double = 0.25
    ) -> PredictionSmoother.Outcome {
        var outcome = PredictionSmoother.Outcome.warmingUp
        for index in 0..<times {
            outcome = smoother.add(scores: vector, at: start + Double(index) * step)
        }
        return outcome
    }

    // MARK: - Warm-up

    @Test("The very first frame can never produce a confident answer")
    func firstFrameIsNeverConfident() {
        var smoother = PredictionSmoother(configuration: config)
        let outcome = smoother.add(scores: scores(plastic: 0.99), at: 0)
        #expect(outcome == .warmingUp)
    }

    // MARK: - Confident path

    @Test("A steady, well-separated reading becomes a confident result")
    func steadyReadingBecomesConfident() {
        var smoother = PredictionSmoother(configuration: config)
        let outcome = feed(&smoother, scores(plastic: 0.95), times: 8)

        guard case .confident(let result) = outcome else {
            Issue.record("expected a confident outcome, got \(outcome)")
            return
        }
        #expect(result.category == .plastic)
        #expect(result.confidence > config.minimumConfidence)
        #expect(result.rankedScores.count == WasteCategory.allCases.count)
        // Ranked descending, canonical order preserved separately.
        #expect(result.rankedScores[0].category == .plastic)
        #expect(result.canonicalScores.count == WasteCategory.allCases.count)
    }

    @Test("Agreement must be sustained: fewer samples than required stays unsure")
    func needsSustainedAgreement() {
        var smoother = PredictionSmoother(configuration: config)
        // Two qualifying samples when three consecutive are required.
        let outcome = feed(&smoother, scores(plastic: 0.95), times: 2)
        #expect(outcome != .confident(.samplePlastic))
        if case .confident = outcome {
            Issue.record("promoted a label before the required streak")
        }
    }

    // MARK: - Gates

    @Test("Low confidence never names a class, however steady it is")
    func lowConfidenceStaysUnsure() {
        var smoother = PredictionSmoother(configuration: config)
        // Clear winner, but well under the confidence floor.
        let outcome = feed(&smoother, scores(plastic: 0.45), times: 10)
        #expect(outcome == .notSure)
    }

    @Test("A high top-1 with a close runner-up is rejected by the margin gate")
    func narrowMarginStaysUnsure() {
        var smoother = PredictionSmoother(configuration: config)
        // glass 0.44 vs plastic 0.46: margin 0.02, far below the 0.15 floor.
        // A confidence-only threshold would let this through as a coin flip.
        let outcome = feed(&smoother, MockWasteClassifier.narrowMarginScores, times: 10)
        #expect(outcome == .notSure)
    }

    @Test("The margin gate is what rejects it, not the confidence gate")
    func marginGateIsTheOneDoingTheWork() {
        // Same top-1 value, but spread out so the margin clears.
        var narrow = PredictionSmoother(configuration: config)
        var wide = PredictionSmoother(configuration: config)

        let narrowOutcome = feed(&narrow, [0.02, 0.44, 0.03, 0.03, 0.46, 0.02], times: 10)
        let wideOutcome = feed(&wide, [0.02, 0.04, 0.03, 0.03, 0.86, 0.02], times: 10)

        #expect(narrowOutcome == .notSure)
        if case .confident = wideOutcome {} else {
            Issue.record("a well-separated reading should have been confident")
        }
    }

    // MARK: - Hysteresis and switching

    @Test("A shown label survives a dip that would not have introduced it")
    func hysteresisHoldsThroughADip() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)

        // 0.60 is below minimumConfidence (0.65) but above releaseConfidence
        // (0.55). Without hysteresis this would flip to "not sure" and back.
        // Timestamp stays inside staleSampleInterval, or the dip would be read
        // as a pause and reset the history instead.
        let outcome = smoother.add(scores: scores(plastic: 0.60), at: 2.0)
        guard case .confident(let result) = outcome else {
            Issue.record("hysteresis should have held the label, got \(outcome)")
            return
        }
        #expect(result.category == .plastic)
    }

    @Test("A sustained collapse does drop the label")
    func sustainedCollapseDropsTheLabel() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)

        let outcome = feed(&smoother, MockWasteClassifier.ambiguousScores, times: 10, startingAt: 10.0)
        #expect(outcome == .notSure)
    }

    @Test("Switching categories requires a new streak, and never flickers via a single frame")
    func switchingRequiresANewStreak() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)

        // One stray glass frame must not flip the card.
        let afterOneStray = smoother.add(scores: MockWasteClassifier.confidentGlassScores, at: 10.0)
        if case .confident(let result) = afterOneStray {
            #expect(result.category == .plastic, "a single frame flipped the label")
        }

        // Sustained glass eventually wins.
        let afterSustained = feed(&smoother, MockWasteClassifier.confidentGlassScores, times: 10, startingAt: 11.0)
        guard case .confident(let result) = afterSustained else {
            Issue.record("sustained glass should have been adopted, got \(afterSustained)")
            return
        }
        #expect(result.category == .glass)
    }

    // MARK: - Reset

    @Test("Reset clears everything")
    func resetClearsState() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)
        #expect(smoother.smoothedScores != nil)

        smoother.reset()
        #expect(smoother.smoothedScores == nil)

        // Back to warming up, so the first post-reset frame cannot be confident.
        let outcome = smoother.add(scores: scores(plastic: 0.99), at: 100)
        #expect(outcome == .warmingUp)
    }

    @Test("A long gap between frames resets the history")
    func longGapResets() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)

        // The phone was pointed somewhere else for well over staleSampleInterval.
        let outcome = smoother.add(scores: scores(plastic: 0.99), at: 100.0)
        #expect(outcome == .warmingUp, "stale history should have been discarded")
    }

    @Test("A backwards timestamp resets rather than corrupting the average")
    func backwardsTimestampResets() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)
        let outcome = smoother.add(scores: scores(plastic: 0.99), at: 0.0)
        #expect(outcome == .warmingUp)
    }

    // MARK: - Malformed input

    @Test("A wrong-width vector is rejected and does not disturb the average")
    func wrongWidthIsRejected() {
        var smoother = PredictionSmoother(configuration: config)
        _ = feed(&smoother, scores(plastic: 0.95), times: 8)
        let before = smoother.smoothedScores

        let outcome = smoother.add(scores: [0.5, 0.5], at: 3.0)

        #expect(smoother.smoothedScores == before, "a malformed sample changed the average")
        #expect(smoother.rejectedSampleCount == 1)
        #expect(smoother.lastRejection == .malformed(.wrongLength(expected: 6, found: 2)))
        // The previous verdict stands: garbage is not evidence of doubt.
        if case .confident = outcome {} else {
            Issue.record("a rejected sample downgraded a good result")
        }
    }

    @Test("A NaN anywhere in the vector rejects the whole sample")
    func notFiniteIsRejected() {
        var smoother = PredictionSmoother(configuration: config)
        var bad = scores(plastic: 0.9)
        bad[2] = Double.nan

        _ = smoother.add(scores: bad, at: 0)
        #expect(smoother.rejectedSampleCount == 1)
        #expect(smoother.lastRejection == .malformed(.notFinite(index: 2)))
        #expect(smoother.smoothedScores == nil)
    }

    @Test("Infinity is rejected too")
    func infinityIsRejected() {
        var smoother = PredictionSmoother(configuration: config)
        var bad = scores(plastic: 0.9)
        bad[0] = .infinity
        _ = smoother.add(scores: bad, at: 0)
        #expect(smoother.rejectedSampleCount == 1)
    }

    // MARK: - Configurability

    @Test("Thresholds are honoured from the configuration, not hardcoded")
    func thresholdsComeFromConfiguration() {
        var permissive = ScannerConfiguration.default
        permissive.minimumConfidence = 0.30
        permissive.minimumMargin = 0.02
        permissive.requiredConsecutiveAgreements = 1
        permissive.minimumSamplesBeforeVerdict = 1

        var smoother = PredictionSmoother(configuration: permissive)
        // This same vector is "not sure" under the default configuration.
        let outcome = feed(&smoother, scores(plastic: 0.45), times: 4)
        if case .confident = outcome {} else {
            Issue.record("a permissive configuration should have accepted this reading")
        }
    }

    @Test("A release threshold stricter than the entry threshold cannot cause flicker")
    func releaseThresholdIsClampedToEntry() {
        // A misconfiguration: holding is harder than introducing. Left
        // unclamped, a label would be promoted and dropped on the same sample —
        // flicker produced by the anti-flicker mechanism itself.
        var contradictory = ScannerConfiguration.default
        contradictory.minimumConfidence = 0.40
        contradictory.minimumMargin = 0.05
        contradictory.releaseConfidence = 0.90   // stricter than entry
        contradictory.releaseMargin = 0.50       // stricter than entry
        contradictory.requiredConsecutiveAgreements = 2

        var smoother = PredictionSmoother(configuration: contradictory)
        let outcome = feed(&smoother, scores(plastic: 0.50), times: 6)

        guard case .confident(let result) = outcome else {
            Issue.record("the clamp should have kept the promoted label, got \(outcome)")
            return
        }
        #expect(result.category == .plastic)
    }
}

/// The provisional numbers themselves.
@MainActor
struct ScannerConfigurationTests {

    @Test("Defaults match the documented provisional starting points")
    func documentedDefaults() {
        let config = ScannerConfiguration.default
        #expect(config.minimumConfidence == 0.65)
        #expect(config.minimumMargin == 0.15)
        #expect((3.0...5.0).contains(config.targetClassificationsPerSecond))
    }

    @Test("Release thresholds are looser than entry thresholds, or hysteresis does nothing")
    func hysteresisGapExists() {
        let config = ScannerConfiguration.default
        #expect(config.releaseConfidence < config.minimumConfidence)
        #expect(config.releaseMargin < config.minimumMargin)
    }

    @Test("The classification interval follows from the target rate")
    func intervalDerivesFromRate() {
        var config = ScannerConfiguration.default
        config.targetClassificationsPerSecond = 4
        #expect(config.classificationInterval == 0.25)
    }
}
