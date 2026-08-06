import Foundation

/// Turns a noisy stream of per-frame probabilities into a label that a person
/// can actually read.
///
/// A raw classifier at 4 frames a second produces a result card that strobes.
/// This does four things to stop that, in order:
///
/// 1. **Averages** — an exponential moving average over recent probability
///    vectors, so one blurred frame cannot move the answer.
/// 2. **Two gates** — a smoothed top-1 threshold *and* a top-1/top-2 margin.
///    Either alone is insufficient: a high score with a close runner-up is a
///    coin flip wearing a big number.
/// 3. **Stability** — a new label must win several consecutive samples before
///    it is shown, so a passing frame cannot flip the card.
/// 4. **Hysteresis** — a label already on screen is held to looser thresholds
///    than one being introduced, so a result hovering at the boundary does not
///    oscillate between a category and "not sure".
///
/// A pure value type on purpose: no camera, no model, no clock of its own. All
/// timing is passed in, so every behaviour above is deterministically testable.
struct PredictionSmoother {

    /// What the screen should show after the latest sample.
    enum Outcome: Equatable, Sendable {
        /// Not enough samples yet — keep showing "scanning", not "not sure".
        case warmingUp
        /// Samples exist but nothing clears the gates.
        case notSure
        case confident(ClassificationResult)
    }

    /// Why a sample was thrown away without affecting the average.
    enum Rejection: Equatable, Sendable {
        case malformed(ScoreVector.Rejection)
    }

    private let configuration: ScannerConfiguration
    private let order = WasteCategory.modelOutputOrder

    // MARK: - State

    private var average: [Double]?
    private var sampleCount = 0
    private var lastTimestamp: TimeInterval?

    /// The class currently building a case for itself, and for how long.
    private var candidate: WasteCategory?
    private var candidateStreak = 0

    /// The class currently on screen, if any.
    private var displayed: WasteCategory?

    private var unconfidentStreak = 0

    /// Set when the most recent sample was discarded. Diagnostic only.
    private(set) var lastRejection: Rejection?

    /// Number of samples discarded as malformed since the last reset.
    private(set) var rejectedSampleCount = 0

    // MARK: - Init

    init(configuration: ScannerConfiguration = .default) {
        self.configuration = configuration
    }

    /// The current smoothed probabilities, in canonical output order.
    /// Exposed for the DEBUG overlay and for tests.
    var smoothedScores: [Double]? { average }

    // MARK: - Input

    /// Feeds one model output in.
    ///
    /// - Parameters:
    ///   - scores: probabilities in canonical output order.
    ///   - timestamp: a **monotonic** time in seconds. Used only for the
    ///     stale-gap reset, never for latency.
    @discardableResult
    mutating func add(scores: [Double], at timestamp: TimeInterval) -> Outcome {
        // --- Reject anything malformed without letting it touch the average.
        if let rejection = ScoreVector.validate(scores, expectedCount: order.count) {
            lastRejection = .malformed(rejection)
            rejectedSampleCount += 1
            // A garbage frame is not evidence of anything, so the previous
            // verdict stands rather than being downgraded.
            return currentOutcome()
        }
        lastRejection = nil

        // --- A long gap means the phone was somewhere else. Start fresh.
        if let last = lastTimestamp, timestamp - last > configuration.staleSampleInterval {
            reset()
        }
        // A timestamp going backwards means the clock rebased; also start fresh.
        if let last = lastTimestamp, timestamp < last {
            reset()
        }
        lastTimestamp = timestamp

        let probabilities = ScoreVector.softmaxIfNeeded(scores)
        updateAverage(with: probabilities)

        return evaluate()
    }

    /// Clears all history.
    ///
    /// Call on camera restart, interruption, error, permission loss, or when
    /// the scanner disappears — anything that makes previous frames describe a
    /// different situation.
    mutating func reset() {
        average = nil
        sampleCount = 0
        lastTimestamp = nil
        candidate = nil
        candidateStreak = 0
        displayed = nil
        unconfidentStreak = 0
        lastRejection = nil
        rejectedSampleCount = 0
    }

    // MARK: - Internals

    private mutating func updateAverage(with probabilities: [Double]) {
        if let previous = average {
            let alpha = configuration.smoothingFactor
            average = zip(previous, probabilities).map { alpha * $1 + (1 - alpha) * $0 }
        } else {
            average = probabilities
        }
        sampleCount += 1
    }

    private mutating func evaluate() -> Outcome {
        guard let average, let ranking = Ranking(scores: average, order: order) else {
            return .warmingUp
        }

        // Gate 1 and 2, at the strict thresholds used to *introduce* a label.
        let qualifies = ranking.topScore >= configuration.minimumConfidence
            && ranking.margin >= configuration.minimumMargin

        if qualifies {
            unconfidentStreak = 0
            if candidate == ranking.topCategory {
                candidateStreak += 1
            } else {
                candidate = ranking.topCategory
                candidateStreak = 1
            }
        } else {
            unconfidentStreak += 1
            candidate = nil
            candidateStreak = 0

            // Nothing recognisable for a long time: old evidence is only
            // dragging the average down.
            if unconfidentStreak >= configuration.resetAfterUnconfidentSamples {
                let keepTimestamp = lastTimestamp
                reset()
                lastTimestamp = keepTimestamp
                return .notSure
            }
        }

        guard sampleCount >= configuration.minimumSamplesBeforeVerdict else {
            return .warmingUp
        }

        // Promote a candidate that has held up across enough samples.
        if let candidate, candidateStreak >= configuration.requiredConsecutiveAgreements {
            displayed = candidate
        }

        // Hysteresis: hold what is on screen at the looser thresholds.
        //
        // The release thresholds are clamped to be no stricter than the entry
        // ones. Without that, a configuration where release > entry would
        // promote a label and drop it on the very same sample — flicker caused
        // by the anti-flicker mechanism.
        if let shown = displayed {
            let holdConfidence = min(configuration.releaseConfidence, configuration.minimumConfidence)
            let holdMargin = min(configuration.releaseMargin, configuration.minimumMargin)

            let stillOnTop = shown == ranking.topCategory
            let stillPlausible = ranking.topScore >= holdConfidence
                && ranking.margin >= holdMargin

            if stillOnTop && stillPlausible {
                return .confident(result(for: shown, ranking: ranking))
            }
            // Dropped. A different class now leading must build its own streak
            // before it appears — showing a stale label would be worse.
            displayed = nil
        }

        return .notSure
    }

    private func currentOutcome() -> Outcome {
        guard let average, let ranking = Ranking(scores: average, order: order) else {
            return sampleCount == 0 ? .warmingUp : .notSure
        }
        guard let shown = displayed else {
            return sampleCount >= configuration.minimumSamplesBeforeVerdict ? .notSure : .warmingUp
        }
        return .confident(result(for: shown, ranking: ranking))
    }

    private func result(for category: WasteCategory, ranking: Ranking) -> ClassificationResult {
        ClassificationResult(
            category: category,
            confidence: ranking.score(for: category),
            rankedScores: ranking.ranked,
            canonicalScores: ranking.scores
        )
    }
}

/// A scored vector with its ordering worked out once.
private struct Ranking {
    let scores: [Double]
    let ranked: [CategoryScore]

    init?(scores: [Double], order: [WasteCategory]) {
        guard scores.count == order.count, !scores.isEmpty else { return nil }
        self.scores = scores
        self.ranked = zip(order, scores)
            .map { CategoryScore(category: $0, score: $1) }
            .sorted { $0.score > $1.score }
    }

    var topCategory: WasteCategory { ranked[0].category }
    var topScore: Double { ranked[0].score }
    var margin: Double { ranked.count >= 2 ? ranked[0].score - ranked[1].score : ranked[0].score }

    func score(for category: WasteCategory) -> Double {
        ranked.first { $0.category == category }?.score ?? 0
    }
}
