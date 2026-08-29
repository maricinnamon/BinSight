import Foundation

/// One class and its score.
struct CategoryScore: Equatable, Sendable {
    let category: WasteCategory
    let score: Double
}

/// One classification of one item: the winning class, how sure the model is,
/// and — once a real model is wired up — the full score vector and how long it
/// took.
///
/// The confidence maths lives here as plain value-type logic so it can be unit
/// tested without touching SwiftUI or a model.
struct ClassificationResult: Equatable, Sendable {
    /// The predicted class.
    let category: WasteCategory

    /// Model confidence in `0...1`. Values outside that range are clamped, so
    /// a malformed prediction can never produce a nonsensical percentage.
    let confidence: Double

    /// Every class and its score, **sorted descending**. Empty when the result
    /// was built from a bare category/confidence pair.
    let rankedScores: [CategoryScore]

    /// The same scores in canonical model output order
    /// (`WasteCategory.modelOutputOrder`), kept for debugging: a ranked list
    /// hides an off-by-one in the label mapping, a canonical one does not.
    let canonicalScores: [Double]

    /// End-to-end time for the inference behind this result, when measured.
    let inferenceDuration: TimeInterval?

    /// Below this, BinSight declines to name a class and asks for a better
    /// angle instead. Showing a shaky guess is worse than admitting doubt.
    ///
    /// Kept for the single-shot case; the live pipeline uses the richer
    /// thresholds in `ScannerConfiguration`.
    static let confidenceThreshold: Double = 0.65

    init(
        category: WasteCategory,
        confidence: Double,
        rankedScores: [CategoryScore] = [],
        canonicalScores: [Double] = [],
        inferenceDuration: TimeInterval? = nil
    ) {
        self.category = category
        self.confidence = min(max(confidence, 0), 1)
        self.rankedScores = rankedScores
        self.canonicalScores = canonicalScores
        self.inferenceDuration = inferenceDuration
    }

    /// Builds a result from a full score vector in canonical output order.
    ///
    /// Returns `nil` for a vector that is the wrong width or holds a non-finite
    /// value — a caller that gets `nil` has a model contract problem, not a
    /// low-confidence frame.
    init?(canonicalScores scores: [Double], inferenceDuration: TimeInterval? = nil) {
        let order = WasteCategory.modelOutputOrder
        guard ScoreVector.validate(scores, expectedCount: order.count) == nil else { return nil }

        let probabilities = ScoreVector.softmaxIfNeeded(scores)
        let ranked = zip(order, probabilities)
            .map { CategoryScore(category: $0, score: $1) }
            .sorted { $0.score > $1.score }

        guard let best = ranked.first else { return nil }

        self.init(
            category: best.category,
            confidence: best.score,
            rankedScores: ranked,
            canonicalScores: probabilities,
            inferenceDuration: inferenceDuration
        )
    }

    var isConfident: Bool {
        confidence >= Self.confidenceThreshold
    }

    /// Confidence as a whole percentage point.
    var confidencePercent: Int {
        Int((confidence * 100).rounded())
    }

    /// Locale-aware — the percent sign is not always a suffix, and not always
    /// separated the same way.
    var confidenceText: String {
        Double(confidence).formatted(.percent.precision(.fractionLength(0)))
    }

    /// Gap between the best and second-best class.
    ///
    /// A high top-1 score with a close runner-up means the model is confident
    /// about two things at once, which is not confidence at all. Returns `nil`
    /// when there is no score vector to compare.
    var topTwoMargin: Double? {
        guard rankedScores.count >= 2 else { return nil }
        return rankedScores[0].score - rankedScores[1].score
    }
}

// MARK: - Fixtures

extension ClassificationResult {
    /// Deterministic samples used by previews, the DEBUG state menu and tests.
    /// Fixed values, never random, so a preview looks the same every time.
    static let samplePlastic = ClassificationResult(category: .plastic, confidence: 0.94)
    static let sampleGlass = ClassificationResult(category: .glass, confidence: 0.87)
    static let sampleCardboard = ClassificationResult(category: .cardboard, confidence: 0.71)
    static let sampleUncertain = ClassificationResult(category: .metal, confidence: 0.41)
}
