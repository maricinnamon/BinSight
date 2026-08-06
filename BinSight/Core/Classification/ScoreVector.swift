import Foundation

/// Validation and small maths for a six-class probability vector.
///
/// Pulled out as free functions so the rules about what counts as a usable
/// model output live in exactly one place — and so they can be tested without
/// a model, a camera or a smoother.
enum ScoreVector {

    /// Why a score vector was rejected.
    enum Rejection: Equatable, Sendable {
        case wrongLength(expected: Int, found: Int)
        case notFinite(index: Int)
        case negative(index: Int)
    }

    /// Checks a raw model output before anything downstream trusts it.
    ///
    /// Values are *not* required to sum to 1: a model emitting logits is
    /// legitimate, and `softmaxIfNeeded` handles that. What is never acceptable
    /// is a NaN, an infinity, or a wrong-width vector.
    static func validate(_ scores: [Double], expectedCount: Int) -> Rejection? {
        guard scores.count == expectedCount else {
            return .wrongLength(expected: expectedCount, found: scores.count)
        }
        for (index, value) in scores.enumerated() {
            guard value.isFinite else { return .notFinite(index: index) }
        }
        return nil
    }

    /// Whether a vector already looks like a probability distribution.
    ///
    /// Used to decide if softmax is needed. Applying softmax to something that
    /// is already softmaxed is a silent bug: it does not crash, it just flattens
    /// every confidence toward 1/n and makes the model look unsure of
    /// everything.
    static func looksLikeProbabilities(_ scores: [Double], tolerance: Double = 0.02) -> Bool {
        guard !scores.isEmpty else { return false }
        guard scores.allSatisfy({ $0 >= -tolerance && $0 <= 1 + tolerance }) else { return false }
        return abs(scores.reduce(0, +) - 1.0) <= tolerance
    }

    /// Softmax, numerically stabilised by subtracting the max.
    static func softmax(_ scores: [Double]) -> [Double] {
        guard let maximum = scores.max() else { return scores }
        let exponentiated = scores.map { Foundation.exp($0 - maximum) }
        let total = exponentiated.reduce(0, +)
        guard total > 0, total.isFinite else {
            // Degenerate input: fall back to a uniform distribution rather than
            // dividing by zero and producing NaNs.
            return Array(repeating: 1.0 / Double(scores.count), count: scores.count)
        }
        return exponentiated.map { $0 / total }
    }

    /// Applies softmax **only** if the vector is not already a distribution.
    ///
    /// The exported BinSight model ends in a softmax layer (see
    /// `model_contract.json`, `output.activation`), so in practice this is a
    /// no-op — it exists so a future logits-output model cannot silently produce
    /// double-softmaxed confidences.
    static func softmaxIfNeeded(_ scores: [Double]) -> [Double] {
        looksLikeProbabilities(scores) ? scores : softmax(scores)
    }

    /// Normalises so the vector sums to 1, leaving an all-zero vector uniform.
    static func normalized(_ scores: [Double]) -> [Double] {
        let total = scores.reduce(0, +)
        guard total > 0, total.isFinite else {
            return Array(repeating: 1.0 / Double(max(scores.count, 1)), count: scores.count)
        }
        return scores.map { $0 / total }
    }
}
