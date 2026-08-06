import Foundation

/// A classifier that returns a fixed script of score vectors.
///
/// Deterministic on purpose: previews, UI tests and the pipeline's unit tests
/// all need repeatable results, and none of them may open a camera or load a
/// model. Nothing here is random.
///
/// An actor because `WasteClassifying` is called concurrently and this keeps a
/// cursor — the same reason the real LiteRT implementation will be one.
actor MockWasteClassifier: WasteClassifying {

    /// Cycles once through `script` then repeats the final entry, so a test can
    /// describe a run-up and then a steady state without listing it 30 times.
    private let script: [[Double]]
    private let error: ClassificationError?
    private let simulatedDuration: TimeInterval
    private var cursor = 0

    private(set) var callCount = 0

    init(
        script: [[Double]],
        error: ClassificationError? = nil,
        simulatedDuration: TimeInterval = 0.02
    ) {
        precondition(!script.isEmpty || error != nil, "a mock needs a script or an error")
        self.script = script
        self.error = error
        self.simulatedDuration = simulatedDuration
    }

    /// Always returns the same vector.
    init(constant scores: [Double], simulatedDuration: TimeInterval = 0.02) {
        self.init(script: [scores], simulatedDuration: simulatedDuration)
    }

    /// Always fails, for the recoverable/unrecoverable error paths.
    init(failingWith error: ClassificationError) {
        self.init(script: [], error: error)
    }

    func classify(_ frame: CameraFrame, crop: FrameCropPlan) async throws -> ClassifierOutput {
        callCount += 1

        if let error {
            throw error
        }

        let scores = script[min(cursor, script.count - 1)]
        cursor += 1

        return ClassifierOutput(
            scores: scores,
            preprocessingDuration: simulatedDuration * 0.3,
            invocationDuration: simulatedDuration * 0.7
        )
    }
}

// MARK: - Fixtures

extension MockWasteClassifier {
    /// A confident plastic reading, in canonical output order
    /// (cardboard, glass, metal, paper, plastic, general_waste).
    static let confidentPlasticScores: [Double] = [0.01, 0.03, 0.01, 0.02, 0.91, 0.02]

    /// A confident glass reading.
    static let confidentGlassScores: [Double] = [0.02, 0.88, 0.03, 0.02, 0.04, 0.01]

    /// Top-1 is high but the runner-up is right behind it — the case a single
    /// confidence threshold waves through and the margin gate catches.
    static let narrowMarginScores: [Double] = [0.02, 0.44, 0.03, 0.03, 0.46, 0.02]

    /// Nothing stands out.
    static let ambiguousScores: [Double] = [0.18, 0.17, 0.16, 0.17, 0.16, 0.16]
}
