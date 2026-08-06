import Foundation

/// One model evaluation.
///
/// Scores are in **canonical model output order** — `WasteCategory.modelOutputOrder`
/// — never sorted. Sorting happens once, at the edge, in `ClassificationResult`.
struct ClassifierOutput: Equatable, Sendable {
    /// Probabilities in output-index order. Always `WasteCategory.allCases.count` long.
    let scores: [Double]
    let preprocessingDuration: TimeInterval
    let invocationDuration: TimeInterval

    var totalDuration: TimeInterval { preprocessingDuration + invocationDuration }

    init(scores: [Double], preprocessingDuration: TimeInterval = 0, invocationDuration: TimeInterval = 0) {
        self.scores = scores
        self.preprocessingDuration = preprocessingDuration
        self.invocationDuration = invocationDuration
    }
}

/// Everything that can go wrong between a camera frame and six numbers.
///
/// Typed rather than a bare `Error` so the UI can tell "the model is missing
/// from the bundle" (a build problem, unrecoverable at runtime) apart from
/// "this frame failed to convert" (transient, worth retrying).
enum ClassificationError: Error, Equatable, Sendable {
    case modelMissingFromBundle(resource: String)
    case labelsMissingFromBundle(resource: String)
    case labelCountMismatch(expected: Int, found: Int)
    case labelNameMismatch(index: Int, expected: String, found: String)
    case interpreterUnavailable(String)
    case inputTensorMismatch(expected: String, found: String)
    case outputTensorMismatch(expected: String, found: String)
    case preprocessingFailed(String)
    case invocationFailed(String)
    case invalidOutput(String)

    /// Whether retrying the same frame could plausibly work.
    ///
    /// A missing model will still be missing next frame; a failed CVPixelBuffer
    /// lock might not be.
    var isRecoverable: Bool {
        switch self {
        case .modelMissingFromBundle, .labelsMissingFromBundle,
             .labelCountMismatch, .labelNameMismatch,
             .interpreterUnavailable, .inputTensorMismatch, .outputTensorMismatch:
            false
        case .preprocessingFailed, .invocationFailed, .invalidOutput:
            true
        }
    }
}

/// The seam between the scanner pipeline and whatever actually runs the model.
///
/// Deliberately one method. The live pipeline, the smoother and every test
/// above this line are written against this protocol, so the real LiteRT
/// implementation drops in without touching any of them — and previews and UI
/// tests use `MockWasteClassifier` and never open the camera or load a model.
///
/// `Sendable` because the pipeline calls it from a non-main context;
/// implementations are expected to be actors (the TFLite interpreter is not
/// assumed thread-safe).
protocol WasteClassifying: Sendable {
    /// Classifies the region of `frame` described by `crop`.
    func classify(_ frame: CameraFrame, crop: FrameCropPlan) async throws -> ClassifierOutput
}
