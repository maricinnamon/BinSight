import Testing
@testable import BinSight

@MainActor
struct ClassificationResultTests {

    // MARK: - Clamping

    @Test("Confidence is clamped into 0...1", arguments: [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.42, 0.42),
        (1.0, 1.0),
        (1.8, 1.0),
    ])
    func confidenceIsClamped(input: Double, expected: Double) {
        let result = ClassificationResult(category: .paper, confidence: input)
        #expect(result.confidence == expected)
    }

    // MARK: - Formatting

    @Test("Confidence renders as a whole percentage", arguments: [
        (0.94, 94),
        (0.871, 87),
        (0.875, 88),
        (0.0, 0),
        (1.0, 100),
    ])
    func percentRounding(confidence: Double, expected: Int) {
        let result = ClassificationResult(category: .glass, confidence: confidence)
        #expect(result.confidencePercent == expected)
        #expect(result.confidenceText == "\(expected)%")
    }

    // MARK: - Threshold

    @Test("Confidence at or above the threshold counts as confident")
    func atThresholdIsConfident() {
        let atThreshold = ClassificationResult(
            category: .metal,
            confidence: ClassificationResult.confidenceThreshold
        )
        #expect(atThreshold.isConfident)
    }

    @Test("Confidence below the threshold is not confident")
    func belowThresholdIsNotConfident() {
        let belowThreshold = ClassificationResult(
            category: .metal,
            confidence: ClassificationResult.confidenceThreshold - 0.01
        )
        #expect(!belowThreshold.isConfident)
    }

    // MARK: - Fixtures

    @Test("Sample fixtures land on the side of the threshold they claim")
    func fixturesMatchTheirIntent() {
        #expect(ClassificationResult.samplePlastic.isConfident)
        #expect(ClassificationResult.sampleGlass.isConfident)
        #expect(ClassificationResult.sampleCardboard.isConfident)
        #expect(!ClassificationResult.sampleUncertain.isConfident)
    }
}
