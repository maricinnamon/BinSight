import Testing
@testable import BinSight

/// Output validation, softmax, and the double-softmax trap.
@MainActor
struct ScoreVectorTests {

    @Test("A well-formed vector passes validation")
    func validVectorPasses() {
        #expect(ScoreVector.validate([0.1, 0.2, 0.3, 0.2, 0.1, 0.1], expectedCount: 6) == nil)
    }

    @Test("Wrong width is reported with both counts")
    func wrongWidthReported() {
        #expect(ScoreVector.validate([0.5, 0.5], expectedCount: 6) == .wrongLength(expected: 6, found: 2))
    }

    @Test("Non-finite values are reported with their index")
    func nonFiniteReported() {
        #expect(ScoreVector.validate([0.1, .nan, 0.3, 0.2, 0.2, 0.2], expectedCount: 6) == .notFinite(index: 1))
        #expect(ScoreVector.validate([0.1, 0.2, -.infinity, 0.2, 0.2, 0.2], expectedCount: 6) == .notFinite(index: 2))
    }

    // MARK: - Probability detection

    @Test("A distribution summing to 1 is recognised as probabilities")
    func recognisesProbabilities() {
        #expect(ScoreVector.looksLikeProbabilities([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]))
    }

    @Test("Logits are not mistaken for probabilities")
    func logitsAreNotProbabilities() {
        #expect(!ScoreVector.looksLikeProbabilities([2.4, -1.0, 0.3, 5.2, 0.0, -3.1]))
        // Sums to 1 but has a negative member - still not a distribution.
        #expect(!ScoreVector.looksLikeProbabilities([1.5, -0.5, 0.0, 0.0, 0.0, 0.0]))
    }

    // MARK: - Softmax

    @Test("Softmax produces a distribution")
    func softmaxProducesDistribution() {
        let result = ScoreVector.softmax([2.0, 1.0, 0.1, 0.0, -1.0, 3.0])
        #expect(abs(result.reduce(0, +) - 1.0) < 1e-9)
        #expect(result.allSatisfy { $0 > 0 && $0 < 1 })
        // Largest logit keeps the largest probability.
        #expect(result.firstIndex(of: result.max()!) == 5)
    }

    @Test("Softmax is numerically stable for large logits")
    func softmaxIsStable() {
        // Naively exponentiating 1000 overflows to infinity and yields NaNs.
        let result = ScoreVector.softmax([1000, 1001, 999, 998, 997, 996])
        #expect(result.allSatisfy { $0.isFinite })
        #expect(abs(result.reduce(0, +) - 1.0) < 1e-9)
    }

    @Test("A degenerate vector falls back to uniform rather than producing NaNs")
    func degenerateFallsBackToUniform() {
        let result = ScoreVector.softmax([])
        #expect(result.isEmpty)
    }

    // MARK: - The double-softmax trap

    @Test("Softmax is not applied to something already softmaxed")
    func doesNotDoubleSoftmax() {
        // The real failure this guards: applying softmax twice does not crash,
        // it silently flattens every confidence toward 1/n and makes a good
        // model look permanently unsure.
        let probabilities = [0.90, 0.04, 0.02, 0.02, 0.01, 0.01]
        let passedThrough = ScoreVector.softmaxIfNeeded(probabilities)
        #expect(passedThrough == probabilities)

        let doubled = ScoreVector.softmax(probabilities)
        #expect(doubled[0] < 0.35, "double softmax should have flattened the peak")
        #expect(passedThrough[0] == 0.90)
    }

    @Test("Logits do get softmaxed")
    func logitsAreSoftmaxed() {
        let logits = [2.4, -1.0, 0.3, 5.2, 0.0, -3.1]
        let result = ScoreVector.softmaxIfNeeded(logits)
        #expect(result != logits)
        #expect(abs(result.reduce(0, +) - 1.0) < 1e-9)
    }
}

/// Building a result from a raw score vector, and the label mapping.
@MainActor
struct ClassificationResultMappingTests {

    @Test("Canonical order maps index to category correctly")
    func canonicalOrderMapsCorrectly() {
        // Index 4 is plastic; index 5 is general waste. If this ever reads as
        // alphabetical, every prediction is silently mislabelled.
        let order = WasteCategory.modelOutputOrder
        #expect(order.count == 6)
        #expect(order[0] == .cardboard)
        #expect(order[1] == .glass)
        #expect(order[2] == .metal)
        #expect(order[3] == .paper)
        #expect(order[4] == .plastic)
        #expect(order[5] == .generalWaste)
    }

    @Test("The canonical order matches the trained model's labels.txt")
    func matchesLabelsFile() {
        // These are TrashNet's own class-directory names, which is what the
        // trained model emits and what ml/experiments/trashnet_tensorflow_baseline/labels.txt holds.
        let labels = WasteCategory.modelOutputOrder.map(\.modelLabel)
        #expect(labels == ["cardboard", "glass", "metal", "paper", "plastic", "trash"])
    }

    @Test("The order is fixed by declaration, not derived from a sort")
    func orderIsNotDerivedFromSorting() {
        // With TrashNet's names the correct order happens to *coincide* with
        // alphabetical — a coincidence, not a guarantee. The app's own internal
        // names show the difference: sorted, generalWaste would land at index 1
        // and every prediction would be silently mislabelled. The order is
        // therefore taken from the declaration, never from `.sorted()`.
        let internalNames = WasteCategory.modelOutputOrder.map(\.rawValue)
        #expect(internalNames != internalNames.sorted())
        #expect(internalNames.last == "generalWaste")
    }

    @Test("A score vector picks the right winner")
    func picksTheRightWinner() {
        let result = ClassificationResult(canonicalScores: MockWasteClassifier.confidentPlasticScores)
        #expect(result?.category == .plastic)
        #expect(result?.confidencePercent == 91)
    }

    @Test("Ranked scores are sorted descending while canonical order is preserved")
    func keepsBothOrderings() {
        let scores = MockWasteClassifier.confidentGlassScores
        guard let result = ClassificationResult(canonicalScores: scores) else {
            Issue.record("expected a result")
            return
        }
        // Ranked: descending.
        let ranked = result.rankedScores.map(\.score)
        #expect(ranked == ranked.sorted(by: >))
        #expect(result.rankedScores[0].category == .glass)
        // Canonical: untouched output order, for debugging a label off-by-one.
        #expect(result.canonicalScores == scores)
    }

    @Test("Top-two margin is computed from the ranked scores")
    func computesMargin() {
        let result = ClassificationResult(canonicalScores: MockWasteClassifier.narrowMarginScores)
        // 0.46 plastic vs 0.44 glass.
        #expect(abs((result?.topTwoMargin ?? 0) - 0.02) < 1e-9)
    }

    @Test("A malformed vector produces no result at all")
    func malformedVectorFailsClearly() {
        #expect(ClassificationResult(canonicalScores: [0.5, 0.5]) == nil)
        #expect(ClassificationResult(canonicalScores: [0.1, .nan, 0.2, 0.2, 0.2, 0.3]) == nil)
    }

    @Test("A bare category/confidence result has no score vector")
    func bareResultHasNoScores() {
        let result = ClassificationResult.samplePlastic
        #expect(result.rankedScores.isEmpty)
        #expect(result.topTwoMargin == nil)
    }
}
