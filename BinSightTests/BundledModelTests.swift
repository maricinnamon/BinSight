import CommonCrypto
import CoreGraphics
import CoreVideo
import Foundation
import Testing
@testable import BinSight

/// The bundled model and its contract.
///
/// These run against the **real** `.tflite` in the app bundle — they are the
/// check that the artefact shipped, that its tensors match what the Swift code
/// assumes, and that the labels line up with `WasteCategory`.
@MainActor
struct BundledModelTests {

    private var bundle: Bundle { Bundle(for: BundleToken.self) }

    /// The app bundle, which unit tests load into via TEST_HOST.
    private var appBundle: Bundle { .main }

    // MARK: - Resources exist

    @Test("The model file is in the app bundle")
    func modelIsBundled() throws {
        let path = appBundle.path(forResource: "binsight_trashnet", ofType: "tflite")
        #expect(path != nil, "binsight_trashnet.tflite is missing from the bundle")
        if let path {
            let size = try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int ?? 0
            #expect(size > 100_000, "model file is implausibly small (\(size) bytes)")
        }
    }

    @Test("The labels file is in the app bundle")
    func labelsAreBundled() {
        #expect(appBundle.path(forResource: "labels", ofType: "txt") != nil)
    }

    @Test("Labels match the model output order, and there are exactly six")
    func labelsMatchCategoryOrder() throws {
        guard let path = appBundle.path(forResource: "labels", ofType: "txt") else {
            Issue.record("labels.txt missing")
            return
        }
        let names = try String(contentsOfFile: path, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        #expect(names.count == WasteCategory.modelOutputOrder.count)
        // Resolve each label the same way the classifier does.
        let categories = names.map(WasteCategory.init(modelLabel:))
        let allResolved = categories.allSatisfy { $0 != nil }
        let resolved = categories.compactMap { $0 }
        #expect(allResolved, "a label did not resolve: \(names)")
        #expect(resolved == WasteCategory.modelOutputOrder)
    }

    @Test("TrashNet's `trash` maps to general waste, never to a recycling stream")
    func trashMapsToGeneralWaste() {
        #expect(WasteCategory(modelLabel: "trash") == .generalWaste)
        #expect(WasteCategory(modelLabel: "general_waste") == .generalWaste)
        #expect(WasteCategory.generalWaste.modelLabel == "trash")
        #expect(WasteCategory.generalWaste.displayName == "General waste")
    }

    @Test("An unknown label is rejected rather than guessed")
    func unknownLabelIsRejected() {
        // Silently mapping an unrecognised label would mislabel every prediction.
        #expect(WasteCategory(modelLabel: "compost") == nil)
        #expect(WasteCategory(modelLabel: "") == nil)
    }

    // MARK: - Contract

    @Test("The bundled contract describes the input the Swift code sends")
    func contractMatchesPreprocessing() throws {
        guard let path = appBundle.path(forResource: "model_contract", ofType: "json"),
              let data = FileManager.default.contents(atPath: path),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let input = json["input"] as? [String: Any],
              let output = json["output"] as? [String: Any]
        else {
            Issue.record("model_contract.json missing or unreadable")
            return
        }

        #expect(input["shape"] as? [Int] == [1, 224, 224, 3])
        #expect(input["dtype"] as? String == "float32")
        #expect(input["color_order"] as? String == "RGB")
        #expect(input["layout"] as? String == "NHWC")
        // [0, 255]: normalisation is baked into the graph, so the client must not repeat it.
        #expect(input["value_range"] as? [Double] == [0.0, 255.0])

        #expect(output["shape"] as? [Int] == [1, 6])
        #expect(output["dtype"] as? String == "float32")
        #expect(output["activation"] as? String == "softmax")
        #expect(output["is_logits"] as? Bool == false)

        // The crop plan targets exactly the contract's input size.
        #expect(FrameCropPlan.modelInputSize == CGSize(width: 224, height: 224))
    }

    @Test("The model does not need Flex ops")
    func contractDoesNotRequireSelectTFOps() throws {
        guard let path = appBundle.path(forResource: "model_contract", ofType: "json"),
              let data = FileManager.default.contents(atPath: path),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let runtime = json["runtime"] as? [String: Any]
        else {
            Issue.record("model_contract.json missing")
            return
        }
        // A Flex-dependent model cannot run on the plain iOS LiteRT runtime.
        #expect(runtime["requires_select_tf_ops"] as? Bool == false)
    }

    @Test("The bundled model's checksum matches the one recorded at export")
    func checksumMatchesContract() throws {
        guard let modelPath = appBundle.path(forResource: "binsight_trashnet", ofType: "tflite"),
              let contractPath = appBundle.path(forResource: "model_contract", ofType: "json"),
              let contractData = FileManager.default.contents(atPath: contractPath),
              let json = try JSONSerialization.jsonObject(with: contractData) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let expected = model["sha256"] as? String
        else {
            Issue.record("model or contract missing from the bundle")
            return
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
        let digest = Self.sha256Hex(data)
        #expect(digest == expected, "bundled model does not match the exported artefact")
    }

    // MARK: - Real inference

    @Test("The classifier loads and produces six finite probabilities")
    func classifierRunsOnASyntheticFrame() async throws {
        let classifier: LiteRTWasteClassifier
        do {
            classifier = try LiteRTWasteClassifier(bundle: appBundle)
        } catch {
            Issue.record("classifier failed to load: \(error)")
            return
        }

        guard let frame = Self.makeFrame(width: 640, height: 480) else {
            Issue.record("could not synthesise a pixel buffer")
            return
        }
        let crop = FrameCropPlan.fullVisibleRegion(
            bufferSize: CGSize(width: 640, height: 480),
            previewSize: CGSize(width: 390, height: 844)
        )

        let output = try await classifier.classify(frame, crop: crop)

        let allFinite = output.scores.allSatisfy { $0.isFinite }
        let allInRange = output.scores.allSatisfy { $0 >= 0 && $0 <= 1 }
        let total = output.scores.reduce(0, +)

        #expect(output.scores.count == WasteCategory.modelOutputOrder.count)
        #expect(allFinite)
        #expect(allInRange)
        #expect(abs(total - 1.0) < 0.01, "scores should be a distribution")
        #expect(output.invocationDuration > 0, "a real invocation takes measurable time")
        #expect(output.preprocessingDuration > 0)
    }

    @Test("A result built from the model's output has a valid top category")
    func resultFromModelOutputIsWellFormed() async throws {
        let classifier = try LiteRTWasteClassifier(bundle: appBundle)
        guard let frame = Self.makeFrame(width: 640, height: 480) else { return }
        let crop = FrameCropPlan.fullVisibleRegion(
            bufferSize: CGSize(width: 640, height: 480),
            previewSize: CGSize(width: 390, height: 844)
        )
        let output = try await classifier.classify(frame, crop: crop)

        guard let result = ClassificationResult(canonicalScores: output.scores) else {
            Issue.record("could not build a result from the model output")
            return
        }
        #expect(WasteCategory.allCases.contains(result.category))
        #expect(result.confidence >= 0 && result.confidence <= 1)
        #expect(result.rankedScores.count == 6)
    }

    @Test("A weak prediction is turned into 'not sure' rather than a class")
    func lowConfidenceBecomesNotSure() {
        let threshold = ScannerConfiguration.default.minimumConfidence
        guard let weak = ClassificationResult(canonicalScores: [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]) else {
            Issue.record("fixture should build a result")
            return
        }
        #expect(weak.confidence < threshold)
        // The app names no class below the threshold.
        let state = ScannerDisplayState.classified(weak)
        #expect(state == .notSure)
    }

    // MARK: - Helpers

    private final class BundleToken {}

    private static func makeFrame(width: Int, height: Int) -> CameraFrame? {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
        guard status == kCVReturnSuccess, let buffer else { return nil }

        // Fill with a deterministic gradient so preprocessing has real data.
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let pointer = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * bytesPerRow + x * 4
                    pointer[offset] = UInt8(x % 256)          // B
                    pointer[offset + 1] = UInt8(y % 256)      // G
                    pointer[offset + 2] = UInt8((x + y) % 256) // R
                    pointer[offset + 3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return CameraFrame(pixelBuffer: buffer, timestamp: 0, orientation: .up)
    }

    private static func sha256Hex(_ data: Data) -> String {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBytes { bytes in
            _ = CC_SHA256(bytes.baseAddress, CC_LONG(data.count), &hash)
        }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}
