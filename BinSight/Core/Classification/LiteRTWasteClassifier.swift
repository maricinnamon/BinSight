import Accelerate
import CoreVideo
import Foundation
import TensorFlowLite
import os

/// The real classifier: a LiteRT interpreter over the bundled TrashNet model.
///
/// An `actor` because `TensorFlowLite.Interpreter` is not documented as
/// thread-safe, and the pipeline calls `classify` from a non-main context. The
/// interpreter is built **once** in `init` and reused for every frame —
/// allocating tensors per prediction would dominate the latency budget.
///
/// The whole preprocessing contract comes from `model_contract.json`:
/// 224×224 RGB, float32, NHWC, values in **[0, 255]**. MobileNetV3's rescaling
/// is baked into the graph, so there is deliberately no normalisation here.
actor LiteRTWasteClassifier: WasteClassifying {

    private let interpreter: Interpreter
    private let labels: [WasteCategory]
    private let inputSide: Int
    private let logger = Logger(subsystem: "com.marynaantonevych.BinSight", category: "classifier")

    /// Scratch buffers, allocated once and reused. Re-allocating 600 KB per
    /// frame at 4 fps is pure churn.
    private var scaledBGRA: [UInt8]
    private var rgbFloat: [Float32]

    // MARK: - Loading

    /// Builds a classifier from the app bundle, or throws a typed error saying
    /// exactly which part of the contract is unsatisfied.
    init(bundle: Bundle = .main, resource: String = "binsight_trashnet") throws {
        guard let modelPath = bundle.path(forResource: resource, ofType: "tflite") else {
            throw ClassificationError.modelMissingFromBundle(resource: "\(resource).tflite")
        }
        guard let labelsPath = bundle.path(forResource: "labels", ofType: "txt"),
              let labelsText = try? String(contentsOfFile: labelsPath, encoding: .utf8)
        else {
            throw ClassificationError.labelsMissingFromBundle(resource: "labels.txt")
        }

        let names = labelsText
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard names.count == WasteCategory.modelOutputOrder.count else {
            throw ClassificationError.labelCountMismatch(
                expected: WasteCategory.modelOutputOrder.count, found: names.count
            )
        }

        // Map each label to a category *by name*, so a reordered labels.txt
        // fails loudly instead of silently mislabelling every prediction.
        var mapped: [WasteCategory] = []
        for (index, name) in names.enumerated() {
            guard let category = WasteCategory(modelLabel: name) else {
                throw ClassificationError.labelNameMismatch(
                    index: index, expected: WasteCategory.modelOutputOrder[index].modelLabel, found: name
                )
            }
            mapped.append(category)
        }
        labels = mapped

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
        } catch {
            throw ClassificationError.interpreterUnavailable(error.localizedDescription)
        }

        // --- Contract checks, before a single frame is classified ---------
        let input = try Self.tensor(interpreter.input(at:), index: 0, isInput: true)
        let output = try Self.tensor(interpreter.output(at:), index: 0, isInput: false)

        guard input.shape.dimensions.count == 4,
              input.shape.dimensions[0] == 1,
              input.shape.dimensions[1] == input.shape.dimensions[2],
              input.shape.dimensions[3] == 3,
              input.dataType == .float32
        else {
            throw ClassificationError.inputTensorMismatch(
                expected: "[1, N, N, 3] float32",
                found: "\(input.shape.dimensions) \(input.dataType)"
            )
        }
        guard output.shape.dimensions.count == 2,
              output.shape.dimensions[1] == mapped.count,
              output.dataType == .float32
        else {
            throw ClassificationError.outputTensorMismatch(
                expected: "[1, \(mapped.count)] float32",
                found: "\(output.shape.dimensions) \(output.dataType)"
            )
        }

        inputSide = input.shape.dimensions[1]
        scaledBGRA = [UInt8](repeating: 0, count: inputSide * inputSide * 4)
        rgbFloat = [Float32](repeating: 0, count: inputSide * inputSide * 3)

        logger.info(
            "Classifier ready: \(input.shape.dimensions, privacy: .public) -> \(output.shape.dimensions, privacy: .public), \(mapped.count, privacy: .public) labels"
        )
    }

    private static func tensor(_ accessor: (Int) throws -> Tensor, index: Int, isInput: Bool) throws -> Tensor {
        do {
            return try accessor(index)
        } catch {
            throw isInput
                ? ClassificationError.inputTensorMismatch(expected: "tensor 0", found: error.localizedDescription)
                : ClassificationError.outputTensorMismatch(expected: "tensor 0", found: error.localizedDescription)
        }
    }

    // MARK: - Inference

    func classify(_ frame: CameraFrame, crop: FrameCropPlan) async throws -> ClassifierOutput {
        let preprocessStart = ContinuousClock.now
        let input = try preprocess(frame.pixelBuffer, crop: crop)
        let preprocessing = (ContinuousClock.now - preprocessStart).seconds

        let invokeStart = ContinuousClock.now
        let raw: [Float32]
        do {
            try input.withUnsafeBufferPointer { pointer in
                try interpreter.copy(Data(buffer: pointer), toInputAt: 0)
            }
            try interpreter.invoke()
            let output = try interpreter.output(at: 0)
            raw = output.data.toArray(of: Float32.self)
        } catch {
            throw ClassificationError.invocationFailed(error.localizedDescription)
        }
        let invocation = (ContinuousClock.now - invokeStart).seconds

        guard raw.count == labels.count else {
            throw ClassificationError.invalidOutput("expected \(labels.count) scores, got \(raw.count)")
        }
        let scores = raw.map(Double.init)
        guard scores.allSatisfy(\.isFinite) else {
            throw ClassificationError.invalidOutput("non-finite value in output")
        }

        return ClassifierOutput(
            scores: scores,
            preprocessingDuration: preprocessing,
            invocationDuration: invocation
        )
    }

    // MARK: - Preprocessing

    /// Crops, scales and converts one pixel buffer into the model's input.
    ///
    /// Done with vImage rather than `UIImage`/`CGImage` round trips: at 4 fps a
    /// CoreGraphics conversion per frame is both slower and allocates heavily.
    /// The camera delivers BGRA (see `CameraSessionController`), so the channel
    /// swizzle to RGB happens here, explicitly.
    private func preprocess(_ pixelBuffer: CVPixelBuffer, crop: FrameCropPlan) throws -> [Float32] {
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
            throw ClassificationError.preprocessingFailed("could not lock the pixel buffer")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw ClassificationError.preprocessingFailed("pixel buffer has no base address")
        }

        let bufferWidth = CVPixelBufferGetWidth(pixelBuffer)
        let bufferHeight = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // The crop rect comes from FrameCropPlan, which is derived from the same
        // geometry the preview layer uses — so the model sees what the person framed.
        let pixelRect = crop.pixelRect(bufferSize: CGSize(width: bufferWidth, height: bufferHeight))
        let x = max(0, min(Int(pixelRect.minX), bufferWidth - 1))
        let y = max(0, min(Int(pixelRect.minY), bufferHeight - 1))
        let width = max(1, min(Int(pixelRect.width), bufferWidth - x))
        let height = max(1, min(Int(pixelRect.height), bufferHeight - y))

        // A view onto the crop region — no copy.
        var source = vImage_Buffer(
            data: base.advanced(by: y * bytesPerRow + x * 4),
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: bytesPerRow
        )

        let side = inputSide
        let scaleStatus = scaledBGRA.withUnsafeMutableBytes { destinationBytes -> vImage_Error in
            guard let destinationBase = destinationBytes.baseAddress else { return kvImageNullPointerArgument }
            var destination = vImage_Buffer(
                data: destinationBase,
                height: vImagePixelCount(side),
                width: vImagePixelCount(side),
                rowBytes: side * 4
            )
            // Bilinear, matching the training pipeline's tf.image.resize.
            return vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageNoFlags))
        }
        guard scaleStatus == kvImageNoError else {
            throw ClassificationError.preprocessingFailed("vImageScale failed (\(scaleStatus))")
        }

        // BGRA (uint8) -> RGB (float32, still 0...255).
        for pixel in 0..<(side * side) {
            let source = pixel * 4
            let destination = pixel * 3
            rgbFloat[destination] = Float32(scaledBGRA[source + 2])      // R
            rgbFloat[destination + 1] = Float32(scaledBGRA[source + 1])  // G
            rgbFloat[destination + 2] = Float32(scaledBGRA[source])      // B
        }
        return rgbFloat
    }
}

// MARK: - Helpers

private extension Data {
    /// Reinterprets the tensor's bytes as an array of `T`.
    func toArray<T>(of type: T.Type) -> [T] {
        withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }
}

private extension Duration {
    var seconds: TimeInterval {
        let (whole, attoseconds) = components
        return TimeInterval(whole) + TimeInterval(attoseconds) / 1e18
    }
}
