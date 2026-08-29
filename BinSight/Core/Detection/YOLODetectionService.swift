import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation
import ImageIO
import os

/// Everything that can go wrong between a camera frame and a list of boxes.
enum DetectionError: Error, Equatable, Sendable {
    case modelMissingFromBundle(resource: String)
    case modelLoadFailed(String)
    case unexpectedInputInterface(String)
    case unexpectedOutputInterface(String)
    case preprocessingFailed(String)
    case invocationFailed(String)
    case invalidOutput(String)

    /// Whether retrying the next frame could plausibly work.
    ///
    /// A model missing from the bundle will still be missing; a pixel-buffer
    /// allocation that failed under memory pressure might not be.
    var isRecoverable: Bool {
        switch self {
        case .modelMissingFromBundle, .modelLoadFailed,
             .unexpectedInputInterface, .unexpectedOutputInterface:
            false
        case .preprocessingFailed, .invocationFailed, .invalidOutput:
            true
        }
    }
}

/// Runs the bundled YOLO26 detector on camera frames.
///
/// An `actor`, so the model and its scratch buffers are accessed serially
/// without locks, and so inference never runs on the main actor.
///
/// ## What is created once, not per frame
///
/// The model, the `CIContext`, and a `CVPixelBufferPool` for the 640x640 input.
/// Re-creating any of these per frame is the classic way to turn a 20 ms
/// inference into a 200 ms one: `MLModel` load costs hundreds of milliseconds,
/// and a fresh `CIContext` rebuilds its Metal pipeline state every time.
///
/// ## The output interface is discovered, not hardcoded
///
/// The exported tensor is currently named `var_1441` — a compiler-generated
/// name that is not stable across re-exports. Rather than depending on that
/// string, or on Xcode's generated model class, the single MultiArray output is
/// resolved from `MLModelDescription` at load time. A re-export that renames the
/// tensor keeps working; one that changes the *shape* fails loudly at load.
actor YOLODetectionService {

    private let configuration: DetectionConfiguration
    private let logger = Logger(subsystem: "com.marynaantonevych.BinSight", category: "detection")

    private let model: MLModel
    private let inputFeatureName: String
    private let outputFeatureName: String

    /// Created once. Software renderer disabled so this uses the GPU.
    private let ciContext: CIContext
    private let pixelBufferPool: CVPixelBufferPool

    /// The grey Ultralytics pads with, as a full-size image to composite over.
    private let paddingImage: CIImage

    /// Scratch for the decoded tensor, reused across frames so a 1,800-element
    /// array is not allocated 4 times a second.
    private var scratch: [Float]

    /// Most recent pure model-invocation time, for the debug overlay.
    private(set) var lastInferenceDuration: TimeInterval = 0
    private(set) var lastPreprocessingDuration: TimeInterval = 0

    // MARK: - Loading

    /// - Parameter bundle: defaults to the app bundle; injectable so unit tests
    ///   can load the model from the test bundle.
    init(
        configuration: DetectionConfiguration = .default,
        bundle: Bundle = .main
    ) throws {
        self.configuration = configuration

        // Xcode compiles the .mlpackage into a .mlmodelc directory in the bundle.
        let name = configuration.modelResourceName
        guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
            throw DetectionError.modelMissingFromBundle(resource: "\(name).mlmodelc")
        }

        let mlConfiguration = MLModelConfiguration()
        // Let CoreML pick: Neural Engine where available, GPU otherwise, CPU as
        // a floor. Forcing CPU would cost roughly an order of magnitude.
        mlConfiguration.computeUnits = .all

        do {
            self.model = try MLModel(contentsOf: url, configuration: mlConfiguration)
        } catch {
            throw DetectionError.modelLoadFailed(String(describing: error))
        }

        let description = model.modelDescription

        // Exactly one image input is expected. Resolved by kind, not by name.
        let imageInputs = description.inputDescriptionsByName.filter { $0.value.type == .image }
        guard imageInputs.count == 1, let imageInput = imageInputs.first else {
            throw DetectionError.unexpectedInputInterface(
                "expected exactly 1 image input, found \(imageInputs.count)"
            )
        }
        self.inputFeatureName = imageInput.key

        if let constraint = imageInput.value.imageConstraint {
            let width = CGFloat(constraint.pixelsWide)
            let height = CGFloat(constraint.pixelsHigh)
            guard width == configuration.inputSize.width,
                  height == configuration.inputSize.height
            else {
                throw DetectionError.unexpectedInputInterface(
                    "model input is \(Int(width))x\(Int(height)), expected "
                    + "\(Int(configuration.inputSize.width))x\(Int(configuration.inputSize.height))"
                )
            }
        }

        // Exactly one MultiArray output, shaped [1, 300, 6].
        let arrayOutputs = description.outputDescriptionsByName.filter { $0.value.type == .multiArray }
        guard arrayOutputs.count == 1, let arrayOutput = arrayOutputs.first else {
            throw DetectionError.unexpectedOutputInterface(
                "expected exactly 1 multiArray output, found \(arrayOutputs.count)"
            )
        }
        self.outputFeatureName = arrayOutput.key

        if let shape = arrayOutput.value.multiArrayConstraint?.shape, !shape.isEmpty {
            let dimensions = shape.map(\.intValue)
            guard dimensions.last == configuration.valuesPerDetection else {
                throw DetectionError.unexpectedOutputInterface(
                    "output shape \(dimensions) does not end in \(configuration.valuesPerDetection)"
                )
            }
        }

        self.ciContext = CIContext(options: [
            .useSoftwareRenderer: false,
            // No colour management: the model was trained on raw sRGB bytes and
            // any working-space conversion would shift every pixel it sees.
            .workingColorSpace: NSNull(),
            .outputColorSpace: NSNull(),
        ])

        let width = Int(configuration.inputSize.width)
        let height = Int(configuration.inputSize.height)
        var pool: CVPixelBufferPool?
        let poolStatus = CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            [kCVPixelBufferPoolMinimumBufferCountKey: 2] as CFDictionary,
            [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            ] as CFDictionary,
            &pool
        )
        guard poolStatus == kCVReturnSuccess, let pool else {
            throw DetectionError.modelLoadFailed("pixel buffer pool creation failed (\(poolStatus))")
        }
        self.pixelBufferPool = pool

        let grey = LetterboxTransform.paddingComponent
        self.paddingImage = CIImage(color: CIColor(red: grey, green: grey, blue: grey, alpha: 1))
        self.scratch = [Float](
            repeating: 0,
            count: configuration.maxDetections * configuration.valuesPerDetection
        )

        logger.info("""
            Detector loaded: input=\(self.inputFeatureName, privacy: .public) \
            output=\(self.outputFeatureName, privacy: .public)
            """)
    }

    // MARK: - Inference

    /// Detects objects in `frame`, returning boxes in the frame's own pixel space.
    func detect(_ frame: CameraFrame) throws -> [Detection] {
        let sourceSize = frame.pixelSize
        guard sourceSize.width > 0, sourceSize.height > 0 else {
            throw DetectionError.preprocessingFailed("degenerate frame \(sourceSize)")
        }

        let transform = LetterboxTransform(
            sourceSize: sourceSize,
            targetSize: configuration.inputSize
        )
        guard transform.scale > 0 else {
            throw DetectionError.preprocessingFailed("degenerate letterbox for \(sourceSize)")
        }

        let preprocessStarted = ContinuousClock.now
        let input = try letterboxedBuffer(from: frame, transform: transform)
        lastPreprocessingDuration = (ContinuousClock.now - preprocessStarted).seconds

        let inferenceStarted = ContinuousClock.now
        let output: MLFeatureProvider
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                inputFeatureName: MLFeatureValue(pixelBuffer: input)
            ])
            output = try model.prediction(from: provider)
        } catch {
            throw DetectionError.invocationFailed(String(describing: error))
        }
        lastInferenceDuration = (ContinuousClock.now - inferenceStarted).seconds

        guard let array = output.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw DetectionError.invalidOutput("no multiArray for \(outputFeatureName)")
        }

        try copyIntoScratch(array)
        return DetectionDecoder.decode(
            values: scratch,
            transform: transform,
            configuration: configuration
        )
    }

    // MARK: - Preprocessing

    /// Scales the frame by a single factor, centres it on Ultralytics grey, and
    /// renders it into a pooled 640x640 buffer.
    ///
    /// **No normalisation happens here, deliberately.** The exported package
    /// carries `scale = 1/255, bias = [0,0,0]` inside its own input layer, so
    /// dividing by 255 again would hand the model an almost-black image and it
    /// would detect nothing. Pixels go in as 8-bit BGRA, untouched.
    private func letterboxedBuffer(
        from frame: CameraFrame,
        transform: LetterboxTransform
    ) throws -> CVPixelBuffer {
        var source = CIImage(cvPixelBuffer: frame.pixelBuffer)

        // Frames are rotated to portrait at the capture connection, so this is
        // normally `.up` and the call is a no-op. Applied anyway rather than
        // assumed: if capture rotation ever changes, the image stays upright
        // instead of the model silently seeing a sideways world.
        if frame.orientation != .up {
            source = source.oriented(frame.orientation)
        }

        let scaled = source.transformed(
            by: CGAffineTransform(scaleX: transform.scale, y: transform.scale)
        )
        // Centred padding, so this offset is the same whichever way the y axis
        // runs — see LetterboxTransform.
        let positioned = scaled.transformed(
            by: CGAffineTransform(translationX: transform.padX, y: transform.padY)
        )

        let canvas = CGRect(origin: .zero, size: configuration.inputSize)
        let composited = positioned.composited(over: paddingImage).cropped(to: canvas)

        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw DetectionError.preprocessingFailed("pixel buffer allocation failed (\(status))")
        }

        ciContext.render(composited, to: buffer)
        return buffer
    }

    // MARK: - Output extraction

    /// Copies the output tensor into `scratch` as a flat row-major `[Float]`.
    ///
    /// Strides are honoured rather than assumed contiguous: a CoreML MultiArray
    /// is allowed to be padded, and reading it as if it were packed would shear
    /// every row by a few elements — which looks like plausible-but-wrong boxes
    /// rather than an obvious crash.
    private func copyIntoScratch(_ array: MLMultiArray) throws {
        guard array.dataType == .float32 else {
            throw DetectionError.invalidOutput("output dataType is \(array.dataType), expected float32")
        }

        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard let lastDimension = shape.last, lastDimension == configuration.valuesPerDetection else {
            throw DetectionError.invalidOutput("unexpected output shape \(shape)")
        }

        // Rows are the second-to-last dimension: [1, 300, 6] -> 300.
        let rowCount = shape.count >= 2 ? shape[shape.count - 2] : 1
        let usableRows = min(rowCount, configuration.maxDetections)
        let rowStride = strides.count >= 2 ? strides[strides.count - 2] : lastDimension
        let valueStride = strides.last ?? 1

        let needed = usableRows * configuration.valuesPerDetection
        if scratch.count < needed {
            scratch = [Float](repeating: 0, count: needed)
        }

        var overran = false
        array.withUnsafeBufferPointer(ofType: Float.self) { source in
            for row in 0..<usableRows {
                let sourceBase = row * rowStride
                // The strides come from the model, not from us. A padded layout
                // — legal for MLMultiArray, and the reason this code honours
                // strides at all — can put the last element of the last row past
                // the end of the buffer. In Release the subscript is not
                // bounds-checked, so this would be a silent out-of-bounds read
                // producing plausible-looking garbage boxes.
                let lastIndex = sourceBase + (configuration.valuesPerDetection - 1) * valueStride
                guard lastIndex < source.count else { overran = true; return }
                let destinationBase = row * configuration.valuesPerDetection
                for value in 0..<configuration.valuesPerDetection {
                    scratch[destinationBase + value] = source[sourceBase + value * valueStride]
                }
            }
        }

        if overran {
            throw DetectionError.invalidOutput(
                "output strides \(strides) overrun a \(array.count)-element buffer for shape \(shape)"
            )
        }

        // Zero any tail left over from a previous, longer frame so stale rows
        // cannot be decoded as detections.
        if scratch.count > needed {
            for index in needed..<scratch.count { scratch[index] = 0 }
        }
    }
}
