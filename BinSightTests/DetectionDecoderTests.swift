import CoreGraphics
import Testing
@testable import BinSight

/// The output contract, exercised with synthetic tensors.
///
/// Everything here is deterministic and model-free: the decoder takes floats and
/// returns domain objects, so the riskiest logic in the app — column order, box
/// interpretation, class mapping, thresholding — can be pinned down without a
/// camera, a simulator, or CoreML.
struct DetectionDecoderTests {

    /// 416x416 square: scale 640/416, no padding. Keeps the arithmetic in these
    /// tests about *decoding* rather than about letterbox offsets, which
    /// `LetterboxTransformTests` covers separately.
    private let squareTransform = LetterboxTransform(
        sourceSize: CGSize(width: 416, height: 416),
        targetSize: CGSize(width: 640, height: 640)
    )

    /// The portrait capture shape, where padX is 140.
    private let portraitTransform = LetterboxTransform(
        sourceSize: CGSize(width: 720, height: 1280),
        targetSize: CGSize(width: 640, height: 640)
    )

    /// One row: `[x1, y1, x2, y2, confidence, class_id]`.
    private func row(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float,
                     _ confidence: Float, _ classID: Float) -> [Float] {
        [x1, y1, x2, y2, confidence, classID]
    }

    private func expectClose(_ a: CGFloat, _ b: CGFloat, tolerance: CGFloat = 0.01) {
        #expect(abs(a - b) < tolerance, "\(a) is not within \(tolerance) of \(b)")
    }

    // MARK: - Class mapping

    @Test("Class ids map to the model's own taxonomy")
    func classMapping() {
        let values = row(0, 0, 100, 100, 0.9, 0)
            + row(100, 100, 200, 200, 0.9, 1)
            + row(200, 200, 300, 300, 0.9, 2)
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform)

        #expect(detections.count == 3)
        #expect(detections.map(\.classID) == [0, 1, 2])
        #expect(detections.map(\.className) == ["paper", "plastic", "metal"])
        #expect(detections.map(\.detectedClass) == [.paper, .plastic, .metal])
    }

    @Test("Unknown class ids are dropped, not clamped")
    func unknownClassesRejected() {
        // Clamping 3 to 2 would silently relabel a decoding bug as `metal`.
        for bad: Float in [-1, 3, 7, 99] {
            let detections = DetectionDecoder.decode(
                values: row(0, 0, 100, 100, 0.9, bad),
                transform: squareTransform
            )
            #expect(detections.isEmpty, "class id \(bad) should be rejected")
        }
    }

    @Test("Class ids arrive as floats and are rounded")
    func classIDsRounded() {
        // FP16 execution does not return exact integers.
        let values = row(0, 0, 100, 100, 0.9, 1.0000001) + row(0, 0, 100, 100, 0.9, 1.9999999)
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform)
        #expect(detections.map(\.detectedClass) == [.plastic, .metal])
    }

    @Test("A non-finite class id cannot become a detection")
    func nonFiniteClassRejected() {
        #expect(DetectedClass(modelValue: .nan) == nil)
        #expect(DetectedClass(modelValue: .infinity) == nil)
    }

    // MARK: - Confidence filtering

    @Test("The threshold is inclusive at 0.25")
    func confidenceThreshold() {
        let cases: [(Float, Bool)] = [
            (0.0, false), (0.10, false), (0.24, false), (0.2499, false),
            (0.25, true), (0.2501, true), (0.50, true), (0.90, true), (1.0, true),
        ]
        for (confidence, shouldPass) in cases {
            let detections = DetectionDecoder.decode(
                values: row(0, 0, 100, 100, confidence, 0),
                transform: squareTransform
            )
            #expect(detections.count == (shouldPass ? 1 : 0),
                    "confidence \(confidence) should \(shouldPass ? "pass" : "be rejected")")
        }
    }

    @Test("A non-finite confidence is rejected")
    func nonFiniteConfidenceRejected() {
        #expect(DetectionDecoder.decode(values: row(0, 0, 100, 100, .nan, 0),
                                        transform: squareTransform).isEmpty)
    }

    @Test("Padding rows below threshold are ignored")
    func paddingRowsIgnored() {
        // The model always returns 300 rows; most are filler.
        var values = row(0, 0, 100, 100, 0.88, 2)
        for _ in 0..<299 { values += row(0, 0, 0, 0, 0.0, 0) }
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform)
        #expect(detections.count == 1)
        #expect(detections[0].detectedClass == .metal)
    }

    @Test("A row below threshold does not stop later rows being read")
    func lowConfidenceRowDoesNotTerminateScan() {
        // Ordering is documented as *approximate*, so the decoder must not bail
        // at the first weak row.
        let values = row(0, 0, 100, 100, 0.10, 0) + row(100, 100, 200, 200, 0.80, 1)
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform)
        #expect(detections.count == 1)
        #expect(detections[0].detectedClass == .plastic)
    }

    // MARK: - Invalid boxes

    @Test("Inverted and degenerate boxes are rejected")
    func invalidBoxesRejected() {
        let cases: [(String, [Float])] = [
            ("x2 < x1", row(200, 0, 100, 100, 0.9, 0)),
            ("x2 == x1", row(100, 0, 100, 100, 0.9, 0)),
            ("y2 < y1", row(0, 200, 100, 100, 0.9, 0)),
            ("y2 == y1", row(0, 100, 100, 100, 0.9, 0)),
            ("all zero", row(0, 0, 0, 0, 0.9, 0)),
        ]
        for (label, values) in cases {
            #expect(DetectionDecoder.decode(values: values, transform: squareTransform).isEmpty,
                    "\(label) should be rejected")
        }
    }

    @Test("Non-finite coordinates are rejected")
    func nonFiniteBoxRejected() {
        for values in [row(.nan, 0, 100, 100, 0.9, 0),
                       row(0, .infinity, 100, 100, 0.9, 0),
                       row(0, 0, -.infinity, 100, 0.9, 0)] {
            #expect(DetectionDecoder.decode(values: values, transform: squareTransform).isEmpty)
        }
    }

    @Test("A box entirely outside the frame is dropped rather than collapsed")
    func offFrameBoxDropped() {
        // Everything left of the padded content in portrait: clamps to zero width.
        let detections = DetectionDecoder.decode(
            values: row(0, 100, 130, 200, 0.9, 0),
            transform: portraitTransform
        )
        #expect(detections.isEmpty)
    }

    // MARK: - Coordinates

    @Test("Boxes are read as xyxy, not xywh")
    func boxesAreCorners() {
        // Read as xywh, `[100, 100, 300, 200]` would be a 300x200 box at (100,100).
        // Read as xyxy — correct for this model — it is a 200x100 box.
        let detections = DetectionDecoder.decode(
            values: row(100, 100, 300, 200, 0.9, 0),
            transform: squareTransform
        )
        #expect(detections.count == 1)
        let scale = 640.0 / 416.0
        expectClose(detections[0].boundingBox.width, 200 / scale)
        expectClose(detections[0].boundingBox.height, 100 / scale)
    }

    @Test("Boxes land in source pixels, with the pad removed")
    func portraitCoordinates() {
        // Model-space content spans x 140...500. A box at x 140 is the frame's
        // left edge; at x 500 it is the right edge.
        let detections = DetectionDecoder.decode(
            values: row(140, 0, 500, 640, 0.9, 1),
            transform: portraitTransform
        )
        #expect(detections.count == 1)
        let box = detections[0].boundingBox
        expectClose(box.minX, 0)
        expectClose(box.minY, 0)
        expectClose(box.width, 720)
        expectClose(box.height, 1280)
    }

    @Test("A centred model box is centred in the source frame")
    func centredBox() {
        let detections = DetectionDecoder.decode(
            values: row(300, 300, 340, 340, 0.9, 2),
            transform: portraitTransform
        )
        #expect(detections.count == 1)
        expectClose(detections[0].boundingBox.midX, 360)
        expectClose(detections[0].boundingBox.midY, 640)
    }

    // MARK: - Shape and limits

    @Test("No NMS is applied — overlapping boxes all survive")
    func overlappingBoxesKept() {
        // The head is end-to-end. Suppressing these would lose real objects.
        let values = row(100, 100, 300, 300, 0.90, 2)
            + row(110, 110, 310, 310, 0.85, 2)
            + row(120, 120, 320, 320, 0.80, 2)
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform)
        #expect(detections.count == 3)
    }

    @Test("Display cap limits how many boxes are returned")
    func displayCap() {
        var values: [Float] = []
        for index in 0..<40 {
            let offset = Float(index)
            values += row(offset, offset, offset + 50, offset + 50, 0.9, 1)
        }
        var configuration = DetectionConfiguration.default
        configuration.maximumDisplayedDetections = 10
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform,
                                                 configuration: configuration)
        #expect(detections.count == 10)
    }

    @Test("Rows beyond maxDetections are never read")
    func maxDetectionsRespected() {
        var configuration = DetectionConfiguration.default
        configuration.maxDetections = 2
        configuration.maximumDisplayedDetections = 100
        let values = row(0, 0, 50, 50, 0.9, 0)
            + row(10, 10, 60, 60, 0.9, 1)
            + row(20, 20, 70, 70, 0.9, 2)
        let detections = DetectionDecoder.decode(values: values, transform: squareTransform,
                                                 configuration: configuration)
        #expect(detections.count == 2)
    }

    @Test("An empty or truncated tensor decodes to nothing")
    func emptyInput() {
        #expect(DetectionDecoder.decode(values: [], transform: squareTransform).isEmpty)
        // Fewer than one full row.
        #expect(DetectionDecoder.decode(values: [1, 2, 3], transform: squareTransform).isEmpty)
    }

    @Test("A degenerate transform decodes to nothing")
    func degenerateTransform() {
        let broken = LetterboxTransform(sourceSize: .zero, targetSize: CGSize(width: 640, height: 640))
        #expect(DetectionDecoder.decode(values: row(0, 0, 100, 100, 0.9, 0),
                                        transform: broken).isEmpty)
    }

    // MARK: - Presentation

    @Test("Labels round to whole percent")
    func labelFormatting() {
        let detection = Detection(detectedClass: .plastic, confidence: 0.819,
                                  boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(detection.label == "Plastic 82%")
    }

    @Test("Detection classes map onto the existing palette")
    func categoryBridge() {
        #expect(DetectedClass.paper.wasteCategory == .paper)
        #expect(DetectedClass.plastic.wasteCategory == .plastic)
        #expect(DetectedClass.metal.wasteCategory == .metal)
    }
}
