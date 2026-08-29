import CoreGraphics
import Foundation

/// Turns the model's raw output tensor into `Detection` values.
///
/// Deliberately free of CoreML, CoreVideo and CoreImage: it takes a flat array
/// of floats and a `LetterboxTransform`, and returns domain objects. That is what
/// makes the riskiest logic in the app — output indexing, box interpretation,
/// class mapping and coordinate conversion — testable without a camera, a
/// simulator, or the model itself.
///
/// ## The output contract
///
/// One tensor, shape `[1, 300, 6]`, FP32. Each of the 300 rows is:
///
/// ```
/// [x1, y1, x2, y2, confidence, class_id]
/// ```
///
/// * `x1, y1, x2, y2` — **corner** coordinates in 640x640 model-input pixels.
///   Not xywh. Ultralytics' own `Detect.postprocess` docstring says
///   `[x, y, w, h, ...]`, and for this checkpoint that is **wrong**: decoding as
///   xywh yields negative coordinates. Verified empirically against the PyTorch
///   model, and that verification is what this decoder implements.
/// * `confidence` — already the maximum class probability. No sigmoid, no
///   softmax, no objectness multiply.
/// * `class_id` — the winning class as a float: 0 paper, 1 plastic, 2 metal.
///
/// ## No NMS
///
/// YOLO26 is end-to-end: `end2end = True` in the package metadata, and the
/// checkpoint carries a one-to-one head. The rows are already deduplicated.
/// Running NMS here would be actively wrong — it would suppress genuinely
/// overlapping objects, which are common (one evaluation image has 11 valid
/// overlapping `metal` boxes).
///
/// All 300 rows are always returned by the model, padded with low-scoring
/// entries, so the confidence threshold is the only thing separating signal
/// from filler.
enum DetectionDecoder {

    /// Decodes raw model output into detections in **source-image pixels**.
    ///
    /// - Parameters:
    ///   - values: the output tensor flattened row-major; at least
    ///     `rowCount * valuesPerDetection` elements.
    ///   - transform: the letterbox used to build the input, so boxes can be
    ///     mapped back to the frame they came from.
    ///   - configuration: thresholds and the expected tensor shape.
    /// - Returns: detections above threshold, capped at
    ///   `maximumDisplayedDetections`, keeping the strongest ones when capped.
    static func decode(
        values: [Float],
        transform: LetterboxTransform,
        configuration: DetectionConfiguration = .default
    ) -> [Detection] {
        let stride = configuration.valuesPerDetection
        guard stride > 0, transform.scale > 0 else { return [] }

        let available = values.count / stride
        let rowCount = min(available, configuration.maxDetections)
        guard rowCount > 0 else { return [] }

        var detections: [Detection] = []
        detections.reserveCapacity(min(rowCount, configuration.maximumDisplayedDetections))

        for row in 0..<rowCount {
            let base = row * stride
            let confidence = values[base + 4]

            // Rows arrive ordered by descending confidence, but that ordering is
            // documented as approximate, so every row is still tested rather
            // than breaking at the first one below threshold.
            guard confidence.isFinite, confidence >= configuration.confidenceThreshold else {
                continue
            }
            guard let detectedClass = DetectedClass(modelValue: values[base + 5]) else {
                continue
            }

            let x1 = values[base + 0]
            let y1 = values[base + 1]
            let x2 = values[base + 2]
            let y2 = values[base + 3]
            guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite else { continue }
            // Degenerate or inverted boxes mean a misread tensor, not a small
            // object. Dropping them keeps a decoding bug visible as "nothing is
            // detected" rather than as boxes with negative area.
            guard x2 > x1, y2 > y1 else { continue }

            let modelRect = CGRect(x: CGFloat(x1),
                                   y: CGFloat(y1),
                                   width: CGFloat(x2 - x1),
                                   height: CGFloat(y2 - y1))

            let sourceRect = transform.unletterbox(modelRect)
            guard !sourceRect.isNull, !sourceRect.isEmpty else { continue }
            // A box clipped almost entirely away by the clamp is not worth
            // drawing and usually means the object left the frame.
            guard sourceRect.width >= configuration.minimumBoxSide,
                  sourceRect.height >= configuration.minimumBoxSide
            else { continue }

            detections.append(
                Detection(detectedClass: detectedClass,
                          confidence: confidence,
                          boundingBox: sourceRect)
            )
        }

        // Take the *strongest* N, not the first N. Row order is documented as
        // only approximately descending, so stopping at the first N above
        // threshold can discard a stronger detection further down — which
        // matters most when N is small.
        //
        // Ties keep their original row order: `sorted(by:)` is not stable, and
        // without the index tiebreak two equally confident detections could
        // swap places between frames and make the overlay flicker.
        guard detections.count > configuration.maximumDisplayedDetections else {
            return detections
        }
        return detections.enumerated()
            .sorted { left, right in
                left.element.confidence == right.element.confidence
                    ? left.offset < right.offset
                    : left.element.confidence > right.element.confidence
            }
            .prefix(configuration.maximumDisplayedDetections)
            .map(\.element)
    }
}
