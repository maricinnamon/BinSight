import CoreGraphics
import Foundation

/// Every tunable number in the detection path, stated once.
///
/// The shape values are not preferences — they are the exported package's
/// contract, verified against `models/coreml/export_summary.json`. Changing one
/// here without re-exporting would silently misread the output tensor.
struct DetectionConfiguration: Equatable, Sendable {

    // MARK: - Model contract (do not change without re-exporting)

    /// Resource name of the bundled package, without extension.
    var modelResourceName: String = "BinSightYOLO26n"

    /// The model's square input. From the package: `image`, 640x640 RGB.
    var inputSize: CGSize = CGSize(width: 640, height: 640)

    /// Rows in the output tensor — `head.max_det`. Output shape is `[1, 300, 6]`.
    var maxDetections: Int = 300

    /// Values per row: `[x1, y1, x2, y2, confidence, class_id]`.
    var valuesPerDetection: Int = 6

    // MARK: - Thresholds

    /// Minimum confidence to surface a detection.
    ///
    /// 0.25 is the threshold every published BinSight metric was measured at —
    /// test precision 0.682, recall 0.578 — so the app's behaviour matches the
    /// numbers in the report. Raising it trades recall for precision.
    var confidenceThreshold: Float = 0.25

    /// Upper bound on boxes drawn at once.
    ///
    /// Purely a presentation limit. A crowded frame can legitimately produce a
    /// dozen overlapping boxes (the evaluation set has an image with 11 valid
    /// `metal` detections), which is unreadable on a phone. Rows are already
    /// ordered by descending confidence, so this keeps the strongest.
    var maximumDisplayedDetections: Int = 10

    /// Boxes thinner than this in source pixels are dropped as noise.
    var minimumBoxSide: CGFloat = 2

    static let `default` = DetectionConfiguration()
}
