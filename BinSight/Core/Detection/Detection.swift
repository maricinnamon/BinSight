import CoreGraphics
import Foundation

/// The three classes the bundled YOLO26 detector can emit.
///
/// This is **not** `WasteCategory`. That enum has six cases and describes the
/// old TrashNet classifier's taxonomy; the detector was trained on three. Mixing
/// them would let a class index from one taxonomy be read with the other's
/// ordering, which mislabels silently rather than failing. The bridge to
/// presentation is `wasteCategory`, and it is deliberately one-way.
///
/// Raw values are the model's own class indices, straight from the package
/// metadata: `names = {0: 'paper', 1: 'plastic', 2: 'metal'}`.
enum DetectedClass: Int, CaseIterable, Identifiable, Sendable {
    case paper = 0
    case plastic = 1
    case metal = 2

    var id: Int { rawValue }

    /// The model's own label string, as written in the `.mlpackage` metadata.
    var modelLabel: String {
        switch self {
        case .paper: "paper"
        case .plastic: "plastic"
        case .metal: "metal"
        }
    }

    /// Title-case for the overlay. Localized; `modelLabel` above is not, because
    /// that one is the model's own contract.
    var displayName: String {
        switch self {
        case .paper: L("detected.paper.name")
        case .plastic: L("detected.plastic.name")
        case .metal: L("detected.metal.name")
        }
    }

    /// Reuses the existing palette and iconography rather than inventing a
    /// second visual language for the same three materials.
    var wasteCategory: WasteCategory {
        switch self {
        case .paper: .paper
        case .plastic: .plastic
        case .metal: .metal
        }
    }

    /// Rejects anything outside 0...2 instead of clamping.
    ///
    /// The model emits the class index as a `Float`. A value that does not round
    /// to a known class means the output was misread — wrong column, wrong
    /// stride, wrong tensor — and inventing a label would hide exactly the bug
    /// worth catching.
    init?(modelValue: Float) {
        guard modelValue.isFinite else { return nil }
        let rounded = Int(modelValue.rounded())
        guard let value = DetectedClass(rawValue: rounded) else { return nil }
        self = value
    }
}

/// One detected object.
///
/// `boundingBox` is in **source-image pixel coordinates** — the camera buffer's
/// own space, origin top-left, y down — never model space and never normalised.
/// Un-letterboxing happens once, in the decoder, so nothing downstream has to
/// know the model ran at 640x640.
struct Detection: Identifiable, Equatable, Sendable {
    let id: UUID
    let classID: Int
    let className: String
    let confidence: Float
    let boundingBox: CGRect

    /// The typed class. Non-optional because a `Detection` cannot be built with
    /// an unknown class id — the decoder drops those rows.
    var detectedClass: DetectedClass {
        // Safe: the only initialiser takes a DetectedClass.
        DetectedClass(rawValue: classID) ?? .paper
    }

    init(id: UUID = UUID(), detectedClass: DetectedClass, confidence: Float, boundingBox: CGRect) {
        self.id = id
        self.classID = detectedClass.rawValue
        self.className = detectedClass.modelLabel
        self.confidence = confidence
        self.boundingBox = boundingBox
    }

    /// e.g. `"Plastic 81%"`. Rounded, not truncated — 0.819 reads as 82%.
    ///
    /// The percentage goes through `FormatStyle` rather than string
    /// interpolation with a literal `%`: several locales put the sign before the
    /// number or separate it with a non-breaking space, and Ukrainian is one of
    /// the ones that does not match the English convention.
    var label: String {
        L("detection.label", detectedClass.displayName, Self.percent(confidence))
    }

    /// Locale-aware percentage, shared by the overlay and the summary card.
    static func percent(_ confidence: Float) -> String {
        Double(confidence).formatted(.percent.precision(.fractionLength(0)))
    }
}
