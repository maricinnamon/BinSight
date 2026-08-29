import SwiftUI

/// The bottom surface for the detector.
///
/// The overlay already answers *where* and *what*; this answers *what do I do
/// with it*. It lists the distinct materials currently in view — not one row per
/// box, because eleven overlapping `metal` detections on a pile of screws is one
/// piece of information, not eleven.
///
/// Confidence shown per material is the strongest box of that material, which is
/// the honest summary: the weakest of eleven detections says nothing about how
/// sure the app is that there is metal in frame.
struct DetectionSummaryCard: View {
    let detections: [Detection]
    /// `false` while the camera is starting, so the empty state does not claim
    /// "nothing found" before anything has been looked at.
    let isRunning: Bool

    /// Strongest detection per class, ordered by confidence.
    private var grouped: [Detection] {
        Dictionary(grouping: detections, by: \.classID)
            .values
            .compactMap { $0.max(by: { $0.confidence < $1.confidence }) }
            .sorted { $0.confidence > $1.confidence }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if !grouped.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(grouped) { detection in
                        row(for: detection)
                    }
                }
                .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .panelSurface(in: RoundedRectangle(cornerRadius: BinSightTheme.cardCornerRadius, style: .continuous))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("detection.summary")
    }

    private var title: String {
        if !isRunning { return L("summary.cameraStarting") }
        if grouped.isEmpty { return L("summary.pointAtItem") }
        // Plural form comes from the string catalog, not a ternary: Ukrainian
        // has three, and "1 материал / 2 материали / 5 матеріалів" cannot be
        // expressed by branching on `count == 1`.
        return L("summary.foundMaterials", grouped.count)
    }

    private var supporting: String? {
        if !isRunning { return nil }
        if grouped.isEmpty {
            return L("summary.recognisedOnDevice")
        }
        return L("summary.objectsDetected", detections.count)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(BinSightTheme.rounded(grouped.isEmpty ? .headline : .subheadline, weight: .semibold))
                .foregroundStyle(BinSightTheme.onSurface.opacity(grouped.isEmpty ? 1 : 0.55))
                .accessibilityIdentifier("detection.title")

            if let supporting {
                Text(supporting)
                    .font(BinSightTheme.rounded(.footnote))
                    .foregroundStyle(BinSightTheme.onSurface.opacity(0.65))
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        // One swipe stop, not two: "Found 2 materials" and "3 objects detected"
        // are one thought.
        .accessibilityElement(children: .combine)
    }

    private func row(for detection: Detection) -> some View {
        let category = detection.detectedClass.wasteCategory
        return HStack(spacing: 12) {
            Image(systemName: category.symbolName)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(category.accent)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 2) {
                Text(detection.detectedClass.displayName)
                    .font(BinSightTheme.rounded(.headline, weight: .bold))
                    .foregroundStyle(BinSightTheme.onSurface)
                Text(category.disposalGuidance)
                    .font(BinSightTheme.rounded(.caption))
                    .foregroundStyle(BinSightTheme.onSurface.opacity(0.65))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 8)

            Text(Detection.percent(detection.confidence))
                .font(BinSightTheme.rounded(.subheadline, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(category.accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        // `.ignore`, not `.combine` with an overriding label. `.combine` merges
        // the children and then the label *replaces* the merge — which silently
        // dropped `disposalGuidance`, the one actionable sentence on this
        // screen, from VoiceOver entirely. Building the label explicitly makes
        // what is spoken visible in the code.
        .accessibilityLabel(
            L("summary.row.accessibility",
              detection.detectedClass.displayName,
              Detection.percent(detection.confidence))
            + ". " + category.disposalGuidance
        )
    }
}

#if DEBUG
#Preview("Detections") {
    ZStack {
        BinSightTheme.canvasBottom
        DetectionSummaryCard(
            detections: [
                Detection(detectedClass: .plastic, confidence: 0.91, boundingBox: .init(x: 0, y: 0, width: 10, height: 10)),
                Detection(detectedClass: .metal, confidence: 0.76, boundingBox: .init(x: 0, y: 0, width: 10, height: 10)),
            ],
            isRunning: true
        )
        .padding()
    }
    .ignoresSafeArea()
}

#Preview("Empty") {
    ZStack {
        BinSightTheme.canvasBottom
        DetectionSummaryCard(detections: [], isRunning: true).padding()
    }
    .ignoresSafeArea()
}
#endif
