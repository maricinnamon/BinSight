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
        if !isRunning { return "Getting the camera ready…" }
        if grouped.isEmpty { return "Point at an item" }
        return grouped.count == 1 ? "Found 1 material" : "Found \(grouped.count) materials"
    }

    private var supporting: String? {
        if !isRunning { return nil }
        if grouped.isEmpty {
            return "Paper, plastic and metal are recognised on this iPhone."
        }
        let boxes = detections.count
        return boxes == 1 ? "1 object detected" : "\(boxes) objects detected"
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

            Text("\(Int((detection.confidence * 100).rounded()))%")
                .font(BinSightTheme.rounded(.subheadline, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(category.accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(detection.detectedClass.displayName), \(Int((detection.confidence * 100).rounded())) percent")
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
