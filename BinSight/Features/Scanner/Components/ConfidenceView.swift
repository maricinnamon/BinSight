import SwiftUI

/// The confidence readout: a percentage and a slim bar in the class accent.
///
/// Deliberately **secondary** — small type, thin bar, sitting under the class
/// name rather than beside it. It is also never labelled "accuracy": this is the
/// model's confidence in one prediction, not a measured accuracy rate, and
/// conflating the two would overclaim.
///
/// Only ever rendered when a result exists, so there is no "0%" empty state.
struct ConfidenceView: View {
    let result: ClassificationResult

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(L("confidence.label", result.confidenceText))
                .font(BinSightTheme.rounded(.footnote, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(BinSightTheme.onSurface.opacity(0.65))
                .lineLimit(1)
                .minimumScaleFactor(0.8)

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(BinSightTheme.onSurface.opacity(0.14))
                    Capsule()
                        .fill(result.category.accent)
                        .frame(width: max(proxy.size.width * result.confidence, 4))
                }
            }
            .frame(height: 5)
            .animation(reduceMotion ? nil : .snappy(duration: 0.4), value: result)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L("confidence.accessibilityLabel"))
        .accessibilityValue(result.confidenceText)
        .accessibilityIdentifier("result.confidence")
    }
}

#Preview {
    VStack(spacing: 24) {
        ConfidenceView(result: .samplePlastic)
        ConfidenceView(result: .sampleGlass)
        ConfidenceView(result: .sampleCardboard)
    }
    .padding(24)
    .background(BinSightTheme.surface)
}
