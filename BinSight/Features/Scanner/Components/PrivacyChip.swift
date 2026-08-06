import SwiftUI

/// A small glass pill for the two on-device claims the app makes: the
/// "ON-DEVICE AI" badge at the top and the frames-stay-here line at the bottom.
///
/// Glass is spent deliberately — on these two status surfaces, the result card,
/// and real actions. Not on every label.
struct PrivacyChip: View {
    let text: String
    let systemImage: String
    var isCompact = false
    /// A class accent, when the chip should carry one.
    var tint: Color?

    var body: some View {
        Label {
            Text(text)
                .font(BinSightTheme.rounded(isCompact ? .caption2 : .caption, weight: .semibold))
                .kerning(isCompact ? 0.6 : 0)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        } icon: {
            Image(systemName: systemImage)
                .font(.system(size: isCompact ? 10 : 12, weight: .semibold))
                .foregroundStyle(tint ?? .primary)
        }
        // No hardcoded colour: iOS 26 glass substitutes its own legible content
        // colour, and `.primary` adapts on the material fallback. Forcing cream
        // here produced cream-on-light-glass, which is unreadable.
        .foregroundStyle(.primary)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        // Applied last, so the effect wraps the padded element.
        .glassSurface(in: Capsule(), tint: tint)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(text)
    }
}

#Preview {
    GlassGroup(spacing: 14) {
        VStack(spacing: 16) {
            PrivacyChip(text: "ON-DEVICE AI", systemImage: "cpu", isCompact: true)
            PrivacyChip(text: "Frames stay on this iPhone.", systemImage: "lock.fill")
            PrivacyChip(text: "Plastic", systemImage: "waterbottle.fill", tint: BinSightTheme.coral)
        }
    }
    .padding(40)
    .background(MockCameraBackground())
}
