import SwiftUI

/// The wordmark: "Bin" in the on-camera colour, "Sight" in lime.
///
/// The split does the branding work without a custom font — "Sight" is the half
/// that means something here, and lime is the palette's highest-energy colour.
/// It sits over the top scrim, which is what makes lime legible: lime on cream
/// is about 1.1:1 and effectively invisible, but lime on the scrim clears 10:1.
struct BinSightWordmark: View {
    var body: some View {
        (
            Text("Bin").foregroundStyle(BinSightTheme.onCamera)
                + Text("Sight").foregroundStyle(BinSightTheme.lime)
        )
        .font(BinSightTheme.rounded(.title3, weight: .heavy))
        .kerning(-0.3)
        .lineLimit(1)
        // Never let the wordmark wrap or truncate at large Dynamic Type.
        .minimumScaleFactor(0.8)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("BinSight")
        .accessibilityAddTraits(.isHeader)
    }
}

#Preview {
    VStack(spacing: 20) {
        BinSightWordmark()
        BinSightWordmark().environment(\.dynamicTypeSize, .accessibility3)
    }
    .padding(40)
    .background {
        ZStack {
            MockCameraBackground()
            LinearGradient(
                colors: [BinSightTheme.scrimStrong, .clear],
                startPoint: .top,
                endPoint: .bottom
            )
        }
    }
}
