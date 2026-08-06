import SwiftUI
import UIKit

/// BinSight's palette, metrics and type ramp.
///
/// Two layers on purpose: fixed *brand* colours that never change, and
/// *semantic* colours that resolve differently in light and dark. Views only
/// ever reach for the semantic names, so appearance support stays in one file.
enum BinSightTheme {

    // MARK: - Brand palette

    /// Warm off-white. The light-mode canvas base.
    static let cream = Color(hex: 0xFFF8EC)
    /// Near-black. Carries text on light surfaces and backs the result card.
    static let ink = Color(hex: 0x171717)
    static let lime = Color(hex: 0xD7FF5F)
    static let coral = Color(hex: 0xFF5C8A)
    static let violet = Color(hex: 0x8B5CF6)
    static let mint = Color(hex: 0x5DE2B8)
    /// Two extensions to the starting palette, because six classes need six
    /// distinguishable accents. Both are tuned to clear 4.5:1 against `ink`.
    static let amber = Color(hex: 0xF5A524)
    static let slate = Color(hex: 0xA8AEB8)

    // MARK: - Semantic colours

    /// The mock camera canvas, top to bottom.
    static let canvasTop = adaptive(light: 0xFFF4E2, dark: 0x1A1424)
    static let canvasMid = adaptive(light: 0xFFE2E9, dark: 0x241629)
    static let canvasBottom = adaptive(light: 0xE9E3FF, dark: 0x101015)

    /// Text drawn straight onto the canvas.
    static let onCanvas = adaptive(light: 0x171717, dark: 0xF5F0E6)

    /// The result card. Dark in both appearances, which is what lets every
    /// class accent keep a single value and still stay legible.
    static let surface = adaptive(light: 0x171717, dark: 0x1F1F24)
    static let onSurface = adaptive(light: 0xFFF8EC, dark: 0xF5F0E6)

    /// The stand-in object in the mock preview: dark on cream, light on ink.
    static let silhouette = adaptive(light: 0x171717, dark: 0xFFF8EC)

    /// Opaque stand-in for glass when Reduce Transparency is on. Deliberately
    /// solid: the point of that setting is to remove see-through surfaces, so a
    /// "slightly less transparent" material would miss it.
    static let opaqueSurface = adaptive(light: 0x1B1B1E, dark: 0x232329)

    /// The one-pixel edge that separates a surface from the camera behind it.
    static let hairline = Color.white.opacity(0.14)

    /// Scrim over the camera behind the top chrome.
    ///
    /// A camera frame can be any brightness, so light text alone is not enough
    /// to guarantee contrast — the scrim is what makes the wordmark and chips
    /// legible against a white wall as well as a dark bin.
    static let scrimStrong = ink.opacity(0.55)
    static let scrimSoft = ink.opacity(0.28)

    /// Text and icons drawn over the camera, above a scrim.
    static let onCamera = cream

    /// Tint pushed into the result panel's glass so it stays dark enough to
    /// carry cream text over any camera frame.
    static let panelTint = ink.opacity(0.55)

    // MARK: - Metrics

    static let screenPadding: CGFloat = 20
    static let cardCornerRadius: CGFloat = 30
    static let chipCornerRadius: CGFloat = 18
    static let spacing: CGFloat = 16

    /// Apple's minimum comfortable hit target.
    static let minimumTapTarget: CGFloat = 44

    // MARK: - Type

    /// Rounded system type throughout — it carries the friendly, non-clinical
    /// tone without introducing a custom font.
    static func rounded(_ style: Font.TextStyle, weight: Font.Weight = .regular) -> Font {
        .system(style, design: .rounded, weight: weight)
    }

    private static func adaptive(light: UInt32, dark: UInt32) -> Color {
        Color(uiColor: UIColor { traits in
            UIColor(Color(hex: traits.userInterfaceStyle == .dark ? dark : light))
        })
    }
}

extension Color {
    /// Builds a colour from a 24-bit RGB literal, e.g. `0xFFF8EC`.
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}
