import SwiftUI

/// BinSight's one glass abstraction.
///
/// Three rendering paths, chosen in this order:
///
/// 1. **Reduce Transparency on** — an opaque, high-contrast fill. Checked
///    first, because an accessibility setting outranks a visual effect.
/// 2. **iOS 26+** — native `glassEffect`, optionally tinted and interactive.
/// 3. **iOS 17–25** — a system material.
///
/// Keeping all three behind one modifier is what stops `#available` from
/// spreading through the views, and guarantees the fallbacks stay in step when
/// a surface changes shape.
///
/// No custom blur is stacked underneath any of these: native glass already
/// samples what is behind it, and a hand-rolled blur below it would both
/// double-cost and muddy the result.
struct GlassSurfaceModifier<S: InsettableShape>: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    let shape: S
    let tint: Color?
    let isInteractive: Bool

    func body(content: Content) -> some View {
        if reduceTransparency {
            content
                .background(BinSightTheme.opaqueSurface, in: shape)
                .overlay(shape.strokeBorder(BinSightTheme.hairline, lineWidth: 1))
        } else if #available(iOS 26.0, *) {
            content.glassEffect(nativeGlass, in: shape)
        } else {
            content
                .background(.regularMaterial, in: shape)
                .overlay(shape.strokeBorder(BinSightTheme.hairline, lineWidth: 1))
        }
    }

    @available(iOS 26.0, *)
    private var nativeGlass: Glass {
        var glass = Glass.regular
        if let tint {
            glass = glass.tint(tint)
        }
        if isInteractive {
            // Interactive glass is reserved for things that are genuinely
            // tappable — it reacts to touch, which is a lie on a static label.
            glass = glass.interactive()
        }
        return glass
    }
}

/// The large dark panel behind the result — a second surface treatment, not a
/// variant flag, because its fallback differs in kind rather than degree.
///
/// On iOS 26 it is glass tinted toward ink. Below that it is a **solid** dark
/// fill, not a material: a system material in light appearance is pale, and
/// cream text on a pale material over a bright camera frame is unreadable. The
/// panel carries the result, so its contrast is not negotiable.
struct PanelSurfaceModifier<S: InsettableShape>: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    let shape: S

    func body(content: Content) -> some View {
        if reduceTransparency {
            content
                .background(BinSightTheme.opaqueSurface, in: shape)
                .overlay(shape.strokeBorder(BinSightTheme.hairline, lineWidth: 1))
        } else if #available(iOS 26.0, *) {
            content
                .glassEffect(.regular.tint(BinSightTheme.panelTint), in: shape)
        } else {
            content
                .background(BinSightTheme.surface, in: shape)
                .overlay(shape.strokeBorder(BinSightTheme.hairline, lineWidth: 1))
        }
    }
}

extension View {
    /// The result panel's surface.
    func panelSurface(in shape: some InsettableShape) -> some View {
        modifier(PanelSurfaceModifier(shape: shape))
    }

    /// Applies BinSight's glass surface, clipped to `shape`.
    ///
    /// Call this **after** layout and appearance modifiers (padding, font,
    /// foreground style) so the effect wraps the finished element rather than
    /// its unpadded content.
    func glassSurface(
        in shape: some InsettableShape,
        tint: Color? = nil,
        interactive: Bool = false
    ) -> some View {
        modifier(GlassSurfaceModifier(shape: shape, tint: tint, isInteractive: interactive))
    }
}

/// Groups nearby glass elements so the system can blend them as one material
/// rather than as separate overlapping panes.
///
/// Collapses to a plain passthrough below iOS 26 and under Reduce Transparency,
/// where there is nothing to blend.
struct GlassGroup<Content: View>: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var spacing: CGFloat? = nil
    @ViewBuilder let content: Content

    var body: some View {
        if #available(iOS 26.0, *), !reduceTransparency {
            GlassEffectContainer(spacing: spacing) { content }
        } else {
            content
        }
    }
}

#Preview("Glass surfaces") {
    VStack(spacing: 20) {
        GlassGroup(spacing: 14) {
            HStack(spacing: 10) {
                Text("ON-DEVICE AI")
                    .font(BinSightTheme.rounded(.caption2, weight: .semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .glassSurface(in: Capsule())

                Text("Tinted")
                    .font(BinSightTheme.rounded(.caption2, weight: .semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .glassSurface(in: Capsule(), tint: BinSightTheme.lime)
            }
        }

        Text("Interactive")
            .font(BinSightTheme.rounded(.subheadline, weight: .semibold))
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .glassSurface(in: Capsule(), interactive: true)
    }
    .padding(40)
    .background(MockCameraBackground())
}
