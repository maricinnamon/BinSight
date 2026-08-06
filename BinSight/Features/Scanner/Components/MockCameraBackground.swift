import SwiftUI

/// A stand-in for the camera feed, drawn entirely with SwiftUI shapes.
///
/// No photography and no bundled imagery — this is a warm, out-of-focus scene
/// suggested by a gradient and two soft colour pools. It exists so the
/// composition can be judged at full-bleed size before AVFoundation lands
/// underneath it.
///
/// The sample object is deliberately *not* drawn here: it belongs inside the
/// reticle, so `ScannerView` composes `SampleObjectSilhouette` into the scanner
/// area where it stays framed by the corner guides.
struct MockCameraBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [BinSightTheme.canvasTop, BinSightTheme.canvasMid, BinSightTheme.canvasBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Two colour pools give the flat gradient depth without costing a
            // blur pass. Two, not five — this has to stay cheap enough to sit
            // under a live preview later.
            EllipticalGradient(
                colors: [BinSightTheme.coral.opacity(0.34), .clear],
                center: UnitPoint(x: 0.14, y: 0.18),
                startRadiusFraction: 0,
                endRadiusFraction: 0.6
            )
            EllipticalGradient(
                colors: [BinSightTheme.mint.opacity(0.34), .clear],
                center: UnitPoint(x: 0.9, y: 0.76),
                startRadiusFraction: 0,
                endRadiusFraction: 0.55
            )

            // A gentle vignette so the reticle and card read against the scene.
            // Kept far out and faint: pulled in any tighter it greys the whole
            // screen instead of framing it.
            RadialGradient(
                colors: [.clear, BinSightTheme.ink.opacity(0.2)],
                center: .center,
                startRadius: 260,
                endRadius: 660
            )
        }
    }
}

#Preview {
    MockCameraBackground()
        .ignoresSafeArea()
}
