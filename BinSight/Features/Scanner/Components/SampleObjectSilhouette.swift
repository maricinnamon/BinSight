import SwiftUI

/// A stylised bottle standing on a surface, built from primitives.
///
/// This is the "there is an item in frame" cue for the mock preview. It sizes
/// itself from its container so it always sits inside the reticle rather than
/// competing with the copy around it, and it is drawn opaque inside a
/// compositing group then faded as one layer, so the overlapping parts never
/// show seams where they meet.
struct SampleObjectSilhouette: View {
    var body: some View {
        GeometryReader { proxy in
            // 2.6 is the drawing's own height-to-body-width ratio, so deriving
            // the width from the height keeps the whole object inside the box.
            let bodyWidth = min(proxy.size.height / 2.9, proxy.size.width * 0.44)

            ZStack {
                ZStack {
                    // Cap
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .frame(width: bodyWidth * 0.34, height: bodyWidth * 0.22)
                        .offset(y: -bodyWidth * 1.24)
                    // Neck
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .frame(width: bodyWidth * 0.28, height: bodyWidth * 0.55)
                        .offset(y: -bodyWidth * 0.92)
                    // Shoulders
                    Ellipse()
                        .frame(width: bodyWidth * 0.94, height: bodyWidth * 0.72)
                        .offset(y: -bodyWidth * 0.48)
                    // Body
                    RoundedRectangle(cornerRadius: bodyWidth * 0.18, style: .continuous)
                        .frame(width: bodyWidth, height: bodyWidth * 1.5)
                        .offset(y: bodyWidth * 0.3)
                }
                .compositingGroup()
                .opacity(0.16)

                // Contact shadow, kept separate so it stays softer than the body.
                Ellipse()
                    .frame(width: bodyWidth * 1.45, height: bodyWidth * 0.18)
                    .opacity(0.08)
                    .offset(y: bodyWidth * 1.14)
            }
            .foregroundStyle(BinSightTheme.silhouette)
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .accessibilityHidden(true)
    }
}

#Preview {
    SampleObjectSilhouette()
        .frame(width: 300, height: 300)
        .background(MockCameraBackground())
}
