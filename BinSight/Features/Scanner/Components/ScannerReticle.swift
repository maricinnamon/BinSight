import SwiftUI

/// Four corner guides marking where to put the item, plus a slow accent sweep
/// while the app is reading it.
///
/// Two deliberate restraints:
///
/// * **No bounding box.** This is image classification, not object detection —
///   the model has no idea where the item is, and drawing a box around it would
///   claim a capability that does not exist.
/// * **Corners only, never a full frame.** The object has to stay visible; a
///   frame or a scrim inside the reticle would hide the thing being judged.
///
/// Contrast is handled by drawing each corner twice: a dark halo underneath and
/// a light core on top. A single-colour stroke disappears against either a
/// white wall or a dark bin, and the camera supplies both.
struct ScannerReticle: View {
    var isScanning: Bool
    /// Tints the guides with the current class accent once there is one.
    var accent: Color?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var sweepAtBottom = false

    private var animatesSweep: Bool { isScanning && !reduceMotion }

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size
            let arm = min(size.width, size.height) * 0.17

            ZStack {
                // Halo: sits under the core stroke so the guides read against a
                // bright frame as well as a dark one.
                CornerGuides(inset: 2, armLength: arm)
                    .stroke(
                        BinSightTheme.ink.opacity(0.35),
                        style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round)
                    )
                    .blur(radius: 2)

                CornerGuides(inset: 2, armLength: arm)
                    .stroke(
                        (accent ?? BinSightTheme.onCamera).opacity(isScanning ? 0.95 : 0.7),
                        style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round)
                    )

                sweep(in: size)
            }
            .frame(width: size.width, height: size.height)
        }
        .onAppear { sweepAtBottom = animatesSweep }
        .onChange(of: animatesSweep) { _, isAnimating in
            sweepAtBottom = isAnimating
        }
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private func sweep(in size: CGSize) -> some View {
        let travel = size.height * 0.32

        Capsule()
            .fill(
                LinearGradient(
                    colors: [.clear, accent ?? BinSightTheme.lime, .clear],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .frame(height: 2.5)
            .padding(.horizontal, size.width * 0.14)
            .offset(y: sweepAtBottom ? travel : -travel)
            .opacity(isScanning ? 0.9 : 0)
            .animation(
                animatesSweep
                    ? .easeInOut(duration: 1.9).repeatForever(autoreverses: true)
                    : .easeOut(duration: 0.25),
                value: sweepAtBottom
            )
            .animation(.easeOut(duration: 0.25), value: isScanning)
    }
}

/// The corner brackets themselves.
private struct CornerGuides: Shape {
    var inset: CGFloat
    var armLength: CGFloat

    func path(in rect: CGRect) -> Path {
        let frame = rect.insetBy(dx: inset, dy: inset)
        let arm = min(armLength, min(frame.width, frame.height) / 2)
        var path = Path()

        // Top-left
        path.move(to: CGPoint(x: frame.minX, y: frame.minY + arm))
        path.addLine(to: CGPoint(x: frame.minX, y: frame.minY))
        path.addLine(to: CGPoint(x: frame.minX + arm, y: frame.minY))

        // Top-right
        path.move(to: CGPoint(x: frame.maxX - arm, y: frame.minY))
        path.addLine(to: CGPoint(x: frame.maxX, y: frame.minY))
        path.addLine(to: CGPoint(x: frame.maxX, y: frame.minY + arm))

        // Bottom-right
        path.move(to: CGPoint(x: frame.maxX, y: frame.maxY - arm))
        path.addLine(to: CGPoint(x: frame.maxX, y: frame.maxY))
        path.addLine(to: CGPoint(x: frame.maxX - arm, y: frame.maxY))

        // Bottom-left
        path.move(to: CGPoint(x: frame.minX + arm, y: frame.maxY))
        path.addLine(to: CGPoint(x: frame.minX, y: frame.maxY))
        path.addLine(to: CGPoint(x: frame.minX, y: frame.maxY - arm))

        return path
    }
}

#Preview("Idle") {
    ScannerReticle(isScanning: false)
        .frame(width: 280, height: 280)
        .background(MockCameraBackground())
}

#Preview("Scanning") {
    ScannerReticle(isScanning: true)
        .frame(width: 280, height: 280)
        .background(MockCameraBackground())
}

#Preview("Accented") {
    ScannerReticle(isScanning: false, accent: BinSightTheme.coral)
        .frame(width: 280, height: 280)
        .background(MockCameraBackground())
}
