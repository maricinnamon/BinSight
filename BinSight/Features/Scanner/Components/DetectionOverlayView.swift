import SwiftUI

/// Draws detection boxes over the live preview.
///
/// The only interesting part is the coordinate mapping, and it is deliberately
/// delegated: boxes arrive in camera-buffer pixels, the preview shows an
/// aspect-fill centre crop of that buffer, and `PreviewCropGeometry` owns the
/// transform between them. Scaling by `previewSize / bufferSize` here — the
/// obvious-looking shortcut — would misplace every box whenever the buffer and
/// the preview have different aspect ratios, which on this app is always
/// (720x1280 into a full-screen preview).
///
/// Boxes for objects the preview has cropped away are drawn off-screen and
/// clipped, rather than clamped to the edge, because pinning a box to the border
/// would claim an object is at the edge of the frame when it is outside it.
struct DetectionOverlayView: View {
    let detections: [Detection]
    /// Pixel size of the buffer `detections` are expressed in.
    let sourceSize: CGSize

    var body: some View {
        GeometryReader { proxy in
            let previewSize = proxy.size
            ZStack(alignment: .topLeading) {
                ForEach(detections) { detection in
                    let rect = PreviewCropGeometry.previewRect(
                        forBufferPixelRect: detection.boundingBox,
                        bufferSize: sourceSize,
                        previewSize: previewSize
                    )
                    if isDrawable(rect, in: previewSize) {
                        DetectionBox(detection: detection, rect: rect)
                    }
                }
            }
            .frame(width: previewSize.width, height: previewSize.height, alignment: .topLeading)
        }
        .clipped()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// Rejects degenerate rects and ones entirely outside the preview.
    ///
    /// A `.null` rect comes from a degenerate geometry (a zero-sized layout pass
    /// during setup); NaN would otherwise propagate straight into SwiftUI layout.
    private func isDrawable(_ rect: CGRect, in previewSize: CGSize) -> Bool {
        guard !rect.isNull, !rect.isEmpty,
              rect.origin.x.isFinite, rect.origin.y.isFinite,
              rect.width.isFinite, rect.height.isFinite
        else { return false }
        return rect.intersects(CGRect(origin: .zero, size: previewSize))
    }
}

/// One box plus its label.
private struct DetectionBox: View {
    let detection: Detection
    let rect: CGRect

    private var accent: Color { detection.detectedClass.wasteCategory.accent }

    /// Keep the label inside the frame when the box is hard against the top,
    /// otherwise it is clipped away exactly when the object is most prominent.
    private var labelSitsInside: Bool { rect.minY < 22 }

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(accent, lineWidth: 2.5)
                // A dark halo under the stroke: an accent line alone disappears
                // against a same-coloured object, and a camera frame can be any
                // colour at all.
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(BinSightTheme.ink.opacity(0.35), lineWidth: 4)
                )
                .frame(width: rect.width, height: rect.height)

            label
                .offset(x: 0, y: labelSitsInside ? 3 : -21)
        }
        .frame(width: rect.width, height: rect.height, alignment: .topLeading)
        .offset(x: rect.minX, y: rect.minY)
    }

    private var label: some View {
        Text(detection.label)
            .font(BinSightTheme.rounded(.caption2, weight: .bold))
            .foregroundStyle(BinSightTheme.ink)
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(accent))
            .padding(.leading, labelSitsInside ? 4 : 0)
    }
}

#if DEBUG
#Preview("Overlay") {
    // 720x1280 buffer, the portrait shape the capture session produces.
    let source = CGSize(width: 720, height: 1280)
    return ZStack {
        Color.black
        DetectionOverlayView(
            detections: [
                Detection(detectedClass: .plastic, confidence: 0.91,
                          boundingBox: CGRect(x: 180, y: 420, width: 330, height: 380)),
                Detection(detectedClass: .metal, confidence: 0.76,
                          boundingBox: CGRect(x: 90, y: 180, width: 200, height: 190)),
                Detection(detectedClass: .paper, confidence: 0.62,
                          boundingBox: CGRect(x: 330, y: 880, width: 300, height: 260)),
            ],
            sourceSize: source
        )
    }
    .ignoresSafeArea()
}
#endif
