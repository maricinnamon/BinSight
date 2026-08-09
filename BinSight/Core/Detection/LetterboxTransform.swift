import CoreGraphics
import Foundation

/// The aspect-preserving fit from a camera frame into the model's square input,
/// and the inverse mapping that brings predicted boxes back.
///
/// The model takes 640x640. A camera frame is 720x1280. Squashing one into the
/// other would change every object's aspect ratio, and the detector was trained
/// on undistorted images — so it would be asked to recognise shapes it has never
/// seen, and every predicted box would be wrong in a way that still *looks*
/// plausible on screen. Instead the frame is scaled by a single factor and
/// centred, with the leftover margin padded.
///
/// This matches Ultralytics' `LetterBox`, which is what the model was trained
/// and validated with, including the grey padding value.
///
/// A worked example, for the portrait buffer this app actually produces:
///
/// ```
/// source 720x1280 -> 640x640
///   scale = min(640/720, 640/1280) = 0.5
///   scaled = 360x640
///   padX = (640 - 360)/2 = 140,  padY = 0
/// ```
///
/// Padding is **centred**, which is worth stating because it makes the transform
/// invariant to y-axis direction: top and bottom pads are equal, so a
/// bottom-left-origin renderer (CoreImage) and a top-left-origin one
/// (CoreGraphics) agree on where the content lands.
///
/// Pure value semantics and no CoreImage/CoreVideo dependency, so the geometry
/// can be tested exhaustively without a camera or a model.
struct LetterboxTransform: Equatable, Sendable {

    /// Ultralytics' padding grey. Matching it matters: the model saw this exact
    /// value in the padded margins during training and validation.
    static let paddingComponent: CGFloat = 114.0 / 255.0

    let sourceSize: CGSize
    let targetSize: CGSize

    /// Single uniform factor applied to both axes.
    let scale: CGFloat

    /// Half the leftover margin, in target (model input) pixels.
    let padX: CGFloat
    let padY: CGFloat

    /// The source content's size once scaled, excluding padding.
    var scaledSize: CGSize {
        CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
    }

    /// Where the scaled content sits inside the target square.
    var contentRect: CGRect {
        CGRect(origin: CGPoint(x: padX, y: padY), size: scaledSize)
    }

    /// - Parameters:
    ///   - sourceSize: the camera frame, in pixels.
    ///   - targetSize: the model input, normally 640x640.
    ///
    /// A degenerate source collapses to `scale == 0`; callers should treat that
    /// as "no usable frame" rather than dividing by it. `unletterbox` guards
    /// against it explicitly.
    init(sourceSize: CGSize, targetSize: CGSize) {
        self.sourceSize = sourceSize
        self.targetSize = targetSize

        guard sourceSize.width > 0, sourceSize.height > 0,
              targetSize.width > 0, targetSize.height > 0
        else {
            self.scale = 0
            self.padX = 0
            self.padY = 0
            return
        }

        let factor = min(targetSize.width / sourceSize.width,
                         targetSize.height / sourceSize.height)
        self.scale = factor
        self.padX = (targetSize.width - sourceSize.width * factor) / 2
        self.padY = (targetSize.height - sourceSize.height * factor) / 2
    }

    /// Maps a box from model-input pixels back to source-image pixels.
    ///
    /// Undoes the centring first, then the scale — the reverse order of the
    /// forward transform. The result is clamped to the source bounds, because
    /// the detector's boxes are not clamped and legitimately overhang the edge
    /// by a few pixels on objects that run out of frame.
    ///
    /// Returns `.null` for a degenerate transform or a non-finite input, so a
    /// bad row cannot become a `CGRect` full of NaN that silently poisons layout.
    func unletterbox(_ rect: CGRect) -> CGRect {
        guard scale > 0,
              rect.minX.isFinite, rect.minY.isFinite,
              rect.maxX.isFinite, rect.maxY.isFinite
        else { return .null }

        let x1 = (rect.minX - padX) / scale
        let y1 = (rect.minY - padY) / scale
        let x2 = (rect.maxX - padX) / scale
        let y2 = (rect.maxY - padY) / scale

        let clampedX1 = min(max(x1, 0), sourceSize.width)
        let clampedY1 = min(max(y1, 0), sourceSize.height)
        let clampedX2 = min(max(x2, 0), sourceSize.width)
        let clampedY2 = min(max(y2, 0), sourceSize.height)

        return CGRect(x: clampedX1,
                      y: clampedY1,
                      width: max(clampedX2 - clampedX1, 0),
                      height: max(clampedY2 - clampedY1, 0))
    }

    /// Forward mapping, source pixels to model-input pixels.
    ///
    /// Not used by the inference path — the image itself is resized by
    /// CoreImage — but it is what makes `unletterbox` testable as a true
    /// inverse rather than by restating its own arithmetic.
    func letterbox(_ rect: CGRect) -> CGRect {
        guard scale > 0 else { return .null }
        return CGRect(x: rect.minX * scale + padX,
                      y: rect.minY * scale + padY,
                      width: rect.width * scale,
                      height: rect.height * scale)
    }
}
