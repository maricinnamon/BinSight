import CoreGraphics

/// The crop contract between what the preview shows and what inference sees.
///
/// **Phase 5 must match this or the model will classify the wrong pixels.**
///
/// The preview layer uses `.resizeAspectFill`, so it shows a *centre crop* of
/// each frame: the buffer is scaled until it covers the preview region, and
/// whatever overhangs is cut off. The video data output, meanwhile, delivers
/// the **whole** buffer — it knows nothing about the preview's shape.
///
/// So a frame straight off the data output contains more of the world than the
/// person can see. Classifying it whole would feed the model image regions the
/// user never framed, including whatever is off to the sides.
///
/// The pipeline is therefore:
///
/// 1. `visibleRect(bufferSize:previewSize:)` — the part of the buffer the
///    preview is actually showing.
/// 2. `normalizedBufferRect(forPreviewRect:previewSize:bufferSize:)` — map the
///    reticle square from preview coordinates into that visible region.
/// 3. Crop the buffer to the result, then scale to 224x224.
///
/// All rects returned here are **normalised** (0...1) in buffer coordinates,
/// origin top-left, matching `CGImage`/`CIImage` conventions after the capture
/// connection has rotated frames to portrait.
///
/// At runtime `AVCaptureVideoPreviewLayer.metadataOutputRectConverted(fromLayerRect:)`
/// computes the same mapping and is the better source of truth when a layer is
/// on screen. This type exists because that API needs a live layer, which makes
/// it untestable and unusable from a background queue — and because getting the
/// maths wrong silently degrades accuracy rather than crashing.
enum PreviewCropGeometry {

    /// The portion of the buffer an aspect-fill preview of `previewSize`
    /// displays, normalised into buffer coordinates.
    ///
    /// Returns the unit rect when either size is degenerate, so a zero-sized
    /// layout pass during view setup cannot produce a NaN crop.
    static func visibleRect(bufferSize: CGSize, previewSize: CGSize) -> CGRect {
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        guard bufferSize.width > 0, bufferSize.height > 0,
              previewSize.width > 0, previewSize.height > 0
        else { return unit }

        let bufferAspect = bufferSize.width / bufferSize.height
        let previewAspect = previewSize.width / previewSize.height

        if bufferAspect > previewAspect {
            // Buffer is wider than the preview: the sides are cropped away.
            let visibleWidth = previewAspect / bufferAspect
            return CGRect(x: (1 - visibleWidth) / 2, y: 0, width: visibleWidth, height: 1)
        } else if bufferAspect < previewAspect {
            // Buffer is taller: top and bottom are cropped away.
            let visibleHeight = bufferAspect / previewAspect
            return CGRect(x: 0, y: (1 - visibleHeight) / 2, width: 1, height: visibleHeight)
        } else {
            return unit
        }
    }

    /// Maps a rect given in preview-layer coordinates (points, origin top-left)
    /// to a normalised rect in buffer coordinates.
    ///
    /// Use this for the reticle: pass the reticle's frame in the preview's
    /// coordinate space and get back the region of the pixel buffer to crop.
    ///
    /// The result is clamped to the unit square, so a reticle that overhangs
    /// the preview cannot produce an out-of-bounds crop.
    static func normalizedBufferRect(
        forPreviewRect previewRect: CGRect,
        previewSize: CGSize,
        bufferSize: CGSize
    ) -> CGRect {
        guard previewSize.width > 0, previewSize.height > 0 else {
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        }

        let visible = visibleRect(bufferSize: bufferSize, previewSize: previewSize)

        // Where the rect sits within the preview, 0...1.
        let relativeX = previewRect.minX / previewSize.width
        let relativeY = previewRect.minY / previewSize.height
        let relativeWidth = previewRect.width / previewSize.width
        let relativeHeight = previewRect.height / previewSize.height

        // Project that onto the visible slice of the buffer.
        let mapped = CGRect(
            x: visible.minX + relativeX * visible.width,
            y: visible.minY + relativeY * visible.height,
            width: relativeWidth * visible.width,
            height: relativeHeight * visible.height
        )

        return clampedToUnitSquare(mapped)
    }

    /// Maps a rect in **buffer pixel** coordinates to the aspect-fill preview's
    /// own coordinate space, in points.
    ///
    /// This is the inverse of `visibleRect` and the transform the detection
    /// overlay needs: the detector returns boxes in the full buffer's pixel
    /// space, but the preview shows only a centre crop of that buffer. Scaling a
    /// box by `previewSize / bufferSize` — the obvious-looking thing — is wrong
    /// whenever the two aspect ratios differ, and it is wrong by a *translation*
    /// as well as a scale, so boxes drift toward the centre rather than simply
    /// being the wrong size.
    ///
    /// Worked through with the sizes this app actually produces:
    ///
    /// ```
    /// buffer 720x1280 (0.5625), preview 393x852 (0.4613)
    ///   buffer is relatively wider -> its sides are cropped
    ///   visible width = 0.4613/0.5625 = 0.820  ->  x from 0.090 to 0.910
    /// a box at buffer x = 360 (dead centre) maps to preview x = 196.5 (centre)
    /// a box at buffer x =  36 (5% in)      maps off-screen, negative
    /// ```
    ///
    /// The result is deliberately **not** clamped: a detection on an object the
    /// preview has cropped away really is off-screen, and clamping would pin a
    /// box to the edge as if the object were there. Callers clip instead.
    static func previewRect(
        forBufferPixelRect pixelRect: CGRect,
        bufferSize: CGSize,
        previewSize: CGSize
    ) -> CGRect {
        guard bufferSize.width > 0, bufferSize.height > 0,
              previewSize.width > 0, previewSize.height > 0
        else { return .null }

        let visible = visibleRect(bufferSize: bufferSize, previewSize: previewSize)
        guard visible.width > 0, visible.height > 0 else { return .null }

        // Buffer pixels -> normalised buffer.
        let normalizedX = pixelRect.minX / bufferSize.width
        let normalizedY = pixelRect.minY / bufferSize.height
        let normalizedWidth = pixelRect.width / bufferSize.width
        let normalizedHeight = pixelRect.height / bufferSize.height

        // Normalised buffer -> normalised visible region -> preview points.
        return CGRect(
            x: (normalizedX - visible.minX) / visible.width * previewSize.width,
            y: (normalizedY - visible.minY) / visible.height * previewSize.height,
            width: normalizedWidth / visible.width * previewSize.width,
            height: normalizedHeight / visible.height * previewSize.height
        )
    }

    /// Converts a normalised buffer rect to pixel coordinates, rounded to whole
    /// pixels so it can index a `CVPixelBuffer` directly.
    static func pixelRect(fromNormalized rect: CGRect, bufferSize: CGSize) -> CGRect {
        CGRect(
            x: (rect.minX * bufferSize.width).rounded(.down),
            y: (rect.minY * bufferSize.height).rounded(.down),
            width: (rect.width * bufferSize.width).rounded(),
            height: (rect.height * bufferSize.height).rounded()
        )
    }

    private static func clampedToUnitSquare(_ rect: CGRect) -> CGRect {
        let minX = min(max(rect.minX, 0), 1)
        let minY = min(max(rect.minY, 0), 1)
        let maxX = min(max(rect.maxX, 0), 1)
        let maxY = min(max(rect.maxY, 0), 1)
        return CGRect(x: minX, y: minY, width: max(maxX - minX, 0), height: max(maxY - minY, 0))
    }
}
