import CoreGraphics

/// Which pixels of a camera frame get classified.
///
/// **This is the answer to "does the preview show what the model sees?"** —
/// and it is answered by construction rather than by comparison: the plan is
/// derived from `PreviewCropGeometry`, the same type the preview's
/// `.resizeAspectFill` behaviour is documented by. There is no second copy of
/// the maths to drift out of step.
///
/// The chain, for a portrait phone:
///
/// ```
/// buffer 1280x720 (landscape, rotated upright at the capture connection)
///   └─ visibleRect(...)      what the aspect-fill preview actually displays
///        └─ reticle square   the region the person is aiming at
///             └─ 224x224     the model input
/// ```
///
/// Without the first step the model would be fed the parts of the frame cropped
/// off the sides of the preview — image the person never saw and never framed.
struct FrameCropPlan: Equatable, Sendable {

    /// The region to crop, normalised (0...1) in buffer coordinates.
    let normalizedRect: CGRect

    /// What the crop is scaled to before it reaches the model.
    let targetSize: CGSize

    /// Everything the preview is showing, scaled to `targetSize`.
    ///
    /// The fallback when the view has not reported its geometry yet: still
    /// correct with respect to the preview, just not narrowed to the reticle.
    static func fullVisibleRegion(
        bufferSize: CGSize,
        previewSize: CGSize,
        targetSize: CGSize = Self.modelInputSize
    ) -> FrameCropPlan {
        FrameCropPlan(
            normalizedRect: PreviewCropGeometry.visibleRect(
                bufferSize: bufferSize,
                previewSize: previewSize
            ),
            targetSize: targetSize
        )
    }

    /// Just the reticle square.
    ///
    /// - Parameters:
    ///   - reticleRect: the reticle's frame in the preview's coordinate space.
    ///   - previewSize: the preview region's size in points.
    ///   - bufferSize: the pixel buffer's dimensions.
    static func forReticle(
        reticleRect: CGRect,
        previewSize: CGSize,
        bufferSize: CGSize,
        targetSize: CGSize = Self.modelInputSize
    ) -> FrameCropPlan {
        FrameCropPlan(
            normalizedRect: PreviewCropGeometry.normalizedBufferRect(
                forPreviewRect: reticleRect,
                previewSize: previewSize,
                bufferSize: bufferSize
            ),
            targetSize: targetSize
        )
    }

    /// The crop in whole pixels, ready to index a `CVPixelBuffer`.
    func pixelRect(bufferSize: CGSize) -> CGRect {
        PreviewCropGeometry.pixelRect(fromNormalized: normalizedRect, bufferSize: bufferSize)
    }

    /// Whether this plan describes a usable region.
    var isUsable: Bool {
        normalizedRect.width > 0
            && normalizedRect.height > 0
            && normalizedRect.minX >= 0
            && normalizedRect.minY >= 0
            && normalizedRect.maxX <= 1.0001
            && normalizedRect.maxY <= 1.0001
    }

    /// 224x224, from `model_contract.json` → `input.shape`.
    static let modelInputSize = CGSize(width: 224, height: 224)
}
