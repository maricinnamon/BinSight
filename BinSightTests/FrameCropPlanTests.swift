import CoreGraphics
import Testing
@testable import BinSight

/// The "does the model see what the preview shows?" contract.
@MainActor
struct FrameCropPlanTests {

    private let buffer = CGSize(width: 1280, height: 720)
    private let preview = CGSize(width: 390, height: 844)

    @Test("The full-region plan is exactly what the aspect-fill preview displays")
    func fullRegionMatchesPreview() {
        // Built from the same PreviewCropGeometry the preview layer's
        // .resizeAspectFill behaviour is documented by — so there is no second
        // copy of the maths that could drift.
        let plan = FrameCropPlan.fullVisibleRegion(bufferSize: buffer, previewSize: preview)
        let visible = PreviewCropGeometry.visibleRect(bufferSize: buffer, previewSize: preview)

        #expect(plan.normalizedRect == visible)
        #expect(plan.isUsable)
    }

    @Test("The reticle plan is strictly inside the visible region")
    func reticleIsInsideVisibleRegion() {
        // The key property: the model must never be fed pixels the preview
        // cropped off the sides, because the person never framed them.
        let side: CGFloat = 300
        let reticle = CGRect(
            x: (preview.width - side) / 2,
            y: (preview.height - side) / 2,
            width: side,
            height: side
        )
        let plan = FrameCropPlan.forReticle(
            reticleRect: reticle,
            previewSize: preview,
            bufferSize: buffer
        )
        let visible = PreviewCropGeometry.visibleRect(bufferSize: buffer, previewSize: preview)

        #expect(plan.isUsable)
        #expect(plan.normalizedRect.minX >= visible.minX - 1e-9)
        #expect(plan.normalizedRect.maxX <= visible.maxX + 1e-9)
        #expect(plan.normalizedRect.minY >= visible.minY - 1e-9)
        #expect(plan.normalizedRect.maxY <= visible.maxY + 1e-9)
    }

    @Test("A centred reticle maps to a centred crop")
    func centredReticleStaysCentred() {
        let side: CGFloat = 300
        let reticle = CGRect(
            x: (preview.width - side) / 2,
            y: (preview.height - side) / 2,
            width: side,
            height: side
        )
        let plan = FrameCropPlan.forReticle(
            reticleRect: reticle,
            previewSize: preview,
            bufferSize: buffer
        )
        #expect(abs(plan.normalizedRect.midX - 0.5) < 1e-9)
        #expect(abs(plan.normalizedRect.midY - 0.5) < 1e-9)
    }

    @Test("The model input size comes from the contract")
    func targetSizeMatchesContract() {
        // model_contract.json -> input.shape [1, 224, 224, 3]
        #expect(FrameCropPlan.modelInputSize == CGSize(width: 224, height: 224))
        let plan = FrameCropPlan.fullVisibleRegion(bufferSize: buffer, previewSize: preview)
        #expect(plan.targetSize == FrameCropPlan.modelInputSize)
    }

    @Test("Plans convert to whole-pixel rects inside the buffer")
    func pixelRectStaysInBounds() {
        let plan = FrameCropPlan.fullVisibleRegion(bufferSize: buffer, previewSize: preview)
        let pixels = plan.pixelRect(bufferSize: buffer)

        #expect(pixels.minX >= 0)
        #expect(pixels.minY >= 0)
        #expect(pixels.maxX <= buffer.width)
        #expect(pixels.maxY <= buffer.height)
        #expect(pixels.width == pixels.width.rounded())
        #expect(pixels.height == pixels.height.rounded())
    }

    @Test("A degenerate crop is reported as unusable rather than silently classified")
    func degenerateCropIsUnusable() {
        let plan = FrameCropPlan(normalizedRect: .zero, targetSize: FrameCropPlan.modelInputSize)
        #expect(!plan.isUsable)
    }
}
