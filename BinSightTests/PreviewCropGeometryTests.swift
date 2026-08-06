import CoreGraphics
import Testing
@testable import BinSight

/// The crop contract Phase 5 inference has to match.
@MainActor
struct PreviewCropGeometryTests {

    private func expectClose(_ a: CGFloat, _ b: CGFloat, tolerance: CGFloat = 0.0001) {
        #expect(abs(a - b) < tolerance, "\(a) is not within \(tolerance) of \(b)")
    }

    // MARK: - visibleRect

    @Test("Matching aspect ratios show the whole buffer")
    func matchingAspectShowsEverything() {
        let rect = PreviewCropGeometry.visibleRect(
            bufferSize: CGSize(width: 1280, height: 720),
            previewSize: CGSize(width: 640, height: 360)
        )
        #expect(rect == CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    @Test("A landscape buffer in a portrait preview is cropped left and right")
    func widerBufferCropsSides() {
        // The real case: 1280x720 frames shown in a portrait phone region.
        let rect = PreviewCropGeometry.visibleRect(
            bufferSize: CGSize(width: 1280, height: 720),
            previewSize: CGSize(width: 390, height: 690)
        )
        // previewAspect / bufferAspect = (390/690) / (1280/720)
        let expectedWidth: CGFloat = (390.0 / 690.0) / (1280.0 / 720.0)
        expectClose(rect.width, expectedWidth)
        #expect(rect.height == 1)
        // Centred: equal amounts lost from each side.
        expectClose(rect.minX, (1 - expectedWidth) / 2)
        expectClose(rect.midX, 0.5)
    }

    @Test("A taller buffer than the preview is cropped top and bottom")
    func tallerBufferCropsTopAndBottom() {
        let rect = PreviewCropGeometry.visibleRect(
            bufferSize: CGSize(width: 720, height: 1280),
            previewSize: CGSize(width: 800, height: 400)
        )
        #expect(rect.width == 1)
        let expectedHeight: CGFloat = (720.0 / 1280.0) / (800.0 / 400.0)
        expectClose(rect.height, expectedHeight)
        expectClose(rect.midY, 0.5)
    }

    @Test("Aspect fill never drops content in both directions at once")
    func onlyOneAxisIsEverCropped() {
        // That is the defining property of aspect-fill versus aspect-fit: one
        // axis always covers fully.
        let cases: [(CGSize, CGSize)] = [
            (CGSize(width: 1280, height: 720), CGSize(width: 390, height: 690)),
            (CGSize(width: 720, height: 1280), CGSize(width: 800, height: 400)),
            (CGSize(width: 1920, height: 1080), CGSize(width: 320, height: 568)),
        ]
        for (buffer, preview) in cases {
            let rect = PreviewCropGeometry.visibleRect(bufferSize: buffer, previewSize: preview)
            #expect(rect.width == 1 || rect.height == 1)
            #expect(rect.width <= 1 && rect.height <= 1)
            #expect(rect.width > 0 && rect.height > 0)
        }
    }

    @Test("Degenerate sizes fall back to the full buffer rather than producing NaN")
    func degenerateSizesAreSafe() {
        // A zero-sized layout pass happens during view setup; it must not
        // produce a NaN crop that poisons everything downstream.
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        #expect(PreviewCropGeometry.visibleRect(bufferSize: .zero, previewSize: CGSize(width: 10, height: 10)) == unit)
        #expect(PreviewCropGeometry.visibleRect(bufferSize: CGSize(width: 10, height: 10), previewSize: .zero) == unit)
    }

    // MARK: - normalizedBufferRect

    @Test("A full-preview rect maps back to exactly the visible region")
    func fullRectMapsToVisibleRegion() {
        let buffer = CGSize(width: 1280, height: 720)
        let preview = CGSize(width: 390, height: 690)
        let visible = PreviewCropGeometry.visibleRect(bufferSize: buffer, previewSize: preview)

        let mapped = PreviewCropGeometry.normalizedBufferRect(
            forPreviewRect: CGRect(origin: .zero, size: preview),
            previewSize: preview,
            bufferSize: buffer
        )

        expectClose(mapped.minX, visible.minX)
        expectClose(mapped.minY, visible.minY)
        expectClose(mapped.width, visible.width)
        expectClose(mapped.height, visible.height)
    }

    @Test("A centred reticle maps to a centred buffer region")
    func centredReticleStaysCentred() {
        let buffer = CGSize(width: 1280, height: 720)
        let preview = CGSize(width: 390, height: 690)
        let side: CGFloat = 300
        let reticle = CGRect(
            x: (preview.width - side) / 2,
            y: (preview.height - side) / 2,
            width: side,
            height: side
        )

        let mapped = PreviewCropGeometry.normalizedBufferRect(
            forPreviewRect: reticle,
            previewSize: preview,
            bufferSize: buffer
        )

        expectClose(mapped.midX, 0.5)
        expectClose(mapped.midY, 0.5)
        #expect(mapped.width > 0 && mapped.height > 0)
    }

    @Test("A reticle overhanging the preview is clamped, never out of bounds")
    func overhangingRectIsClamped() {
        let mapped = PreviewCropGeometry.normalizedBufferRect(
            forPreviewRect: CGRect(x: -200, y: -200, width: 2000, height: 2000),
            previewSize: CGSize(width: 390, height: 690),
            bufferSize: CGSize(width: 1280, height: 720)
        )
        #expect(mapped.minX >= 0)
        #expect(mapped.minY >= 0)
        #expect(mapped.maxX <= 1)
        #expect(mapped.maxY <= 1)
    }

    // MARK: - pixelRect

    @Test("Normalised rects convert to whole-pixel rects")
    func pixelConversion() {
        let pixels = PreviewCropGeometry.pixelRect(
            fromNormalized: CGRect(x: 0.25, y: 0.0, width: 0.5, height: 1.0),
            bufferSize: CGSize(width: 1280, height: 720)
        )
        #expect(pixels == CGRect(x: 320, y: 0, width: 640, height: 720))
        #expect(pixels.minX == pixels.minX.rounded())
        #expect(pixels.width == pixels.width.rounded())
    }
}
