import CoreGraphics
import Testing
@testable import BinSight

/// The last hop: camera-buffer pixels to points on the preview.
///
/// This is where a box can be geometrically perfect and still be drawn in the
/// wrong place, because the preview shows an aspect-fill *centre crop* of the
/// buffer rather than the whole thing. The failure mode is subtle — boxes drift
/// toward the middle and shrink slightly — so the tests pin exact numbers.
struct DetectionPreviewProjectionTests {

    private func expectClose(_ a: CGFloat, _ b: CGFloat, tolerance: CGFloat = 0.01) {
        #expect(abs(a - b) < tolerance, "\(a) is not within \(tolerance) of \(b)")
    }

    /// The real pairing: a 720x1280 buffer in a 393x852 preview (iPhone 16 Pro).
    private let buffer = CGSize(width: 720, height: 1280)
    private let preview = CGSize(width: 393, height: 852)

    @Test("Identical aspect ratios reduce to a plain scale")
    func matchingAspect() {
        // 720x1280 into 360x640 is exactly half, with nothing cropped.
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: CGRect(x: 100, y: 200, width: 200, height: 400),
            bufferSize: buffer,
            previewSize: CGSize(width: 360, height: 640)
        )
        expectClose(rect.minX, 50)
        expectClose(rect.minY, 100)
        expectClose(rect.width, 100)
        expectClose(rect.height, 200)
    }

    @Test("The buffer centre maps to the preview centre")
    func centreMapsToCentre() {
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: CGRect(x: 360, y: 640, width: 0.001, height: 0.001),
            bufferSize: buffer,
            previewSize: preview
        )
        expectClose(rect.minX, preview.width / 2, tolerance: 0.1)
        expectClose(rect.minY, preview.height / 2, tolerance: 0.1)
    }

    @Test("A full-buffer box overflows the preview horizontally")
    func fullBufferOverflows() {
        // The preview is relatively narrower, so it crops the buffer's sides:
        // a full-width box must extend beyond both edges, not fit inside them.
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: CGRect(origin: .zero, size: buffer),
            bufferSize: buffer,
            previewSize: preview
        )
        #expect(rect.minX < 0)
        #expect(rect.maxX > preview.width)
        expectClose(rect.minY, 0)
        expectClose(rect.maxY, preview.height)
    }

    @Test("Cropped-away content maps off-screen rather than being clamped")
    func offScreenStaysOffScreen() {
        // Clamping would pin the box to the edge and assert the object is
        // visible at the border, which it is not.
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: CGRect(x: 0, y: 600, width: 20, height: 20),
            bufferSize: buffer,
            previewSize: preview
        )
        #expect(rect.maxX < 0)
    }

    @Test("Projection is the inverse of the visible-region crop")
    func inverseOfVisibleRect() {
        // Whatever `visibleRect` says is on screen must map onto the full
        // preview, edge to edge. The two functions have to agree or boxes sit
        // in different pixels from the image they describe.
        let visible = PreviewCropGeometry.visibleRect(bufferSize: buffer, previewSize: preview)
        let visibleInPixels = CGRect(
            x: visible.minX * buffer.width,
            y: visible.minY * buffer.height,
            width: visible.width * buffer.width,
            height: visible.height * buffer.height
        )
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: visibleInPixels,
            bufferSize: buffer,
            previewSize: preview
        )
        expectClose(rect.minX, 0)
        expectClose(rect.minY, 0)
        expectClose(rect.width, preview.width)
        expectClose(rect.height, preview.height)
    }

    @Test("Aspect ratio survives the projection")
    func aspectPreserved() {
        // Aspect-fill scales both axes equally, so a square object stays square.
        let square = CGRect(x: 260, y: 500, width: 200, height: 200)
        let rect = PreviewCropGeometry.previewRect(
            forBufferPixelRect: square, bufferSize: buffer, previewSize: preview
        )
        expectClose(rect.width / rect.height, 1, tolerance: 0.001)
    }

    @Test("The naive scale is measurably wrong, which is why this exists")
    func naiveScalingIsWrong() {
        let box = CGRect(x: 100, y: 500, width: 200, height: 200)
        let correct = PreviewCropGeometry.previewRect(
            forBufferPixelRect: box, bufferSize: buffer, previewSize: preview
        )
        let naive = CGRect(
            x: box.minX / buffer.width * preview.width,
            y: box.minY / buffer.height * preview.height,
            width: box.width / buffer.width * preview.width,
            height: box.height / buffer.height * preview.height
        )
        // Off by tens of points — enough to sit beside the object, not on it.
        #expect(abs(correct.minX - naive.minX) > 15)
        #expect(abs(correct.width - naive.width) > 10)
    }

    @Test("Degenerate sizes produce a null rect, never NaN")
    func degenerateSizes() {
        let box = CGRect(x: 0, y: 0, width: 10, height: 10)
        #expect(PreviewCropGeometry.previewRect(forBufferPixelRect: box,
                                                bufferSize: .zero,
                                                previewSize: preview).isNull)
        #expect(PreviewCropGeometry.previewRect(forBufferPixelRect: box,
                                                bufferSize: buffer,
                                                previewSize: .zero).isNull)
    }
}
