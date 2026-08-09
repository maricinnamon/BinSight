import CoreGraphics
import Testing
@testable import BinSight

/// The geometry that decides whether a box lands on the object or next to it.
///
/// Worth testing hard: a wrong letterbox produces boxes that look entirely
/// plausible — right size, right shape — but sit in the wrong place. Nothing
/// crashes and no value is obviously bad, so only arithmetic catches it.
struct LetterboxTransformTests {

    private let net = CGSize(width: 640, height: 640)

    private func expectClose(_ a: CGFloat, _ b: CGFloat, tolerance: CGFloat = 0.001) {
        #expect(abs(a - b) < tolerance, "\(a) is not within \(tolerance) of \(b)")
    }

    // MARK: - Forward geometry

    @Test("A square source needs no padding at all")
    func squareSourceHasNoPadding() {
        let transform = LetterboxTransform(sourceSize: CGSize(width: 416, height: 416), targetSize: net)
        expectClose(transform.scale, 640.0 / 416.0)
        expectClose(transform.padX, 0)
        expectClose(transform.padY, 0)
        expectClose(transform.scaledSize.width, 640)
        expectClose(transform.scaledSize.height, 640)
    }

    @Test("Portrait 720x1280 — the shape the capture session delivers")
    func portraitCaptureShape() {
        let transform = LetterboxTransform(sourceSize: CGSize(width: 720, height: 1280), targetSize: net)
        // min(640/720, 640/1280) = min(0.888…, 0.5) = 0.5 — height is the binding side.
        expectClose(transform.scale, 0.5)
        expectClose(transform.scaledSize.width, 360)
        expectClose(transform.scaledSize.height, 640)
        expectClose(transform.padX, 140)
        expectClose(transform.padY, 0)
    }

    @Test("Landscape 1920x1080 pads the top and bottom")
    func landscapeHD() {
        let transform = LetterboxTransform(sourceSize: CGSize(width: 1920, height: 1080), targetSize: net)
        expectClose(transform.scale, 640.0 / 1920.0)
        expectClose(transform.scaledSize.width, 640)
        expectClose(transform.scaledSize.height, 360)
        expectClose(transform.padX, 0)
        expectClose(transform.padY, 140)
    }

    @Test("Portrait 1080x1920 pads the sides")
    func portraitHD() {
        let transform = LetterboxTransform(sourceSize: CGSize(width: 1080, height: 1920), targetSize: net)
        expectClose(transform.scale, 640.0 / 1920.0)
        expectClose(transform.padX, 140)
        expectClose(transform.padY, 0)
    }

    @Test("Padding is always centred, so top and bottom margins are equal")
    func paddingIsCentred() {
        for size in [CGSize(width: 1920, height: 1080),
                     CGSize(width: 720, height: 1280),
                     CGSize(width: 300, height: 100)] {
            let transform = LetterboxTransform(sourceSize: size, targetSize: net)
            let leftover = CGSize(
                width: net.width - transform.scaledSize.width,
                height: net.height - transform.scaledSize.height
            )
            // This is what makes the transform invariant to y-axis direction.
            expectClose(transform.padX * 2, leftover.width)
            expectClose(transform.padY * 2, leftover.height)
        }
    }

    @Test("Aspect ratio is preserved — the whole point of letterboxing")
    func aspectRatioPreserved() {
        let source = CGSize(width: 1920, height: 1080)
        let transform = LetterboxTransform(sourceSize: source, targetSize: net)
        expectClose(transform.scaledSize.width / transform.scaledSize.height,
                    source.width / source.height)
    }

    // MARK: - Inverse mapping

    @Test("Un-letterboxing inverts letterboxing", arguments: [
        CGSize(width: 720, height: 1280),
        CGSize(width: 1920, height: 1080),
        CGSize(width: 1080, height: 1920),
        CGSize(width: 640, height: 640),
        CGSize(width: 416, height: 416),
    ])
    func roundTrip(source: CGSize) {
        let transform = LetterboxTransform(sourceSize: source, targetSize: net)
        // A box comfortably inside the frame, so clamping cannot mask an error.
        let original = CGRect(x: source.width * 0.2, y: source.height * 0.25,
                              width: source.width * 0.5, height: source.height * 0.4)
        let recovered = transform.unletterbox(transform.letterbox(original))
        expectClose(recovered.minX, original.minX, tolerance: 0.01)
        expectClose(recovered.minY, original.minY, tolerance: 0.01)
        expectClose(recovered.width, original.width, tolerance: 0.01)
        expectClose(recovered.height, original.height, tolerance: 0.01)
    }

    @Test("A full-frame model box maps to the full source frame")
    func fullFrameMapsBack() {
        let source = CGSize(width: 720, height: 1280)
        let transform = LetterboxTransform(sourceSize: source, targetSize: net)
        // The content occupies x 140...500, y 0...640 in model space.
        let content = CGRect(x: 140, y: 0, width: 360, height: 640)
        let mapped = transform.unletterbox(content)
        expectClose(mapped.minX, 0, tolerance: 0.01)
        expectClose(mapped.minY, 0, tolerance: 0.01)
        expectClose(mapped.width, 720, tolerance: 0.01)
        expectClose(mapped.height, 1280, tolerance: 0.01)
    }

    @Test("A centred model box stays centred in the source")
    func centreStaysCentred() {
        let source = CGSize(width: 720, height: 1280)
        let transform = LetterboxTransform(sourceSize: source, targetSize: net)
        let mapped = transform.unletterbox(CGRect(x: 300, y: 300, width: 40, height: 40))
        expectClose(mapped.midX, 360, tolerance: 0.01)   // 720/2
        expectClose(mapped.midY, 640, tolerance: 0.01)   // 1280/2
    }

    @Test("Ignoring padX would misplace a box by 280 source pixels")
    func paddingActuallyMatters() {
        // Guards the specific bug of dividing by scale without subtracting the
        // pad — the version that "looks right" until you notice everything is
        // shifted toward one side.
        let transform = LetterboxTransform(sourceSize: CGSize(width: 720, height: 1280), targetSize: net)
        let correct = transform.unletterbox(CGRect(x: 140, y: 0, width: 100, height: 100))
        let naive = CGRect(x: 140 / transform.scale, y: 0, width: 100, height: 100)
        expectClose(correct.minX, 0, tolerance: 0.01)
        expectClose(naive.minX, 280, tolerance: 0.01)
        #expect(abs(correct.minX - naive.minX) > 200)
    }

    // MARK: - Clamping and degenerate input

    @Test("Boxes that overhang the frame are clamped to it")
    func overhangIsClamped() {
        let source = CGSize(width: 720, height: 1280)
        let transform = LetterboxTransform(sourceSize: source, targetSize: net)
        // The detector does not clamp its own output; boxes can leave the canvas.
        let mapped = transform.unletterbox(CGRect(x: -20, y: -20, width: 700, height: 700))
        #expect(mapped.minX >= 0)
        #expect(mapped.minY >= 0)
        #expect(mapped.maxX <= source.width + 0.001)
        #expect(mapped.maxY <= source.height + 0.001)
    }

    @Test("A degenerate source yields no usable transform")
    func degenerateSource() {
        for size in [CGSize(width: 0, height: 0), CGSize(width: 720, height: 0), CGSize.zero] {
            let transform = LetterboxTransform(sourceSize: size, targetSize: net)
            #expect(transform.scale == 0)
            #expect(transform.unletterbox(CGRect(x: 0, y: 0, width: 10, height: 10)).isNull)
        }
    }

    @Test("Non-finite input cannot produce a NaN rect")
    func nonFiniteRejected() {
        let transform = LetterboxTransform(sourceSize: CGSize(width: 720, height: 1280), targetSize: net)
        #expect(transform.unletterbox(CGRect(x: CGFloat.nan, y: 0, width: 10, height: 10)).isNull)
        #expect(transform.unletterbox(CGRect(x: 0, y: CGFloat.infinity, width: 10, height: 10)).isNull)
    }

    @Test("Ultralytics' padding grey is matched exactly")
    func paddingColourMatchesTraining() {
        // 114/255. The model saw this value in every padded margin it trained on.
        expectClose(LetterboxTransform.paddingComponent, 114.0 / 255.0, tolerance: 1e-9)
    }
}
