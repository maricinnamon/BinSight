import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Testing
import UIKit
@testable import BinSight

/// End-to-end static inference: real bundled model, real images, no camera.
///
/// The unit tests above prove each stage in isolation. This proves they compose
/// — and it is the only test that can catch the failures that isolated tests
/// structurally cannot:
///
/// * **double normalisation** — the package already carries `scale = 1/255`, so
///   dividing again would hand the model a near-black image. Every confidence
///   would collapse and nothing would be detected.
/// * **a vertically flipped letterbox** — CoreImage renders bottom-left-origin.
///   A flip produces perfectly-shaped boxes in mirrored positions, which no
///   amount of synthetic-tensor testing would reveal.
/// * **wrong output indexing** — reading column 4 as the class or column 5 as
///   confidence yields plausible-looking nonsense.
/// * **wrong class mapping** — paper/plastic/metal transposed.
///
/// Expected values come from `scripts/make_ios_test_fixtures.py`, which runs the
/// *same* `.mlpackage` through coremltools. A failure here therefore means the
/// Swift path is wrong, not that the model changed.
///
/// Tolerances are deliberate: CoreImage's bilinear resampling is not identical
/// to OpenCV's, so confidences move by a few hundredths. Geometry, by contrast,
/// should be near-exact, so IoU is held to a much tighter bound than confidence.
struct YOLODetectionServiceTests {

    // MARK: - Fixtures

    private struct ExpectedDetection: Decodable {
        let className: String
        let classID: Int
        let confidence: Float
        let x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat
        var rect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
    }

    private struct Fixture: Decodable {
        let image: String
        let sourceWidth: CGFloat
        let sourceHeight: CGFloat
        let note: String
        let expected: [ExpectedDetection]
    }

    private struct Manifest: Decodable {
        let outputFeature: String
        let confidenceThreshold: Float
        let fixtures: [Fixture]
    }

    /// `Bundle(for:)` needs a class; swift-testing suites are structs.
    private final class BundleToken {}
    private var testBundle: Bundle { Bundle(for: BundleToken.self) }

    private func loadManifest() throws -> Manifest {
        guard let url = testBundle.url(forResource: "expected", withExtension: "json") else {
            throw TestFailure("expected.json is not in the test bundle — "
                              + "run scripts/make_ios_test_fixtures.py")
        }
        return try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
    }

    private struct TestFailure: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// Loads a fixture JPEG as a BGRA pixel buffer — the format the capture
    /// session delivers, so the service sees exactly what it sees at runtime.
    private func pixelBuffer(named name: String) throws -> CVPixelBuffer {
        let stem = (name as NSString).deletingPathExtension
        guard let url = testBundle.url(forResource: stem, withExtension: "jpg"),
              let data = try? Data(contentsOf: url),
              let image = UIImage(data: data)?.cgImage
        else { throw TestFailure("fixture image \(name) is not in the test bundle") }

        let width = image.width, height = image.height
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw TestFailure("CVPixelBufferCreate failed (\(status))")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { throw TestFailure("CGContext creation failed") }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func frame(named name: String) throws -> CameraFrame {
        CameraFrame(pixelBuffer: try pixelBuffer(named: name), timestamp: 0, orientation: .up)
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull, !intersection.isEmpty else { return 0 }
        let overlap = intersection.width * intersection.height
        let union = a.width * a.height + b.width * b.height - overlap
        return union > 0 ? overlap / union : 0
    }

    // MARK: - Loading

    @Test("The bundled model loads and its interface matches the contract")
    func modelLoads() throws {
        // The initialiser validates input kind, 640x640 size, a single
        // multiArray output and its trailing dimension. Not throwing *is* the
        // assertion.
        _ = try YOLODetectionService()
    }

    @Test("The compiled model is actually in the app bundle")
    func modelIsBundled() {
        #expect(Bundle.main.url(forResource: "BinSightYOLO26n", withExtension: "mlmodelc") != nil,
                "BinSightYOLO26n.mlmodelc is missing from the app bundle")
    }

    // MARK: - Static inference

    @Test("Swift inference reproduces the Python CoreML results")
    func staticInferenceMatchesPython() async throws {
        let manifest = try loadManifest()
        let service = try YOLODetectionService()
        #expect(!manifest.fixtures.isEmpty)

        for fixture in manifest.fixtures {
            let cameraFrame = try frame(named: fixture.image)
            #expect(cameraFrame.pixelSize.width == fixture.sourceWidth)
            #expect(cameraFrame.pixelSize.height == fixture.sourceHeight)

            let detections = try await service.detect(cameraFrame)

            // Every expected detection must be reproduced. Extra low-confidence
            // detections are tolerated — FP16 and resampling move borderline
            // rows across the 0.25 cutoff either way — but a *missing* strong
            // detection means the Swift path is broken.
            for expected in fixture.expected where expected.confidence >= 0.40 {
                let candidates = detections.filter { $0.classID == expected.classID }
                #expect(!candidates.isEmpty,
                        "\(fixture.image): no \(expected.className) detected at all")

                guard let best = candidates.max(by: { iou(expected.rect, $0.boundingBox)
                                                     < iou(expected.rect, $1.boundingBox) })
                else { continue }

                let overlap = iou(expected.rect, best.boundingBox)
                let geometry = "\(fixture.image): \(expected.className) IoU \(overlap) too low — "
                    + "expected \(expected.rect), got \(best.boundingBox). \(fixture.note)"
                #expect(overlap > 0.85, Comment(rawValue: geometry))

                let drift = abs(best.confidence - expected.confidence)
                let score = "\(fixture.image): \(expected.className) confidence "
                    + "\(best.confidence) differs from \(expected.confidence) by \(drift)"
                #expect(drift < 0.15, Comment(rawValue: score))
            }
        }
    }

    @Test("Each class is detected confidently in its own fixture")
    func everyClassIsReachable() async throws {
        let service = try YOLODetectionService()
        let cases: [(String, DetectedClass)] = [
            ("confident_paper.jpg", .paper),
            ("confident_plastic.jpg", .plastic),
            ("confident_metal.jpg", .metal),
        ]
        for (image, expected) in cases {
            let detections = try await service.detect(try frame(named: image))
            let strongest = detections.max(by: { $0.confidence < $1.confidence })
            let got = strongest?.className ?? "nothing"
            #expect(strongest?.detectedClass == expected,
                    Comment(rawValue: "\(image): expected \(expected.modelLabel), got \(got)"))
            // These fixtures were chosen as the most confident example of each
            // class; anything much below 0.8 means preprocessing has degraded.
            let score = strongest?.confidence ?? 0
            #expect(score > 0.80,
                    Comment(rawValue: "\(image): confidence \(score) is unexpectedly low — "
                                      + "the usual cause is normalising the pixels twice"))
        }
    }

    @Test("A padded portrait frame maps boxes back correctly")
    func portraitLetterboxRoundTrip() async throws {
        // 720x1280 -> padX 140. If the pad were ignored or the image flipped,
        // the box would land in the wrong half of the frame while still looking
        // like a sensible detection.
        let manifest = try loadManifest()
        guard let fixture = manifest.fixtures.first(where: { $0.image == "portrait_frame.jpg" }),
              let expected = fixture.expected.first
        else { throw TestFailure("portrait fixture missing") }

        let service = try YOLODetectionService()
        let detections = try await service.detect(try frame(named: fixture.image))
        guard let best = detections.max(by: { $0.confidence < $1.confidence }) else {
            throw TestFailure("no detections in the portrait frame")
        }

        #expect(best.classID == expected.classID)
        #expect(iou(expected.rect, best.boundingBox) > 0.85,
                "expected \(expected.rect), got \(best.boundingBox)")

        // The object was composited into rows 280...1000 of a 1280-tall frame.
        // A vertical flip would put the box around row 280 from the *bottom*.
        #expect(best.boundingBox.midY > 400 && best.boundingBox.midY < 900,
                "box centre \(best.boundingBox.midY) suggests a vertical flip")
        // And it must stay inside the frame.
        #expect(best.boundingBox.minX >= 0)
        #expect(best.boundingBox.maxX <= fixture.sourceWidth + 0.5)
        #expect(best.boundingBox.maxY <= fixture.sourceHeight + 0.5)
    }

    @Test("Detections stay inside the source frame for every fixture")
    func detectionsAreInBounds() async throws {
        let manifest = try loadManifest()
        let service = try YOLODetectionService()
        for fixture in manifest.fixtures {
            for detection in try await service.detect(try frame(named: fixture.image)) {
                let box = detection.boundingBox
                #expect(box.minX >= 0 && box.minY >= 0)
                #expect(box.maxX <= fixture.sourceWidth + 0.5)
                #expect(box.maxY <= fixture.sourceHeight + 0.5)
                #expect(box.width > 0 && box.height > 0)
                #expect(detection.confidence >= manifest.confidenceThreshold)
            }
        }
    }

    @Test("Repeated inference on one frame is deterministic")
    func inferenceIsStable() async throws {
        // Catches state leaking between frames — a scratch buffer that is not
        // fully overwritten, or a pooled pixel buffer reused before rendering.
        let service = try YOLODetectionService()
        let cameraFrame = try frame(named: "confident_metal.jpg")
        let first = try await service.detect(cameraFrame)
        for _ in 0..<3 {
            let again = try await service.detect(cameraFrame)
            #expect(again.count == first.count)
            for (a, b) in zip(first, again) {
                #expect(a.classID == b.classID)
                #expect(abs(a.confidence - b.confidence) < 0.0001)
                #expect(abs(a.boundingBox.minX - b.boundingBox.minX) < 0.01)
            }
        }
    }

    @Test("Alternating between frames does not leak detections between them")
    func noStateLeakBetweenFrames() async throws {
        let service = try YOLODetectionService()
        let metal = try frame(named: "confident_metal.jpg")
        let paper = try frame(named: "confident_paper.jpg")

        let metalFirst = try await service.detect(metal)
        _ = try await service.detect(paper)
        let metalAgain = try await service.detect(metal)

        #expect(metalFirst.count == metalAgain.count)
        #expect(metalFirst.first?.classID == metalAgain.first?.classID)
    }

    // MARK: - Performance

    @Test("Inference is fast enough for a live preview")
    func inferenceLatency() async throws {
        let service = try YOLODetectionService()
        let cameraFrame = try frame(named: "portrait_frame.jpg")
        _ = try await service.detect(cameraFrame)   // warm up: first call compiles

        var total: TimeInterval = 0
        let runs = 10
        for _ in 0..<runs {
            let started = ContinuousClock.now
            _ = try await service.detect(cameraFrame)
            total += (ContinuousClock.now - started).seconds
        }
        let average = total / Double(runs)
        let inference = await service.lastInferenceDuration
        let preprocessing = await service.lastPreprocessingDuration
        print("""
            detection latency — end-to-end \(String(format: "%.1f", average * 1000))ms \
            (model \(String(format: "%.1f", inference * 1000))ms, \
            letterbox \(String(format: "%.1f", preprocessing * 1000))ms) \
            ≈ \(String(format: "%.1f", 1 / average))/s
            """)

        // Generous: this runs on a simulator, which has no Neural Engine. The
        // pipeline only needs 4/s, and the bound exists to catch an order-of-
        // magnitude regression such as reloading the model every frame.
        #expect(average < 1.0, "average inference \(average)s is far too slow")
    }
}
