import CoreMedia
import CoreVideo
import ImageIO

/// One camera frame handed across the capture boundary.
///
/// This is the whole contract Phase 5 inference will consume. It carries the
/// pixel buffer, when it was captured, and — stated explicitly rather than
/// assumed — which way up it is.
///
/// `@unchecked Sendable`: `CVPixelBuffer` is a CoreFoundation type with no
/// `Sendable` conformance. It is safe to move here because the capture pipeline
/// hands the buffer over and never touches it again, and because the delivery
/// stream retains at most one at a time (see `CameraSessionController`).
struct CameraFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer

    /// Presentation timestamp from the sample buffer, in seconds.
    let timestamp: TimeInterval

    /// The buffer's orientation *as delivered*.
    ///
    /// The capture connection is rotated to portrait before frames arrive, so
    /// this is `.up` and the consumer needs no further rotation. It is a stored
    /// property rather than a hardcoded assumption so that if rotation handling
    /// ever changes, the consumer breaks loudly instead of silently classifying
    /// sideways images.
    let orientation: CGImagePropertyOrientation

    var pixelSize: CGSize {
        CGSize(
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer)
        )
    }
}

/// Drops frames at the capture boundary so downstream work runs at a target
/// rate rather than at the camera's 30 fps.
///
/// Pure value logic, deliberately: throttling is easy to get subtly wrong, and
/// this way it can be tested without a camera.
struct FrameSampler {
    /// Minimum gap between accepted frames.
    let minimumInterval: TimeInterval

    private var lastAcceptedTimestamp: TimeInterval?

    /// - Parameter targetFramesPerSecond: how many frames a second should get
    ///   through. Phase 5 targets 3–5 inferences per second; more than that
    ///   burns battery without making the reading feel any more live.
    init(targetFramesPerSecond: Double) {
        precondition(targetFramesPerSecond > 0, "target rate must be positive")
        self.minimumInterval = 1.0 / targetFramesPerSecond
    }

    /// Whether the frame at `timestamp` should be passed on.
    ///
    /// The first frame is always accepted. A timestamp that goes backwards —
    /// which happens when the session restarts and the clock rebases — resets
    /// the sampler rather than stalling it until the old clock catches up.
    mutating func shouldAccept(at timestamp: TimeInterval) -> Bool {
        guard let last = lastAcceptedTimestamp else {
            lastAcceptedTimestamp = timestamp
            return true
        }

        if timestamp < last {
            lastAcceptedTimestamp = timestamp
            return true
        }

        guard timestamp - last >= minimumInterval else { return false }

        lastAcceptedTimestamp = timestamp
        return true
    }

    /// Forgets the last accepted timestamp, so the next frame is accepted.
    mutating func reset() {
        lastAcceptedTimestamp = nil
    }
}
