import AVFoundation
import CoreMedia
import ImageIO
import os

/// Owns the `AVCaptureSession`. Knows nothing about SwiftUI.
///
/// Threading, which is most of the job here:
///
/// * `sessionQueue` — every `startRunning`, `stopRunning` and
///   `beginConfiguration` block. These are synchronous and slow (hundreds of
///   milliseconds), so none of them may touch the main thread.
/// * `videoQueue` — the sample buffer delegate. Separate from `sessionQueue` so
///   frame delivery cannot be blocked behind a reconfiguration.
/// * The main actor — nothing. `ScannerCameraModel` does the hopping.
///
/// `@unchecked Sendable` because `AVCaptureSession` is not `Sendable`. All
/// mutable state is confined to `sessionQueue` (session, inputs, outputs) or
/// `videoQueue` (`sampler`), and the two never touch each other's state.
final class CameraSessionController: NSObject, @unchecked Sendable {

    /// Handed to the preview layer on the main thread. Reading this property is
    /// safe; mutating the session is not, and only happens on `sessionQueue`.
    let session = AVCaptureSession()

    /// Creates a **fresh** frame stream, replacing any previous one.
    ///
    /// A new stream per run, rather than one shared property, because
    /// `AsyncStream` can only be iterated once: handing the same stream to a
    /// second `start()` produces a consumer that silently receives nothing.
    /// That is exactly what happened on device — `model: loaded` but
    /// `frames: 0 ok · 0 dropped`.
    ///
    /// `bufferingNewest(1)` is the "do not retain an unbounded frame queue"
    /// guarantee: at most one pixel buffer waits for a consumer, and a slow
    /// consumer drops old frames instead of growing a backlog.
    func makeFrameStream() -> AsyncStream<CameraFrame> {
        let (stream, continuation) = AsyncStream.makeStream(
            of: CameraFrame.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        continuationLock.lock()
        framesContinuation?.finish()
        framesContinuation = continuation
        continuationLock.unlock()

        videoQueue.async { [self] in
            // A new consumer should not be throttled against the old clock.
            sampler.reset()
        }
        return stream
    }

    /// Fires when the session is interrupted, resumes, or hits a runtime error.
    /// The model turns these into UI state.
    var onStatusEvent: (@Sendable (StatusEvent) -> Void)?

    enum StatusEvent: Equatable, Sendable {
        case interrupted
        case interruptionEnded
        case runtimeError(String)
    }

    /// Guarded by `continuationLock`: written on the main actor by
    /// `makeFrameStream`, read on `videoQueue` by the delegate.
    private var framesContinuation: AsyncStream<CameraFrame>.Continuation?
    private let continuationLock = NSLock()
    private let sessionQueue = DispatchQueue(label: "com.marynaantonevych.BinSight.camera.session")
    private let videoQueue = DispatchQueue(
        label: "com.marynaantonevych.BinSight.camera.video",
        qos: .userInitiated
    )
    private let videoOutput = AVCaptureVideoDataOutput()
    private let logger = Logger(subsystem: "com.marynaantonevych.BinSight", category: "camera")

    /// `videoQueue` only.
    private var sampler: FrameSampler

    /// `sessionQueue` only.
    private var isConfigured = false

    /// Portrait. Explicit, so nothing depends on whatever the connection
    /// happens to default to on a given device.
    private static let portraitRotationAngle: CGFloat = 90

    init(targetFramesPerSecond: Double = 4) {
        self.sampler = FrameSampler(targetFramesPerSecond: targetFramesPerSecond)
        super.init()
        observeSessionNotifications()
    }

    deinit {
        framesContinuation?.finish()
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Lifecycle

    enum StartResult: Equatable, Sendable {
        case running
        case noCameraAvailable
        case configurationFailed(String)
    }

    /// Configures the session on first call, then starts it. Idempotent.
    func start() async -> StartResult {
        await withCheckedContinuation { continuation in
            sessionQueue.async { [self] in
                if !isConfigured {
                    if let failure = configureSession() {
                        continuation.resume(returning: failure)
                        return
                    }
                    isConfigured = true
                }
                if !session.isRunning {
                    session.startRunning()
                }
                continuation.resume(returning: .running)
            }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            guard session.isRunning else { return }
            session.stopRunning()
        }
        videoQueue.async { [self] in
            // Next start should not be throttled against the old clock.
            sampler.reset()
        }
    }

    // MARK: - Configuration

    /// Returns `nil` on success, or the failure to report.
    private func configureSession() -> StartResult? {
        guard let device = Self.preferredCaptureDevice() else {
            logger.error("No rear capture device available")
            return .noCameraAvailable
        }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // 1280x720 is plenty of detail for a 224x224 classifier and a phone
        // preview. Anything higher costs bandwidth, thermals and battery for
        // pixels that get thrown away in the downscale.
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                return .configurationFailed("Could not add the camera input.")
            }
            session.addInput(input)
        } catch {
            logger.error("Camera input failed: \(error.localizedDescription, privacy: .public)")
            return .configurationFailed(error.localizedDescription)
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // Never queue up stale frames behind a slow consumer.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(videoOutput) else {
            return .configurationFailed("Could not add the video output.")
        }
        session.addOutput(videoOutput)

        // Rotate at the connection so buffers arrive portrait-upright and the
        // frame contract can honestly say `.up`.
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(Self.portraitRotationAngle) {
            connection.videoRotationAngle = Self.portraitRotationAngle
        }

        logger.info("Session configured with \(device.localizedName, privacy: .public)")
        return nil
    }

    /// Rear wide-angle first, then progressively broader fallbacks.
    static func preferredCaptureDevice() -> AVCaptureDevice? {
        let preferredTypes: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .builtInDualWideCamera,
            .builtInDualCamera,
            .builtInTripleCamera,
        ]

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: preferredTypes,
            mediaType: .video,
            position: .back
        )

        for type in preferredTypes {
            if let match = discovery.devices.first(where: { $0.deviceType == type }) {
                return match
            }
        }

        // Last resort: any video device at all. Keeps unusual hardware working
        // rather than showing "no camera" on a device that has one.
        return discovery.devices.first ?? AVCaptureDevice.default(for: .video)
    }

    // MARK: - Notifications

    private func observeSessionNotifications() {
        let center = NotificationCenter.default
        center.addObserver(
            self,
            selector: #selector(sessionWasInterrupted(_:)),
            name: AVCaptureSession.wasInterruptedNotification,
            object: session
        )
        center.addObserver(
            self,
            selector: #selector(sessionInterruptionEnded(_:)),
            name: AVCaptureSession.interruptionEndedNotification,
            object: session
        )
        center.addObserver(
            self,
            selector: #selector(sessionRuntimeError(_:)),
            name: AVCaptureSession.runtimeErrorNotification,
            object: session
        )
    }

    @objc private func sessionWasInterrupted(_ notification: Notification) {
        logger.notice("Capture session interrupted")
        onStatusEvent?(.interrupted)
    }

    @objc private func sessionInterruptionEnded(_ notification: Notification) {
        logger.notice("Capture session interruption ended")
        onStatusEvent?(.interruptionEnded)
    }

    @objc private func sessionRuntimeError(_ notification: Notification) {
        let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        let description = error?.localizedDescription ?? "Unknown capture error"
        logger.error("Capture runtime error: \(description, privacy: .public)")
        onStatusEvent?(.runtimeError(description))
    }
}

// MARK: - Frame delivery

extension CameraSessionController: AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Called on `videoQueue` at the full capture rate.
    ///
    /// This does three cheap things and nothing else: check the throttle, pull
    /// the pixel buffer, yield. No image conversion, no `UIImage`, and above
    /// all **no SwiftUI state** — a `@Observable` write here would invalidate
    /// the view 30 times a second.
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard timestamp.isFinite, sampler.shouldAccept(at: timestamp) else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let frame = CameraFrame(
            pixelBuffer: pixelBuffer,
            timestamp: timestamp,
            // Frames are rotated at the connection, so they arrive upright.
            orientation: .up
        )
        continuationLock.lock()
        let continuation = framesContinuation
        continuationLock.unlock()
        continuation?.yield(frame)
    }
}
