import Foundation

/// Every tunable number in the live scanner, in one place.
///
/// > **These values are provisional.** They were chosen to be conservative —
/// > biased toward saying "not sure" rather than naming a class — and they have
/// > **not** been calibrated against a trained model or validated on a physical
/// > device. Expect them to move once `docs/DEVICE_TESTING.md` has real
/// > measurements in it. Nothing here is a scientific result.
struct ScannerConfiguration: Equatable, Sendable {

    // MARK: - Frame rate

    /// How many classifications a second the pipeline aims for.
    ///
    /// 4/s reads as live to a person holding a phone, and leaves the CPU
    /// headroom that 30/s would not. Applied at the capture boundary by
    /// `FrameSampler`, so surplus frames are dropped before any work happens.
    var targetClassificationsPerSecond: Double = 4.0

    var classificationInterval: TimeInterval { 1.0 / targetClassificationsPerSecond }

    // MARK: - Smoothing

    /// Exponential moving average weight for the newest sample.
    ///
    /// 0.4 keeps roughly the last 4–5 frames meaningfully in play — about a
    /// second at 4/s. Higher reacts faster and flickers more; lower is calmer
    /// and feels laggy when you move the phone to a new item.
    var smoothingFactor: Double = 0.4

    /// Smoothed top-1 probability needed to *name* a class.
    var minimumConfidence: Double = 0.65

    /// Required gap between the smoothed top-1 and top-2 classes.
    ///
    /// Guards the case a single threshold misses: 0.66 vs 0.64 is not
    /// confidence, it is a coin flip with a high number attached.
    var minimumMargin: Double = 0.15

    /// Consecutive qualifying samples before a new label is shown.
    ///
    /// At 4/s this is ~0.75 s of agreement — long enough to ignore a single
    /// blurred frame, short enough not to feel sticky.
    var requiredConsecutiveAgreements: Int = 3

    // MARK: - Hysteresis
    //
    // Showing a label uses the thresholds above; *keeping* one uses these,
    // which are looser. Without the gap, a result hovering at the boundary
    // flickers between a category and "not sure" several times a second.

    /// Smoothed top-1 probability below which a shown label is dropped.
    var releaseConfidence: Double = 0.55

    /// Top-1/top-2 margin below which a shown label is dropped.
    var releaseMargin: Double = 0.08

    // MARK: - Reset

    /// Gap between frames that counts as a meaningful pause and clears history.
    ///
    /// Longer than this and the previous samples describe whatever the phone was
    /// pointed at before, which is worse than starting fresh.
    var staleSampleInterval: TimeInterval = 1.5

    /// Consecutive unconfident samples after which smoothing is reset.
    ///
    /// ~5 s at 4/s. If nothing has been recognisable for that long, old
    /// evidence is just holding the average down.
    var resetAfterUnconfidentSamples: Int = 20

    /// Samples needed before the smoother will commit to anything at all,
    /// so the very first frame cannot produce a confident answer.
    var minimumSamplesBeforeVerdict: Int = 2

    static let `default` = ScannerConfiguration()
}
