import Foundation

/// A fixed-size ring of recent durations, with the summary statistics the
/// DEBUG overlay and the device-testing notes need.
///
/// Bounded on purpose: an unbounded array of every latency since launch is a
/// slow leak, and averaging over a whole session hides the thermal throttling
/// that actually matters on a phone.
struct LatencyTracker: Equatable, Sendable {

    private(set) var samples: [TimeInterval] = []
    let capacity: Int

    init(capacity: Int = 60) {
        precondition(capacity > 0, "capacity must be positive")
        self.capacity = capacity
        samples.reserveCapacity(capacity)
    }

    mutating func record(_ duration: TimeInterval) {
        guard duration.isFinite, duration >= 0 else { return }
        samples.append(duration)
        if samples.count > capacity {
            samples.removeFirst(samples.count - capacity)
        }
    }

    mutating func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    var count: Int { samples.count }
    var isEmpty: Bool { samples.isEmpty }

    var average: TimeInterval? {
        guard !samples.isEmpty else { return nil }
        return samples.reduce(0, +) / Double(samples.count)
    }

    var median: TimeInterval? { percentile(0.5) }

    /// The value below which `fraction` of samples fall, by nearest rank.
    ///
    /// Nearest-rank rather than interpolated: with 20-60 samples the difference
    /// is noise, and this cannot produce a value that was never measured.
    func percentile(_ fraction: Double) -> TimeInterval? {
        guard !samples.isEmpty else { return nil }
        let sorted = samples.sorted()
        let clamped = min(max(fraction, 0), 1)
        let rank = Int((clamped * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank - 1, 0), sorted.count - 1)]
    }

    var p95: TimeInterval? { percentile(0.95) }
}

/// What the DEBUG overlay shows. Compact by design — this crosses to the main
/// actor on every published update.
struct ScanMetrics: Equatable, Sendable {
    var processedFrames = 0
    /// Frames the pipeline never saw because it was busy, plus frames dropped
    /// at the capture boundary by the sampler.
    var droppedFrames = 0
    var rejectedOutputs = 0

    var lastPreprocessing: TimeInterval?
    var lastInvocation: TimeInterval?
    var lastEndToEnd: TimeInterval?

    var averageEndToEnd: TimeInterval?
    var medianEndToEnd: TimeInterval?
    var p95EndToEnd: TimeInterval?

    /// Classifications a second, over the tracked window.
    var effectiveRate: Double?

    /// The model's raw top class before smoothing, and its score.
    ///
    /// Diagnostic: the smoother deliberately withholds a verdict until it is
    /// confident, so a screen showing "No confident match" is ambiguous — it
    /// could mean the model is working and unsure, or not running at all. This
    /// distinguishes the two.
    var rawTopLabel: String?
    var rawTopScore: Double?

    /// Whether a classifier was constructed at all.
    var classifierLoaded = false
}
