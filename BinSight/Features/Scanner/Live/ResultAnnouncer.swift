import Foundation

/// Decides what VoiceOver should say, and when.
///
/// A classifier running at 4/s would otherwise produce a stream of interruptions
/// that makes the screen unusable with VoiceOver on. The rules:
///
/// * Announce only a **meaningful** change — a different category, or crossing
///   into "not sure" or an error. Re-confirming the same category says nothing.
/// * Never announce a confidence change on its own. The number moves constantly
///   and the category is what matters.
/// * Throttle, so even genuine changes cannot arrive faster than they can be
///   listened to.
/// * Say nothing for transient states (`scanning`, `modelLoading`) — they are
///   noise, and the result that follows is the news.
///
/// Pure value logic with an injected clock, so all of that is testable.
struct ResultAnnouncer {

    /// Minimum gap between two spoken announcements.
    let minimumInterval: TimeInterval

    /// What was last announced, so a repeat can be suppressed.
    private var lastAnnouncedKey: String?
    private var lastAnnouncedAt: TimeInterval?

    init(minimumInterval: TimeInterval = 1.5) {
        self.minimumInterval = minimumInterval
    }

    /// The phrase to announce for `state`, or `nil` to stay silent.
    ///
    /// - Parameter now: a monotonic time in seconds.
    mutating func announcement(for state: ScannerDisplayState, at now: TimeInterval) -> String? {
        guard let key = Self.key(for: state), let phrase = Self.phrase(for: state) else {
            // Transient state: say nothing, and do not reset the throttle —
            // a brief dip through `scanning` between two results should not
            // unlock an immediate re-announcement of the same category.
            return nil
        }

        // Same news as last time.
        guard key != lastAnnouncedKey else { return nil }

        if let last = lastAnnouncedAt, now - last < minimumInterval, now >= last {
            return nil
        }

        lastAnnouncedKey = key
        lastAnnouncedAt = now
        return phrase
    }

    /// Forgets history, so the next meaningful state is announced immediately.
    /// Call when the scanner restarts or the screen reappears.
    mutating func reset() {
        lastAnnouncedKey = nil
        lastAnnouncedAt = nil
    }

    /// Identity of the *news*, deliberately excluding confidence: a category
    /// holding steady while its percentage drifts is not a new announcement.
    private static func key(for state: ScannerDisplayState) -> String? {
        switch state {
        case .ready: "ready"
        case .scanning, .modelLoading: nil
        case .result(let result): "result.\(result.category.rawValue)"
        case .notSure: "notSure"
        case .unavailable(let reason): "unavailable.\(reason.rawValue)"
        }
    }

    private static func phrase(for state: ScannerDisplayState) -> String? {
        switch state {
        case .ready: L("announce.readyToScan")
        case .scanning, .modelLoading: nil
        case .result(let result): L("announce.result", result.category.displayName, result.confidenceText)
        case .notSure: ScannerDisplayState.notSureMessage
        case .unavailable(let reason): reason.title
        }
    }
}
