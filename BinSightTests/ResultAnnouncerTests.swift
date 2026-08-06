import Testing
@testable import BinSight

/// VoiceOver announcement policy.
///
/// The failure this guards against: a classifier running at 4/s announcing
/// every inference, which makes the screen unusable with VoiceOver on.
@MainActor
struct ResultAnnouncerTests {

    @Test("A confident result is announced with its category and confidence")
    func announcesConfidentResult() {
        var announcer = ResultAnnouncer()
        let phrase = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        #expect(phrase?.contains("Plastic") == true)
        #expect(phrase?.contains("94") == true)
    }

    @Test("The same category is not announced twice")
    func doesNotRepeatTheSameCategory() {
        var announcer = ResultAnnouncer()
        let first = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        // Well past the throttle window - still nothing new to say.
        let second = announcer.announcement(for: .classified(.samplePlastic), at: 60)
        #expect(first != nil)
        #expect(second == nil)
    }

    @Test("A changing confidence on the same category says nothing")
    func ignoresConfidenceDrift() {
        // The percentage moves every frame; the category is the news.
        var announcer = ResultAnnouncer()
        _ = announcer.announcement(for: .classified(.samplePlastic), at: 0)

        let drifted = ClassificationResult(category: .plastic, confidence: 0.71)
        let phrase = announcer.announcement(for: .classified(drifted), at: 60)
        #expect(phrase == nil)
    }

    @Test("A genuinely different category is announced")
    func announcesCategoryChange() {
        var announcer = ResultAnnouncer()
        _ = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        let phrase = announcer.announcement(for: .classified(.sampleGlass), at: 10)
        #expect(phrase?.contains("Glass") == true)
    }

    @Test("Changes arriving faster than the throttle are suppressed")
    func throttlesRapidChanges() {
        var announcer = ResultAnnouncer(minimumInterval: 1.5)
        let first = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        // A different category, but only 0.3s later.
        let tooSoon = announcer.announcement(for: .classified(.sampleGlass), at: 0.3)
        let later = announcer.announcement(for: .classified(.sampleGlass), at: 2.0)

        #expect(first != nil)
        #expect(tooSoon == nil)
        #expect(later != nil)
    }

    @Test("Transient states are never announced")
    func staysSilentForTransientStates() {
        var announcer = ResultAnnouncer()
        #expect(announcer.announcement(for: .scanning, at: 0) == nil)
        #expect(announcer.announcement(for: .modelLoading, at: 1) == nil)
    }

    @Test("A dip through scanning does not unlock a repeat announcement")
    func transientStateDoesNotResetSuppression() {
        // Real sequence: result -> a frame that momentarily reads as scanning
        // -> the same result. That must not speak twice.
        var announcer = ResultAnnouncer()
        _ = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        _ = announcer.announcement(for: .scanning, at: 5)
        let repeated = announcer.announcement(for: .classified(.samplePlastic), at: 10)
        #expect(repeated == nil)
    }

    @Test("Crossing into not-sure is announced once")
    func announcesNotSureOnce() {
        var announcer = ResultAnnouncer()
        _ = announcer.announcement(for: .classified(.samplePlastic), at: 0)

        let first = announcer.announcement(for: .notSure, at: 5)
        let second = announcer.announcement(for: .notSure, at: 10)
        #expect(first == ScannerDisplayState.notSureMessage)
        #expect(second == nil)
    }

    @Test("Each error state is announced with its own wording")
    func announcesErrors() {
        for reason in ScannerUnavailableReason.allCases {
            var announcer = ResultAnnouncer()
            let phrase = announcer.announcement(for: .unavailable(reason), at: 0)
            #expect(phrase == reason.title)
        }
    }

    @Test("Reset allows the same state to be announced again")
    func resetReenablesAnnouncement() {
        var announcer = ResultAnnouncer()
        _ = announcer.announcement(for: .classified(.samplePlastic), at: 0)
        announcer.reset()
        let again = announcer.announcement(for: .classified(.samplePlastic), at: 1)
        #expect(again != nil)
    }
}
