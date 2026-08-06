import Testing
@testable import BinSight

/// Frame throttling at the capture boundary.
///
/// `shouldAccept` is `mutating`, and the `#expect` macro cannot call a mutating
/// member on its captured expression — so every call is hoisted into a `let`
/// first. That is a macro constraint, not a style choice.
@MainActor
struct FrameSamplerTests {

    @Test("The target rate becomes the minimum gap between frames")
    func intervalFromTargetRate() {
        #expect(FrameSampler(targetFramesPerSecond: 4).minimumInterval == 0.25)
        #expect(FrameSampler(targetFramesPerSecond: 5).minimumInterval == 0.2)
    }

    @Test("The first frame is always accepted")
    func firstFrameAccepted() {
        var sampler = FrameSampler(targetFramesPerSecond: 4)
        let accepted = sampler.shouldAccept(at: 123.456)
        #expect(accepted)
    }

    @Test("Frames arriving faster than the target rate are dropped")
    func fastFramesDropped() {
        var sampler = FrameSampler(targetFramesPerSecond: 4)   // one per 0.25s
        let decisions = [0.0, 0.10, 0.20, 0.25, 0.40, 0.50].map { sampler.shouldAccept(at: $0) }
        #expect(decisions == [true, false, false, true, false, true])
    }

    @Test("A 30 fps stream is thinned to roughly the target rate")
    func thirtyFpsIsThinnedToTarget() {
        var sampler = FrameSampler(targetFramesPerSecond: 4)
        var accepted = 0
        // One second of 30 fps input.
        for step in 0..<30 {
            if sampler.shouldAccept(at: Double(step) / 30.0) { accepted += 1 }
        }
        // 4 fps requested; frame timing granularity permits one extra.
        #expect((4...5).contains(accepted))
    }

    @Test("The gap is measured from the last accepted frame, not the last seen one")
    func gapMeasuredFromAcceptance() {
        var sampler = FrameSampler(targetFramesPerSecond: 2)   // one per 0.5s
        let first = sampler.shouldAccept(at: 0.0)
        let rejected = sampler.shouldAccept(at: 0.4)   // must not reset the clock
        let second = sampler.shouldAccept(at: 0.5)     // still 0.5s after the accepted frame
        #expect(first)
        #expect(!rejected)
        #expect(second)
    }

    @Test("A backwards timestamp resets rather than stalling the sampler")
    func backwardsTimestampResets() {
        // The presentation clock rebases when the session restarts. Without
        // this, the sampler would reject everything until the new clock passed
        // the old one - potentially a very long time.
        var sampler = FrameSampler(targetFramesPerSecond: 4)
        let beforeRestart = sampler.shouldAccept(at: 1000.0)
        let afterRebase = sampler.shouldAccept(at: 0.0)
        let throttledAgain = sampler.shouldAccept(at: 0.1)
        #expect(beforeRestart)
        #expect(afterRebase)
        #expect(!throttledAgain)
    }

    @Test("Reset makes the next frame accepted again")
    func resetAcceptsNextFrame() {
        var sampler = FrameSampler(targetFramesPerSecond: 4)
        let first = sampler.shouldAccept(at: 0.0)
        let throttled = sampler.shouldAccept(at: 0.1)
        sampler.reset()
        let afterReset = sampler.shouldAccept(at: 0.1)
        #expect(first)
        #expect(!throttled)
        #expect(afterReset)
    }
}
