import Foundation

extension Duration {
    /// Seconds as a `Double`, from the component representation.
    ///
    /// `ContinuousClock` is used for every measurement in the app because it is
    /// monotonic — it keeps running across system clock changes and cannot go
    /// backwards, unlike `Date()`. This is the one place its `Duration` is
    /// flattened to a `TimeInterval` for reporting.
    var seconds: TimeInterval {
        let (whole, attoseconds) = components
        return TimeInterval(whole) + TimeInterval(attoseconds) / 1e18
    }
}
