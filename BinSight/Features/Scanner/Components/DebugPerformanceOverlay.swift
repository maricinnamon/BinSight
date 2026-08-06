#if DEBUG
import SwiftUI

/// Rolling inference latency and frame accounting, for development only.
///
/// The whole file is inside `#if DEBUG`, so none of it — not the view, not the
/// formatting — is compiled into a Release build. There is no runtime flag to
/// get wrong.
struct DebugPerformanceOverlay: View {
    let metrics: ScanMetrics

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            row("model", metrics.classifierLoaded ? "loaded" : "NOT LOADED")
            if let label = metrics.rawTopLabel, let score = metrics.rawTopScore {
                // Pre-smoothing, so it moves every frame. This is the line that
                // shows the model is alive when the card still says "not sure".
                row("raw", String(format: "%@ %.2f", label, score))
            }
            row("frames", "\(metrics.processedFrames) ok · \(metrics.droppedFrames) dropped")
            if metrics.rejectedOutputs > 0 {
                row("rejected", "\(metrics.rejectedOutputs)")
            }
            row("pre", milliseconds(metrics.lastPreprocessing))
            row("invoke", milliseconds(metrics.lastInvocation))
            row("e2e", milliseconds(metrics.lastEndToEnd))
            row("avg/med", "\(milliseconds(metrics.averageEndToEnd)) / \(milliseconds(metrics.medianEndToEnd))")
            row("p95", milliseconds(metrics.p95EndToEnd))
            if let rate = metrics.effectiveRate {
                row("rate", String(format: "%.1f/s", rate))
            }
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
        .foregroundStyle(BinSightTheme.cream)
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(BinSightTheme.ink.opacity(0.72))
        )
        // Noise for VoiceOver, and never present in Release anyway.
        .accessibilityHidden(true)
        .accessibilityIdentifier("debug.performance")
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(label).opacity(0.6)
            Spacer(minLength: 8)
            Text(value)
        }
    }

    private func milliseconds(_ duration: TimeInterval?) -> String {
        guard let duration else { return "—" }
        return String(format: "%.1fms", duration * 1000)
    }
}

#Preview {
    DebugPerformanceOverlay(
        metrics: ScanMetrics(
            processedFrames: 128,
            droppedFrames: 7,
            rejectedOutputs: 0,
            lastPreprocessing: 0.004,
            lastInvocation: 0.021,
            lastEndToEnd: 0.026,
            averageEndToEnd: 0.028,
            medianEndToEnd: 0.026,
            p95EndToEnd: 0.041,
            effectiveRate: 4.0
        )
    )
    .padding(40)
    .background(MockCameraBackground())
}
#endif
