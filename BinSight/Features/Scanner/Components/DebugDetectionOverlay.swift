#if DEBUG
import SwiftUI

/// Rolling detector latency and frame accounting, for development only.
///
/// The whole file is inside `#if DEBUG`, so none of it is compiled into a
/// Release build. There is no runtime flag to get wrong.
///
/// `pre` and `infer` are reported separately on purpose: they answer different
/// questions when the app feels slow. `pre` is the CoreImage letterbox, `infer`
/// is CoreML alone, and `e2e` includes the actor hop and decoding. A large gap
/// between `infer` and `e2e` means the bottleneck is not the model.
struct DebugDetectionOverlay: View {
    let metrics: DetectionMetrics

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            row("model", metrics.detectorLoaded ? "loaded" : "NOT LOADED")
            row("boxes", "\(metrics.lastDetectionCount)")
            if let label = metrics.topLabel {
                row("top", label)
            }
            row("frames", "\(metrics.processedFrames) ok · \(metrics.droppedFrames) dropped")
            row("pre", milliseconds(metrics.lastPreprocessing))
            row("infer", milliseconds(metrics.lastInference))
            row("e2e", milliseconds(metrics.lastEndToEnd))
            row("avg/med", "\(milliseconds(metrics.averageEndToEnd)) / \(milliseconds(metrics.medianEndToEnd))")
            row("p95", milliseconds(metrics.p95EndToEnd))
            if metrics.effectiveRate > 0 {
                row("rate", String(format: "%.1f/s", metrics.effectiveRate))
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
        .accessibilityIdentifier("debug.detection")
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(label).opacity(0.6)
            Spacer(minLength: 8)
            Text(value)
        }
    }

    private func milliseconds(_ duration: TimeInterval?) -> String {
        guard let duration, duration > 0 else { return "—" }
        return String(format: "%.1fms", duration * 1000)
    }
}

#Preview {
    DebugDetectionOverlay(
        metrics: DetectionMetrics(
            detectorLoaded: true,
            processedFrames: 128,
            droppedFrames: 7,
            lastDetectionCount: 3,
            lastPreprocessing: 0.004,
            lastInference: 0.021,
            lastEndToEnd: 0.028,
            averageEndToEnd: 0.030,
            medianEndToEnd: 0.028,
            p95EndToEnd: 0.044,
            effectiveRate: 4.0,
            topLabel: "Plastic 91%"
        )
    )
    .padding(40)
    .background(MockCameraBackground())
}
#endif
