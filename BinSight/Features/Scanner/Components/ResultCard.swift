import SwiftUI

/// The bottom surface that answers the question.
///
/// One container for every state. Only its inner rows come and go, which keeps
/// the transition to a small animatable change rather than a rebuild of the
/// screen — and stops the layout jumping as the classifier changes its mind.
///
/// VoiceOver order is the source order: status → category → confidence →
/// guidance → action.
struct ResultCard: View {
    let state: ScannerDisplayState
    /// Invoked when the person taps the recovery button on a camera or model
    /// problem. No-op by default so previews need not supply one.
    var onRecover: (ScannerUnavailableReason.Recovery) -> Void = { _ in }

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    private var isAccessibilitySize: Bool { dynamicTypeSize.isAccessibilitySize }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            if let result = state.result {
                categoryRow(result)
                guidance(for: result.category)
            }

            if state == .notSure {
                notSureRow
            }

            if case .unavailable(let reason) = state, let recovery = reason.recovery {
                recoveryButton(recovery)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .panelSurface(in: RoundedRectangle(cornerRadius: BinSightTheme.cardCornerRadius, style: .continuous))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(cardIdentifier)
    }

    /// Camera and model problems get their own identifier so a UI test can
    /// target a specific failure, not just "the card".
    private var cardIdentifier: String {
        if case .unavailable(let reason) = state {
            return reason.accessibilityIdentifier
        }
        return "scanner.state"
    }

    // MARK: - Rows

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(state.title)
                // Secondary when a category is about to shout underneath it.
                .font(BinSightTheme.rounded(state.result == nil ? .headline : .subheadline, weight: .semibold))
                .foregroundStyle(BinSightTheme.onSurface.opacity(state.result == nil ? 1 : 0.55))

            if let supporting = state.supportingText {
                Text(supporting)
                    .font(BinSightTheme.rounded(.footnote))
                    .foregroundStyle(BinSightTheme.onSurface.opacity(0.65))
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private func categoryRow(_ result: ClassificationResult) -> some View {
        // At accessibility text sizes the icon and a large title cannot share a
        // row without one of them truncating, so they stack instead.
        let layout = isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 12))
            : AnyLayout(HStackLayout(alignment: .center, spacing: 14))

        return VStack(alignment: .leading, spacing: 10) {
            layout {
                Image(systemName: result.category.symbolName)
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(result.category.accent)
                    .frame(width: 54, height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 17, style: .continuous)
                            .fill(result.category.accent.opacity(0.16))
                    )
                    .accessibilityHidden(true)

                Text(result.category.displayName)
                    .font(BinSightTheme.rounded(.largeTitle, weight: .bold))
                    .foregroundStyle(BinSightTheme.onSurface)
                    // Essential text: wrap and shrink rather than truncate.
                    .lineLimit(2)
                    .minimumScaleFactor(0.6)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(result.category.displayName)
            .accessibilityIdentifier("result.category")

            ConfidenceView(result: result)
        }
    }

    private func guidance(for category: WasteCategory) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
                .overlay(BinSightTheme.onSurface.opacity(0.12))
                .padding(.bottom, 2)

            Label {
                Text("Generic guidance · local rules vary")
                    .font(BinSightTheme.rounded(.caption2, weight: .semibold))
            } icon: {
                Image(systemName: "info.circle")
                    .font(.system(size: 10, weight: .semibold))
            }
            .foregroundStyle(BinSightTheme.onSurface.opacity(0.5))

            Text(category.disposalGuidance)
                .font(BinSightTheme.rounded(.footnote))
                .foregroundStyle(BinSightTheme.onSurface.opacity(0.85))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Generic guidance, local rules vary. \(category.disposalGuidance)")
    }

    private var notSureRow: some View {
        // An icon as well as the wording, so the state is not carried by
        // colour or tone alone.
        Label {
            Text(ScannerDisplayState.notSureMessage)
                .font(BinSightTheme.rounded(.subheadline, weight: .medium))
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "questionmark.circle.fill")
                .foregroundStyle(BinSightTheme.lime)
        }
        .foregroundStyle(BinSightTheme.onSurface.opacity(0.9))
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(ScannerDisplayState.notSureMessage)
        .accessibilityIdentifier("result.notSure")
    }

    private func recoveryButton(_ recovery: ScannerUnavailableReason.Recovery) -> some View {
        Button {
            onRecover(recovery)
        } label: {
            Text(recovery.title)
                .font(BinSightTheme.rounded(.subheadline, weight: .semibold))
                .frame(maxWidth: .infinity)
                // Guarantees the 44pt minimum target at every text size.
                .frame(minHeight: BinSightTheme.minimumTapTarget)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(BinSightTheme.ink)
        .background(Capsule().fill(BinSightTheme.lime))
        .accessibilityIdentifier(recovery == .retry ? "camera.retry" : "camera.openSettings")
    }
}

#Preview("States") {
    ScrollView {
        VStack(spacing: 16) {
            ResultCard(state: .ready)
            ResultCard(state: .scanning)
            ResultCard(state: .classified(.samplePlastic))
            ResultCard(state: .notSure)
            ResultCard(state: .unavailable(.permissionDenied))
            ResultCard(state: .unavailable(.modelUnavailable))
        }
        .padding(BinSightTheme.screenPadding)
    }
    .background(MockCameraBackground().ignoresSafeArea())
}

#Preview("Accessibility text") {
    ResultCard(state: .classified(.sampleGlass))
        .padding(BinSightTheme.screenPadding)
        .environment(\.dynamicTypeSize, .accessibility3)
        .background(MockCameraBackground().ignoresSafeArea())
}
