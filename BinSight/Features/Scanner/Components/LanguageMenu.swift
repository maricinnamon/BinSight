import SwiftUI

/// The language picker in the top bar.
///
/// A `Menu` rather than a settings screen: BinSight has exactly one screen and
/// exactly one setting, and a whole navigation layer for a three-item list would
/// be more chrome than the app has content.
///
/// It sits beside the "on-device" chip, styled as another glass control, so the
/// top bar stays one row of related affordances rather than growing a second.
struct LanguageMenu: View {
    let store: LocalizationStore

    /// Grows with Dynamic Type instead of staying a 30pt dot beside text that
    /// has doubled.
    @ScaledMetric(relativeTo: .footnote) private var diameter: CGFloat = 30

    var body: some View {
        Menu {
            Picker(L("settings.language.title"), selection: selection) {
                ForEach(store.available) { language in
                    Text(language.pickerLabel)
                        .accessibilityLabel(language.accessibilityLabel)
                        .tag(language)
                }
            }
            .pickerStyle(.inline)
        } label: {
            Image(systemName: "globe")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(BinSightTheme.onCamera)
                // The tap target is the full 44pt minimum even though the glass
                // circle is smaller — a 30pt control is below the floor.
                .frame(width: max(diameter, BinSightTheme.minimumTapTarget),
                       height: max(diameter, BinSightTheme.minimumTapTarget))
                .background(
                    Circle()
                        .fill(BinSightTheme.ink.opacity(0.45))
                        .frame(width: diameter, height: diameter)
                )
                .contentShape(Circle())
        }
        .accessibilityLabel(L("settings.language.title"))
        .accessibilityHint(L("settings.language.accessibilityHint"))
        .accessibilityValue(store.language.accessibilityLabel)
        .accessibilityIdentifier("settings.language")
    }

    private var selection: Binding<AppLanguage> {
        Binding(get: { store.language }, set: { store.select($0) })
    }
}

#Preview {
    ZStack {
        MockCameraBackground()
        LanguageMenu(store: LocalizationStore())
    }
    .ignoresSafeArea()
}
