import SwiftUI

@main
struct BinSightApp: App {
    /// Owns the language choice for the process.
    @State private var localization = LocalizationStore.shared

    var body: some Scene {
        WindowGroup {
            AppView()
                // Rebuilds the whole tree when the language changes. Every
                // string in the app is resolved eagerly by `L(_:)` at render
                // time, so without a new identity SwiftUI has no reason to
                // re-run the bodies that produced them and the old language
                // stays on screen until something else happens to invalidate.
                .id(localization.language)
                .environment(\.locale, localization.language.localeCode.map(Locale.init(identifier:))
                             ?? Locale.current)
        }
    }
}
