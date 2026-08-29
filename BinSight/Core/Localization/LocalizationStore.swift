import Foundation
import Observation

/// Holds the chosen language and makes it stick across launches.
///
/// The default is `.system`, so a fresh install simply follows the device and
/// most people never open this setting at all. A manual choice is remembered
/// because the reason someone overrides their device language — a Ukrainian
/// speaker on an English phone — does not go away between launches.
@Observable
@MainActor
final class LocalizationStore {

    /// One store for the app. The chosen language is process-wide state — it
    /// decides what `L(_:)` returns everywhere, including inside value types
    /// that no view owns — so a second instance could only disagree with the
    /// first.
    static let shared = LocalizationStore()

    private static let defaultsKey = "binsight.language"

    private let defaults: UserDefaults

    private(set) var language: AppLanguage {
        didSet {
            guard language != oldValue else { return }
            Localizer.setLanguage(language)
            persist()
        }
    }

    /// Only languages actually built into the bundle, so the picker cannot
    /// offer something that would silently fall back.
    var available: [AppLanguage] { Localizer.availableLanguages }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let stored = defaults.string(forKey: Self.defaultsKey)
        // An unknown stored value means the app once shipped a language it no
        // longer does. Following the device is the safe reading.
        self.language = stored.flatMap(AppLanguage.init(rawValue:)) ?? .system
        Localizer.setLanguage(self.language)
    }

    func select(_ language: AppLanguage) {
        self.language = language
    }

    private func persist() {
        if language == .system {
            // Remove rather than store "system": a later version that changes
            // the default should be free to apply it.
            defaults.removeObject(forKey: Self.defaultsKey)
        } else {
            defaults.set(language.rawValue, forKey: Self.defaultsKey)
        }
    }
}
