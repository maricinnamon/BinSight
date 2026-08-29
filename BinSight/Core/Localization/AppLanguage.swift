import Foundation

/// The languages BinSight ships, plus following the device.
///
/// `system` is the default and the honest one: a person who set their phone to
/// Ukrainian should not have to find a setting inside every app. The explicit
/// choices exist because BinSight's two audiences do not always match their
/// device language — a Ukrainian speaker with an English phone is common.
enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case system
    case english = "en"
    case ukrainian = "uk"

    var id: String { rawValue }

    /// The `.lproj` code, or `nil` when following the device.
    var localeCode: String? {
        self == .system ? nil : rawValue
    }

    /// Shown in the picker.
    ///
    /// Each language is written **in itself** — "Українська", not "Ukrainian" —
    /// because someone looking for their own language scans for the word they
    /// recognise, not for its English name. `system` is the exception: it
    /// describes a behaviour rather than a language, so it is translated.
    var pickerLabel: String {
        switch self {
        case .system: L("settings.language.system")
        case .english: "English"
        case .ukrainian: "Українська"
        }
    }

    /// Spoken by VoiceOver. The display name alone ("English") is ambiguous
    /// when the control is itself called "Language".
    var accessibilityLabel: String {
        switch self {
        case .system: L("settings.language.system")
        case .english: "English"
        case .ukrainian: "Ukrainian, Українська"
        }
    }
}
