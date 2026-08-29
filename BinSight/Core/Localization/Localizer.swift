import Foundation
import os

/// Resolves localized strings against the language the person chose.
///
/// ## Why not just `NSLocalizedString`
///
/// The plain call resolves against `Bundle.main`, which is fixed to the device
/// language for the lifetime of the process. That is correct for an app with no
/// in-app language setting; it is wrong for BinSight, which has one. Overriding
/// `AppleLanguages` in `UserDefaults` is the other common trick, but it only
/// takes effect on the *next* launch — the person taps "Українська" and nothing
/// happens until they force-quit, which reads as a bug.
///
/// So the language-specific `.lproj` bundle is resolved once and every lookup
/// goes through it. Switching swaps the bundle and the UI re-renders.
///
/// ## Thread safety
///
/// The bundle is read from wherever a string is needed — including value types
/// like `ScannerDisplayState` that are not actor-isolated — and written only
/// from the main actor when the setting changes. An `OSAllocatedUnfairLock`
/// keeps that safe without making every caller async.
enum Localizer {

    private static let storage = OSAllocatedUnfairLock<Bundle>(initialState: .main)
    private static let logger = Logger(
        subsystem: "com.marynaantonevych.BinSight",
        category: "localization"
    )

    /// The bundle every lookup goes through. `Bundle.main` while following the
    /// device language.
    static var bundle: Bundle {
        storage.withLock { $0 }
    }

    /// Points lookups at `language`.
    ///
    /// Falls back to `Bundle.main` — i.e. the device language — if the requested
    /// `.lproj` is missing from the app bundle. That can only happen if a
    /// language is offered in the picker but was never actually built, which is
    /// a packaging mistake worth logging rather than crashing over.
    static func setLanguage(_ language: AppLanguage) {
        guard let code = language.localeCode else {
            storage.withLock { $0 = .main }
            return
        }
        guard let path = Bundle.main.path(forResource: code, ofType: "lproj"),
              let localized = Bundle(path: path)
        else {
            logger.error("No \(code, privacy: .public).lproj in the bundle; using the device language")
            storage.withLock { $0 = .main }
            return
        }
        storage.withLock { $0 = localized }
    }

    /// Which languages were actually built into this bundle.
    ///
    /// Used by the picker so it can only ever offer something that resolves.
    static var availableLanguages: [AppLanguage] {
        AppLanguage.allCases.filter { language in
            guard let code = language.localeCode else { return true }
            return Bundle.main.path(forResource: code, ofType: "lproj") != nil
        }
    }
}

/// Looks up `key` in the chosen language.
///
/// Deliberately terse: it appears at every user-facing string in the app, and a
/// longer name would bury the string itself. The key is the argument label-free
/// first parameter so call sites read as `L("scanner.instruction")`.
func L(_ key: String) -> String {
    NSLocalizedString(key, bundle: Localizer.bundle, comment: "")
}

/// Looks up `key` and substitutes `arguments`.
///
/// Positional (`%1$@`, `%2$lld`) rather than concatenated, because word order
/// differs between English and Ukrainian and a translator must be able to move
/// the pieces.
func L(_ key: String, _ arguments: CVarArg...) -> String {
    String(format: NSLocalizedString(key, bundle: Localizer.bundle, comment: ""),
           locale: Locale.current,
           arguments: arguments)
}
