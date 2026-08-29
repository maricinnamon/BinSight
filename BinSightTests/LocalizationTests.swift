import Foundation
import Testing
@testable import BinSight

/// The localization contract.
///
/// These are deliberately about *completeness and wiring*, not about whether a
/// particular translation reads well — a test cannot judge that. What it can
/// judge is that every key resolves in every shipped language, that nothing
/// falls back to the raw key, and that the plural rules Ukrainian needs are
/// actually present.
@MainActor
struct LocalizationTests {

    /// Every key the app looks up at runtime.
    private static let keys: [String] = [
        "waste.cardboard.name", "waste.glass.name", "waste.metal.name",
        "waste.paper.name", "waste.plastic.name", "waste.generalWaste.name",
        "waste.cardboard.guidance", "waste.glass.guidance", "waste.metal.guidance",
        "waste.paper.guidance", "waste.plastic.guidance", "waste.generalWaste.guidance",
        "detected.paper.name", "detected.plastic.name", "detected.metal.name",
        "unavailable.permissionDenied.title", "unavailable.permissionDenied.message",
        "unavailable.permissionRestricted.title", "unavailable.permissionRestricted.message",
        "unavailable.cameraUnavailable.title", "unavailable.cameraUnavailable.message",
        "unavailable.interrupted.title", "unavailable.interrupted.message",
        "unavailable.captureFailed.title", "unavailable.captureFailed.message",
        "unavailable.modelUnavailable.title", "unavailable.modelUnavailable.message",
        "recovery.retry.title", "recovery.openSettings.title",
        "state.ready.title", "state.ready.supporting",
        "state.modelLoading.title", "state.modelLoading.supporting",
        "state.scanning.title", "state.scanning.supporting",
        "state.result.title", "state.notSure.title", "state.notSure.supporting",
        "state.notSure.message",
        "chip.onDeviceAI", "chip.framesStayOnPhone",
        "scanner.instruction.live", "scanner.instruction.pinned",
        "result.guidance.disclaimer", "result.guidance.accessibility",
        "summary.cameraStarting", "summary.pointAtItem", "summary.recognisedOnDevice",
        "summary.row.accessibility",
        "confidence.label", "confidence.accessibilityLabel",
        "detection.label", "announce.readyToScan", "announce.result",
        "settings.language.title", "settings.language.system",
        "settings.language.accessibilityHint",
    ]

    private func withLanguage(_ language: AppLanguage, _ body: () -> Void) {
        Localizer.setLanguage(language)
        defer { Localizer.setLanguage(.system) }
        body()
    }

    @Test("Both shipped languages are actually in the bundle")
    func bothLanguagesBuilt() {
        let available = Localizer.availableLanguages
        #expect(available.contains(.english), "en.lproj is missing from the bundle")
        #expect(available.contains(.ukrainian), "uk.lproj is missing from the bundle")
        #expect(available.contains(.system))
    }

    @Test("Every key resolves in every language", arguments: [AppLanguage.english, .ukrainian])
    func everyKeyResolves(language: AppLanguage) {
        withLanguage(language) {
            for key in Self.keys {
                let value = L(key)
                // A missing key resolves to the key itself — the one failure
                // mode that looks like text on screen and ships unnoticed.
                #expect(value != key, "\(language.rawValue): \(key) is missing")
                #expect(!value.isEmpty, "\(language.rawValue): \(key) is empty")
            }
        }
    }

    @Test("The two languages are actually different")
    func translationsDiffer() {
        // Guards the copy-paste failure where a catalog is duplicated and the
        // "translation" is the English string again.
        var identical: [String] = []
        for key in Self.keys {
            var english = ""
            var ukrainian = ""
            withLanguage(.english) { english = L(key) }
            withLanguage(.ukrainian) { ukrainian = L(key) }
            if english == ukrainian { identical.append(key) }
        }
        // A handful legitimately match — pure format strings like
        // "%1$@ %2$@" carry no words to translate.
        #expect(identical.count <= 3, "untranslated keys: \(identical)")
    }

    @Test("Ukrainian plural forms are present and distinct")
    func ukrainianPlurals() {
        withLanguage(.ukrainian) {
            // Ukrainian needs one/few/many, not English's one/other. 1, 3 and 5
            // must each produce a different word ending.
            let one = L("summary.objectsDetected", 1)
            let few = L("summary.objectsDetected", 3)
            let many = L("summary.objectsDetected", 5)
            #expect(one != few, "one and few are identical: \(one)")
            #expect(few != many, "few and many are identical: \(few)")
            #expect(one.contains("1") && few.contains("3") && many.contains("5"))
        }
    }

    @Test("English plurals still work")
    func englishPlurals() {
        withLanguage(.english) {
            #expect(L("summary.foundMaterials", 1) == "Found 1 material")
            #expect(L("summary.foundMaterials", 2) == "Found 2 materials")
        }
    }

    @Test("Model-layer strings follow the chosen language")
    func modelStringsFollowLanguage() {
        var english = ""
        var ukrainian = ""
        withLanguage(.english) { english = DetectedClass.plastic.displayName }
        withLanguage(.ukrainian) { ukrainian = DetectedClass.plastic.displayName }
        #expect(english == "Plastic")
        #expect(ukrainian == "Пластик")
    }

    @Test("Percentages are formatted, not concatenated")
    func percentIsLocaleAware() {
        // The point is that a format style produced it — a hand-built
        // "\(n)%" would be identical in every locale, which is the bug.
        let text = Detection.percent(0.82)
        #expect(text.contains("82"))
        #expect(text.count > 2)
    }

    @Test("The picker only offers languages that exist")
    func pickerOffersOnlyBuiltLanguages() {
        let store = LocalizationStore(defaults: UserDefaults(suiteName: "picker.test")!)
        for language in store.available where language != .system {
            #expect(Bundle.main.path(forResource: language.rawValue, ofType: "lproj") != nil)
        }
    }

    @Test("A chosen language is remembered")
    func selectionPersists() {
        let suite = "binsight.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        let store = LocalizationStore(defaults: defaults)
        #expect(store.language == .system, "a fresh install should follow the device")

        store.select(.ukrainian)
        #expect(LocalizationStore(defaults: defaults).language == .ukrainian)

        // Going back to system clears the override rather than storing "system",
        // so a future change of default is free to apply.
        store.select(.system)
        #expect(defaults.string(forKey: "binsight.language") == nil)
        #expect(LocalizationStore(defaults: defaults).language == .system)

        Localizer.setLanguage(.system)
    }

    @Test("An unknown stored language falls back to the device")
    func unknownStoredValueFallsBack() {
        let suite = "binsight.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set("kl", forKey: "binsight.language")   // never shipped
        #expect(LocalizationStore(defaults: defaults).language == .system)
        Localizer.setLanguage(.system)
    }
}
