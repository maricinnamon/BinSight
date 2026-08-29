import Testing
@testable import BinSight

@MainActor
struct ScannerDisplayStateTests {

    @Test("A confident prediction is presented as a named result")
    func confidentPredictionBecomesResult() {
        let state = ScannerDisplayState.classified(.samplePlastic)
        #expect(state == .result(.samplePlastic))
        #expect(state.result == .samplePlastic)
    }

    @Test("A weak prediction is presented as 'not sure', and names no class")
    func weakPredictionBecomesNotSure() {
        let state = ScannerDisplayState.classified(.sampleUncertain)
        #expect(state == .notSure)
        #expect(state.result == nil)
    }

    @Test("A percentage is only offered when a result exists")
    func confidenceIsShownOnlyForResults() {
        #expect(ScannerDisplayState.classified(.samplePlastic).showsConfidence)
        #expect(!ScannerDisplayState.ready.showsConfidence)
        #expect(!ScannerDisplayState.scanning.showsConfidence)
        #expect(!ScannerDisplayState.notSure.showsConfidence)
        #expect(!ScannerDisplayState.unavailable(.permissionDenied).showsConfidence)
    }

    @Test("Only the scanning state reports itself as scanning")
    func isScanningIsExclusive() {
        #expect(ScannerDisplayState.scanning.isScanning)
        #expect(!ScannerDisplayState.ready.isScanning)
        #expect(!ScannerDisplayState.classified(.sampleGlass).isScanning)
    }

    @Test("Every state has a title")
    func everyStateHasATitle() {
        let states: [ScannerDisplayState] = [
            .ready,
            .scanning,
            .classified(.samplePlastic),
            .notSure,
            .unavailable(.permissionDenied),
            .unavailable(.cameraUnavailable),
        ]
        for state in states {
            #expect(!state.title.isEmpty)
        }
    }

    @Test("Unavailable states explain themselves")
    func unavailableStatesCarryAMessage() {
        for reason in ScannerUnavailableReason.allCases {
            #expect(!reason.title.isEmpty)
            #expect(!reason.message.isEmpty)
            #expect(ScannerDisplayState.unavailable(reason).supportingText == reason.message)
        }
    }

    @Test("The low-confidence message is the exact copy the design calls for")
    func notSureMessageCopy() {
        // Pins English explicitly. Without this the test asserts whatever
        // language the simulator happens to be set to, which is how it started
        // failing once the app was localized — the copy was fine, the test was
        // reading the Ukrainian translation.
        Localizer.setLanguage(.english)
        defer { Localizer.setLanguage(.system) }
        #expect(ScannerDisplayState.notSureMessage == "Not sure — try a clearer angle")
    }
}
