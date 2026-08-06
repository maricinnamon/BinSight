import XCTest

/// Screen-level assertions driven entirely by launch arguments.
///
/// Every scenario pins the scanner to a known state via
/// `UITestLaunchConfiguration`, so none of these open the camera, trigger a
/// permission prompt, or need a model. That makes them deterministic on a
/// Simulator with no camera hardware.
final class ScannerScreenUITests: XCTestCase {

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
    }

    private func launch(_ scenario: String) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments += ["-BinSightUITestScenario", scenario]
        app.launch()
        return app
    }

    /// Identifiers are attached to containers and to `.accessibilityElement`
    /// wrappers, which surface as different element types depending on the
    /// wrapping — so match on any descendant.
    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any)[identifier]
    }

    // MARK: - Always present

    func testChromeIsPresentOnEveryScreen() {
        let app = launch("ready")
        XCTAssertTrue(app.staticTexts["BinSight"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["ON-DEVICE AI"].exists)
        // The privacy claim must never quietly disappear.
        XCTAssertTrue(app.staticTexts["Frames stay on this iPhone."].exists)
        XCTAssertTrue(app.staticTexts["Point at one item"].exists)
    }

    // MARK: - Scenarios

    func testConfidentResultShowsCategoryAndConfidence() {
        let app = launch("confidentResult")

        let card = element("scanner.state", in: app)
        XCTAssertTrue(card.waitForExistence(timeout: 5))

        XCTAssertTrue(element("result.category", in: app).exists, "category row missing")
        XCTAssertTrue(element("result.confidence", in: app).exists, "confidence readout missing")

        // Guidance must be framed as generic and must flag that local rules
        // vary. Matched on substrings so a copy tweak does not fail the test,
        // but dropping either caveat does.
        XCTAssertTrue(
            app.staticTexts
                .matching(NSPredicate(format: "label CONTAINS[c] %@", "generic guidance"))
                .firstMatch.exists,
            "disposal guidance is not labelled as generic"
        )
        XCTAssertTrue(
            app.staticTexts
                .matching(NSPredicate(format: "label CONTAINS[c] %@", "local rules"))
                .firstMatch.exists,
            "disposal guidance does not carry a local-rules caveat"
        )
    }

    func testNotSureShowsGuidanceAndNoCategory() {
        let app = launch("notSure")

        XCTAssertTrue(element("result.notSure", in: app).waitForExistence(timeout: 5))
        // A low-confidence reading must never name a class or show a percentage.
        XCTAssertFalse(element("result.category", in: app).exists, "named a class while unsure")
        XCTAssertFalse(element("result.confidence", in: app).exists, "showed a percentage while unsure")
    }

    func testCameraDeniedOffersSettingsNotRetry() {
        let app = launch("cameraDenied")

        XCTAssertTrue(element("camera.denied", in: app).waitForExistence(timeout: 5))
        // iOS will not prompt twice, so Settings is the only useful route.
        XCTAssertTrue(element("camera.openSettings", in: app).exists)
        XCTAssertFalse(element("camera.retry", in: app).exists, "offered a retry that cannot work")
    }

    func testCameraUnavailableOffersNoRecovery() {
        let app = launch("cameraUnavailable")

        XCTAssertTrue(element("camera.unavailable", in: app).waitForExistence(timeout: 5))
        // Retrying will not conjure hardware.
        XCTAssertFalse(element("camera.retry", in: app).exists)
        XCTAssertFalse(element("camera.openSettings", in: app).exists)
    }

    func testModelErrorIsReportedDistinctlyFromCameraErrors() {
        let app = launch("modelError")

        XCTAssertTrue(element("model.unavailable", in: app).waitForExistence(timeout: 5))
        XCTAssertFalse(element("camera.denied", in: app).exists)
        XCTAssertFalse(element("result.category", in: app).exists, "showed a result with no model")
    }

    func testReadyStateExposesItsHook() {
        let app = launch("ready")
        // "scanner.ready" is only attached when the scanner is genuinely ready;
        // with a pinned idle camera the instruction hook is used instead.
        XCTAssertTrue(element("scanner.instruction", in: app).waitForExistence(timeout: 5))
    }
}
