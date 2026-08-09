import SwiftUI

/// Root view of the app.
///
/// BinSight has a single primary screen, so this stays a thin, stable shell
/// around `ScannerView`. No navigation stack, no tabs — there is nowhere else
/// to go.
struct AppView: View {
    var body: some View {
        #if DEBUG
        // A UI test can pin the scanner to a known screen with a launch
        // argument, so assertions run without a camera, a permission prompt or
        // a model. A normal Debug run takes the same path as Release.
        if let scenario = UITestLaunchConfiguration.scenario {
            ScannerView(
                initialState: scenario.scannerState,
                camera: .preview(scenario.cameraState),
                // Pinned: never loads CoreML, so UI tests stay hardware-free.
                engine: .preview()
            )
        } else {
            ScannerView()
        }
        #else
        ScannerView()
        #endif
    }
}

#Preview {
    AppView()
}
