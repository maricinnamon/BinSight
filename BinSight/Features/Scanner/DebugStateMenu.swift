#if DEBUG
import SwiftUI

/// Development-only switch for driving the scanner through its states.
///
/// The whole file is inside `#if DEBUG`, so the menu is not compiled into a
/// Release build at all. It exists so the mock states can be demoed on a device
/// before there is a camera to produce them.
struct DebugStateMenu: View {
    /// `nil` means "show the live classifier". Optional on purpose: without a
    /// way back to nil, picking any mock once would hide the real model for the
    /// rest of the session.
    @Binding var state: ScannerDisplayState?

    private struct Option: Identifiable {
        let id: String
        let state: ScannerDisplayState?
    }

    private let options: [Option] = [
        Option(id: "▶︎ Live camera + model", state: nil),
        Option(id: "Ready", state: .ready),
        Option(id: "Scanning", state: .scanning),
        Option(id: "Plastic · 94%", state: .classified(.samplePlastic)),
        Option(id: "Glass · 87%", state: .classified(.sampleGlass)),
        Option(id: "Cardboard · 71%", state: .classified(.sampleCardboard)),
        Option(id: "Not sure · 41%", state: .classified(.sampleUncertain)),
        Option(id: "Permission denied", state: .unavailable(.permissionDenied)),
        Option(id: "Camera unavailable", state: .unavailable(.cameraUnavailable)),
    ]

    var body: some View {
        Menu {
            ForEach(options) { option in
                Button(option.id) { state = option.state }
            }
        } label: {
            Image(systemName: state == nil ? "ladybug" : "ladybug.fill")
                .font(.system(size: 12, weight: .semibold))
                // Filled and tinted while a mock is overriding the live model,
                // so it is obvious the screen is not showing real predictions.
                .foregroundStyle(state == nil ? .primary : Color(BinSightTheme.coral))
                // 44pt hit target even though the glyph is small.
                .frame(width: BinSightTheme.minimumTapTarget, height: BinSightTheme.minimumTapTarget)
                .contentShape(Circle())
                .glassSurface(in: Circle(), interactive: true)
        }
        .accessibilityLabel("Debug: choose scanner state")
        .accessibilityIdentifier("debug.stateMenu")
    }
}

#Preview {
    @Previewable @State var state: ScannerDisplayState? = nil
    DebugStateMenu(state: $state)
        .padding(40)
        .background(MockCameraBackground())
}
#endif
