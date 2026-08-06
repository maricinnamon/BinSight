import AVFoundation
import Testing
@testable import BinSight

/// Permission-to-display-state mapping. All pure logic — no camera, no session,
/// no permission prompt.
@MainActor
struct CameraStateTests {

    @Test("Authorized means the camera can run")
    func authorizedMapsToRunning() {
        #expect(CameraAuthorization.state(for: .authorized) == .running)
    }

    @Test("Not-determined shows as preparing, not as an error")
    func notDeterminedMapsToPreparing() {
        // The system prompt is about to appear; showing "camera access needed"
        // before the person has been asked would be wrong.
        #expect(CameraAuthorization.state(for: .notDetermined) == .preparing)
    }

    @Test("Denied and restricted map to their own distinct states")
    func refusalsMapDistinctly() {
        #expect(CameraAuthorization.state(for: .denied) == .denied)
        #expect(CameraAuthorization.state(for: .restricted) == .restricted)
    }

    @Test("Permission is requested only when the status is not determined")
    func onlyPromptsWhenUndetermined() {
        #expect(CameraAuthorization.shouldRequestPermission(for: .notDetermined))
        // iOS will not prompt a second time, so asking again is pure waste.
        #expect(!CameraAuthorization.shouldRequestPermission(for: .denied))
        #expect(!CameraAuthorization.shouldRequestPermission(for: .restricted))
        #expect(!CameraAuthorization.shouldRequestPermission(for: .authorized))
    }

    @Test("The prompt result maps to running or denied")
    func promptResultMapping() {
        #expect(CameraAuthorization.state(afterPromptGranted: true) == .running)
        #expect(CameraAuthorization.state(afterPromptGranted: false) == .denied)
    }

    // MARK: - CameraState -> screen state

    @Test("Healthy states show no error card")
    func healthyStatesHaveNoReason() {
        #expect(CameraState.idle.unavailableReason == nil)
        #expect(CameraState.preparing.unavailableReason == nil)
        #expect(CameraState.running.unavailableReason == nil)
    }

    @Test("Every failure state maps to a distinct scanner reason")
    func failureStatesMapDistinctly() {
        let mapping: [(CameraState, ScannerUnavailableReason)] = [
            (.denied, .permissionDenied),
            (.restricted, .permissionRestricted),
            (.unavailable, .cameraUnavailable),
            (.interrupted, .interrupted),
            (.failed, .captureFailed),
        ]
        for (state, expected) in mapping {
            #expect(state.unavailableReason == expected)
        }
        // No two failure states collapse onto the same message.
        let reasons = mapping.map(\.1)
        #expect(Set(reasons.map(\.rawValue)).count == reasons.count)
    }

    @Test("Only the running state reports itself as running")
    func isRunningIsExclusive() {
        #expect(CameraState.running.isRunning)
        for state in [CameraState.idle, .preparing, .denied, .restricted, .unavailable, .interrupted, .failed] {
            #expect(!state.isRunning)
        }
    }
}
