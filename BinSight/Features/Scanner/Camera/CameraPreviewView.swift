import AVFoundation
import SwiftUI
import UIKit

/// The live preview, wrapped as narrowly as possible.
///
/// The representable does one job: put an `AVCaptureVideoPreviewLayer` on
/// screen and keep its session and rotation correct. No gestures, no overlays,
/// no drawing — the reticle and everything else stay in SwiftUI above this.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer.session = session
        // Aspect-fill: fill the region, crop the overhang, never distort.
        // PreviewCropGeometry documents what this means for Phase 5 inference.
        view.previewLayer.videoGravity = .resizeAspectFill
        view.backgroundColor = .black
        view.applyPortraitRotation()
        return view
    }

    func updateUIView(_ view: PreviewContainerView, context: Context) {
        if view.previewLayer.session !== session {
            view.previewLayer.session = session
        }
        view.applyPortraitRotation()
    }
}

/// A view whose backing layer *is* the preview layer.
///
/// Using `layerClass` rather than adding a sublayer means UIKit resizes the
/// preview for us — a manually added sublayer needs its frame updated in
/// `layoutSubviews` and lags by a frame during rotation and sheet transitions.
final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        // Safe: `layerClass` above guarantees the type.
        layer as! AVCaptureVideoPreviewLayer
    }

    /// BinSight is portrait-only, so the preview connection is pinned to
    /// portrait explicitly rather than inheriting whatever the connection
    /// defaults to.
    func applyPortraitRotation() {
        guard let connection = previewLayer.connection else { return }
        let portrait: CGFloat = 90
        if connection.isVideoRotationAngleSupported(portrait) {
            connection.videoRotationAngle = portrait
        }
    }

    /// Maps a rect in this view's coordinate space to a normalised rect in the
    /// capture buffer — the runtime counterpart to `PreviewCropGeometry`.
    ///
    /// Phase 5 should prefer this when a layer is on screen, because it accounts
    /// for the actual device format rather than an assumed buffer size.
    func normalizedBufferRect(for rect: CGRect) -> CGRect {
        previewLayer.metadataOutputRectConverted(fromLayerRect: rect)
    }
}
