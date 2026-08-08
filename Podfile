# BinSight — CocoaPods manifest.
#
# The app currently ships WITHOUT an on-device ML runtime and without a model.
# `ScannerView.makeClassifier()` returns nil, so the scanner runs the camera and
# reports `.modelUnavailable` rather than inventing labels.
#
# When a model returns, add its runtime here. For LiteRT/TFLite that was
# `pod 'TensorFlowLiteSwift', '2.17.0'` — CocoaPods rather than SwiftPM, because
# Google's iOS quickstart distributes the runtime only as a pod and there is no
# `LiteRT` pod on trunk; the Swift API is still `TensorFlowLiteSwift`. Anything
# after 2.17.0 on trunk is a 0.0.1-nightly.
#
# Build from BinSight.xcworkspace, not the .xcodeproj.

platform :ios, '17.0'
use_frameworks!

target 'BinSight' do
  # No ML runtime is bundled: the app ships without a classifier and shows its
  # `.modelUnavailable` state. Re-add a runtime pod here when a model returns.

  target 'BinSightTests' do
    inherit! :search_paths
  end

  target 'BinSightUITests' do
    inherit! :search_paths
  end
end
