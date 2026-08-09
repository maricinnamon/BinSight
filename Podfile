# BinSight — CocoaPods manifest.
#
# The app has **no third-party runtime dependencies**. Detection runs on Apple's
# own CoreML: `BinSight/Resources/Models/BinSightYOLO26n.mlpackage` is compiled
# by Xcode into the bundle and loaded by `YOLODetectionService`.
#
# This file is kept, rather than deleted, only because the workspace references
# it. Earlier phases used `pod 'TensorFlowLiteSwift'` for a LiteRT classifier;
# that path is retired and no TFLite runtime is linked.
#
# Build from BinSight.xcworkspace, not the .xcodeproj.

platform :ios, '17.0'
use_frameworks!

target 'BinSight' do
  # Intentionally empty — CoreML is an Apple framework and needs no pod.

  target 'BinSightTests' do
    inherit! :search_paths
  end

  target 'BinSightUITests' do
    inherit! :search_paths
  end
end
