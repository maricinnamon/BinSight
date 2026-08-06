# BinSight — LiteRT runtime for on-device waste classification.
#
# CocoaPods, not Swift Package Manager: Google's official iOS quickstart
# (developers.google.com/edge/litert/ios/quickstart) still distributes the
# runtime only as a pod, and there is no `LiteRT` pod on CocoaPods trunk — the
# Swift API remains `TensorFlowLiteSwift`.
#
# Pinned to the newest stable release. Everything after 2.17.0 on trunk is a
# 0.0.1-nightly, which the project brief rules out.
#
# After running `pod install`, build from BinSight.xcworkspace, not the
# .xcodeproj.

platform :ios, '17.0'
use_frameworks!

target 'BinSight' do
  pod 'TensorFlowLiteSwift', '2.17.0'

  target 'BinSightTests' do
    inherit! :search_paths
  end

  target 'BinSightUITests' do
    inherit! :search_paths
  end
end
