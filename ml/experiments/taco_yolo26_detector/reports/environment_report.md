# YOLO environment report

Environment: **BinSight_YOLO** (separate from the TensorFlow baseline's
`binsight-trashnet-metal`; no TensorFlow is installed here).

| | |
|---|---|
| Python | 3.11.15 |
| Architecture | arm64 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| Ultralytics | 8.4.9 |
| MPS built | True |
| MPS available | True |

## MPS verification

`is_available()` reports capability, not correctness, so both of these run
real computation.

**Matmul** 2048×2048: shape (2048, 2048), mean -0.003121, all finite True, 534.7 ms.

**Conv2d forward/backward**: output (4, 8, 128, 128), loss 0.123001, gradients finite True, grad norm 0.000396.

## YOLO26n

Loaded successfully: task `detect`, **2,572,280 parameters**, 80 pretrained COCO classes.

Weights downloaded for a load smoke test only. No training was run.
