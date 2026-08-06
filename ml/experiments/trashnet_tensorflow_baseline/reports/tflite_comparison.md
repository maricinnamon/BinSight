# TFLite candidate comparison

> Latency was measured on the **training Mac**, CPU only. It is a relative
> comparison between candidates and says nothing about iPhone performance.

| model | size | accuracy | macro F1 | top-1 agreement vs Keras | mean ms | p95 ms | input/output |
|---|---:|---:|---:|---:|---:|---:|---|
| float32 | 3.76 MB | 0.8338 | 0.8046 | 1.0000 | 3.56 | 4.20 | float32/float32 |
| float16 **(selected)** | 1.94 MB | 0.8311 | 0.8022 | 0.9974 | 3.46 | 3.72 | float32/float32 |
| int8 | 1.24 MB | 0.5567 | 0.4843 | 0.5963 | 6.04 | 7.52 | float32/float32 |

Keras reference: accuracy 0.8338, macro F1 0.8046.

## Selection rule

A quantised model is rejected if it fails contract verification, or if it
loses more than 2.0 percentage points of
accuracy **or** macro F1 against float32. Macro F1 is checked separately because
a quantised model can hold overall accuracy while quietly dropping the smallest
class. Among survivors, a float input/output contract wins (no dequantisation in
Swift), then the smallest file.
