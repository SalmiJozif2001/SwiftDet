# Live Benchmark — Intel Core i5-3320M

Live-stream benchmark of the SwiftDet deployment model (single stream, CPU-only,
onnxruntime `CPUExecutionProvider`).

- **Host**: Intel Core i5-3320M (supplied host).
- **Device**: CPU-only.
- **Model**: `swiftdet_voc480_prep.onnx` (the preprocessed FP32 deployment model).
- **Input**: 480×480.
- **Threads**: 2.
- **Duration**: 4.5 s.
- **Displayed frames**: 60 (average displayed FPS **13.4**).
- **Dropped camera frames**: 19.
- **Model latency**: P50 **50.8 ms**, P95 **73.5 ms**.
- **Preprocessing**: P50 **5.3 ms**.
- **Full preprocessing + model + NMS**: P50 **58.7 ms**.
- **End-to-end processing**: P95 **80.4 ms**.
- **Capture-to-display latency**: P50 **93 ms**, P95 **122 ms**.
- **Peak reported memory**: **132 MB**.
- **Approximate CPU usage**: ~1.89 CPU cores.

## Interpretation

- This is a **single-stream measurement of the demo loop** — do not treat it as a sustained 30 FPS system claim.
- The demo loop drops camera frames when the model is slower than the camera (~19 frames dropped in the run).
- Separate, unidentified-host benchmarks reported 50.7 FPS (FP32) and 30.5 FPS (INT8) on the Kaggle host; they are **not** attributed to the i5-3320M.
