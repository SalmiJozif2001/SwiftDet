# INT8 vs FP32 — Export, Quantization and Speed

## Export facts (verified)

- Backbone merged from BatchNorm (QARepVGG fusion) before export.
- ONNX opset: **17**.
- Calibration images: **300** (`--calib-num 300`; sampled with seed 0 from `VOC2007/JPEGImages`).
- Quantization: ONNX Runtime static post-training quantization, **QDQ format**, per-channel weights, `reduce_range: true` (the supplied run; the config exposes `reduce_range`).
- Fusion verification: `max_abs_diff = 3.457e-06`.
- Sizes: FP32 ONNX 6.95 MB; INT8 ONNX 2.02 MB (≈3.4× smaller).

## Accuracy on VOC2007 test (verified)

| Metric | FP32 | INT8 | Δ (INT8 − FP32) |
|---|---:|---:|---:|
| AP50 | 64.62% | 64.23% | −0.39 pp |
| mAP50-95 | 38.86% | 38.51% | −0.35 pp |

The accuracy loss is small.

## Speed on the supplied host (Kaggle, identified by logs)

The host CPU of these runs is **not identified** in the supplied logs, so the numbers below are not attributed to the i5-3320M.

| Model | Model-only p50 | End-to-end p50 | FPS (e2e) |
|---|---:|---:|---:|
| FP32 ONNX | 13.92 ms | 19.74 ms | 50.7 |
| INT8 QDQ ONNX | 26.90 ms | 32.82 ms | 30.5 |

Observation: INT8 was slower than FP32 on this host; consistent with the small model size and the slower QDQ kernels. The INT8 artifacts were 2.02 MB vs 6.95 MB.
