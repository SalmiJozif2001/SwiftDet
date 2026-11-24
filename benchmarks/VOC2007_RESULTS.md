# VOC2007 Results — Evaluation on Pascal VOC 2007 test

- Evaluation implementation: `swiftdet/eval_voc.py` (`evaluate_onnx`, CPU, `onnxruntime` `CPUExecutionProvider`).
- Model: SwiftDet (deployment name: SwiftDet-PConv-CPU), input 480×480.
- Images evaluated: **4,952** (the full Pascal VOC 2007 test set).

| Model | AP50 | mAP50-95 | Size (MB) |
|---|---:|---:|---:|
| FP32 ONNX (`swiftdet_voc480_fp32.onnx`) | **64.62%** | **38.86%** | 6.95 |
| INT8 QDQ ONNX (`swiftdet_voc480_int8_qdq.onnx`) | **64.23%** | **38.51%** | 2.02 |

(The same numbers are stored as fractions in `swiftdet-voc480/results/metrics.json`.)

## Disclosure — not a completely untouched holdout

During training, periodic EMA validation was run on the **first 500 images of the Pascal VOC 2007 test set** (`VAL_EVERY = 20`, `val_images = 500`; the notebook explicitly notes this for the epochs-279 and epoch-299 validations). The values above are therefore full-set evaluations, but the first 500 images were used during training. Examples from `results/train_log.csv`: epoch 19: AP50 0.2204; epoch 279: AP50 0.6350; epoch 299: AP50 0.6353.

## Fusion verification

`swiftdet/export_quantize.py` verifies the fused (deploy) graph against the training graph: `max_abs_diff = 3.457e-06` (assert limit: < 1e-3).

## See also

- [docs/MODEL_EVALUATION.md](../docs/MODEL_EVALUATION.md) — AP50 and mAP50-95 definitions.
- [benchmarks/INT8_VS_FP32.md](INT8_VS_FP32.md) — INT8 vs FP32 speed/accuracy.
