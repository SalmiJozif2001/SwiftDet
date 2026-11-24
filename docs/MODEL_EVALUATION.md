# Model Evaluation

This document describes the evaluation process of the SwiftDet models and explains **AP50** and **mAP50-95**.

## Evaluation pipeline

The evaluator is `swiftdet/eval_voc.py` (`evaluate_onnx` for the ONNX files):

1. **Load the Pascal VOC 2007 test split** (`2007/test`) with `VOCDataset` — 4,952 images.
2. **Per image**:
   - letterbox-resize the image to 480×480 (ImageNet normalization, padding value 114);
   - run the model (`images` → `output` (1, 4725, 52));
   - decode predictions with `decode_and_nms`: sigmoid on the class logits, softmax over the DFL bins for the regression values, per-class greedy NMS with confidence threshold 0.05, IoU threshold 0.5, max 300 per class.
3. **Matching**: per class, detections sorted by descending score are matched greedily against GT boxes (same image) at each IoU threshold; a GT can only be matched once.

## Metrics

- **AP50** — Average Precision of each class computed at the **IoU threshold 0.50**, averaged over the 20 classes that have at least one ground-truth box (classes without GT are skipped). Reported as a percentage.
- **mAP50-95** — the standard VOC2007+ metric: the **mean of AP over the 10 IoU thresholds 0.50, 0.55, …, 0.95** (`np.linspace(0.5, 0.95, 10)`). Also reported as a percentage.
- Both are returned as fractions (0–1) by `evaluate_fn` and converted to percentages in the tables below.

## Evaluation results (verified from the run logs)

| Model | AP50 | mAP50-95 | Images evaluated |
|---|---:|---:|---:|
| FP32 ONNX (`swiftdet_voc480_fp32.onnx`) | **64.62%** | **38.86%** | 4,952 |
| INT8 QDQ ONNX (`swiftdet_voc480_int8_qdq.onnx`) | **64.23%** | **38.51%** | 4,952 |

(The same numbers are stored as fractions in `swiftdet-voc480/results/metrics.json`.)

## Disclosure — the evaluation set was not a completely untouched holdout

Per the notebook and the evaluation tool: during training, periodic EMA validation was run on the **first 500 images of the Pascal VOC 2007 test set** (`VAL_EVERY = 20`, `val_images = 500`; the notebook explicitly notes this for the epochs-279 and epoch-299 validations). Periodic validation examples from `results/train_log.csv`: epoch 19: AP50 0.2204; epoch 279: AP50 0.6350; epoch 299: AP50 0.6353.

Therefore the final numbers above are **full-set evaluations on all 4,952 images of VOC2007 test**, but this set was not untouched: the first 500 images were used during training for periodic validation. Do not treat the table as a completely independent holdout evaluation.

## Notes

- The final periodic validation (epoch 299, first 500 images) reports AP50 63.53% — slightly below the full-set FP32 evaluation (64.62%), as expected for a different set.
- INT8 vs FP32: AP50 −0.39 pp, mAP50-95 −0.35 pp; see [benchmarks/INT8_VS_FP32.md](../benchmarks/INT8_VS_FP32.md).
- The evaluation is CPU-only (`onnxruntime` `CPUExecutionProvider`).
