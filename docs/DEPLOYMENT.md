# Deployment

This document describes how the SwiftDet models are deployed for inference.

## Files (in the Hugging Face model repository `swiftdet-voc480`)

| File | Type | Size | Purpose |
|---|---|---:|---|
| `onnx/swiftdet_voc480_fp32.onnx` | FP32 ONNX | 6.95 MB | FP32 deployment model — fused graph (BN folded into convs, QARepVGG fused), opset 17 |
| `onnx/swiftdet_voc480_int8_qdq.onnx` | INT8 QDQ ONNX | 2.02 MB | Static INT8 (QDQ format, `reduce_range`) quantization of the FP32 model |
| `onnx/swiftdet_voc480_prep.onnx` | FP32 ONNX | 6.96 MB | FP32 model after shape preprocessing (the file measured in the live benchmark) |
| `pytorch/ema_ep299.pt` | PyTorch EMA checkpoint | 7.28 MB | Final-epoch EMA state (training graph + EMA weights) |

Checksums of the artifacts: [manifest.json](../manifest.json).

## Model interface (all three ONNX files)

- **Input**: `images`, shape `(1, 3, 480, 480)`, float32, RGB.
  - Letterbox resize to 480×480, padding value 114.
  - Normalization: `(x/255 − [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]` (ImageNet mean/std).
- **Output**: `output`, shape `(1, 4725, 52)`:
  - 4725 anchors = 60² (stride 8) + 30² (stride 16) + 15² (stride 32);
  - 52 values = 20 class logits + 4 × 8 DFL regression values (l, t, r, b).
- The models run on **CPU** with `onnxruntime` (`CPUExecutionProvider`); no CUDA is required.

## Preprocessing and decoding

- Preprocessing: `inference/main.py::preprocess` (or equivalent code); note the live demo uses cached buffers.
- Decoding and NMS: `inference/main.py::decode_and_nms` — confidence threshold on the logits (`sigmoid(x) >= t`), softmax over the DFL bins, greedy per-class NMS (IoU 0.5, max 300 per class).

## Running inference

From the SwiftDet repository root. The examples assume that `SwiftDet` and `swiftdet-voc480` are sibling folders:

```powershell
# Download the model artifacts into the sibling folder
python -m pip install -U huggingface_hub
hf download Youssef2Salmi/swiftdet-voc480 --repo-type model --local-dir ..\swiftdet-voc480

# Benchmark + detections (on the prepared FP32 model)
python inference\main.py --onnx ..\swiftdet-voc480\onnx\swiftdet_voc480_prep.onnx --threads 2

# Live camera demo
python inference\live.py --onnx ..\swiftdet-voc480\onnx\swiftdet_voc480_prep.onnx --camera 0 --conf 0.35 --threads 2
```

`inference/main.py` defaults: `--img-size 480`, `--threads` = physical core count, `--conf 0.25`, `--iters 100`, `--warmup 20`.
`inference/live.py` defaults: `--camera 0`, `--display auto`, `--conf 0.35`, `--threads` = physical core count.

Supplied live-benchmark results on the Intel Core i5-3320M: 13.4 displayed FPS, P50 model latency 50.8 ms — see [benchmarks/I5_3320M_LIVE.md](../benchmarks/I5_3320M_LIVE.md).

## Notes

- INT8 was slower than FP32 on the supplied benchmark host (see [benchmarks/INT8_VS_FP32.md](../benchmarks/INT8_VS_FP32.md)); accuracy loss was small.
- Deployment expects static input shapes; there is no dynamic-batch variant in the current export.
