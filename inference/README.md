# Inference Tools (SwiftDet ONNX)

Command-line tools for running the SwiftDet deployment models in inference mode.
Run from this directory (they use local module imports).

Prerequisites (see [requirements.txt](../requirements.txt)): `onnxruntime`, `numpy`, `psutil`, `opencv-python-headless`.

## `main.py` — ONNX benchmark and detections tool

```powershell
python main.py --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_prep.onnx --threads 2 --iters 100
```

Common options: `--onnx`, `--all` (benchmark every `.onnx` in the model folder), `--image`,
`--img-size 480`, `--threads`, `--warmup 20`, `--iters 100`, `--conf 0.25`, `--out`, `--no-show`.

Reports: model load time, model-only latency (p50/p95), end-to-end (+NMS) latency, CPU cores used,
peak RSS memory, number of detections.

## `live.py` — live camera / stream demo

```powershell
python live.py --camera 0 --conf 0.35 --threads 2
```

Common options: `--camera` (index, video file, or stream URL), `--display` (auto/window/web/none),
`--every`, `--mirror`, `--seconds`, `--loop`, `--verbose`, `--width`, `--height`, `--port`.

Runs on CPU (`CPUExecutionProvider`); needs `opencv-python-headless` for the window mode,
otherwise the stream is served in the browser (`--display web`).

Supplied live-benchmark results on the Intel Core i5-3320M: 13.4 displayed FPS, P50 model latency
50.8 ms — see [benchmarks/I5_3320M_LIVE.md](../benchmarks/I5_3320M_LIVE.md).

## Notes

- The tools are equivalent to the Kaggle session's benchmark and live cells (`swiftdet/benchmark_cpu.py`
  and `inference/live.py`), re-implemented for the packaged tools.
- All three ONNX models expose the same interface: input `images` `(1, 3, 480, 480)` float32 RGB
  (letterboxed, ImageNet-normalized), output `output` `(1, 4725, 52)`.
