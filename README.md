# SwiftDet — SwiftDet-PConv-CPU (v0.1.0 Research Preview)

SwiftDet is a custom lightweight object-detection architecture (deployment name: **SwiftDet-PConv-CPU**) for efficient CPU inference. It was trained from scratch on Pascal VOC 2007 + 2012, then fused and exported to static ONNX for CPU deployment.

> **Release status: SwiftDet v0.1.0 — Research Preview.** Not production-certified. No license has been specified yet (see [Known limitations](#known-limitations)).

## Architecture

| Property | Value |
|---|---|
| Model name | SwiftDet-PConv-CPU |
| Input size | 480 × 480 |
| Backbone | SwiftDetBackbone — QARepVGG stems + PConv blocks |
| Neck | SlimPANPConv (PANPConv) |
| Detection head | DecoupledGFLHead (cls + reg branches per level) |
| Feature strides | 8, 16, 32 |
| Regression | DFL (`reg_max` = 7) |
| Classes | 20 (Pascal VOC) |
| Assignment | SimOTA |
| Parameters | 1,776,716 (training graph) / 1,725,980 (deploy) |

The exported model outputs `(1, 4725, 52)`: 4725 anchors = 60² + 30² + 15².

## Main features

- Single-stream CPU inference ≈ **13.4 displayed FPS** on an Intel Core i5-3320M at 480×480 (see [benchmarks/I5_3320M_LIVE.md](benchmarks/I5_3320M_LIVE.md)).
- Static ONNX (opset 17), INT8 QDQ deployment, CPU-only via onnxruntime `CPUExecutionProvider` (no CUDA required).
- Full training code: AMP fp16, channels-last, EMA (decay 0.9998), periodic validation, resume, atomic checkpoints.
- CPU benchmark (`swiftdet/benchmark_cpu.py`) and live-camera tools (`inference/main.py`, `inference/live.py`).

## Installation

Windows PowerShell; run from the repository root:

```powershell
python -m pip install -r requirements.txt            # inference (ONNX Runtime, no torch)
python -m pip install -r requirements-training.txt   # training (torch, yaml, opencv-headless)
python -m pip install -r requirements-export.txt    # export & quantization (torch, onnxruntime)
```

## Quick start — inference

Model artifacts are in the Hugging Face model repository (see [Links](#links)):

```powershell
# Benchmark + detections (ONNX Runtime CPU, 480x480, threads=2)
python inference/main.py --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_prep.onnx --threads 2

# Live camera demo
cd inference
python live.py --camera 0 --conf 0.35 --threads 2
```

## Training

```powershell
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root <VOC_ROOT> --epochs 300 --batch-size 128 --workers 4 --val-every 20
```

All three training stages are documented in [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md); exact commands and resume instructions in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Evaluation

```powershell
python -m swiftdet.eval_voc --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_fp32.onnx --voc-root <VOC_ROOT>
```

Results (4,952 images of the Pascal VOC 2007 test set):

| Model | AP50 | mAP50-95 |
|---|---:|---:|
| FP32 ONNX | 64.62% | 38.86% |
| INT8 QDQ ONNX | 64.23% | 38.51% |

See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) for details on AP50 / mAP50-95 and the disclosure about periodic validation using the first 500 VOC2007 test images.

## Links

- Hugging Face model repository: `https://huggingface.co/Youssef2Salmi/swiftdet-voc480` (placeholder — replace `<hf-username>` with your Hugging Face username).
- Documentation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/AUDIT_SUMMARY.md](docs/AUDIT_SUMMARY.md).
- Benchmarks: [benchmarks/README.md](benchmarks/README.md).

## Benchmark summary

| Metric | FP32 ONNX | INT8 QDQ ONNX | Live (i5-3320M) |
|---|---:|---:|---:|
| AP50 | 64.62% | 64.23% | — |
| mAP50-95 | 38.86% | 38.51% | — |
| Model-only p50 (480×480) | 13.92 ms | 26.90 ms | 50.8 ms |
| End-to-end p50 | 19.74 ms | 32.82 ms | 58.7 ms |
| Displayed FPS | — | — | 13.4 (single stream) |

Details: [benchmarks/VOC2007_RESULTS.md](benchmarks/VOC2007_RESULTS.md), [benchmarks/INT8_VS_FP32.md](benchmarks/INT8_VS_FP32.md), [benchmarks/I5_3320M_LIVE.md](benchmarks/I5_3320M_LIVE.md).

## Known limitations

- **Not production-certified** — research preview (v0.1.0).
- **No license specified yet** — decision pending (do not assume any license).
- **VOC-only**: 20 Pascal VOC classes; not a general-purpose detector.
- **CPU deployment only**: static ONNX, `CPUExecutionProvider`; the models require no CUDA, but the code paths for deployment assume CPU.
- **Fixed input size**: 480×480 letterboxed input.
- **INT8 was slower** than FP32 on the supplied benchmark host; small accuracy loss.
- **Not an untouched holdout**: the first 500 images of VOC2007 test were also used for periodic EMA validation during training (`val_images = 500`); final tables are full-set (4,952 images).
- **Kaggle benchmark numbers** (50.7 / 30.5 FPS) are not attributed to the i5-3320M host because their host CPU was not identified in the supplied logs.
- Pascal VOC source images/annotations are owned by the VOC authors; derived training data must follow the official dataset terms — nothing is redistributed here.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — model architecture from the source code.
- [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) — the three training stages.
- [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) — evaluation process, AP50 and mAP50-95.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — deployment guide for the ONNX artifacts.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — exact commands for dataset, training, export, benchmark.
- [docs/AUDIT_SUMMARY.md](docs/AUDIT_SUMMARY.md) — audit findings (verified facts).
- [benchmarks/](benchmarks/) — evaluation and live-benchmark reports.

## Citation

See [CITATION.cff](CITATION.cff) for the machine-readable citation metadata (v0.1.0). Maintainer information is marked as placeholders to be filled in.

## About this release

SwiftDet v0.1.0 was prepared from the source and artifacts already in this workspace: the GitHub repository (`SwiftDet`) contains the code, documentation and benchmark reports; the Hugging Face model repository (`swiftdet-voc480`) contains the three ONNX files, the EMA checkpoint and the evaluation results.
