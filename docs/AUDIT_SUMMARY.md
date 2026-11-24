# SwiftDet v0.1.0 — Audit Summary

This document records the results of the pre-release audit performed for SwiftDet v0.1.0 (Research Preview). Everything below was obtained by reading the actual source files, configs, run logs and notebook cells. Nothing was inferred without evidence, and no file was modified.

## 1. Repositories audited

| Repository | Path | Purpose |
|---|---|---|
| Source repository | `SwiftDet` | GitHub project — code, configs, docs, benchmark docs |
| Model repository | `swiftdet-voc480` | Hugging Face model card, ONNX/PyTorch artifacts, evaluation results |

## 2. Source audit — `SwiftDet/`

### 2.1 Python package (`SwiftDet/swiftdet/`)

| File | Contents |
|---|---|
| `models/swiftdet.py` | `SwiftDetBackbone`, `SlimPANPConv`, `HeadBranch`, `DecoupledGFLHead`, `SwiftDet` (with `deploy()` for fused ONNX export) |
| `models/blocks.py` | `ConvBNReLU`, `QARepConv2d` (Quantization-Aware RepVGG, Chu et al. AAAI 2024), `PConvBlock` (FasterNet PConv), `SPPFLight` |
| `losses/gfl_simota.py` | `SwiftDetLoss` (Varifocal + GIoU + DFL), SimOTA (`simota_assign`), ATSS (`atss_assign`) |
| `data/voc_dataset.py` | VOC 2007/2012 loader, mosaic/mixup augmentation, letterbox resize, `collate_fn` |
| `train.py` | Training loop: AMP fp16 autocast (loss in fp32), channels-last, EMA, resume, periodic validation, checkpoints |
| `eval_voc.py` | VOC2007-test evaluation (`evaluate_fn`, `evaluate_onnx`) |
| `export_quantize.py` | QARepVGG deploy fusion, ONNX export, PTQ INT8 QDQ quantization |
| `benchmark_cpu.py` | ONNX CPU latency/memory benchmark |
| `configs/voc_swiftdet.yaml` | Model/train/deploy config |

### 2.2 Inference (`SwiftDet/inference/`)

- `main.py` — ONNX speed/memory benchmark + detections visualization (Arabic interface).
- `live.py` — live camera/video detection with FPS/latency HUD, snapshot, browser stream mode (MJPEG).

### 2.3 Notebook (`SwiftDet/training/fork-of-fork-of-swiftdet-claude.ipynb`)

32 cells. Structure:
1. Cell 1: `!kaggle kernels output youssefsalmi/fork-of-swiftdet-claude` — downloads session outputs.
2. Cell 2: settings (`RUN_SANITY_CHECK=True`, `RUN_MINI_ABLATIONS=False`, `RUN_FULL_TRAINING=True`, `RUN_EXPORT_QUANTIZE=True`, `RUN_EVAL_ONNX=True`, `RUN_CPU_BENCHMARK=True`, `EPOCHS=300`, `BATCH_SIZE=128`, `NUM_WORKERS=4`, `TIME_LIMIT_HOURS=10`, `VAL_EVERY=20`, `MANUAL_RESUME_PATH=""`).
3. Cells 3–14: install deps, package skeleton, write `swiftdet/*` files, write `voc_download` for VOC.
4. Cell 16: VOC download (pjreddie + robots.ox.ac.uk mirrors).
5. Cell 18 (Stage 1): overfit sanity check.
6. Cell 20 (Stage 2): mini ablations (SimOTA vs ATSS, 80 epochs each on VOC2007 trainval).
7. Cell 22 (Stage 3): full training from scratch (resumed).
8. Cell 25 (after Stage 3): export + quantize (opset 17, `--reduce-range`).
9. Cell 27 (after Stage 3): ONNX evaluation on 4,952 images.
10. Cell 29 (after Stage 3): Kaggle CPU benchmark on FP32 and INT8 models.
11. Cell 31: archive outputs, list artifacts.

Notebook outputs were captured from the Kaggle session; the notebook itself was not executed at creation (stated in notebook markdown).

### 2.4 Checkpoints / logs in `SwiftDet/runs/` (referenced by the notebook)

- `runs/swiftdet_voc480/last.pt` — latest full checkpoint (model + EMA + optimizer + scaler + epoch).
- `runs/swiftdet_voc480/ema_epNNN.pt` — EMA-only checkpoints, saved every 25 epochs and at the final epoch (`ema_ep299.pt`).
- `runs/swiftdet_voc480/train_log.csv` — epoch-by-epoch log (epochs 0–299, columns: epoch, lr, loss, vfl, giou, dfl, epoch_sec, val_ap50, val_map).
- Model repository `swiftdet-voc480/pytorch/ema_ep299.pt` — the final EMA checkpoint (SHA256 `D893E9FCFEEFCB75C46C72E8188AA92A6B0897132A018144E27C906F4E43AC9E`, 7,276,487 bytes).

## 3. Architecture (verified from source)

`SwiftDet` (deployment name **SwiftDet-PConv-CPU**), input 480×480 → `SwiftDetBackbone` → `SlimPANPConv` → `DecoupledGFLHead` → output `(1, 4725, 52)` (verified by loading the ONNX artifacts with onnxruntime: input `images[1,3,480,480]`, output `output[1,4725,52]`).

- Backbone `SwiftDetBackbone`: `stem_1` ConvBNReLU(3→16, 3×3 s2) → `stem_2` QARepConv2d(16→32, 3×3 s2) → `stage1` PConvBlock(32) → `down2` QARepConv2d(32→64 s2) → `stage2` 2×PConvBlock(64) [c3, stride 8, 64 ch] → `down3` QARepConv2d(64→128 s2) → `stage3` 3×PConvBlock(128) [c4, stride 16, 128 ch] → `down4` QARepConv2d(128→256 s2) → `stage4` 2×PConvBlock(256) → `sppf` SPPFLight(256, k5) [c5, stride 32, 256 ch].
- Neck `SlimPANPConv(in=(64,128,256), cn=64)`: 1×1 lat layers to 64 ch; top-down P3/P4 (`td_p4_red`+`td_p4`, `td_p3_red`+`td_p3` PConvBlocks); bottom-up P4/P5 (`bu_p3_down`, `bu_p4_red`+`bu_p4`, `bu_p4_down`, `bu_p5_red`+`bu_p5` PConvBlocks); outputs `p3_out`/`p4_out`/`p5_out` (64 ch, strides 8/16/32).
- Head `DecoupledGFLHead(cn=64, num_classes=20, reg_max=7)`: 3 levels × (`HeadBranch` cls → 20 logits; `HeadBranch` reg → 32 values).
- `HeadBranch`: split 64 → 32 (PConv 3×3) + 32 (passthrough); `pw` ConvBNReLU 64→64; `out_conv` 1×1 to out channels.
- `PConvBlock` (FasterNet): PConv(c/4) on the dynamic branch; static PW1 c→2c + PW2 2c→c + residual.
- `QARepConv2d`: 3×3 conv + BN + 1×1 conv (+identity when cin==cout, s==1) + post BN + ReLU; fused to one 3×3 conv at deploy.
- `SPPFLight`: cv1 1×1 (256→128), MaxPool2d(5, s1, p2) ×3, cv2 1×1 (512→256).
- Feature strides: 8, 16, 32 (config `strides: [8,16,32]`); anchors at `(x+0.5)*stride` centers → 60²+30²+15² = 4725.
- Regression representation: Distribution Focal Loss — 4 sides × (reg_max+1) = 8 bins; decoded by softmax expectation; box = center ± ltrb.
- Classes: 20 Pascal VOC classes.
- Parameters: training graph **1,776,716** (notebook print, `[Model] training-graph params = 1,776,716`); deployment (fused) **1,725,980** (notebook export print).

## 4. Training stages (verified from the notebook)

See [docs/TRAINING_PIPELINE.md](../SwiftDet/docs/TRAINING_PIPELINE.md) for the full documentation. Stage names and order as written in the notebook:

1. **Stage 1 — Safety check (overfit on one batch)** — `RUN_SANITY_CHECK = True`; executed; PASS.
2. **Stage 2 — Mini ablation (SimOTA vs ATSS)** — `RUN_MINI_ABLATIONS = False`; not executed in the supplied run.
3. **Stage 3 — Full training from scratch (resumable)** — `RUN_FULL_TRAINING = True`; executed; resumed at epoch 267, completed at epoch 299.

Export/quantization/evaluation (notebook sections 6–9) are not training stages.

## 5. Verified results

- Training: VOC2007 trainval + VOC2012 trainval, 16,551 images; 300 epochs; batch 128; workers 4; SimOTA; AMP fp16 (autocast, loss in fp32); channels-last; SGD momentum 0.937 / wd 5e-4 / nesterov; lr0 0.01, warmup 5, cosine decay; EMA decay 0.9998 (2,000-update ramp); grad clip 10.0; mosaic 0.8 / mixup 0.15 for the first 280 epochs (`no_aug_epochs=20`); periodic EMA validation on first 500 images of VOC2007 test every 20 epochs.
- Evaluation: FP32 AP50 **64.62%**, mAP50-95 **38.86%**; INT8 AP50 **64.23%**, mAP50-95 **38.51%** (4,952 images).
- Export: fusion max_abs_diff 3.457e-06; 300 calibration images; FP32 ONNX 6.95 MB; INT8 ONNX 2.02 MB; `reduce_range: true`; opset 17.
- Live benchmark (Intel Core i5-3320M): prep model, threads 2, 480×480, displayed FPS 13.4, P50 50.8 ms, peak 132 MB (details in [benchmarks/I5_3320M_LIVE.md](../SwiftDet/benchmarks/I5_3320M_LIVE.md)).

## 6. Not verified / cannot be confirmed

- The host CPU of the separate Kaggle benchmarks (50.7 FPS / 30.5 FPS) — not identified in the supplied logs.
- The ablation runs (Stage 2) — disabled by default and not executed.
- Exact per-stage timing of Stage 3's earlier sessions — only the final session's tail (epochs 267–299) is in the outputs; the log CSV (epochs 0–299) was copied into the model repo.
- The Pascal VOC 2007/2012 ground-truth license terms — VOC is a research dataset whose source is not redistributed; official terms must be checked with the dataset authors.

## 7. Minor notes from the notebook

- The notebook author states the numbers in the report are estimates.
- "The notebook was not executed at creation; only syntax/import checks were performed."
