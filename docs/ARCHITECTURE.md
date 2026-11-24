# SwiftDet Architecture

SwiftDet — a custom lightweight object-detection architecture (deployment name: **SwiftDet-PConv-CPU**).

This document describes the architecture exactly as implemented in `swiftdet/models/swiftdet.py`, `swiftdet/models/blocks.py` and `swiftdet/losses/gfl_simota.py`. All shapes are at the configured input size of 480×480. The exported model (verified by loading the ONNX artifacts) outputs `(1, 4725, 52)`: 4725 anchors = 60² + 30² + 15², 52 = 20 class logits + 4 × 8 DFL regression values.

```
480×480
   │  SwiftDetBackbone                → (c3: 64ch@8, c4: 128ch@16, c5: 256ch@32)
   │  SlimPANPConv (neck, cn=64)      → (p3_out: 64ch@8, p4_out: 64ch@16, p5_out: 64ch@32)
   │  DecoupledGFLHead                → (1, 4725, 52)
```

## Backbone — `SwiftDetBackbone`

| Module | Block | In → Out | Stride (at 480×480) |
|---|---|---|---|
| `stem_1` | `ConvBNReLU(3→16, 3×3, s2)` + ReLU | 3 → 16 ch | 2 |
| `stem_2` | `QARepConv2d(16→32, 3×3, s2)` | 16 → 32 ch | 4 |
| `stage1` | `PConvBlock(32)` | 32 → 32 ch | 4 |
| `down2` | `QARepConv2d(32→64, 3×3, s2)` | 32 → 64 ch | 8 |
| `stage2` | `PConvBlock(64)` ×2 | 64 → 64 ch | 8 |
| `down3` | `QARepConv2d(64→128, 3×3, s2)` | 64 → 128 ch | 16 |
| `stage3` | `PConvBlock(128)` ×3 | 128 → 128 ch | 16 |
| `down4` | `QARepConv2d(128→256, 3×3, s2)` | 128 → 256 ch | 32 |
| `stage4` | `PConvBlock(256)` ×2 | 256 → 256 ch | 32 |
| `sppf` | `SPPFLight(256, k=5)` | 256 → 256 ch | 32 |

Backbone returns `(c3, c4, c5)`: the **feature strides are 8, 16, 32** (config `model.strides: [8, 16, 32]`). At 480×480: c3 = 60×60×64, c4 = 30×30×128, c5 = 15×15×256.

## Blocks — `swiftdet/models/blocks.py`

### `ConvBNReLU`
Plain conv (bias-free) + BatchNorm + ReLU. In deploy mode (`fuse_for_deploy`) the conv+BN pair is fused into a single conv with bias.

### `QARepConv2d` (Quantization-Aware RepVGG, Chu et al., AAAI 2024)
RepVGG-style block: `ReLU(post_bn(bn3(conv3(x)) + conv1(x) [+ x]))`.
- `conv3` (3×3, bias-free) + `bn3`,
- `conv1` (1×1, with bias),
- identity shortcut `x` when `cin == cout and stride == 1` (`has_identity`),
- `post_bn`, then ReLU.
At deploy time `fuse_for_deploy()` merges conv3+bn3, the padded conv1, the identity kernel and post_bn into a single 3×3 conv with bias.

### `PConvBlock` (FasterNet)
FasterNet-style partial convolution block. With `ratio=0.25` and `mlp_ratio=2.0`:
- **Dynamic branch**: `PConv(c/4)` — a 3×3 conv on `cp = c/4` channels of the input;
- **Static branch**: `PConv` output is concatenated with the untouched `3c/4` channels, then `pw1` = `ConvBNReLU(c→2c, 1×1)` + ReLU, then `pw2` = `ConvBNReLU(2c→c, 1×1)` (no activation);
- **Skip**: `out = x + pw2(pw1(cat(pconv(x1), x2)))`.

### `SPPFLight`
- `cv1` = `ConvBNReLU(c→c/2, 1×1)`;
- three sequential `MaxPool2d(kernel_size=k, stride=1, padding=k//2)` applied to produce `y1, y2, y3`;
- `cv2` = `ConvBNReLU(2c→c, 1×1)` on `cat([x, y1, y2, y3])`.

## Neck — `SlimPANPConv` (`in_channels=(64,128,256), cn=64`)

Top-down (P4, P3):
- `lat_p5` = `ConvBNReLU(256→64, 1×1)`, `lat_p4` = `ConvBNReLU(128→64, 1×1)`, `lat_p3` = `ConvBNReLU(64→64, 1×1)`;
- `up5 = F.interpolate(lat_p5, size=l4, mode="nearest")`;
- `p4_td = td_p4(td_p4_red(cat([up5, l4])))` — `td_p4_red` = `ConvBNReLU(128→64, 1×1)`, `td_p4` = `PConvBlock(64)`;
- `up4 = F.interpolate(p4_td, size=l3, mode="nearest")`;
- `p3_out = td_p3(td_p3_red(cat([up4, l3])))` — `td_p3_red` = `ConvBNReLU(128→64, 1×1)`, `td_p3` = `PConvBlock(64)`.

Bottom-up (P4, P5):
- `bu_p3_down` = `ConvBNReLU(64→64, 3×3, s2)` (halves P3 to the P4 grid);
- `p4_out = bu_p4(bu_p4_red(cat([bu_p3_down(p3_out), p4_td])))` — `bu_p4_red` = `ConvBNReLU(128→64, 1×1)`, `bu_p4` = `PConvBlock(64)`;
- `bu_p4_down` = `ConvBNReLU(64→64, 3×3, s2)` (halves P4 to the P5 grid);
- `p5_out = bu_p5(bu_p5_red(cat([bu_p4_down(p4_out), l5])))` — `bu_p5_red` = `ConvBNReLU(128→64, 1×1)`, `bu_p5` = `PConvBlock(64)`.

Neck outputs: `p3_out` (64 ch @ stride 8, 60×60), `p4_out` (64 ch @ stride 16, 30×30), `p5_out` (64 ch @ stride 32, 15×15).

## Head — `DecoupledGFLHead` (`cn=64, num_classes=20, reg_max=7`)

Per level: `cls_branches[i]` and `reg_branches[i]`, each a `HeadBranch`:
- cls branch output: 20 logits;
- reg branch output: `4 * (reg_max + 1)` = 32 values (4 sides × 8 DFL bins).

`HeadBranch(cn=64, out_ch)`: split the 64-channel input into `cp = 32` (PConv) and 32 (passthrough); `pw` = `ConvBNReLU(64→64, 1×1)`; `out_conv` = `Conv2d(64→out_ch, 1×1)`.

Per-level prediction is `cat([cls_logit, reg_dist])` reshaped to `(B, H·W, 52)`; the three levels are concatenated → `(B, 4725, 52)`.

## Regression representation

Distribution Focal Loss (DFL): each anchor has 4 sides (l, t, r, b), each with `reg_max + 1 = 8` logits over bin indices 0..7.
- Decoding (`SwiftDetLoss.decode_boxes`): softmax over the 8 bins, then `ltrb = sum(prob * [0..7]) * stride`; box = `center − ltrb[:2]`, `center + ltrb[2:]`.
- Target side length: `t_ltrb = (centers − gt[:2], gt[2:] − centers) / stride`.
- Training target: `t_ltrb.clamp(0, reg_max − 0.01)`; DFL loss uses floor/ceil bins with interpolated weights.

## Classes

20 classes — the Pascal VOC class names in `swiftdet/data/voc_dataset.py::VOC_CLASSES`:
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

## Parameters

- Training graph: **1,776,716** parameters (printed by the training log: `[Model] training-graph params = 1,776,716`).
- Deployment (after `model.deploy()` fusion): **1,725,980** parameters (printed by the export: `[Deploy Model] Total Params = 1,725,980`).

## Losses and assignment — `swiftdet/losses/gfl_simota.py`

- `SwiftDetLoss(vfl_w=1.0, giou_w=2.0, dfl_w=0.25, assigner="simota")` — weights from `swiftdet/configs/voc_swiftdet.yaml` (`loss_weights`).
- Varifocal Loss (`varifocal_loss`): `BCEWithLogits(pred, target_score) * focal_weight` with `alpha=0.75, gamma=2.0`; positive anchors get weight = target score, negatives get `alpha * (sigmoid − target)^gamma`.
- GIoU loss on positives: `(1 − giou) * weights`, weights = clamped max class score.
- DFL loss on positives (as above).
- Total: `(vfl_w * vfl + giou_w * giou + dfl_w * dfl) / total_pos` — always computed in FP32.
- Assignment: **SimOTA** (`simota_assign`, default) — candidate anchors inside the GT box or within 2.5×stride of the GT center, cost = `cls_cost + 3.0*iou_cost + 1e5*not_in_box_and_center`, dynamic top-k per GT, conflict resolution by lower cost. ATSS (`atss_assign`) is also implemented and selectable via `--assigner`.

## Training-time preprocessing (from `swiftdet/data/voc_dataset.py`)

- Letterbox to 480×480, padding value 114; horizontal flip with p=0.5; normalization `(x/255 − [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]` (ImageNet stats).
- Mosaic + MixUp (mosaic p=0.8, mixup p=0.15) for the first 280 epochs (`no_aug_epochs: 20`).

## Deployment (ONNX)

- `model.deploy()` fuses `ConvBNReLU` and `QARepConv2d`; `torch.onnx.export` with `input_names=["images"]`, `output_names=["output"]`, `opset_version=17`, `do_constant_folding=True`.
- Input: `images`, shape `(1, 3, 480, 480)`, float32, RGB, letterboxed.
- Output: `output`, shape `(1, 4725, 52)`.
- Decoding + NMS (`swiftdet/eval_voc.py::decode_and_nms`): per-class greedy NMS with confidence threshold 0.05, IoU 0.5, max 300 per class; box decode from DFL.
- CPU-only: the ONNX models run on `CPUExecutionProvider`; no CUDA is required.
