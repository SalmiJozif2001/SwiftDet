# SwiftDet Training Pipeline

This document describes the three training stages of SwiftDet v0.1.0, executed by the Kaggle notebook `training/fork-of-fork-of-swiftdet-claude.ipynb`. Stage names and order follow the notebook sections (the notebook sections on export, evaluation and benchmarking are **not** training stages):

| # | Stage | Notebook section | Default | Executed in the supplied run |
|---|---|---|---|---|
| 1 | Safety check (overfit on one batch) | `RUN_SANITY_CHECK` | `True` | Yes |
| 2 | Mini ablation (SimOTA vs ATSS) | `RUN_MINI_ABLATIONS` | `False` | No |
| 3 | Full training from scratch (resumable) | `RUN_FULL_TRAINING` | `True` | Yes |

## Stage 1 — Safety check (overfit on one batch)

- **Purpose**: sanity check that the training pipeline runs end-to-end. It trains on a single batch for 200 steps.
- **Command** (notebook cell 18):
  ```powershell
  python -u -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root <DATA_ROOT> --batch-size 16 --workers 4 --overfit-one-batch
  ```
- **Dataset**: the configured VOC training set — Pascal VOC 2007 trainval + VOC 2012 trainval (16,551 images).
- **Hyperparameters**: the config values (`swiftdet/configs/voc_swiftdet.yaml`), run with batch size 16.
- **Result** (from the notebook output): `[Model] training-graph params = 1,776,716`; overfit loss 2.1110 → 0.3904 over 200 steps (vfl 1.320→0.142, giou 0.256→0.045, dfl 1.115→0.635); verdict **PASS**.
- **Checkpoints**: none — the overfit path returns before the resume logic and before the epoch loop.
- **Transition**: passed; full training follows.

## Stage 2 — Mini ablation (SimOTA vs ATSS)

- **Purpose**: comparison of the assignment strategy (SimOTA vs ATSS); run on the Pascal VOC 2007 trainval split only.
- **Command** (notebook cell 20, run once per assigner name `simota` and `atss`):
  ```powershell
  python -u -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root <DATA_ROOT> --subset-voc07 --epochs 80 --assigner <simota|atss> --save-dir runs/abl_<assigner> --batch-size 128 --workers 4 --val-every 20
  ```
- **Dataset**: Pascal VOC 2007 trainval only.
- **Epochs**: 80 per run (~1–1.5 h per run per the notebook comment).
- **Checkpoints**: separate per run — `runs/abl_simota/last.pt`, `runs/abl_atss/last.pt`, plus periodic `ema_ep*.pt` every 25 epochs.
- **Executed**: **no** — `RUN_MINI_ABLATIONS = False` by default; both runs are disabled and were not executed in the supplied run.
- **Note**: the main training (Stage 3) used the SimOTA assigner.

## Stage 3 — Full training from scratch (resumable)

- **Purpose**: the production training run — training the SwiftDet model from random initialization on Pascal VOC 2007 + 2012 trainval.
- **Command** (notebook cell 22):
  ```powershell
  python -u -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root <DATA_ROOT> --save-dir runs/swiftdet_voc480 --epochs 300 --batch-size 128 --workers 4 --val-every 20 --time-limit-hours 10 --resume runs/swiftdet_voc480/last.pt
  ```
- **Dataset**: Pascal VOC 2007 trainval (5,011 images) + Pascal VOC 2012 trainval (11,540 images) = **16,551 training images**.
- **Epochs**: **300** (`EPOCHS = 300`; config `train.epochs: 300`); `--time-limit-hours 10` (graceful stop before Kaggle's 12 h session limit).
- **Steps**: one optimizer step per batch; 16,551 images / batch 128 with `drop_last=True` ≈ 129 steps per epoch.
- **Hyperparameters** (config + notebook, all verified):
  - Optimizer: SGD, momentum 0.937, weight decay 5e-4, nesterov.
  - `lr0` 0.01, warmup 5 epochs, cosine decay (`lr_at` in `swiftdet/train.py`).
  - EMA decay 0.9998 with an exponential warmup over 2,000 updates (`ModelEMA`); the EMA model is used for periodic validation and for the export.
  - Gradient clip 10.0; seed 42.
  - Augmentation: Mosaic (p = 0.8) and MixUp (p = 0.15) for the first 280 epochs; disabled for the last 20 epochs (`no_aug_epochs: 20`). Letterbox to 480×480 (gray-114 padding); random horizontal flip p = 0.5; ImageNet-normalized input.
  - Assignment: SimOTA (default; `assigner: simota` in the config).
  - Loss weights: VFL 1.0, GIoU 2.0, DFL 0.25.
  - AMP: fp16 autocast on the network forward; the loss is always computed in FP32 (config `amp_fp16: true`; enabled only when CUDA is available).
  - Channels-last: enabled when CUDA is available (config `channels_last: true`).
  - Periodic validation: every 20 epochs (`val_every: 20`) on the first 500 images of VOC2007 test (`val_images: 500`).
- **Executed**: **Yes** (`RUN_FULL_TRAINING = True`). From the notebook output: `Resume: /kaggle/working/runs/swiftdet_voc480/last.pt` → `[Resume] from /kaggle/working/runs/swiftdet_voc480/last.pt -> starting at epoch 267`; the run completed: `TRAINING_COMPLETE` at epoch 299; notebook report: `Completed epochs: 300/300`.
- **Training-log evidence** (`results/train_log.csv` in the model repository):
  - Epoch 0: loss 2.0064 (loss then decays; the last pre-no-aug epoch, 279, logged `loss=1.1831`).
  - Augmentation-disabled phase begins at epoch 280 (`Epoch 280 | lr=0.000212 | loss=0.9380 | vfl=0.5243 giou=0.1246 dfl=0.6578`).
  - Periodic validations (first 500 images of VOC2007 test, EMA): epoch 19: AP50 0.2204; epoch 279: AP50 0.6350; final epoch 299: AP50 0.6353 (full-set evaluation: 64.62%).
  - Final epoch (299): `loss=0.8496 | vfl=0.4592 giou=0.1155 dfl=0.6377 | lr=0.000100`.

### Checkpoints, resume and final selection

- `runs/swiftdet_voc480/last.pt` — saved every epoch (atomic save; contains `model`, `ema`, `ema_updates`, `optimizer`, `scaler`, `epoch`). Latest checkpoint at the end of training is epoch 299.
- `runs/swiftdet_voc480/ema_epNNN.pt` — EMA-only checkpoints, saved every 25 epochs and at the final epoch (`ema_ep299.pt`).
- Resume mechanism (`swiftdet/train.py`): `--resume <path>` loads model, EMA, optimizer and GradScaler state and sets `start_epoch = saved_epoch + 1`; validation must never kill a long training run. The notebook auto-detects `runs/swiftdet_voc480/last.pt` and resumes from it (the final session started at epoch 267 after an earlier session's graceful stop).
- Final checkpoint selection: the final-epoch EMA checkpoint is `ema_ep299.pt` (notebook export cell: `--weights '/kaggle/working/runs/swiftdet_voc480/ema_ep299.pt'`). `swiftdet-voc480/pytorch/ema_ep299.pt` is the artifact shipped in the model repository.

### Same weights or separate checkpoints per stage?

- **Stage 1 (overfit check)**: a fresh randomly-initialized model; no checkpoint is loaded or saved.
- **Stage 2 (mini ablation)**: would start each run from a fresh initialization with separate checkpoints (`runs/abl_simota`, `runs/abl_atss`); not executed in the supplied run.
- **Stage 3 (full training)**: the production run. The final session resumed from the previous session's `last.pt` (starting at epoch 267); the final artifact is `ema_ep299.pt`.

## Notes

- The notebook author marks the numbers in the report as estimates.
- The notebook was not executed at creation (per the notebook markdown): only syntax, import and static checks were performed.
- No stages 4+ exist: export, quantization, evaluation, benchmark and archival are separate notebook sections and are documented in [MODEL_EVALUATION.md](MODEL_EVALUATION.md), [DEPLOYMENT.md](DEPLOYMENT.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the [benchmarks](../benchmarks/README.md).
