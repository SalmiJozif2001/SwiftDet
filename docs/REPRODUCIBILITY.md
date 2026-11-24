# Reproducibility

Exact commands for the SwiftDet v0.1.0 workflow. Assumes PowerShell, the repository layout

```
C:\Users\salmi\Documents\Swift\SwiftDet          # source repository (code, docs)
C:\Users\salmi\Documents\Swift\swiftdet-voc480   # model repository (artifacts)
```

and the required Python packages installed (see [requirements.txt](../requirements.txt),
[requirements-training.txt](../requirements-training.txt), [requirements-export.txt](../requirements-export.txt)).
No training run is executed for the release; the artifacts in `swiftdet-voc480/` are as shipped.

## 1. Dataset preparation

The code discovers VOC folders automatically (`find_voc_dirs`/`detect_year` in `swiftdet/data/voc_dataset.py`): a VOC folder has `Annotations/`, `JPEGImages/` and `ImageSets/`, and the year is detected from the folder name or `ImageSets/Main/*.txt`. Required splits: `2007/trainval`, `2007/test`, `2012/trainval`.

Download from the official sources (not redistributed by this repository):

```powershell
$dir = "C:\data\VOC"; New-Item -ItemType Directory -Force $dir | Out-Null
Invoke-WebRequest "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar" -OutFile "$dir\VOCtrainval_06-Nov-2007.tar"
Invoke-WebRequest "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar"     -OutFile "$dir\VOCtest_06-Nov-2007.tar"
Invoke-WebRequest "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"  -OutFile "$dir\VOCtrainval_11-May-2012.tar"
# Extract the archives; the layout should then contain VOC2007\ and VOC2012\ folders with Annotations/, JPEGImages/, ImageSets/.
```

(The notebook downloader additionally used the `http://host.robots.ox.ac.uk/pascal/VOC/...` mirrors; the same files.)

Pass the root to the commands below via `--data-root C:\data\VOC`.

## 2. Training (full run from scratch)

```powershell
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root C:\data\VOC --epochs 300 --batch-size 128 --workers 4 --val-every 20
```

Config defaults apply for anything not passed (see `swiftdet/configs/voc_swiftdet.yaml`): seed 42, warmup 5, lr0 0.01, EMA decay 0.9998, `no_aug_epochs: 20`, `mosaic_prob: 0.8`, `mixup_prob: 0.15`, assigner `simota`, loss weights VFL 1.0 / GIoU 2.0 / DFL 0.25. AMP and channels-last are auto-enabled when CUDA is available.

Expected log lines:

```
[Data] train images = 16551
[Model] training-graph params = 1,776,716
Epoch 000/300 | ...
```

Outputs go to `runs/swiftdet_voc480/`: `last.pt` (every epoch), `ema_epNNN.pt` (every 25 epochs and the final one), `train_log.csv`.

## 3. Resuming training

```powershell
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root C:\data\VOC --save-dir runs/swiftdet_voc480 --resume runs/swiftdet_voc480/last.pt
```

`--resume` loads model, EMA, optimizer and GradScaler state and continues at the next epoch (`[Resume] from ... -> starting at epoch N`). `--time-limit-hours` triggers a graceful stop after the current epoch ("TIME_LIMIT_REACHED ... Checkpoint saved to last.pt").

## 4. Overfit sanity check (optional, Stage 1)

```powershell
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root C:\data\VOC --batch-size 16 --workers 4 --overfit-one-batch
```

## 5. Mini ablation — SimOTA vs ATSS (optional, Stage 2; disabled by default)

```powershell
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root C:\data\VOC --subset-voc07 --epochs 80 --assigner simota --save-dir runs/abl_simota --batch-size 128 --workers 4 --val-every 20
python -m swiftdet.train --cfg swiftdet/configs/voc_swiftdet.yaml --data-root C:\data\VOC --subset-voc07 --epochs 80 --assigner atss   --save-dir runs/abl_atss   --batch-size 128 --workers 4 --val-every 20
```

## 6. Evaluation (VOC2007 test)

```powershell
python -m swiftdet.eval_voc --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_fp32.onnx --voc-root C:\data\VOC
python -m swiftdet.eval_voc --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_int8_qdq.onnx --voc-root C:\data\VOC
```

Output on the shipped artifacts (verified): FP32: AP50 64.62%, mAP50-95 38.86%; INT8: AP50 64.23%, mAP50-95 38.51% — on 4,952 images.

## 7. Export ONNX (FP32, fused)

```powershell
python -m swiftdet.export_quantize --weights ../swiftdet-voc480/pytorch/ema_ep299.pt --calib-dir C:\data\VOC\VOC2007\JPEGImages --out-dir outputs --reduce-range
```

- `--weights` accepts `ema_ep299.pt` or `last.pt`; the loader reads `ckpt["ema"]` when present.
- Calibration: first 300 sorted `.jpg` files of `--calib-dir` (`--calib-num`, sampled with seed 0).
- Outputs `outputs/swiftdet_voc480_fp32.onnx` (+ `swiftdet_voc480_prep.onnx` from `quant_pre_process`, + `swiftdet_voc480_int8_qdq.onnx`).
- Fusion verification: `[Fusion Verification] max_abs_diff = 3.457e-06` (assert limit 1e-3); `[Deploy Model] Total Params = 1,725,980`.

## 8. INT8 quantization

Same command as export — the INT8 QDQ file is produced by `quantize_static`:

```powershell
python -m swiftdet.export_quantize --weights <ckpt> --calib-dir C:\data\VOC\VOC2007\JPEGImages --out-dir outputs --reduce-range
```

Optional: `--u8u8` (uint8 activations+weights), or drop `--reduce-range` (the supplied run used `reduce_range: true`).

## 9. CPU benchmark

```powershell
python -m swiftdet.benchmark_cpu --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_prep.onnx --threads 4 --iters 200
```

The live-stream benchmark (threads=2, i5-3320M) is documented in [benchmarks/I5_3320M_LIVE.md](../benchmarks/I5_3320M_LIVE.md).

## 10. Live detection

```powershell
cd inference
python main.py --onnx ../swiftdet-voc480/onnx/swiftdet_voc480_prep.onnx --threads 2
python live.py --camera 0 --conf 0.35 --threads 2
```

## 11. Verifying the shipped artifacts

`../swiftdet-voc480/manifest.json` contains the SHA256 and size of every artifact. Example:

```powershell
Get-FileHash -Algorithm SHA256 ..\swiftdet-voc480\onnx\swiftdet_voc480_fp32.onnx
```
