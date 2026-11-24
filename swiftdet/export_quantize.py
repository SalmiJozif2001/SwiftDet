import argparse
import os
import random

import cv2
import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from swiftdet.models.swiftdet import SwiftDet

try:  # location differs between onnxruntime versions
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError:
    try:
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError:
        quant_pre_process = None


class VOCCalibReader(CalibrationDataReader):
    def __init__(self, image_paths, img_size=480):
        self.image_paths = image_paths
        self.img_size = img_size
        self.idx = 0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        img = cv2.imread(self.image_paths[self.idx])
        self.idx += 1
        if img is None:
            return self.get_next()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        r = min(self.img_size / h, self.img_size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dy, dx = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return {"images": np.ascontiguousarray(x)}


def export_onnx(model, dummy, path, opset=17):
    kwargs = dict(input_names=["images"], output_names=["output"],
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, path, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--calib-dir", type=str, required=True)
    parser.add_argument("--calib-num", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--img-size", type=int, default=480)
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--u8u8", action="store_true", help="use uint8 activations+weights (fallback for saturation)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = SwiftDet(num_classes=20, reg_max=7, neck_channels=64).eval()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
    model.load_state_dict(state)

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        out_train_graph = model(dummy)
    model.deploy()
    with torch.no_grad():
        out_fused_graph = model(dummy)
    max_diff = (out_train_graph - out_fused_graph).abs().max().item()
    print(f"[Fusion Verification] max_abs_diff = {max_diff:.3e}")
    assert max_diff < 1e-3, f"Fusion mismatch too large: {max_diff}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Deploy Model] Total Params = {total_params:,} ({total_params / 1e6:.4f} M)")

    fp32_path = os.path.join(args.out_dir, "swiftdet_voc480_fp32.onnx")
    prep_path = os.path.join(args.out_dir, "swiftdet_voc480_prep.onnx")
    int8_path = os.path.join(args.out_dir, "swiftdet_voc480_int8_qdq.onnx")

    export_onnx(model, dummy, fp32_path, opset=17)

    quant_input = fp32_path
    if quant_pre_process is not None:
        try:
            quant_pre_process(fp32_path, prep_path)
            quant_input = prep_path
        except Exception as e:
            print(f"[Warn] quant_pre_process failed ({e}); quantizing the raw FP32 model instead.")

    all_imgs = sorted(f for f in os.listdir(args.calib_dir) if f.lower().endswith(".jpg"))
    rng = random.Random(0)
    if len(all_imgs) > args.calib_num:
        all_imgs = rng.sample(all_imgs, args.calib_num)
    imgs = [os.path.join(args.calib_dir, f) for f in all_imgs]
    print(f"[Calibration] {len(imgs)} images from {args.calib_dir}")
    reader = VOCCalibReader(imgs, img_size=args.img_size)

    act_type = QuantType.QUInt8 if args.u8u8 else QuantType.QInt8
    w_type = QuantType.QUInt8 if args.u8u8 else QuantType.QInt8
    quantize_static(
        model_input=quant_input,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=act_type,
        weight_type=w_type,
        per_channel=True,
        reduce_range=args.reduce_range,
    )
    print(f"FP32 ONNX: {os.path.getsize(fp32_path) / 1e6:.2f} MB")
    print(f"INT8 ONNX: {os.path.getsize(int8_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
