import argparse
import os
import platform
import time

import numpy as np
import onnxruntime as ort
import psutil

from swiftdet.eval_voc import decode_and_nms


def run_bench(model_path, img_size=480, threads=4, warmup=30, iters=200):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    for _ in range(warmup):
        _ = sess.run(None, {"images": x})

    proc = psutil.Process(os.getpid())
    lat_infer, lat_total = [], []
    rss_peak_mb = 0.0

    for _ in range(iters):
        t0 = time.perf_counter()
        out = sess.run(None, {"images": x})[0]
        t1 = time.perf_counter()
        _ = decode_and_nms(out, img_size=img_size, conf_thresh=0.25, iou_thresh=0.5)
        t2 = time.perf_counter()
        lat_infer.append((t1 - t0) * 1000.0)
        lat_total.append((t2 - t0) * 1000.0)
        rss_peak_mb = max(rss_peak_mb, proc.memory_info().rss / (1024 ** 2))

    p50_inf, p95_inf = np.percentile(lat_infer, [50, 95])
    p50_tot, p95_tot = np.percentile(lat_total, [50, 95])
    print(f"Model: {model_path} ({os.path.getsize(model_path) / 1e6:.2f} MB)")
    print(f"  Host: {platform.processor()} | Threads={threads} | Warmup={warmup} | Iters={iters}")
    print(f"  Model-Only Latency -> p50: {p50_inf:.2f} ms | p95: {p95_inf:.2f} ms ({1000.0 / p50_inf:.1f} FPS)")
    print(f"  End-to-End (+NMS)  -> p50: {p50_tot:.2f} ms | p95: {p95_tot:.2f} ms ({1000.0 / p50_tot:.1f} FPS)")
    print(f"  Peak RSS Memory    -> {rss_peak_mb:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--img-size", type=int, default=480)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()
    run_bench(a.onnx, a.img_size, a.threads, a.warmup, a.iters)
