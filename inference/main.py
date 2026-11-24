"""
اختبار سرعة واستهلاك موارد نموذج SwiftDet (ONNX) على جهازك.

المتطلبات:
    pip install onnxruntime numpy psutil opencv-python-headless

أمثلة:
    python main.py                                   # يختبر swiftdet_voc480_int8_qdq.onnx في نفس المجلد
    python main.py --onnx path/to/model.onnx
    python main.py --image test.jpg --threads 2
    python main.py --all                             # يقارن كل ملفات .onnx الموجودة في مجلد النموذج
"""
import argparse
import glob
import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_MODEL = "swiftdet_voc480_int8_qdq.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor")


# ------------------------------------------------------------------ pre / post processing
# (pix/255 - mean) / std  ==  pix * SCALE + SHIFT  (قيم RGB)
_SCALE_RGB = (1.0 / (255.0 * STD.reshape(3))).astype(np.float32)
_SHIFT_RGB = (-MEAN.reshape(3) / STD.reshape(3)).astype(np.float32)
_PRE_BUFFERS = {}


def preprocess(img_bgr, size):
    """Letterbox + normalisation, same result as training/eval but faster.

    - لا يوجد cvtColor: عكس القنوات يتم أثناء النسخ.
    - التطبيع يُنفَّذ على المنطقة المصغّرة فقط، أما الحشو (114) فقيمته المطبَّعة محسوبة مسبقًا.
    - المصفوفة الناتجة تُعاد استخدامها بين الاستدعاءات: استهلكها (sess.run) قبل الاستدعاء التالي.
    """
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dy, dx = (size - nh) // 2, (size - nw) // 2

    key = (size, nh, nw)
    buf = _PRE_BUFFERS.get(key)
    if buf is None:
        buf = np.empty((1, 3, size, size), dtype=np.float32)
        buf[0] = (114.0 * _SCALE_RGB + _SHIFT_RGB).reshape(3, 1, 1)
        _PRE_BUFFERS[key] = buf

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rf = resized.astype(np.float32)
    rf *= _SCALE_RGB[::-1]      # القنوات هنا BGR
    rf += _SHIFT_RGB[::-1]
    buf[0, :, dy:dy + nh, dx:dx + nw] = rf.transpose(2, 0, 1)[::-1]   # [::-1] => RGB
    return buf, r, dx, dy


_GRID_CACHE = {}


def _grid(img_size, strides):
    """مراكز الأنكرات والخطوات: تُحسب مرة واحدة فقط (كانت تُعاد في كل إطار)."""
    key = (img_size, tuple(strides))
    g = _GRID_CACHE.get(key)
    if g is None:
        centers, steps = [], []
        for s in strides:
            n = img_size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            centers.append(np.stack([(xv + 0.5) * s, (yv + 0.5) * s], axis=-1).reshape(-1, 2))
            steps.append(np.full((n * n, 1), s, dtype=np.float32))
        g = (np.concatenate(centers, axis=0), np.concatenate(steps, axis=0))
        _GRID_CACHE[key] = g
    return g


def decode_and_nms(raw, img_size=480, strides=(8, 16, 32), num_classes=20, reg_max=7,
                   conf_thresh=0.25, iou_thresh=0.5, max_per_class=300):
    """نفس نتائج النسخة القديمة، لكن:
    - العتبة تُطبَّق على logits مباشرة (sigmoid(x) >= t  <=>  x >= logit(t)) بدل sigmoid على كل الأنكرات.
    - softmax وفك الصناديق يجريان فقط للأنكرات التي اجتازت العتبة (غالبًا عشرات من 4725).
    - الشبكة تُخزَّن مؤقتًا.
    """
    pred = raw[0]
    centers, steps = _grid(img_size, strides)
    t = min(max(float(conf_thresh), 1e-6), 1.0 - 1e-6)
    thr_logit = float(np.log(t / (1.0 - t)))

    logits = pred[:, :num_classes]
    a_idx, c_idx = np.nonzero(logits >= thr_logit)
    if a_idx.size == 0:
        return []

    uniq, inv = np.unique(a_idx, return_inverse=True)
    reg = pred[uniq, num_classes:].reshape(-1, 4, reg_max + 1)
    reg_exp = np.exp(reg - np.max(reg, axis=-1, keepdims=True))
    reg_prob = reg_exp / np.sum(reg_exp, axis=-1, keepdims=True)
    ltrb = np.sum(reg_prob * np.arange(reg_max + 1, dtype=np.float32), axis=-1) * steps[uniq]
    ctr = centers[uniq]
    boxes_u = np.clip(np.concatenate([ctr - ltrb[:, :2], ctr + ltrb[:, 2:]], axis=-1), 0, img_size)
    boxes_all = boxes_u[inv.reshape(-1)]
    scores_all = 1.0 / (1.0 + np.exp(-np.clip(logits[a_idx, c_idx], -40, 40)))

    out = []
    for c in np.unique(c_idx):
        m = c_idx == c
        cb, cs = boxes_all[m], scores_all[m]
        c = int(c)
        if cs.size > max_per_class:
            top = np.argpartition(-cs, max_per_class)[:max_per_class]
            cb, cs = cb[top], cs[top]
        order = np.argsort(-cs)
        while order.size > 0:
            i = order[0]
            out.append((cb[i].copy(), float(cs[i]), c))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(cb[i, 0], cb[rest, 0])
            yy1 = np.maximum(cb[i, 1], cb[rest, 1])
            xx2 = np.minimum(cb[i, 2], cb[rest, 2])
            yy2 = np.minimum(cb[i, 3], cb[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
            a_o = (cb[rest, 2] - cb[rest, 0]) * (cb[rest, 3] - cb[rest, 1])
            iou = inter / np.maximum(a_i + a_o - inter, 1e-6)
            order = rest[iou <= iou_thresh]
    return out


# ------------------------------------------------------------------ drawing
def class_color(c):
    rs = np.random.RandomState(c * 7 + 3)
    return tuple(int(v) for v in rs.randint(40, 230, 3))


def draw_detections(img, dets, r, dx, dy):
    """Draw boxes on the ORIGINAL image (undo letterbox: x = (x' - pad) / scale)."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(0.004 * (h + w) / 2)))
    fscale = max(0.5, 0.0009 * (h + w) / 2)
    mapped = []
    for box, score, c in dets:
        x1 = int(np.clip((box[0] - dx) / r, 0, w - 1))
        y1 = int(np.clip((box[1] - dy) / r, 0, h - 1))
        x2 = int(np.clip((box[2] - dx) / r, 0, w - 1))
        y2 = int(np.clip((box[3] - dy) / r, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = class_color(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{VOC_CLASSES[c]} {score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        ytop = max(y1 - th - base - 4, 0)
        cv2.rectangle(out, (x1, ytop), (x1 + tw + 6, ytop + th + base + 4), color, -1)
        cv2.putText(out, label, (x1 + 3, ytop + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        mapped.append((VOC_CLASSES[c], score, (x1, y1, x2, y2)))
    return out, mapped


def open_image(path):
    """opencv-headless has no imshow, so open the saved file with the OS viewer."""
    import subprocess
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"  [!] تعذّر فتح الصورة تلقائيًا ({e}). افتح الملف يدويًا: {path}")


# ------------------------------------------------------------------ benchmark
def cpu_name():
    name = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            import subprocess
            out = subprocess.check_output("wmic cpu get name", shell=True, text=True)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
    except Exception:
        pass
    return name or "unknown"


def peak_rss_mb(proc):
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", None)  # Windows exact peak
    return (peak if peak else mi.rss) / 1024 ** 2


def bench(model_path, args):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / 1024 ** 2

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000.0
    rss_loaded = proc.memory_info().rss / 1024 ** 2
    inp = sess.get_inputs()[0].name

    if args.image and cv2 is not None:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Cannot read image: {args.image}")
    else:
        if args.image:
            print("  [!] opencv غير مثبت؛ سيُستخدم إدخال عشوائي بدل الصورة.")
        img = None

    def make_input():
        if img is not None:
            return preprocess(img, args.img_size)[0]
        return np.random.randn(1, 3, args.img_size, args.img_size).astype(np.float32)

    x = make_input()
    for _ in range(args.warmup):
        sess.run(None, {inp: x})

    lat_model, lat_pre, lat_post, lat_total = [], [], [], []
    peak = rss_loaded
    cpu0, wall0 = proc.cpu_times(), time.perf_counter()
    n_det = 0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        xi = make_input()
        t1 = time.perf_counter()
        out = sess.run(None, {inp: xi})[0]
        t2 = time.perf_counter()
        dets = decode_and_nms(out, img_size=args.img_size, conf_thresh=args.conf)
        t3 = time.perf_counter()
        n_det = len(dets)
        lat_pre.append((t1 - t0) * 1000)
        lat_model.append((t2 - t1) * 1000)
        lat_post.append((t3 - t2) * 1000)
        lat_total.append((t3 - t0) * 1000)
        peak = max(peak, peak_rss_mb(proc))
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    cpu_sec = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)

    def stats(a):
        return np.percentile(a, 50), np.percentile(a, 95), float(np.mean(a))

    m50, m95, mavg = stats(lat_model)
    t50, t95, tavg = stats(lat_total)
    res = {
        "name": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / 1e6,
        "load_ms": load_ms,
        "m50": m50, "m95": m95, "mavg": mavg,
        "t50": t50, "t95": t95,
        "pre50": np.percentile(lat_pre, 50), "post50": np.percentile(lat_post, 50),
        "fps_model": 1000.0 / m50, "fps_total": 1000.0 / t50,
        "rss_model_mb": rss_loaded - rss_before,
        "rss_peak_mb": peak,
        "cpu_cores_used": cpu_sec / wall,
        "cpu_util_pct": 100.0 * cpu_sec / wall / (psutil.cpu_count(logical=True) or 1),
        "n_det": n_det,
    }

    print(f"\n=== {res['name']} ({res['size_mb']:.2f} MB) ===")
    print(f"  زمن التحميل           : {load_ms:.1f} ms")
    print(f"  النموذج فقط           : p50 {m50:.2f} ms | p95 {m95:.2f} ms | متوسط {mavg:.2f} ms  ({res['fps_model']:.1f} FPS)")
    print(f"  شامل (pre+model+NMS)  : p50 {t50:.2f} ms | p95 {t95:.2f} ms  ({res['fps_total']:.1f} FPS)")
    print(f"    - المعالجة المسبقة  : {res['pre50']:.2f} ms (p50)")
    print(f"    - فك الترميز + NMS  : {res['post50']:.2f} ms (p50)")
    print(f"  الذاكرة (RAM)         : زيادة بعد تحميل النموذج {res['rss_model_mb']:.1f} MB | الذروة الكلية {peak:.1f} MB")
    print(f"  المعالج               : ~{res['cpu_cores_used']:.2f} نواة مستخدمة ({res['cpu_util_pct']:.0f}% من كل الأنوية المنطقية)")
    print(f"  عدد الكشوفات (آخر تمريرة، conf>={args.conf}): {n_det}")

    # ---- رسم المربعات على الصورة (بعد انتهاء القياس حتى لا يؤثر على الأزمنة) ----
    if img is not None:
        xi, r, dx, dy = preprocess(img, args.img_size)
        out = sess.run(None, {inp: xi})[0]
        dets = decode_and_nms(out, img_size=args.img_size, conf_thresh=args.conf)
        drawn, mapped = draw_detections(img, dets, r, dx, dy)
        print("  الكشوفات على الصورة الأصلية:")
        for name, score, (x1, y1, x2, y2) in mapped:
            print(f"    - {name:12s} {score:.2f}  box=({x1},{y1},{x2},{y2})")
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if args.out and not args.all:
            out_path = args.out
        else:
            out_path = os.path.splitext(args.image)[0] + f"_{stem}_det.jpg"
        cv2.imwrite(out_path, drawn)
        print(f"  تم حفظ الصورة مع المربعات: {out_path}")
        if not args.no_show and not args.all:  # في وضع --all لا نفتح 3 نوافذ
            open_image(os.path.abspath(out_path))
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="SwiftDet ONNX speed / memory benchmark")
    ap.add_argument("--onnx", default=os.path.join(here, DEFAULT_MODEL))
    ap.add_argument("--all", action="store_true", help="اختبر كل ملفات .onnx في مجلد النموذج للمقارنة")
    ap.add_argument("--image", default="", help="صورة حقيقية للاختبار (اختياري)")
    ap.add_argument("--img-size", type=int, default=480)
    ap.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False) or 2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="", help="مسار حفظ الصورة الناتجة (افتراضيًا بجانب الصورة الأصلية)")
    ap.add_argument("--no-show", action="store_true", help="احفظ الصورة فقط بدون فتحها")
    args = ap.parse_args()

    print("=" * 64)
    print(f"CPU     : {cpu_name()}")
    print(f"Cores   : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical")
    print(f"RAM     : {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")
    print(f"OS      : {platform.platform()} | Python {platform.python_version()} | onnxruntime {ort.__version__}")
    print(f"Threads : {args.threads} | warmup {args.warmup} | iters {args.iters} | input {args.img_size}x{args.img_size}")
    print("=" * 64)

    if args.all:
        folder = os.path.dirname(os.path.abspath(args.onnx))
        models = sorted(glob.glob(os.path.join(folder, "*.onnx")))
    else:
        models = [args.onnx]
    models = [m for m in models if os.path.isfile(m)]
    if not models:
        sys.exit(f"لم يتم العثور على النموذج: {args.onnx}\nضع ملف .onnx بجانب main.py أو مرّر --onnx.")

    results = [bench(m, args) for m in models]

    if len(results) > 1:
        ranked = sorted(results, key=lambda r: r["m50"])
        best = ranked[0]
        print("\n" + "=" * 72)
        print(f"المقارنة (مرتبة من الأسرع)  | threads={args.threads}")
        print("=" * 72)
        print(f"{'model':34s}{'model p50':>10s}{'FPS':>7s}{'total p50':>10s}{'peak MB':>9s}{'x':>6s}")
        for r in ranked:
            slow = r["m50"] / best["m50"]
            mark = "  <== الأسرع" if r is best else ""
            print(f"{r['name']:34s}{r['m50']:8.2f}ms{r['fps_model']:7.1f}{r['t50']:8.2f}ms"
                  f"{r['rss_peak_mb']:9.1f}{slow:6.2f}{mark}")
        print("-" * 72)
        print(f"الأفضل على معالجك (زمن النموذج): {best['name']}")
        print("ملاحظة: عمود x = كم مرة أبطأ من الأسرع. ملف prep هو نفس FP32 بعد تبسيط الرسم فقط.")


if __name__ == "__main__":
    main()
