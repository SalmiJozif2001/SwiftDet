"""
live.py — كشف مباشر من الكاميرا (SwiftDet ONNX) مع عرض FPS والكشوفات.

ضعه في نفس مجلد main.py وملفات .onnx ثم شغّل:
    python live.py                       # الكاميرا 0، النموذج prep، عرض تلقائي
    python live.py --camera 1            # كاميرا أخرى
    python live.py --display web         # عرض البث في المتصفح (يعمل مع opencv-headless)
    python live.py --conf 0.4 --threads 2
    python live.py --camera http://192.168.1.5:8080/video    # كاميرا هاتف (IP Webcam)
    python live.py --camera clip.mp4 --display none --seconds 20   # اختبار على فيديو بدون عرض

العرض:
    - opencv-python-headless لا يدعم النوافذ (cv2.imshow). لذلك في الوضع auto يحاول فتح نافذة،
      وإن فشل يشغّل خادمًا محليًا ويفتح المتصفح على http://127.0.0.1:8080 .
    - للحصول على نافذة حقيقية: pip uninstall opencv-python-headless  ثم  pip install opencv-python

المفاتيح (وضع النافذة): q أو ESC للخروج، s لحفظ لقطة. في وضع المتصفح: Ctrl+C في الطرفية.
"""
import argparse
import collections
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
import psutil

try:
    import cv2
except ImportError:
    sys.exit("opencv غير مثبت:  pip install opencv-python-headless")

try:
    from main import (DEFAULT_MODEL, cpu_name, decode_and_nms, draw_detections,
                      peak_rss_mb, preprocess)
except ImportError as e:
    sys.exit(f"ضع live.py في نفس مجلد main.py ({e})")

WIN = "SwiftDet live  (q = exit, s = snapshot)"
MODEL_CANDIDATES = ("swiftdet_voc480_prep.onnx", "swiftdet_voc480_fp32.onnx", DEFAULT_MODEL)


# ------------------------------------------------------------------ مصادر الفيديو
def parse_source(s):
    return int(s) if str(s).isdigit() else s


def open_source(source, width, height):
    """يفتح كاميرا/بثًا/ملفًا ويتأكد أنه يعطي إطارًا. يعيد (cap, first_frame, live)."""
    is_index = isinstance(source, int)
    live = is_index or not os.path.isfile(str(source))
    if is_index and sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]   # DSHOW أسرع فتحًا غالبًا
    else:
        backends = [cv2.CAP_ANY]
    for b in backends:
        cap = cv2.VideoCapture(source, b)
        if not cap.isOpened():
            cap.release()
            continue
        if is_index:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # أقل تأخيرًا (لا تدعمه كل الواجهات)
        except Exception:
            pass
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, frame, live
        cap.release()
    return None, None, live


class LatestFrameReader:
    """خيط يقرأ الكاميرا باستمرار ويحتفظ بآخر إطار فقط، فلا يتراكم التأخير إذا كان النموذج أبطأ من الكاميرا."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.ts, self.idx = None, 0.0, 0
        self.ended = False
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails = 0
        while not self._stop:
            ok, f = self.cap.read()
            if not ok or f is None:
                fails += 1
                if fails >= 30:
                    with self.cond:
                        self.ended = True
                        self.cond.notify_all()
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self.cond:
                self.frame, self.ts, self.idx = f, time.perf_counter(), self.idx + 1
                self.cond.notify_all()

    def read(self, last_idx, timeout=3.0):
        with self.cond:
            self.cond.wait_for(lambda: self.idx != last_idx or self.ended, timeout)
            if self.idx != last_idx:
                return self.frame, self.ts, self.idx
        return None, 0.0, last_idx

    def stop(self):
        self._stop = True


class FileReader:
    """قراءة متزامنة لملف فيديو (كل إطار يُعالَج، بأقصى سرعة)."""

    def __init__(self, cap, loop):
        self.cap, self.loop, self.idx, self.ended = cap, loop, 0, False

    def read(self, last_idx, timeout=0.0):
        ok, f = self.cap.read()
        if (not ok or f is None) and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok or f is None:
            self.ended = True
            return None, 0.0, last_idx
        self.idx += 1
        return f, time.perf_counter(), self.idx

    def stop(self):
        pass


# ------------------------------------------------------------------ عرض المتصفح (MJPEG)
PAGE = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<title>SwiftDet live</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100%;max-height:92vh;margin-top:8px}p{margin:6px;font-size:14px;opacity:.8}</style></head>
<body><p>SwiftDet — بث مباشر (أوقف البرنامج بـ Ctrl+C في الطرفية)</p><img src="/stream" alt="stream"></body></html>
""".encode("utf-8")


class StreamServer:
    def __init__(self, port):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients, self.closed = None, 0, 0, False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with outer.cond:
                        outer.clients += 1
                    last = -1
                    try:
                        while True:
                            with outer.cond:
                                outer.cond.wait_for(
                                    lambda: (outer.seq != last and outer.jpg is not None) or outer.closed,
                                    2.0,
                                )
                                if outer.closed:
                                    break
                                if outer.seq == last or outer.jpg is None:
                                    continue
                                jpg, last = outer.jpg, outer.seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                    except OSError:
                        pass
                    finally:
                        with outer.cond:
                            outer.clients -= 1
                else:
                    self.send_error(404)

        self.httpd = None
        for p in range(port, port + 10):
            try:
                self.httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                self.port = p
                break
            except OSError:
                continue
        if self.httpd is None:
            raise OSError(f"لا يوجد منفذ حر بين {port} و{port + 9}")
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def push(self, jpg):
        with self.cond:
            self.jpg = jpg
            self.seq += 1
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


# ------------------------------------------------------------------ مساعدات
def draw_hud(img, lines):
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    width = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines) + 16
    height = 18 * len(lines) + 8
    roi = img[0:height, 0:min(width, img.shape[1])]
    roi[:] = cv2.convertScaleAbs(roi, alpha=0.35)          # خلفية داكنة شفافة
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 20 + 18 * i), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def class_summary(mapped):
    cnt = collections.Counter(name for name, _, _ in mapped)
    return ", ".join(f"{k} x{v}" for k, v in cnt.most_common()) or "-"


def fps_of(stamps):
    return (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def pick_model(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    for name in MODEL_CANDIDATES:
        p = os.path.join(here, name)
        if os.path.isfile(p):
            return p
    return os.path.join(here, MODEL_CANDIDATES[0])


# ------------------------------------------------------------------ البرنامج
def main():
    ap = argparse.ArgumentParser(description="SwiftDet ONNX live camera demo")
    ap.add_argument("--onnx", default="", help="افتراضيًا: prep ثم fp32 ثم int8 (الأول الموجود)")
    ap.add_argument("--camera", default="0", help="رقم الكاميرا، أو مسار فيديو، أو رابط بث")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False) or 2)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--img-size", type=int, default=480)
    ap.add_argument("--every", type=int, default=1, help="شغّل النموذج كل N إطار (الباقي يعرض آخر مربعات)")
    ap.add_argument("--mirror", action="store_true", help="قلب أفقي (مثل المرآة)")
    ap.add_argument("--display", choices=["auto", "window", "web", "none"], default="auto")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-open", action="store_true", help="لا تفتح المتصفح تلقائيًا في وضع web")
    ap.add_argument("--seconds", type=float, default=0, help="أوقف بعد N ثانية (0 = بلا حد)")
    ap.add_argument("--loop", action="store_true", help="إعادة تشغيل ملف الفيديو عند نهايته")
    ap.add_argument("--verbose", action="store_true", help="اطبع كل كشف (فئة/ثقة/صندوق) كل ثانية")
    args = ap.parse_args()
    args.every = max(1, args.every)

    model_path = pick_model(args.onnx)
    if not os.path.isfile(model_path):
        sys.exit(f"لم يتم العثور على النموذج: {model_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shp = inp.shape
    size = int(shp[-1]) if isinstance(shp[-1], int) else args.img_size
    for _ in range(3):   # إحماء
        sess.run(None, {inp.name: np.zeros((1, 3, size, size), np.float32)})

    source = parse_source(args.camera)
    cap, first, live = open_source(source, args.width, args.height)
    if cap is None:
        sys.exit(f"تعذّر فتح المصدر: {args.camera}\n"
                 "- تأكد أن الكاميرا غير مستخدمة في برنامج آخر وأن Windows يسمح للتطبيقات بالوصول إليها.\n"
                 "- جرّب --camera 1 أو مسار فيديو.")
    try:
        backend = cap.getBackendName()
    except Exception:
        backend = "?"
    fh, fw = first.shape[:2]

    print("=" * 64)
    print(f"CPU      : {cpu_name()}")
    print(f"Model    : {os.path.basename(model_path)} | input {size}x{size} | threads {args.threads}")
    print(f"Source   : {args.camera} | {fw}x{fh} | backend {backend} | {'live' if live else 'file'}")
    print(f"Settings : conf>={args.conf} | detect every {args.every} frame(s) | display {args.display}")
    print("=" * 64)

    # ---- اختيار طريقة العرض
    mode = args.display
    if mode in ("auto", "window"):
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            mode = "window"
        except cv2.error:
            if args.display == "window":
                sys.exit("هذه النسخة من opencv بلا نوافذ (headless).\n"
                         "ثبّت:  pip uninstall opencv-python-headless  ثم  pip install opencv-python\n"
                         "أو استخدم  --display web")
            print("[!] opencv-headless لا يدعم النوافذ -> سيُعرض البث في المتصفح.")
            mode = "web"
    server = None
    if mode == "web":
        server = StreamServer(args.port)
        url = f"http://127.0.0.1:{server.port}"
        print(f"البث المباشر: {url}   (Ctrl+C للإيقاف)")
        if not args.no_open:
            webbrowser.open(url)

    reader = LatestFrameReader(cap) if live else FileReader(cap, args.loop)

    # ---- الحلقة
    proc = psutil.Process(os.getpid())
    cpu0, t_start = proc.cpu_times(), time.perf_counter()
    disp_ts, inf_ts = collections.deque(maxlen=30), collections.deque(maxlen=30)
    t_pre, t_mod, t_post, t_e2e, lags, ndets = [], [], [], [], [], []
    ema = {"pre": 0.0, "mod": 0.0, "post": 0.0}
    state = ([], 1.0, 0, 0)
    mapped = []
    last_idx, frame_no, dropped, timeouts, shown = 0, 0, 0, 0, 0
    last_report = t_start
    stop_reason = ""

    try:
        while True:
            now = time.perf_counter()
            if args.seconds and now - t_start >= args.seconds:
                stop_reason = "انتهت المدة المحددة"
                break
            frame, ts, idx = reader.read(last_idx)
            if frame is None:
                if reader.ended:
                    stop_reason = "انتهى البث/الملف"
                    break
                timeouts += 1
                if timeouts >= 5:
                    stop_reason = "لا توجد إطارات من الكاميرا"
                    break
                continue
            timeouts = 0
            if last_idx and live:
                dropped += idx - last_idx - 1
            last_idx = idx
            if args.mirror:
                frame = cv2.flip(frame, 1)

            if frame_no % args.every == 0:
                a = time.perf_counter()
                x, r, dx, dy = preprocess(frame, size)
                b = time.perf_counter()
                out = sess.run(None, {inp.name: x})[0]
                c = time.perf_counter()
                dets = decode_and_nms(out, img_size=size, conf_thresh=args.conf)
                d = time.perf_counter()
                state = (dets, r, dx, dy)
                pre_ms, mod_ms, post_ms = (b - a) * 1000, (c - b) * 1000, (d - c) * 1000
                for k, v in (("pre", pre_ms), ("mod", mod_ms), ("post", post_ms)):
                    ema[k] = v if ema[k] == 0.0 else 0.9 * ema[k] + 0.1 * v
                t_pre.append(pre_ms)
                t_mod.append(mod_ms)
                t_post.append(post_ms)
                t_e2e.append((d - a) * 1000)
                inf_ts.append(d)
            frame_no += 1

            vis, mapped = draw_detections(frame, *state)
            ndets.append(len(mapped))
            lag_ms = (time.perf_counter() - ts) * 1000
            lags.append(lag_ms)
            disp_ts.append(time.perf_counter())
            fps, ifps = fps_of(disp_ts), fps_of(inf_ts)

            draw_hud(vis, [
                f"FPS {fps:5.1f}  (detect {ifps:4.1f}/s)",
                f"model {ema['mod']:5.1f} ms  pre {ema['pre']:4.1f}  post {ema['post']:4.1f}",
                f"lag {lag_ms:4.0f} ms   dropped {dropped}",
                f"detections {len(mapped)}: {class_summary(mapped)}"[:60],
            ])

            if mode == "window":
                cv2.imshow(WIN, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    stop_reason = "تم الخروج (q/ESC)"
                    break
                if k == ord("s"):
                    fn = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(fn, vis)
                    print(f"\nتم حفظ اللقطة: {fn}")
                if shown > 2 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "أُغلقت النافذة"
                    break
            elif mode == "web" and (server.clients > 0 or shown < 3):
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    server.push(buf.tobytes())
            shown += 1

            now = time.perf_counter()
            if now - last_report >= 1.0:
                last_report = now
                line = (f"FPS {fps:5.1f} | model {ema['mod']:5.1f} ms | pre {ema['pre']:4.1f} | post {ema['post']:4.1f} "
                        f"| lag {lag_ms:4.0f} ms | dets {len(mapped)}: {class_summary(mapped)}")
                if args.verbose:
                    print(line)
                    for name, score, box in mapped:
                        print(f"    - {name:12s} {score:.2f} box={box}")
                else:
                    print("\r" + line[:118].ljust(118), end="", flush=True)
    except KeyboardInterrupt:
        stop_reason = "تم الإيقاف (Ctrl+C)"
    finally:
        reader.stop()
        if server:
            server.close()
        if mode == "window":
            cv2.destroyAllWindows()
        cap.release()

    # ---- الملخص
    wall = time.perf_counter() - t_start
    cpu1 = proc.cpu_times()
    cpu_sec = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    print("\n" + "=" * 64)
    print(f"انتهى: {stop_reason or 'تم الإيقاف'}")
    if shown == 0:
        print("لم تُعالَج أي إطارات.")
        return
    print(f"الإطارات المعروضة : {shown} خلال {wall:.1f} ث  => متوسط {shown / wall:.1f} FPS")
    print(f"إطارات أُسقطت (الكاميرا أسرع من المعالجة): {dropped}")
    if t_mod:
        print(f"النموذج فقط       : p50 {pct(t_mod, 50):.1f} ms | p95 {pct(t_mod, 95):.1f} ms")
        print(f"pre+model+NMS     : p50 {pct(t_e2e, 50):.1f} ms | p95 {pct(t_e2e, 95):.1f} ms  (مرات التشغيل: {len(t_mod)})")
        print(f"  - المعالجة المسبقة: {pct(t_pre, 50):.1f} ms | فك الترميز+NMS: {pct(t_post, 50):.1f} ms")
    print(f"التأخير (من الالتقاط للعرض): p50 {pct(lags, 50):.0f} ms | p95 {pct(lags, 95):.0f} ms")
    print(f"الكشوفات لكل إطار : متوسط {np.mean(ndets):.2f} | أقصى {max(ndets)}")
    print(f"المعالج           : ~{cpu_sec / wall:.2f} نواة مستخدمة | الذاكرة (ذروة): {peak_rss_mb(proc):.0f} MB")


if __name__ == "__main__":
    main()
