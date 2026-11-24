import argparse

import numpy as np

from swiftdet.data.voc_dataset import VOCDataset, VOC_CLASSES


def decode_and_nms(raw_output, img_size=480, strides=(8, 16, 32), num_classes=20,
                   reg_max=7, conf_thresh=0.05, iou_thresh=0.5, max_per_class=300):
    pred = raw_output[0]
    cls_scores = 1.0 / (1.0 + np.exp(-np.clip(pred[:, :num_classes], -40, 40)))
    reg = pred[:, num_classes:].reshape(-1, 4, reg_max + 1)
    reg_exp = np.exp(reg - np.max(reg, axis=-1, keepdims=True))
    reg_prob = reg_exp / np.sum(reg_exp, axis=-1, keepdims=True)
    ltrb = np.sum(reg_prob * np.arange(reg_max + 1, dtype=np.float32), axis=-1)

    centers, step_arr = [], []
    for s in strides:
        g = img_size // s
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        pts = np.stack([(xv + 0.5) * s, (yv + 0.5) * s], axis=-1).reshape(-1, 2)
        centers.append(pts)
        step_arr.append(np.full((pts.shape[0], 1), s, dtype=np.float32))
    centers = np.concatenate(centers, axis=0)
    ltrb = ltrb * np.concatenate(step_arr, axis=0)
    boxes = np.concatenate([centers - ltrb[:, :2], centers + ltrb[:, 2:]], axis=-1)
    boxes = np.clip(boxes, 0, img_size)

    out_boxes, out_scores, out_cls = [], [], []
    for c in range(num_classes):
        scores = cls_scores[:, c]
        mask = scores >= conf_thresh
        if not np.any(mask):
            continue
        cb, cs = boxes[mask], scores[mask]
        if cs.size > max_per_class:
            top = np.argpartition(-cs, max_per_class)[:max_per_class]
            cb, cs = cb[top], cs[top]
        order = np.argsort(-cs)
        while order.size > 0:
            i = order[0]
            out_boxes.append(cb[i])
            out_scores.append(cs[i])
            out_cls.append(c)
            if order.size == 1:
                break
            xx1 = np.maximum(cb[i, 0], cb[order[1:], 0])
            yy1 = np.maximum(cb[i, 1], cb[order[1:], 1])
            xx2 = np.minimum(cb[i, 2], cb[order[1:], 2])
            yy2 = np.minimum(cb[i, 3], cb[order[1:], 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
            area_o = (cb[order[1:], 2] - cb[order[1:], 0]) * (cb[order[1:], 3] - cb[order[1:], 1])
            iou = inter / np.maximum(area_i + area_o - inter, 1e-6)
            order = order[1:][iou <= iou_thresh]

    if not out_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)
    return np.array(out_boxes), np.array(out_scores), np.array(out_cls, dtype=np.int64)


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_fn(predict_fn, voc_root, img_size=480, max_images=0, conf_thresh=0.05, verbose=True):
    """Generic VOC07-test evaluation.

    predict_fn: callable taking a float32 numpy array (1,3,H,W) and returning a numpy array (1,N,52).
    Returns (AP50, mAP50-95) as fractions in [0, 1].
    """
    ds = VOCDataset(voc_root, sets=[("2007", "test")], img_size=img_size, train=False)
    n_total = len(ds)
    if max_images and max_images < n_total:
        indices = np.linspace(0, n_total - 1, max_images).astype(int).tolist()
    else:
        indices = list(range(n_total))

    n_cls = len(VOC_CLASSES)
    preds_by_cls = {c: [] for c in range(n_cls)}
    gts_by_cls = {c: {} for c in range(n_cls)}

    for k, idx in enumerate(indices):
        tensor, tgt = ds[idx]
        out = predict_fn(tensor.unsqueeze(0).numpy())
        pb, ps, pc = decode_and_nms(out, img_size=img_size, conf_thresh=conf_thresh)
        for b, s, c in zip(pb, ps, pc):
            preds_by_cls[int(c)].append((k, float(s), b))
        gb, gl = tgt["boxes"].numpy(), tgt["labels"].numpy()
        for b, c in zip(gb, gl):
            gts_by_cls[int(c)].setdefault(k, []).append(b)
        if verbose and (k + 1) % 500 == 0:
            print(f"  evaluated {k + 1}/{len(indices)} images", flush=True)

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps_per_thresh = []
    for thr in iou_thresholds:
        cls_aps = []
        for c in range(n_cls):
            npos = sum(len(v) for v in gts_by_cls[c].values())
            if npos == 0:
                continue
            used = {key: np.zeros(len(v), dtype=bool) for key, v in gts_by_cls[c].items()}
            c_preds = sorted(preds_by_cls[c], key=lambda item: -item[1])
            tp, fp = np.zeros(len(c_preds)), np.zeros(len(c_preds))
            for j, (img_idx, _, box) in enumerate(c_preds):
                gt_list = gts_by_cls[c].get(img_idx, [])
                if not gt_list:
                    fp[j] = 1.0
                    continue
                gt_arr = np.array(gt_list)
                ix1 = np.maximum(gt_arr[:, 0], box[0])
                iy1 = np.maximum(gt_arr[:, 1], box[1])
                ix2 = np.minimum(gt_arr[:, 2], box[2])
                iy2 = np.minimum(gt_arr[:, 3], box[3])
                inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                uni = (box[2] - box[0]) * (box[3] - box[1]) + \
                      (gt_arr[:, 2] - gt_arr[:, 0]) * (gt_arr[:, 3] - gt_arr[:, 1]) - inter
                ious = inter / np.maximum(uni, 1e-6)
                jmax = int(np.argmax(ious))
                if ious[jmax] >= thr and not used[img_idx][jmax]:
                    tp[j] = 1.0
                    used[img_idx][jmax] = True
                else:
                    fp[j] = 1.0
            fp_cum, tp_cum = np.cumsum(fp), np.cumsum(tp)
            rec = tp_cum / max(npos, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            cls_aps.append(compute_ap(rec, prec))
        aps_per_thresh.append(float(np.mean(cls_aps)) if cls_aps else 0.0)

    return aps_per_thresh[0], float(np.mean(aps_per_thresh))


def evaluate_onnx(onnx_path, voc_root, img_size=480, max_images=0):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict(x):
        return sess.run(None, {"images": x})[0]

    ap50, map_all = evaluate_fn(predict, voc_root, img_size=img_size, max_images=max_images)
    print(f"ONNX: {onnx_path} | AP50 = {ap50 * 100:.2f}% | mAP50-95 = {map_all * 100:.2f}%")
    return ap50, map_all


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--voc-root", type=str, default="/kaggle/input")
    p.add_argument("--img-size", type=int, default=480)
    p.add_argument("--max-images", type=int, default=0)
    a = p.parse_args()
    evaluate_onnx(a.onnx, a.voc_root, a.img_size, a.max_images)
