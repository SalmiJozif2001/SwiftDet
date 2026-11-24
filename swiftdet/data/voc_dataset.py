import os
import random
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def find_voc_dirs(root):
    """Find every VOC folder (has Annotations + JPEGImages + ImageSets) under one or more roots.

    `root` may contain several roots separated by os.pathsep.
    """
    found = []
    for r in str(root).split(os.pathsep):
        if not r or not os.path.isdir(r):
            continue
        for dirpath, dirnames, _ in os.walk(r, followlinks=True):
            if "Annotations" in dirnames and "JPEGImages" in dirnames and "ImageSets" in dirnames:
                found.append(dirpath)
                dirnames[:] = []  # do not descend into a VOC folder
    return found


def detect_year(voc_dir):
    base = os.path.basename(os.path.normpath(voc_dir))
    for y in ("2007", "2012"):
        if y in base:
            return y
    main = os.path.join(voc_dir, "ImageSets", "Main")
    for split in ("trainval", "train", "val", "test"):
        p = os.path.join(main, f"{split}.txt")
        if os.path.isfile(p):
            with open(p, "r") as f:
                first = f.readline().strip()
            if first:
                return "2007" if re.fullmatch(r"\d{6}", first) else "2012"
    return None


def resolve_sets(root, sets):
    dirs = [(d, detect_year(d)) for d in find_voc_dirs(root)]
    resolved = []
    for year, split in sets:
        hit = None
        for d, y in dirs:
            if y == year and os.path.isfile(os.path.join(d, "ImageSets", "Main", f"{split}.txt")):
                hit = d
                break
        if hit is None:
            raise FileNotFoundError(f"VOC{year} '{split}' not found under: {root}")
        resolved.append((hit, split))
    return resolved


class VOCDataset(Dataset):
    def __init__(self, root, sets=(("2007", "trainval"), ("2012", "trainval")),
                 img_size=480, train=True, mosaic_prob=0.8, mixup_prob=0.15):
        self.root = root
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cls_map = {c: i for i, c in enumerate(VOC_CLASSES)}
        self.samples = []

        for voc_dir, split in resolve_sets(root, sets):
            txt_path = os.path.join(voc_dir, "ImageSets", "Main", f"{split}.txt")
            with open(txt_path, "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            for img_id in ids:
                self.samples.append((voc_dir, img_id))

    def __len__(self):
        return len(self.samples)

    def _load_raw(self, idx):
        voc_dir, img_id = self.samples[idx]
        img_path = os.path.join(voc_dir, "JPEGImages", f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tree = ET.parse(os.path.join(voc_dir, "Annotations", f"{img_id}.xml"))
        boxes, labels = [], []
        for obj in tree.findall("object"):
            if int(obj.findtext("difficult", "0")) == 1 and self.train:
                continue
            name = obj.findtext("name")
            if name not in self.cls_map:
                continue
            bb = obj.find("bndbox")
            x1, y1 = float(bb.findtext("xmin")) - 1, float(bb.findtext("ymin")) - 1
            x2, y2 = float(bb.findtext("xmax")) - 1, float(bb.findtext("ymax")) - 1
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cls_map[name])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return img, boxes, labels

    def _resize_letterbox(self, img, boxes, size):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dy, dx = (size - nh) // 2, (size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + dy
        return canvas, boxes

    def _load_mosaic(self, idx):
        s = self.img_size
        xc, yc = random.randint(s // 2, 3 * s // 2), random.randint(s // 2, 3 * s // 2)
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        indices = [idx] + [random.randint(0, len(self.samples) - 1) for _ in range(3)]
        all_boxes, all_labels = [], []

        for i, qidx in enumerate(indices):
            img, boxes, labels = self._load_raw(qidx)
            h, w = img.shape[:2]
            scale = s / max(h, w) * random.uniform(0.7, 1.2)
            nh, nw = max(int(h * scale), 2), max(int(w * scale), 2)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), nh - (y2a - y1a), nw, nh
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(xc + nw, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, nh - (y2a - y1a), x2a - x1a, nh
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(yc + nh, 2 * s)
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), 0, nw, y2a - y1a
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, 2 * s), min(yc + nh, 2 * s)
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(boxes):
                b = boxes.copy() * scale
                b[:, [0, 2]] += x1a - x1b
                b[:, [1, 3]] += y1a - y1b
                # keep only the part of each box that lies inside the pasted tile
                b[:, [0, 2]] = b[:, [0, 2]].clip(x1a, x2a)
                b[:, [1, 3]] = b[:, [1, 3]].clip(y1a, y2a)
                keep = ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
                if keep.any():
                    all_boxes.append(b[keep])
                    all_labels.append(labels[keep])

        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
        x0, y0 = random.randint(0, s), random.randint(0, s)
        crop = np.ascontiguousarray(canvas[y0:y0 + s, x0:x0 + s])
        if len(boxes):
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - x0, 0, s)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - y0, 0, s)
            keep = (boxes[:, 2] - boxes[:, 0] > 4) & (boxes[:, 3] - boxes[:, 1] > 4)
            boxes, labels = boxes[keep], labels[keep]
        return crop, boxes.astype(np.float32), labels

    def __getitem__(self, idx):
        if self.train and random.random() < self.mosaic_prob:
            img, boxes, labels = self._load_mosaic(idx)
            if random.random() < self.mixup_prob:
                img2, boxes2, labels2 = self._load_mosaic(random.randint(0, len(self.samples) - 1))
                lam = float(np.random.beta(8.0, 8.0))
                img = (img.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam)).astype(np.uint8)
                boxes = np.concatenate([boxes, boxes2], axis=0)
                labels = np.concatenate([labels, labels2], axis=0)
        else:
            img, boxes, labels = self._load_raw(idx)
            img, boxes = self._resize_letterbox(img, boxes, self.img_size)

        if self.train and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = self.img_size - boxes[:, [2, 0]]

        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - _MEAN) / _STD
        return tensor, {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
        }


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
