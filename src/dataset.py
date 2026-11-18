"""COCO-2017 person keypoints for top-down pose estimation (SimpleBaseline / HRNet data pipeline).

Each sample is one annotated person: the box is expanded to the 3:4 aspect ratio and by 1.25, the crop is
warped to 256x192 with an affine transform, and the targets are 17 Gaussian heatmaps at 64x48 (sigma 2).
Training augmentation: random scale (+-35 %), rotation (+-45 deg, p=0.6) and horizontal flip with
left/right joint swapping.
"""
import os
import random

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_ROOT = os.environ.get("COCO_ROOT", os.path.expanduser("~/datasets/portfolio/coco"))
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def box_to_center_scale(x, y, w, h, aspect=192 / 256, pad=1.25):
    c = np.array([x + w * 0.5, y + h * 0.5], np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32) * pad


def affine(center, size, rot, out_wh, inv=False):
    """Affine map from the rotated box (center, size in pixels) to an out_wh image."""
    # same construction as the HRNet / SimpleBaseline reference code (get_affine_transform)
    rad = np.deg2rad(rot)
    sn, cs = np.sin(rad), np.cos(rad)
    p = np.array([0.0, -0.5 * size[0]])
    src_dir = np.array([p[0] * cs - p[1] * sn, p[0] * sn + p[1] * cs])
    dst_dir = np.array([0.0, -0.5 * out_wh[0]])
    third = lambda a, b: b + np.array([-(a - b)[1], (a - b)[0]])
    c_dst = np.array(out_wh, np.float64) * 0.5
    src = np.stack([center, center + src_dir, third(center, center + src_dir)]).astype(np.float32)
    dst = np.stack([c_dst, c_dst + dst_dir, third(c_dst, c_dst + dst_dir)]).astype(np.float32)
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


def apply(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


class CocoKeypoints(Dataset):
    def __init__(self, split="train", in_wh=(192, 256), hm_wh=(48, 64), sigma=2.0, root=COCO_ROOT):
        self.split, self.train = split, split == "train"
        self.img_dir = os.path.join(root, f"{split}2017")
        self.coco = COCO(os.path.join(root, "annotations", f"person_keypoints_{split}2017.json"))
        self.in_wh, self.hm_wh, self.sigma = in_wh, hm_wh, sigma
        self.items = []
        for img_id in self.coco.getImgIds(catIds=[1]):
            info = self.coco.imgs[img_id]
            for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)):
                if a["num_keypoints"] == 0:
                    continue
                x, y, w, h = a["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(info["width"] - 1, x1 + max(0, w - 1)), min(info["height"] - 1, y1 + max(0, h - 1))
                if a["area"] <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                c, s = box_to_center_scale(x1, y1, x2 - x1, y2 - y1)
                self.items.append({"img_id": img_id, "file": info["file_name"], "center": c, "size": s,
                                   "kpts": np.array(a["keypoints"], np.float32).reshape(17, 3),
                                   "ann_id": a["id"], "area": a["area"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, it["file"])), cv2.COLOR_BGR2RGB)
        c, s = it["center"].copy(), it["size"].copy()
        k = it["kpts"].copy()
        rot = 0.0
        if self.train:
            s = s * np.clip(np.random.randn() * 0.35 + 1, 0.65, 1.35)
            if random.random() < 0.6:
                rot = float(np.clip(np.random.randn() * 45, -90, 90))
            if random.random() < 0.5:
                img = img[:, ::-1]
                w = img.shape[1]
                k[:, 0] = w - 1 - k[:, 0]
                for a, b in FLIP_PAIRS:
                    k[[a, b]] = k[[b, a]]
                c[0] = w - 1 - c[0]
        M = affine(c, s, rot, self.in_wh)
        crop = cv2.warpAffine(np.ascontiguousarray(img), M, self.in_wh, flags=cv2.INTER_LINEAR)
        x = torch.from_numpy(((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())

        pts = apply(M, k[:, :2]) * (self.hm_wh[0] / self.in_wh[0])
        vis = (k[:, 2] > 0).astype(np.float32)
        target = np.zeros((17, self.hm_wh[1], self.hm_wh[0]), np.float32)
        xs, ys = np.arange(self.hm_wh[0]), np.arange(self.hm_wh[1])[:, None]
        for j in range(17):
            px, py = pts[j]
            if vis[j] == 0 or px < 0 or py < 0 or px >= self.hm_wh[0] or py >= self.hm_wh[1]:
                vis[j] = 0
                continue
            target[j] = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * self.sigma ** 2))
        meta = {"center": it["center"], "size": it["size"], "img_id": it["img_id"], "area": it["area"]}
        return x, torch.from_numpy(target), torch.from_numpy(vis), meta
