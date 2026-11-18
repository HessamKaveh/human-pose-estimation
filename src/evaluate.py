"""COCO keypoint evaluation (OKS AP) on val2017 with ground-truth person boxes, plus a qualitative figure.

Decoding follows SimpleBaseline: flip test (average with the flipped, joint-swapped, 1-px shifted heatmap),
arg-max with a quarter-pixel shift toward the higher neighbour, and the inverse affine transform back to
image coordinates. Instance score = box score (1.0 for GT boxes) x mean confidence of joints above 0.2.
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import FLIP_PAIRS, SKELETON, CocoKeypoints, affine, apply
from model import SimpleBaseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def flip_back(hm):
    hm = hm.flip(3)
    idx = list(range(17))
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    hm = hm[:, idx]
    hm[..., 1:] = hm[..., :-1].clone()
    return hm


def decode(hm):
    """hm: (B, 17, H, W) numpy -> coords (B, 17, 2) in heatmap pixels, maxvals (B, 17)."""
    B, J, H, W = hm.shape
    flat = hm.reshape(B, J, -1)
    idx, mx = flat.argmax(2), flat.max(2)
    xs, ys = (idx % W).astype(np.float32), (idx // W).astype(np.float32)
    for b in range(B):
        for j in range(J):
            x, y = int(xs[b, j]), int(ys[b, j])
            if 1 < x < W - 1 and 1 < y < H - 1:
                xs[b, j] += 0.25 * np.sign(hm[b, j, y, x + 1] - hm[b, j, y, x - 1])
                ys[b, j] += 0.25 * np.sign(hm[b, j, y + 1, x] - hm[b, j, y - 1, x])
    return np.stack([xs, ys], 2), mx


def collate(batch):
    x, _, _, meta = zip(*batch)
    return torch.stack(x), list(meta)


@torch.no_grad()
def predict(model, ds, dev):
    dl = DataLoader(ds, batch_size=64, num_workers=12, collate_fn=collate)
    results = []
    for x, meta in dl:
        x = x.to(dev).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            hm = model(x).float()
            hm = 0.5 * (hm + flip_back(model(x.flip(3)).float()))
        coords, mx = decode(hm.cpu().numpy())
        coords *= ds.in_wh[0] / ds.hm_wh[0]
        for b, m in enumerate(meta):
            Minv = affine(m["center"], m["size"], 0, ds.in_wh, inv=True)
            xy = apply(Minv, coords[b])
            conf = mx[b]
            good = conf > 0.2
            score = float(conf[good].mean()) if good.any() else 0.0
            kp = np.concatenate([xy, conf[:, None]], 1).reshape(-1)
            results.append({"image_id": int(m["img_id"]), "category_id": 1,
                            "keypoints": [float(v) for v in kp], "score": score})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["resnet50"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    ds = CocoKeypoints("val")
    names = ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L"]
    os.makedirs(RES, exist_ok=True)
    allm = {}
    for bb in args.backbones:
        model = SimpleBaseline(bb, pretrained=False).to(dev).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"simplebaseline_{bb}.pt"), map_location=dev))
        model.eval()
        res = predict(model, ds, dev)
        rf = os.path.join(ROOT, "checkpoints", f"val2017_{bb}_results.json")
        json.dump(res, open(rf, "w"))
        E = COCOeval(ds.coco, ds.coco.loadRes(rf), "keypoints")
        E.params.useSegm = None
        E.evaluate(); E.accumulate(); E.summarize()
        allm[bb] = {n: round(100 * float(v), 1) for n, v in zip(names, E.stats)}
        print(bb, allm[bb], flush=True)
    json.dump(allm, open(os.path.join(RES, "coco_val2017_gt_boxes.json"), "w"), indent=2)

    # qualitative figure: predictions of the last model on a few crowded val images
    by_img = {}
    for r in res:
        by_img.setdefault(r["image_id"], []).append(r)
    ids = [i for i in sorted(by_img, key=lambda i: -len(by_img[i])) if len(by_img[i]) >= 2][5:200:33][:6]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, iid in zip(axs.ravel(), ids):
        img = cv2.cvtColor(cv2.imread(os.path.join(ds.img_dir, ds.coco.imgs[iid]["file_name"])), cv2.COLOR_BGR2RGB)
        ax.imshow(img); ax.axis("off")
        for r in by_img[iid]:
            k = np.array(r["keypoints"]).reshape(17, 3)
            for a, b in SKELETON:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    ax.plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], lw=2, color="#2ca02c" if a % 2 else "#1f77b4")
            m = k[:, 2] > 0.3
            ax.scatter(k[m, 0], k[m, 1], s=9, c="#d62728", zorder=3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "val2017_predictions.png"), dpi=100); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for bb in args.backbones:
        p = os.path.join(RES, f"history_{bb}.json")
        if os.path.exists(p):
            ax.plot(range(1, len(json.load(open(p))["loss"]) + 1), json.load(open(p))["loss"], marker="o", label=bb)
    ax.set(xlabel="epoch", ylabel="heatmap MSE", title="Training loss"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, "training_curves.png"), dpi=110)


if __name__ == "__main__":
    main()
