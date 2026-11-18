"""Train SimpleBaseline on COCO-2017 train keypoints.

    python src/train.py --backbone resnet50 --epochs 20
"""
import argparse
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import CocoKeypoints
from model import SimpleBaseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collate(batch):
    x, t, v, _ = zip(*batch)
    return torch.stack(x), torch.stack(t), torch.stack(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hours", type=float, default=None, help="time budget; overrides --epochs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    ds = CocoKeypoints("train")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    model = SimpleBaseline(args.backbone).to(dev).to(memory_format=torch.channels_last)
    if args.hours and not args.resume:
        # time budget: measure the real throughput (data loading included), then pick the epoch count
        init = copy.deepcopy(model.state_dict())
        probe = torch.optim.SGD(model.parameters(), lr=0.0)
        it_dl = iter(dl)
        for k in range(60):
            if k == 10:
                torch.cuda.synchronize(); t_start = time.time()
            x, t, v = next(it_dl)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x.to(dev).to(memory_format=torch.channels_last))
            l = ((out.float() - t.to(dev)) ** 2).mean()
            probe.zero_grad(); l.backward(); probe.step()
        torch.cuda.synchronize()
        sec_it = (time.time() - t_start) / 50
        model.load_state_dict(init); del probe, it_dl
        args.epochs = max(1, int(args.hours * 3600 / (sec_it * len(dl))))
        print(f"{sec_it:.3f} s/it -> {args.epochs} epochs in {args.hours} h", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.03)
    scaler = torch.amp.GradScaler()
    ckpt = os.path.join(ROOT, "checkpoints", f"simplebaseline_{args.backbone}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    start, hist = 0, {"loss": []}
    if args.resume and os.path.exists(ckpt + ".state"):
        st = torch.load(ckpt + ".state", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, hist = st["epoch"], st["hist"]

    t0 = time.time()
    for ep in range(start, args.epochs):
        model.train()
        s, n = 0.0, 0
        for it, (x, t, v) in enumerate(dl):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            t, v = t.to(dev, non_blocking=True), v.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            # joints-MSE with target weights (invisible / out-of-crop joints are ignored)
            loss = 0.5 * (((out.float() - t) ** 2).mean((2, 3)) * v).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            s += loss.item(); n += 1
            if it % 500 == 0:
                print(f"ep {ep + 1} it {it}/{len(dl)} loss {s / n:.6f} {(time.time() - t0) / 60:.1f} min", flush=True)
        hist["loss"].append(s / n)
        torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "hist": hist}, ckpt + ".state")
        print(f"epoch {ep + 1} done, loss {s / n:.6f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(hist, open(os.path.join(ROOT, "results", f"history_{args.backbone}.json"), "w"))


if __name__ == "__main__":
    main()
