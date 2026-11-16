"""Train the floor segmentation model on synthetic renders (floor_seg/thor_render.py output). Needs a CUDA GPU.

    # v2 recipe (released model): ProcTHOR renders + floor retexturing, 30 epochs
    python -m floor_seg.train --data data/synth/thor_v1 --out models/floor_v2 --tex data/textures/ambientcg

Split by house (house % 10 == 0 -> validation), so validation measures unseen houses.
Augmentation imitates our camera, not the renderer: colour / gamma, sensor noise, defocus and motion blur,
JPEG artefacts, small zoom-crops, horizontal flip, roll +-15 deg (exposed corners ignored in the loss).
With --tex, floors are first retextured (tiles, rugs, sun patches: floor_seg/floor_tex.py), using the lens
calibration the renders were made with (--calib, default: the bundled Reachy Mini calibration).
Metrics on validation: floor IoU, and false-floor rate = obstacle pixels predicted as floor / obstacle
pixels (the error that drives the robot into things).
--under (thor_v3 renders, with _under.png): 3-channel model, the third channel says "floor with something lower
than the robot above it" (see model.FloorNet); its loss is on floor pixels only, its metrics (under IoU,
precision, recall) are reported but best.pt is still chosen on floor IoU.

    python -m floor_seg.train --data data/synth/thor_v3 --out models/floor_v5_under --tex data/textures/ambientcg --under
Saves best.pt (by floor IoU), last.pt, floor.onnx (fp32, from best) and metrics.jsonl.
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from floor_seg import DEFAULT_CALIB
from floor_seg.export_onnx import export
from floor_seg.floor_tex import FloorTex
from floor_seg.model import IN_H, IN_W, MEAN, STD, FloorNet

cv2.setNumThreads(1)  # parallelism comes from DataLoader workers; per-worker OpenCV threads only contend
ROT_P, ROT_DEG = 0.7, 15.0
# One label image carries both masks, so every geometric augmentation transforms them together:
NOT_FLOOR, FLOOR, UNDER, FLOOR_NA = 0, 1, 2, 3  # UNDER = floor with something low above it; FLOOR_NA = floor,
IGNORE = 255                                     # 'under' unknown (no _under.png, e.g. thor_v1); IGNORE: no label


def encode_labels(floor, under=None):
    """floor, under: 0/1 masks -> one uint8 label image (NOT_FLOOR / FLOOR / UNDER, or FLOOR_NA if under is None)."""
    if under is None:
        return (floor * FLOOR_NA).astype(np.uint8)
    return (floor * np.where(under > 0, UNDER, FLOOR)).astype(np.uint8)


def seg_loss(logits, lab, w_under=1.0):
    """Floor: cross-entropy on channels 0-1 (target: any floor label). Under (3-channel model): binary
    cross-entropy on channel 2, on pixels labelled FLOOR or UNDER only (not on obstacles, unknown or ignored)."""
    floor_t = torch.where(lab == IGNORE, lab, (lab != NOT_FLOOR).long())
    loss = torch.nn.functional.cross_entropy(logits[:, :2], floor_t, ignore_index=IGNORE)
    if logits.shape[1] > 2:
        m = (lab == FLOOR) | (lab == UNDER)
        if m.any():
            loss = loss + w_under * torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 2][m].float(), (lab[m] == UNDER).float())
    return loss


class Renders(Dataset):
    def __init__(self, root, names, train, poses=None, ftex=None, under=False):
        """poses: name -> (pitch deg, roll deg, camera height m); ftex: FloorTex for floor retexturing (train only,
        needs poses); under: also read <name>_under.png (labels: encode_labels)."""
        self.root, self.names, self.train = root, names, train
        self.poses, self.ftex, self.under = poses, ftex, under

    def __len__(self):
        return len(self.names)

    def augment(self, img, mask, rng):
        h, w = mask.shape
        if rng.random() < 0.5:  # small zoom-crop (keeps aspect)
            s = rng.uniform(0.8, 1.0)
            cw, ch = int(w * s), int(h * s)
            x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
            img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
        if rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if rng.random() < ROT_P:  # roll: our unit's head rests tilted, the base tilts; be roll-invariant
            h, w = mask.shape
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-ROT_DEG, ROT_DEG), 1.0)
            img = cv2.warpAffine(np.ascontiguousarray(img), rot, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(np.ascontiguousarray(mask), rot, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)  # exposed corners: no label
        img = img.astype(np.float32) / 255
        # colour: per-channel gain (white balance), brightness, contrast, saturation, gamma
        img = img * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
        img = (img - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15)
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.6, 1.3)
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.3:  # defocus
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < 0.2:  # motion blur, random direction
            k = rng.choice([5, 7, 9])
            kern = np.zeros((k, k), np.float32)
            kern[k // 2] = 1 / k
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
            kern = cv2.warpAffine(kern, rot, (k, k))
            img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))
        img = img + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, 0.03), img.shape)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if rng.random() < 0.5:  # JPEG artefacts
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        base = os.path.join(self.root, self.names[i])
        img = cv2.imread(base + ".jpg")  # BGR
        floor = (cv2.imread(base + "_floor.png", 0) > 127).astype(np.uint8)
        mask = floor  # 0 / 1, IGNORE added by augment
        if self.under:
            u = cv2.imread(base + "_under.png", 0)
            mask = encode_labels(floor, None if u is None else (u > 127).astype(np.uint8))
        if self.train:
            rng = random.Random(random.getrandbits(32))
            if self.ftex is not None:  # first, while pixels still match the stored depth and pose
                depth = cv2.imread(base + "_depth.png", cv2.IMREAD_UNCHANGED)
                pitch, roll, h = self.poses[self.names[i]]
                img = self.ftex.apply(img, floor, depth, pitch, roll, rng, cam_h=h)
            img, mask = self.augment(img, mask, rng)
        img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
        mask = cv2.resize(mask, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        x = ((img - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(x.copy()), torch.from_numpy(mask.astype(np.int64))


def evaluate(model, loader, dev):
    """Floor: IoU, false-floor and missed-floor rates (labels 0 = not floor, any other floor label = floor).
    Under (3-channel model): IoU, precision, recall of sigmoid > 0.5 over GT FLOOR / UNDER pixels."""
    model.eval()
    tp = fp = fn = obst = 0
    utp = ufp = ufn = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            z = model(x.to(dev)).float().cpu()
            pred = z[:, :2].argmax(1)
            gt = (y != NOT_FLOOR) & (y != IGNORE)
            tp += ((pred == 1) & gt).sum().item()
            fp += ((pred == 1) & (y == NOT_FLOOR)).sum().item()
            fn += ((pred == 0) & gt).sum().item()
            obst += (y == NOT_FLOOR).sum().item()
            if z.shape[1] > 2:
                m = (y == FLOOR) | (y == UNDER)
                pu, gu = (z[:, 2] > 0)[m], (y == UNDER)[m]
                utp, ufp, ufn = utp + (pu & gu).sum().item(), ufp + (pu & ~gu).sum().item(), ufn + (~pu & gu).sum().item()
    m = {"floor_iou": tp / max(tp + fp + fn, 1), "false_floor_rate": fp / max(obst, 1),
         "missed_floor_rate": fn / max(tp + fn, 1)}
    if utp + ufp + ufn:
        m.update(under_iou=utp / (utp + ufp + ufn), under_precision=utp / max(utp + ufp, 1),
                 under_recall=utp / max(utp + ufn, 1))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tex", help="texture root (data/textures/ambientcg): enables floor retexturing")
    ap.add_argument("--calib", default=DEFAULT_CALIB, help="lens calibration the renders were made with (for --tex)")
    ap.add_argument("--kinds", nargs="+", help="keep only these view kinds of stats.jsonl (thor_v2: floor, table)")
    ap.add_argument("--under", action="store_true", help="3-channel model with the 'under' output (needs _under.png)")
    ap.add_argument("--w-under", type=float, default=1.0, help="weight of the under loss (with --under)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)

    stats = [json.loads(line) for line in open(os.path.join(args.data, "stats.jsonl"))]
    if args.kinds:  # thor_v1 has no "kind": all its views are floor views
        stats = [s for s in stats if s.get("kind", "floor") in args.kinds]
    train_names = [s["name"] for s in stats if s["house"] % 10 != 0]
    val_names = [s["name"] for s in stats if s["house"] % 10 == 0]
    print(f"train {len(train_names)} views, val {len(val_names)} views (by house)", flush=True)
    ftex = FloorTex(args.tex, args.calib) if args.tex else None
    poses = {s["name"]: (s["pitch"], s["roll"], s["h"]) for s in stats}
    tl = DataLoader(Renders(args.data, train_names, True, poses, ftex, under=args.under), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
    vl = DataLoader(Renders(args.data, val_names, False, under=args.under), batch_size=args.batch,
                    num_workers=args.workers)

    dev = "cuda"
    model = FloorNet(under=args.under).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best = -1.0
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = seg_loss(model(x), y, args.w_under)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        m = evaluate(model, vl, dev)
        m.update(epoch=ep, loss=tot / len(tl), s=round(time.time() - t0, 1))
        log.write(json.dumps(m) + "\n")
        log.flush()
        under = (f" | under IoU {m['under_iou']:.4f} P {m['under_precision']:.3f} R {m['under_recall']:.3f}"
                 if "under_iou" in m else "")
        print(f"ep {ep:2d} loss {m['loss']:.4f} | val floor IoU {m['floor_iou']:.4f} false-floor "
              f"{m['false_floor_rate']:.4f} missed-floor {m['missed_floor_rate']:.4f}{under} | {m['s']} s", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if m["floor_iou"] > best:
            best = m["floor_iou"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt")))
    export(model, os.path.join(args.out, "floor.onnx"))
    print(f"best val floor IoU {best:.4f}; exported floor.onnx", flush=True)


if __name__ == "__main__":
    main()
