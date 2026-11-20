"""Score floor models on hand-checked real labels (floor_seg.label_real output: <name>.jpg + <name>_floor.png).

    python -m floor_seg.eval_real --data data/real/test_v1 models/floor_v1/best.pt models/floor_v2/best.pt
    python -m floor_seg.eval_real --data data/real/test_v1 --thr 0.5 0.6 0.7 0.8 models/floor_v2/floor.onnx
    python -m floor_seg.eval_real --data data/real/test_v1 --thr 0.8 pollen-robotics/reachy-mini-floor-seg
    # per group too (CSV: name + category, or name + reason as written by frame mining)
    python -m floor_seg.eval_real --data data/real/test_v2/labels --groups data/real/test_v2/candidates.csv \
        data/real/test_v2/picked_band.csv --thr 0.5 0.7 models/hf_dev/v7_floor_under/best.pt

Models: .pt state dicts, save_pretrained folders / Hub repo ids, or .onnx exports (ONNX Runtime CPU).
A pixel is floor when its floor probability > thr (0.5 = argmax).
Metrics pooled over all labelled pixels (255 = ignore), same definitions as floor_seg/train.py:
floor IoU, false-floor rate (obstacle pixels predicted floor: drives into things), missed-floor rate.
Also mean per-frame floor IoU and the worst frames (at the first threshold).
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from floor_seg.infer import floor_prob, load_model


def read_groups(paths):
    """Frame name -> group, from CSV files with a `name` column and a `category` column (or a `reason` column,
    whose first word is taken: candidate lists from frame mining). Later files win."""
    groups = {}
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                g = r.get("category") or (r.get("reason") or "").replace("+", " ").replace(":", " ").split(" ")[0]
                if g:
                    groups[r["name"]] = g
    return groups


def scores(frames, thr):
    """Pooled floor IoU, false-floor rate, missed-floor rate (nan when undefined) and per-frame IoU."""
    tp = fp = fn = obst = 0
    per = []
    for n, prob, y in frames:
        p = prob > thr
        a, b, c = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
        tp, fp, fn, obst = tp + a, fp + b, fn + c, obst + (y == 0).sum()
        per.append((a / max(a + b + c, 1), n))
    div = lambda u, v: u / v if v else float("nan")  # noqa: E731
    return div(tp, tp + fp + fn), div(fp, obst), div(fn, tp + fn), per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--groups", nargs="*", default=[],
                    help="CSV files naming each frame's group (name + category or reason columns): also report "
                         "the metrics per group (frames in no file: 'other')")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    names = sorted(os.path.basename(p)[:-len("_floor.png")] for p in glob.glob(os.path.join(args.data, "*_floor.png")))
    groups = read_groups(args.groups)
    print(f"{len(names)} labelled frames")
    for mp in args.models:
        model = load_model(mp)
        frames = [(n, floor_prob(model, cv2.imread(os.path.join(args.data, n + ".jpg"))),
                   cv2.imread(os.path.join(args.data, n + "_floor.png"), 0)) for n in names]
        print(mp)
        for k, thr in enumerate(args.thr):
            iou, ff, mf, per = scores(frames, thr)
            print(f"  thr {thr:.2f}: floor IoU {iou:.3f}  false-floor {ff:.3f}  "
                  f"missed-floor {mf:.3f}  | mean per-frame IoU {np.mean([v for v, _ in per]):.3f}")
            if k == 0:
                print("    worst:", ", ".join(f"{n} {v:.2f}" for v, n in sorted(per)[:args.worst]))
            if groups:
                for g in sorted({groups.get(n, "other") for n, _, _ in frames}):
                    sub = [f for f in frames if groups.get(f[0], "other") == g]
                    iou, ff, mf, _ = scores(sub, thr)
                    print(f"    {g:<20} {len(sub):3d} frames: floor IoU {iou:.3f}  false-floor {ff:.3f}  "
                          f"missed-floor {mf:.3f}")


if __name__ == "__main__":
    main()
