"""Score floor models on hand-checked real labels (floor_seg.label_real output: <name>.jpg + <name>_floor.png).

    python -m floor_seg.eval_real --data data/real/test_v1 models/floor_v1/best.pt models/floor_v2/best.pt
    python -m floor_seg.eval_real --data data/real/test_v1 --thr 0.5 0.6 0.7 0.8 models/floor_v2/floor.onnx
    python -m floor_seg.eval_real --data data/real/test_v1 --thr 0.8 pollen-robotics/reachy-mini-floor-seg

Models: .pt state dicts, save_pretrained folders / Hub repo ids, or .onnx exports (ONNX Runtime CPU).
A pixel is floor when its floor probability > thr (0.5 = argmax).
Metrics pooled over all labelled pixels (255 = ignore), same definitions as floor_seg/train.py:
floor IoU, false-floor rate (obstacle pixels predicted floor: drives into things), missed-floor rate.
Also mean per-frame floor IoU and the worst frames (at the first threshold).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from floor_seg.infer import floor_prob, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    names = sorted(os.path.basename(p)[:-len("_floor.png")] for p in glob.glob(os.path.join(args.data, "*_floor.png")))
    print(f"{len(names)} labelled frames")
    for mp in args.models:
        model = load_model(mp)
        frames = [(n, floor_prob(model, cv2.imread(os.path.join(args.data, n + ".jpg"))),
                   cv2.imread(os.path.join(args.data, n + "_floor.png"), 0)) for n in names]
        print(mp)
        for k, thr in enumerate(args.thr):
            tp = fp = fn = obst = 0
            per = []
            for n, prob, y in frames:
                p = prob > thr
                a, b, c = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
                tp, fp, fn, obst = tp + a, fp + b, fn + c, obst + (y == 0).sum()
                per.append((a / max(a + b + c, 1), n))
            print(f"  thr {thr:.2f}: floor IoU {tp / (tp + fp + fn):.3f}  false-floor {fp / obst:.3f}  "
                  f"missed-floor {fn / (tp + fn):.3f}  | mean per-frame IoU {np.mean([v for v, _ in per]):.3f}")
            if k == 0:
                print("    worst:", ", ".join(f"{n} {v:.2f}" for v, n in sorted(per)[:args.worst]))


if __name__ == "__main__":
    main()
