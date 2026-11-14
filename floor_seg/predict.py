"""Floor mask and overlay for images (visual check; no labels needed).

    python -m floor_seg.predict --model pollen-robotics/reachy-mini-floor-seg --out out/ img1.jpg img2.jpg
    python -m floor_seg.predict --model floor.onnx --out out/ frames/*.jpg

Model: .onnx (ONNX Runtime), .pt state dict, save_pretrained folder or Hub repo id.
Per image, writes <name>_floor.png (255 floor, 0 not floor, image size) and <name>_overlay.jpg
(floor tinted green, outline yellow). A pixel is floor when its floor probability > --thr.
"""
import argparse
import os

import cv2
import numpy as np

from floor_seg.infer import floor_prob, load_model
from floor_seg.model import THRESHOLD


def overlay(bgr, floor):
    """Floor pixels tinted green, mask outline in yellow (BGR)."""
    ov = bgr.copy()
    ov[floor] = (0.6 * ov[floor] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    cnt, _ = cv2.findContours(floor.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (0, 255, 255), 1)
    return ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=THRESHOLD)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model)
    for path in args.images:
        bgr = cv2.imread(path)
        floor = floor_prob(model, bgr) > args.thr
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, name + "_floor.png"), floor.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"), overlay(bgr, floor))
        print(f"{path}: floor {floor.mean():.1%}")


if __name__ == "__main__":
    main()
