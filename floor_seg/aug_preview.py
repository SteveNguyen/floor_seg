"""Preview grid of the structure augmentations (floor_seg/aug_structures.py) on a few renders, with labels.

    python -m floor_seg.aug_preview --data data/synth/thor_v3 data/synth/thor_v2 --n 6 --out aug_preview.jpg

One row per render (floor views whose files exist locally), columns: original, plinth, flush threshold,
raised step, sun streak + specular highlight, all (training defaults). Each cell: the image above its label
(green floor, red obstacle, blue 'under' when the render has _under.png). "n/a": nothing could be placed
(e.g. no wall base in view). --tex also runs the floor retexturing first, as in training.
"""
import argparse
import json
import os
import random
from dataclasses import replace

import cv2
import numpy as np

from floor_seg import DEFAULT_CALIB
from floor_seg.aug_structures import UNDER, StructAug, StructAugCfg

OFF = dict(p_plinth=0.0, p_threshold=0.0, p_sun=0.0, p_specular=0.0)
COLUMNS = [("original", OFF), ("plinth", dict(OFF, p_plinth=1.0)),
           ("flush threshold", dict(OFF, p_threshold=1.0, p_step=0.0)),
           ("raised step", dict(OFF, p_threshold=1.0, p_step=1.0)),
           ("sun + specular", dict(OFF, p_sun=1.0, p_specular=1.0)), ("all (defaults)", {})]


def label_view(img, lab):
    """Label overlay: floor green, obstacle red, under blue."""
    col = np.zeros_like(img)
    col[lab == 0] = (0, 0, 255)
    col[(lab != 0) & (lab != UNDER)] = (0, 200, 0)
    col[lab == UNDER] = (255, 80, 0)
    return cv2.addWeighted(img, 0.45, col, 0.55, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="grid image (.jpg)")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--tex", help="texture root: retexture floors first (floor_seg.floor_tex)")
    args = ap.parse_args()

    views = []
    for root in args.data:
        for line in open(os.path.join(root, "stats.jsonl")):
            s = json.loads(line)
            base = os.path.join(root, s["name"])
            if s.get("kind", "floor") == "floor" and os.path.exists(base + ".jpg") and os.path.exists(base + "_depth.png"):
                views.append((base, s))
    rng = random.Random(args.seed)
    views = rng.sample(views, min(args.n, len(views)))
    print(f"{len(views)} views")
    ftex = None
    if args.tex:
        from floor_seg.floor_tex import FloorTex
        ftex = FloorTex(args.tex, args.calib)
    augs = [StructAug(replace(StructAugCfg(), **kw), args.calib) for _, kw in COLUMNS]

    rows = []
    for k, (base, s) in enumerate(views):
        img0 = cv2.imread(base + ".jpg")
        floor = (cv2.imread(base + "_floor.png", 0) > 127).astype(np.uint8)
        lab0 = floor.copy()
        if os.path.exists(base + "_under.png"):
            lab0[(cv2.imread(base + "_under.png", 0) > 127) & (floor > 0)] = UNDER
        depth = cv2.imread(base + "_depth.png", cv2.IMREAD_UNCHANGED)
        if ftex is not None:
            img0 = ftex.apply(img0, floor, depth, s["pitch"], s["roll"], random.Random(k), cam_h=s["h"])
        cells = []
        for (title, kw), sa in zip(COLUMNS, augs):
            img, lab, info = sa.apply(img0, lab0, depth, s["pitch"], s["roll"], s["h"], random.Random(1000 * args.seed + k))
            cell = np.vstack([img, label_view(img, lab)])
            painted = {key: v for key, v in info.items() if v}
            text = title if (not kw or kw is OFF or painted) else f"{title}: n/a"
            if "plinth" in painted:
                text += f" {painted['plinth']['h'] * 100:.0f} cm" + (" plain" if painted["plinth"]["plain"] else "")
            if "threshold" in painted:
                text += f" {painted['threshold']['h'] * 100:.1f} cm"
            cv2.putText(cell, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(cell, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cells.append(cv2.resize(cell, (cell.shape[1] // 2, cell.shape[0] // 2), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(cells))
        cv2.putText(rows[-1], os.path.basename(base), (8, rows[-1].shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(args.out)


if __name__ == "__main__":
    main()
