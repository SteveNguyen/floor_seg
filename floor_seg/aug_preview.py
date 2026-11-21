"""Preview grid of the structure augmentations (floor_seg/aug_structures.py) on a few renders, with labels.

    python -m floor_seg.aug_preview --data data/synth/thor_v3 data/synth/thor_v2 --n 6 --out aug_preview.jpg

One row per render (floor views whose files exist locally), columns: original, plinth, wall tiles, flush
threshold, level up, level down, level +-1 cm (drivable), sun streak + specular highlight, all (training defaults).
Each cell: the image above its label
(green floor, red obstacle, blue 'under' when the render has _under.png). "n/a": nothing could be placed
(e.g. no wall base in view). --tex also runs the floor retexturing first, as in training.

A training configuration only (--all-only, 4 views per row), with overrides as train.py's flags set them:

    python -m floor_seg.aug_preview --data data/synth/thor_v3 --n 12 --all-only --set p_plinth=0 p_wall_tiles=0 --out a.jpg
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

OFF = dict(p_plinth=0.0, p_threshold=0.0, p_level=0.0, p_sun=0.0, p_specular=0.0, p_wall_tiles=0.0,
           p_plinth_render=0.0, p_door_threshold=0.0)
# plinth: the rendered one recoloured where the render has one (_plinth.png), else painted; threshold: in a real
# doorway where the render lists them, else on open floor
COLUMNS = [("original", OFF), ("plinth", dict(OFF, p_plinth=1.0, p_plinth_render=1.0)),
           ("wall tiles", dict(OFF, p_wall_tiles=1.0)),
           ("flush threshold", dict(OFF, p_threshold=1.0, p_door_threshold=1.0)),
           ("level up", dict(OFF, p_level=1.0, p_level_up=1.0, p_level_flat=0.0)),
           ("level down", dict(OFF, p_level=1.0, p_level_up=0.0, p_level_flat=0.0)),
           ("level +-1 cm", dict(OFF, p_level=1.0, p_level_flat=1.0)),
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
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="StructAugCfg overrides for the 'all' column (e.g. p_plinth=0)")
    ap.add_argument("--all-only", action="store_true", help="only the 'all' column, 4 views per row")
    args = ap.parse_args()
    over = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    columns = [(t, kw if kw else over) for t, kw in COLUMNS]
    if args.all_only:
        columns = [("all", over)]

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
    augs = [StructAug(replace(StructAugCfg(), **kw), args.calib) for _, kw in columns]

    rows, cells_all = [], []
    for k, (base, s) in enumerate(views):
        img0 = cv2.imread(base + ".jpg")
        floor = (cv2.imread(base + "_floor.png", 0) > 127).astype(np.uint8)
        lab0 = floor.copy()
        if os.path.exists(base + "_under.png"):
            lab0[(cv2.imread(base + "_under.png", 0) > 127) & (floor > 0)] = UNDER
        depth = cv2.imread(base + "_depth.png", cv2.IMREAD_UNCHANGED)
        plinth = cv2.imread(base + "_plinth.png", 0) if os.path.exists(base + "_plinth.png") else None
        src = img0.copy()   # before retexturing: the rendered plinth's own light
        if ftex is not None:
            img0 = ftex.apply(img0, floor, depth, s["pitch"], s["roll"], random.Random(k), cam_h=s["h"])
        cells = []
        for (title, kw), sa in zip(columns, augs):
            img, lab, info = sa.apply(img0, lab0, depth, s["pitch"], s["roll"], s["h"], random.Random(1000 * args.seed + k),
                                      plinth=plinth, plinth_src=src, doorways=s.get("doorways"))
            cell = np.vstack([img, label_view(img, lab)])
            painted = {key: v for key, v in info.items() if v}
            text = title if (kw is over or kw is OFF or painted) else f"{title}: n/a"
            if "plinth" in painted:
                text += f" {painted['plinth']['h'] * 100:.0f} cm {painted['plinth']['style']}" + \
                    (" rendered" if painted["plinth"].get("rendered") else "")
            if "wall_tiles" in painted:
                text += f" tiles {painted['wall_tiles']['h']:.1f} m"
            if "threshold" in painted:
                text += f" {painted['threshold']['h'] * 100:.1f} cm" + \
                    (" doorway" if painted["threshold"].get("doorway") else "")
            if "level" in painted:
                lv = painted["level"]
                text += f" {lv['dz'] * 100:+.1f} cm" + (" door" if lv["door"] else "") + (" other" if lv["other"] else "")
            cv2.putText(cell, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(cell, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cells.append(cv2.resize(cell, (cell.shape[1] // 2, cell.shape[0] // 2), interpolation=cv2.INTER_AREA))
        if args.all_only:
            cells_all += cells
            continue
        rows.append(np.hstack(cells))
        cv2.putText(rows[-1], os.path.basename(base), (8, rows[-1].shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    if args.all_only:   # 4 views per row, the last row padded with black
        cells_all += [np.zeros_like(cells_all[0])] * (-len(cells_all) % 4)
        rows = [np.hstack(cells_all[i:i + 4]) for i in range(0, len(cells_all), 4)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(args.out)


if __name__ == "__main__":
    main()
