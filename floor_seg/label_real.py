"""Semi-automatic floor labels for real frames: SAM2 regions, pre-labelled by a floor model, corrected by review.

    # 1. pick frames, segment with SAM2, pre-label each region (mean floor prob of the model > 0.5)
    #    (each spot folder holds a recording sweep as frames/*.jpg; needs a GPU and `transformers`)
    python -m floor_seg.label_real prelabel --model models/floor_v1/best.pt \
        --out data/real/test_v1 --per-spot 5 recordings/*/spot*
    # 2. review with the click-to-label page (python -m floor_seg.label_page data/real/test_v1), which writes
    #    edits.json + reviewed.json; or review data/real/test_v1/review/*.jpg and write edits.json by hand:
    #    {"<name>": {"all": "floor"|"obstacle" (optional reset), "floor": [ids], "obstacle": [ids], "ignore": [ids],
    #                "poly": [{"label": "obstacle", "pts": [[x, y], ...]}]}}   (polygons in 640x360 px, applied last)
    # 3. write final labels <name>_floor.png (0 obstacle, 1 floor, 255 ignore)
    python -m floor_seg.label_real apply --out data/real/test_v1 [--reviewed-only]

Pixels that no SAM2 region covers become extra regions when they form blobs (>= MIN_GAP px); thin seams are ignore.
Label rule (project decision): anything > 1 cm above the floor is an obstacle; rugs (edges and surface), cables
and door thresholds are floor.
SAM2 weights: facebook/sam2.1-hiera-large (Apache-2.0).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from floor_seg.model import load_torch, preprocess

W, H = 640, 360        # labels live at the model's working resolution (camera frames resized)
IGNORE = 255
MIN_ID_AREA = 0.003    # regions smaller than this fraction of the image get no id printed (label from the model)
MIN_GAP = 150          # px; uncovered blobs at least this big become their own region, smaller stay ignore


def sharpness(path):
    g = cv2.resize(cv2.imread(path, 0), (W, H), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def pick_frames(spot, n, window=15):
    """n frames spread over the spot's sweep (first 10 % skipped: head settling); at each sample point the
    sharpest frame within +-window (sweep frames are often motion-blurred, unfair and hard to label)."""
    fs = sorted(glob.glob(os.path.join(spot, "frames", "*.jpg")))
    picks = []
    for c in np.linspace(len(fs) // 10, len(fs) - 1, n).astype(int):
        near = fs[max(0, c - window):c + window + 1]
        picks.append(max(near, key=sharpness))
    return picks


def region_map(masks):
    """Paint SAM2 masks largest first so smaller (finer) regions win; 0 = uncovered."""
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())
    reg = np.zeros((H, W), np.uint16)
    for k, i in enumerate(order, start=1):
        reg[masks[i]] = k
    # uncovered blobs (baseboards, cables, fur...) become regions too; thin seams (< MIN_GAP px) stay 0 = ignore
    n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(
        (reg == 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), connectivity=4)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= MIN_GAP:
            reg[cc == c] = reg.max() + 1
    # relabel to consecutive ids, small ids for big regions (easier to read)
    ids, counts = np.unique(reg[reg > 0], return_counts=True)
    lut = np.zeros(reg.max() + 1, np.uint16)
    for new, old in enumerate(ids[np.argsort(-counts)], start=1):
        lut[old] = new
    return lut[reg]


def review_image(rgb, reg, lab):
    """Left: frame. Right: floor regions tinted green, obstacle red, region ids at centroids."""
    over = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[lab == 1] = (0, 200, 0)
    tint[lab == 0] = (220, 0, 0)
    m = lab != IGNORE
    over[m] = (0.55 * rgb[m] + 0.45 * tint[m]).astype(np.uint8)
    edges = cv2.morphologyEx((reg % 251).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0
    over[edges] = (255, 255, 0)
    big = cv2.resize(over, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)  # 2x so ids are readable
    for r in np.unique(reg[reg > 0]):
        m = reg == r
        if m.mean() < MIN_ID_AREA:
            continue
        # label position: the region pixel furthest from its border (inside even for non-convex regions)
        dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(big, str(r), (2 * x - 10, 2 * y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th)
    left = cv2.resize(rgb, (2 * W, 2 * H), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, big], axis=1)


def labels_from(reg, region_lab):
    lab = np.full((H, W), IGNORE, np.uint8)
    for r, v in region_lab.items():
        lab[reg == int(r)] = v
    return lab


def prelabel(args):
    from transformers import pipeline
    gen = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)
    model = load_torch(args.model)
    os.makedirs(os.path.join(args.out, "review"), exist_ok=True)
    for spot in args.spots:
        for f in pick_frames(spot, args.per_spot):
            name = f"{os.path.basename(spot)}_{os.path.splitext(os.path.basename(f))[0]}"
            bgr = cv2.resize(cv2.imread(f), (W, H), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            # looser than the SAM defaults: more, finer masks, so fewer regions mix floor and obstacle
            out = gen(Image.fromarray(rgb), points_per_batch=64, pred_iou_thresh=0.6, stability_score_thresh=0.85)
            masks = [np.asarray(m, bool) for m in out["masks"]]
            reg = region_map(masks)
            with torch.no_grad():
                logits = model(torch.from_numpy(preprocess(rgb)))[0]
            prob = torch.softmax(logits, 0)[1].numpy()
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            region_lab = {int(r): int(prob[reg == r].mean() > 0.5) for r in np.unique(reg[reg > 0])}
            cv2.imwrite(os.path.join(args.out, name + ".jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(args.out, name + "_regions.png"), reg)
            json.dump(region_lab, open(os.path.join(args.out, name + "_init.json"), "w"))
            rev = review_image(rgb, reg, labels_from(reg, region_lab))
            cv2.imwrite(os.path.join(args.out, "review", name + ".jpg"), rev[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{name}: {len(masks)} masks, uncovered {np.mean(reg == 0):.1%}", flush=True)


def apply(args):
    """Write <name>_floor.png for every frame, or only args.names (a set) when given: label_page passes its reviewed
    frames, so unreviewed model guesses do not become labels. Returns the number of frames written."""
    edits_path = os.path.join(args.out, "edits.json")
    edits = json.load(open(edits_path)) if os.path.exists(edits_path) else {}
    names = getattr(args, "names", None)
    if getattr(args, "reviewed_only", False):  # list written by floor_seg.label_page
        names = set(json.load(open(os.path.join(args.out, "reviewed.json"))))
    written = 0
    for p in sorted(glob.glob(os.path.join(args.out, "*_init.json"))):
        name = os.path.basename(p)[:-len("_init.json")]
        if names is not None and name not in names:
            continue
        region_lab = json.load(open(p))
        e = edits.get(name, {})
        if e.get("all") in ("floor", "obstacle"):  # reset every region first, then the lists below apply
            region_lab = {r: int(e["all"] == "floor") for r in region_lab}
        for key, val in (("floor", 1), ("obstacle", 0), ("ignore", IGNORE)):
            for r in edits.get(name, {}).get(key, []):
                region_lab[str(r)] = val
        reg = cv2.imread(os.path.join(args.out, name + "_regions.png"), cv2.IMREAD_UNCHANGED)
        lab = labels_from(reg, region_lab)
        for poly in e.get("poly", []):  # for objects SAM2 merged into another region: hand polygon, W x H pixels
            val = {"floor": 1, "obstacle": 0, "ignore": IGNORE}[poly["label"]]
            cv2.fillPoly(lab, [np.array(poly["pts"], np.int32)], val)
        cv2.imwrite(os.path.join(args.out, name + "_floor.png"), lab)
        rgb = cv2.imread(os.path.join(args.out, name + ".jpg"))[:, :, ::-1]
        rev = review_image(np.ascontiguousarray(rgb), reg, lab)
        cv2.imwrite(os.path.join(args.out, "review", name + ".jpg"), rev[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        written += 1
    print(f"applied: {written} label pngs written ({len(edits)} frames in edits.json)", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prelabel")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-spot", type=int, default=5)
    p.add_argument("spots", nargs="+")
    a = sub.add_parser("apply")
    a.add_argument("--out", required=True)
    a.add_argument("--reviewed-only", action="store_true", help="only frames in reviewed.json (floor_seg.label_page)")
    args = ap.parse_args()
    prelabel(args) if args.cmd == "prelabel" else apply(args)


if __name__ == "__main__":
    main()
