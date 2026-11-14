"""Synthetic floor-segmentation data: low-camera views of ProcTHOR-10k houses (Apache-2.0).

    PYTHONPATH=. .venv-thor/bin/python -m floor_seg.thor_render --out data/synth/thor_v1 --houses 2000 --views 5

Needs its own env (AI2-THOR 5 wants Python 3.11; no torch needed):
    uv venv --python 3.11 .venv-thor && uv pip install --python .venv-thor/bin/python \
        ai2thor==5.0.0 prior==1.0.3 opencv-python-headless numpy
Rendering uses Vulkan headless (CloudRendering); the Unity build (~800 MB) is downloaded on first run.
--calib defaults to the bundled Reachy Mini camera calibration (floor_seg/assets/reachy_mini_camera.json).

Per house: drop clutter objects on free floor (AI2-THOR assets), randomize materials; per view: random
lighting, a camera at H_RANGE above the floor, head pitch 0..24 deg plus base tilt +-5, roll +-3, yaw
toward clutter 70 % of the time.

Floor label rule (project decision): a pixel is floor if its 3D point is < FLOOR_TOL (1 cm) above the
floor AND it belongs to a room floor surface (instance segmentation). The AND removes depth that passes
through glass (vases, windows). Frames where the two disagree (IoU < MIN_AGREE: see-through or broken
geometry) or that mostly face a wall are dropped.

Lens: output 640x360, K = calibrated K / 2 with the calibrated rational distortion. A larger pinhole
image with the same focal length is rendered and remapped (RGB bilinear, mask and depth nearest).
Noise / blur / colour jitter are left to training-time augmentation.

Output per view: <name>.jpg (RGB), <name>_floor.png (0/255), <name>_depth.png (uint16 mm, optical z);
one JSON line per view in stats.jsonl (written as it goes; re-running skips houses already done).
"""
import argparse
import json
import math
import os
import random

import cv2
import numpy as np
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from floor_seg import DEFAULT_CALIB

W, H = 640, 360
H_RANGE = (0.25, 0.55)  # camera height above the floor, m (Mini on the floor .. on the hoverboard)
FLOOR_TOL = 0.01
MIN_AGREE = 0.85
CLUTTER_PER_CELL = (0.03, 0.12)  # fraction of reachable 0.25 m cells that get an object
MIN_SPACING = 0.35  # m between clutter objects
CLUTTER_TYPES = ["Box", "Pillow", "Book", "GarbageCan", "LaundryHamper", "DogBed", "BasketBall", "TeddyBear",
                 "Boots", "Dumbbell", "Cloth", "Towel", "Newspaper", "GarbageBag", "VacuumCleaner", "HousePlant",
                 "Footstool", "Ottoman", "Stool", "WateringCan", "TennisRacket", "BaseballBat", "Laptop", "Bowl",
                 "Pot", "Vase", "Statue", "Cart", "Kettle", "Mug", "Bottle", "Plunger", "TissueBox", "Desktop"]


def lens_maps(calib_path):
    """Output pixel -> render pixel maps, render size and vertical FOV for the calibrated lens at W x H."""
    cal = json.load(open(calib_path))
    sx, sy = W / cal["image_resolution"][0], H / cal["image_resolution"][1]
    K = np.array(cal["K"]) * np.array([[sx], [sy], [1]])
    D = np.array(cal["D"])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], -1)[:, None, :]
    xn = cv2.undistortPoints(pts, K, D, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6))[:, 0]
    f = K[1, 1]
    wr = 2 * int(math.ceil(np.abs(xn[:, 0]).max() * f)) + 4
    hr = 2 * int(math.ceil(np.abs(xn[:, 1]).max() * f)) + 4
    map_x = (xn[:, 0] * f + wr / 2).reshape(H, W).astype(np.float32)
    map_y = (xn[:, 1] * f + hr / 2).reshape(H, W).astype(np.float32)
    return map_x, map_y, wr, hr, f, math.degrees(2 * math.atan(hr / 2 / f))


def height_above_floor(depth, f, cam_y, pitch, roll):
    """Height (m) of each render pixel's 3D point (Unity world, y up, floor at 0); depth = optical z."""
    hr, wr = depth.shape
    v, u = np.mgrid[0:hr, 0:wr].astype(np.float32)
    x = (u - wr / 2) / f * depth
    y = -(v - hr / 2) / f * depth  # Unity camera: y up
    r, p = math.radians(roll), math.radians(pitch)
    y1 = x * math.sin(r) + y * math.cos(r)  # roll about the optical axis
    return cam_y + y1 * math.cos(p) - depth * math.sin(p)  # pitch about x, positive = look down


def add_clutter(house, reach, by_type, rng):
    """Copy of the house with clutter objects resting on free floor (positions from reachable cells)."""
    house = json.loads(json.dumps(house))
    placed = []
    n_target = max(3, int(len(reach) * rng.uniform(*CLUTTER_PER_CELL)))
    rng.shuffle(reach)
    for p in reach:
        if len(placed) >= n_target:
            break
        x, z = p["x"] + rng.uniform(-0.1, 0.1), p["z"] + rng.uniform(-0.1, 0.1)
        if any((x - a) ** 2 + (z - b) ** 2 < MIN_SPACING ** 2 for a, b in placed):
            continue
        t = rng.choice(sorted(by_type))
        aid, bb = rng.choice(by_type[t])
        house["objects"].append({"assetId": aid, "id": f"{t}|clutter|{len(placed)}", "kinematic": True,
                                 "position": {"x": x, "y": -bb["min"]["y"] + 0.002, "z": z},
                                 "rotation": {"x": 0, "y": rng.uniform(0, 360), "z": 0}})
        placed.append((x, z))
    return house, placed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--houses", type=int, default=2000)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats_path = os.path.join(args.out, "stats.jsonl")
    done = set()
    if os.path.exists(stats_path):
        done = {json.loads(line)["house"] for line in open(stats_path)}

    map_x, map_y, wr, hr, f, vfov = lens_maps(args.calib)
    print(f"render {wr}x{hr} vfov {vfov:.1f} deg -> {W}x{H} distorted; {len(done)} houses already done", flush=True)
    houses = prior.load_dataset("procthor-10k")["train"]
    order = list(range(len(houses)))
    random.Random(args.seed).shuffle(order)
    ctrl = Controller(platform=CloudRendering, scene=houses[0], width=wr, height=hr, renderDepthImage=True,
                      renderInstanceSegmentation=True, snapToGrid=False)
    db = ctrl.step("GetAssetDatabase").metadata["actionReturn"]
    by_type = {}
    for aid, a in db.items():
        if a.get("objectType") in CLUTTER_TYPES:
            by_type.setdefault(a["objectType"], []).append((aid, a["boundingBox"]))

    stats_f = open(stats_path, "a")
    for hi in order[:args.houses]:
        if hi in done:
            continue
        rng = random.Random(args.seed * 100003 + hi)  # per house: reproducible and resumable
        try:
            ctrl.reset(scene=houses[hi])
            reach = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []
            if len(reach) < 10:
                continue
            house, placed = add_clutter(houses[hi], list(reach), by_type, rng)
            ctrl.reset(scene=house)
            ctrl.step("RandomizeMaterials")
            reach = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []  # around the clutter
            if not reach:
                continue
            ctrl.step("AddThirdPartyCamera", position=dict(x=0, y=1, z=0), rotation=dict(x=0, y=0, z=0),
                      fieldOfView=vfov)
        except Exception as e:  # a broken house must not stop a long run
            print(f"house {hi}: setup failed ({e})", flush=True)
            continue
        n_ok, tries = 0, 0
        while n_ok < args.views and tries < 5 * args.views:
            tries += 1
            ctrl.step("RandomizeLighting", synchronized=False)
            pos = rng.choice(reach)
            h = rng.uniform(*H_RANGE)
            pitch = rng.uniform(0, 24) + rng.uniform(-5, 5)
            roll = rng.uniform(-3, 3)
            if placed and rng.random() < 0.7:
                cx, cz = rng.choice(placed)
                yaw = math.degrees(math.atan2(cx - pos["x"], cz - pos["z"])) + rng.uniform(-25, 25)
            else:
                yaw = rng.uniform(0, 360)
            ev = ctrl.step("UpdateThirdPartyCamera", thirdPartyCameraId=0, position=dict(x=pos["x"], y=h, z=pos["z"]),
                           rotation=dict(x=pitch, y=yaw, z=roll), fieldOfView=vfov)
            rgb_r = ev.third_party_camera_frames[0][:, :, :3]  # RGBA -> RGB
            depth_r = ev.third_party_depth_frames[0]
            if (depth_r < 0.8).mean() > 0.6:  # staring at a wall / furniture
                continue
            seg = ev.third_party_instance_segmentation_frames[0][:, :, :3].astype(np.int32)
            seg_code = (seg[:, :, 0] << 16) | (seg[:, :, 1] << 8) | seg[:, :, 2]
            floor_codes = [(c[0] << 16) | (c[1] << 8) | c[2] for c, oid in ev.color_to_object_id.items()
                           if oid.lower().startswith("room")]
            ids_r = np.isin(seg_code, floor_codes)
            geo_r = height_above_floor(depth_r, f, h, pitch, roll) < FLOOR_TOL
            union = (geo_r | ids_r).sum()
            agree = (geo_r & ids_r).sum() / union if union else 0.0
            floor = cv2.remap((geo_r & ids_r).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
            if floor.mean() < 0.15 or agree < MIN_AGREE:
                continue
            rgb = cv2.remap(rgb_r, map_x, map_y, cv2.INTER_LINEAR)
            depth_mm = cv2.remap(np.clip(depth_r * 1000, 0, 65535).astype(np.uint16), map_x, map_y,
                                 cv2.INTER_NEAREST)
            name = f"h{hi:05d}_{n_ok}"
            cv2.imwrite(os.path.join(args.out, name + ".jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(args.out, name + "_floor.png"), floor * 255)
            cv2.imwrite(os.path.join(args.out, name + "_depth.png"), depth_mm)
            stats_f.write(json.dumps(dict(name=name, house=hi, h=h, pitch=pitch, roll=roll, yaw=yaw,
                                          floor_frac=float(floor.mean()), agree=float(agree),
                                          n_clutter=len(placed))) + "\n")
            stats_f.flush()
            n_ok += 1
        print(f"house {hi}: {len(placed)} clutter, {n_ok} views", flush=True)
    ctrl.stop()


if __name__ == "__main__":
    main()
