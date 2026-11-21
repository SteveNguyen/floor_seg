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

Labels (project decision), two masks per view:
  floor  a pixel is floor if
    1. its 3D point is within FLOOR_TOL (1 cm) of the plane the robot stands on, AND
    2. it belongs to that surface (instance segmentation: a room floor, or the table top in table views).
  under  (room floor only) floor pixels with something lower than CLEARANCE above them (bed, sofa, chair
         seat, low table...; measured from the geometry, see UndersideMap). Visible floor the robot cannot
         drive to. A separate mask, NOT removed from floor: mixed into the floor label it made
         the model doubt plain open floor (floor_v4_thor2); a separate output keeps the floor task as is.
The AND in 1-2 removes depth that passes through glass (vases, windows). Frames where 1 and 2 disagree
(IoU < MIN_AGREE: see-through or broken geometry) or that mostly face a wall are dropped.

Table views (thor_v2, P_TABLE per house): the robot stands on a large table / counter top; the top is floor,
the room floor below it is not (a drop edge). They teach "a floor-looking surface below the support plane
is not floor". stats.jsonl: kind = "floor" | "table", support = height of the support plane (m).

Lens: output 640x360, K = calibrated K / 2 with the calibrated rational distortion. A larger pinhole
image with the same focal length is rendered and remapped (RGB bilinear, mask and depth nearest).
Noise / blur / colour jitter are left to training-time augmentation.

Output per view: <name>.jpg (RGB), <name>_floor.png and <name>_under.png (0/255), <name>_depth.png (uint16
mm, optical z), floor views also <name>_above.png (uint16 mm: height of the lowest surface above each floor
pixel, for metrics at other clearances; see shoot); one JSON line per view in stats.jsonl (written as it goes; re-running skips houses already
done). --plans: one top-down QA plan per house (plan_image).
--plinths: plinths rendered along the room walls (plinth_walls; labelled not floor by the rules above),
<name>_plinth.png (0/255) their pixels, and in stats.jsonl plinth_h, plinth_frac and doorways (the door
floor segments near the camera, in its yaw frame) for training-time styling and doorway thresholds.
"""
import argparse
import json
import math
import os
import random

import cv2
import numpy as np

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
# "under" mask (UndersideMap)
CLEARANCE = 0.45      # m: Reachy Mini head on the hoverboard <= 0.40 m, + 5 cm margin
FP_RES = 0.02         # m, top-down raster
UNDER_D = 10.0        # m, underside camera below the floor ...
UNDER_FOV = 20.0      # deg (vertical): nearly vertical rays (<= ~18 deg off at a tile corner)
STRUCTURE = {"Floor", "Wall", "Doorway", "Doorframe", "Window", "Painting"}  # not named on the QA plan
STRUCTURE_IDS = ("wall", "door", "window")  # instance ids seen from below that are not overhangs
CONTACT_M = 0.03      # m: a lowest surface below this rests on the floor (covers it), it is not an overhang
ABOVE_NONE = 65535    # _above.png value: height above unknown (or > 65 m)
# table views
SUPPORT_TYPES = {"DiningTable", "CounterTop", "SideTable", "TVStand", "CoffeeTable"}
SUPPORT_MIN = (0.6, 0.45)  # m, smallest top (long x short side) the base can stand on
ROBOT_R = 0.13             # m, camera kept this far inside the top's edges
P_TABLE = 1.0              # chance per house of one extra view from a table top (if it has one)
MIN_FLOOR = {"floor": 0.15, "table": 0.05}  # smallest labelled-floor fraction kept (a table top is small)


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


def world_xz(depth, f, cam_x, cam_z, yaw, pitch, roll):
    """World x, z (m) of each render pixel's 3D point; depth = optical z.

    Unity is left-handed, y up. The camera euler angles apply roll (about the optical axis), then pitch
    (about x, positive looks down), then yaw (about y, positive turns right seen from above), so after
    roll + pitch a point has a lateral offset x1 and a horizontal forward distance fwd in the camera's yaw
    frame (same as height_above_floor), and yaw maps them to the world:
    forward = (sin yaw, cos yaw), right = (cos yaw, -sin yaw) in (x, z).
    """
    hr, wr = depth.shape
    v, u = np.mgrid[0:hr, 0:wr].astype(np.float32)
    x = (u - wr / 2) / f * depth
    y = -(v - hr / 2) / f * depth
    r, p, w = math.radians(roll), math.radians(pitch), math.radians(yaw)
    x1 = x * math.cos(r) - y * math.sin(r)
    y1 = x * math.sin(r) + y * math.cos(r)
    fwd = depth * math.cos(p) + y1 * math.sin(p)
    return (cam_x + x1 * math.cos(w) + fwd * math.sin(w),
            cam_z - x1 * math.sin(w) + fwd * math.cos(w))


def box_corners(o):
    """(8, 3) world corners of an object's box: oriented if THOR gives one, else axis-aligned (cabinets,
    counters, structure have no oriented box)."""
    b = o["objectOrientedBoundingBox"] or o["axisAlignedBoundingBox"]
    return np.array(b["cornerPoints"], np.float64)


class UndersideMap:
    """Top-down raster (FP_RES) of the height of the LOWEST surface above each floor point; grid = "under"
    (that height < CLEARANCE: the robot cannot drive there although the camera may see the floor).

    Measured from the geometry, not from object boxes: a box cannot tell a loft bed (1.5 m underside) from a
    bed (0.1 m), and meshes stick out of THOR's boxes. measure_underside renders depth from far below the
    floor looking up (the floor mesh is one-sided, so invisible from below); every depth pixel is an exact
    3D point on the underside of something (bed slats, chair seat, table top, ceiling), added with add().
    Cells hidden behind a lower surface get no sample: finish() fills them from their neighbours (FILL_PX),
    and what is still unknown counts as under (safe side). Gaps narrower than CLOSE_PX cells between under
    cells (slats) are closed, and enclosed pockets of free floor smaller than POCKET_M2 become under. Cells
    whose lowest surface is below CONTACT_M are contact (an object standing there), not under.
    """
    FILL_PX = 3
    CLOSE_PX = 5       # 10 cm
    POCKET_M2 = 0.25   # m^2, e.g. 50 x 50 cm: a bar stool ring or chair stretchers, not a room

    def __init__(self, xmin, zmin, xmax, zmax):
        self.x0, self.z0 = xmin, zmin
        nx, nz = int(np.ceil((xmax - xmin) / FP_RES)) + 1, int(np.ceil((zmax - zmin) / FP_RES)) + 1
        self.h = np.full((nz, nx), np.inf, np.float32)  # rows = z, cols = x
        self.grid = np.zeros((nz, nx), bool)
        self.objs = []  # (objectType, x-z box corners), only to name things on the QA plan (plan_image)

    def _ij(self, x, z):
        i = np.round((np.asarray(z) - self.z0) / FP_RES).astype(np.int64)
        j = np.round((np.asarray(x) - self.x0) / FP_RES).astype(np.int64)
        return i, j, (i >= 0) & (i < self.h.shape[0]) & (j >= 0) & (j < self.h.shape[1])

    def add(self, x, z, height):
        """Surface points (world x, z, height above the floor): keep the lowest per cell."""
        i, j, ok = self._ij(x, z)
        np.minimum.at(self.h, (i[ok], j[ok]), np.maximum(height[ok], 0.0).astype(np.float32))

    def finish(self):
        unknown = ~np.isfinite(self.h)
        for _ in range(self.FILL_PX):  # grow known heights into unsampled cells, lowest neighbour first
            if not unknown.any():
                break
            grown = cv2.erode(np.where(unknown, np.float32(1e9), self.h), np.ones((3, 3), np.uint8))
            take = unknown & (grown < 1e9)
            self.h[take] = grown[take]
            unknown = ~np.isfinite(self.h) | (self.h >= 1e9)
        self.unknown = unknown
        # close gaps narrower than ~CLOSE_PX cells (rays passing between the slats of a bed or a shelf would
        # stripe the label; the robot cannot use a gap that narrow anyway)
        low = (self.h < CLEARANCE).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.CLOSE_PX, self.CLOSE_PX))
        grid = unknown | cv2.morphologyEx(low, cv2.MORPH_CLOSE, k).astype(bool)
        # small pockets of free floor fully enclosed by under cells (inside a bar stool's foot ring, a chair's
        # stretchers) cannot be reached: under too. 4-connected free regions, so a diagonal step in a
        # rasterised ring does not count as a way in. Rooms are enclosed by walls too, hence the size limit.
        n, lab, st, _ = cv2.connectedComponentsWithStats((~grid).astype(np.uint8), connectivity=4)
        small = np.r_[False, st[1:, cv2.CC_STAT_AREA] * FP_RES ** 2 < self.POCKET_M2]
        # contact cells (lowest surface < CONTACT_M: an object standing on the floor) are not under: the floor
        # there is covered, and visible floor pixels landing in them are rims around object bases
        self.grid = (grid | small[lab]) & ~(self.h < CONTACT_M)

    def lookup(self, x, z):
        """True where world (x, z) is under something lower than CLEARANCE (outside the raster: False)."""
        i, j, ok = self._ij(x, z)
        out = np.zeros(np.shape(x), bool)
        out[ok] = self.grid[i[ok], j[ok]]
        return out

    def lookup_h(self, x, z):
        """Height (m) of the lowest surface above world (x, z), after finish()'s fill; inf outside the raster
        or where unknown. Raw geometry: no gap closing or pockets (those only shape the 'under' mask)."""
        i, j, ok = self._ij(x, z)
        out = np.full(np.shape(x), np.inf, np.float32)
        out[ok] = np.where(self.unknown[i[ok], j[ok]], np.inf, self.h[i[ok], j[ok]])
        return out


def house_bounds(reach, objects, margin=0.5):
    """(xmin, zmin, xmax, zmax) of everything in the house: reachable cells and every object's box. Any floor
    a camera can see is inside (sizing from the reachable cells alone missed far furniture in a house with
    few reachable cells, whose floor was then never 'under')."""
    pts = [(p["x"], p["z"]) for p in reach]
    for o in objects:
        pts += [(c[0], c[2]) for c in o["axisAlignedBoundingBox"]["cornerPoints"]]
    pts = np.array(pts)
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    return lo[0], lo[1], hi[0], hi[1]


def measure_underside(ctrl, cam_id, reach, objects, wr, hr):
    """UndersideMap of the current house: tiles of depth renders from UNDER_D below the floor looking up
    (third-party camera cam_id, wr x hr pixels), over the whole house (house_bounds)."""
    x0, z0, x1, z1 = house_bounds(reach, objects)
    um = UndersideMap(x0, z0, x1, z1)
    f = hr / 2 / math.tan(math.radians(UNDER_FOV) / 2)
    tile_z, tile_x = 0.8 * hr / f * UNDER_D, 0.8 * wr / f * UNDER_D  # floor extent per tile, 20 % overlap
    for cz in np.arange(z0 + tile_z / 2, z1 + tile_z / 2, tile_z):
        for cx in np.arange(x0 + tile_x / 2, x1 + tile_x / 2, tile_x):
            ev = ctrl.step("UpdateThirdPartyCamera", thirdPartyCameraId=cam_id,
                           position=dict(x=float(cx), y=-UNDER_D, z=float(cz)), rotation=dict(x=-90, y=0, z=0),
                           fieldOfView=UNDER_FOV)
            depth = ev.third_party_depth_frames[cam_id]
            hgt = height_above_floor(depth, f, -UNDER_D, -90.0, 0.0)  # looking up: = depth - UNDER_D
            wx, wz = world_xz(depth, f, float(cx), float(cz), 0.0, -90.0, 0.0)
            # rays that hit nothing return the far plane (20 m: height ~10 m). Not "height < 5 m": a 5.05 m
            # ceiling was then dropped and its whole house stayed unmeasured, i.e. all under.
            hit = hgt < UNDER_D - 0.5
            # walls, doors, windows start a few cm above the floor but are structure, not overhangs: their
            # cells count as clear above (else the floor at every wall base is under)
            seg = ev.third_party_instance_segmentation_frames[cam_id][:, :, :3].astype(np.int32)
            code = (seg[:, :, 0] << 16) | (seg[:, :, 1] << 8) | seg[:, :, 2]
            wall = np.isin(code, [(c[0] << 16) | (c[1] << 8) | c[2] for c, oid in ev.color_to_object_id.items()
                                  if oid.lower().startswith(STRUCTURE_IDS)])
            hgt = np.where(wall, np.float32(UNDER_D), hgt)
            um.add(wx[hit], wz[hit], hgt[hit])
    um.finish()
    for o in objects:
        if o["objectType"] not in STRUCTURE:
            um.objs.append((o["objectType"], box_corners(o)[:, [0, 2]]))
    return um


def plan_image(um, reach, views):
    """Top-down QA plan of one house at the raster scale (FP_RES, 1 px = 2 cm; image right = +x, image down
    = +z): grey = under (UndersideMap), pink = under because never measured, yellow = object box hulls with
    their type (names only: the label does not use boxes), dots = reachable positions, red = cameras (name,
    0.5 m line along the view yaw)."""
    img = np.full(um.grid.shape + (3,), 255, np.uint8)
    img[um.grid] = (170, 170, 170)
    img[um.unknown] = (200, 170, 255)

    def px(x, z):
        return int(round((x - um.x0) / FP_RES)), int(round((z - um.z0) / FP_RES))

    for p in reach:
        cv2.circle(img, px(p["x"], p["z"]), 1, (200, 120, 0), -1)
    for t, c in um.objs:
        hull = cv2.convexHull(np.array([px(*q) for q in c], np.int32))
        cv2.polylines(img, [hull], True, (0, 180, 230), 1)
        cx, cz = hull[:, 0].mean(0).astype(int)
        cv2.putText(img, t, (int(cx) - 15, int(cz)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    for name, x, z, yaw in views:
        w = math.radians(yaw)
        cv2.circle(img, px(x, z), 4, (0, 0, 255), -1)
        cv2.line(img, px(x, z), px(x + 0.5 * math.sin(w), z + 0.5 * math.cos(w)), (0, 0, 255), 2)
        cv2.putText(img, name, px(x, z), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    return img


def support_tops(objects):
    """Table / counter tops the base can stand on: (object, top y, corner c0, edge e1, edge e2) in x-z,
    the top being the parallelogram c0 + s e1 + t e2, s, t in [0, 1]. Oriented-box corners 0-3 are the
    bottom face in order (checked on THOR 5 metadata); axis-aligned boxes are rebuilt from min / max."""
    out = []
    for o in objects:
        if o["objectType"] not in SUPPORT_TYPES:
            continue
        c = box_corners(o)
        if o["objectOrientedBoundingBox"]:
            c0, e1, e2 = c[0, [0, 2]], c[1, [0, 2]] - c[0, [0, 2]], c[3, [0, 2]] - c[0, [0, 2]]
        else:
            lo, hi = c[:, [0, 2]].min(0), c[:, [0, 2]].max(0)
            c0, e1, e2 = lo, np.array([hi[0] - lo[0], 0.0]), np.array([0.0, hi[1] - lo[1]])
        a, b = sorted((np.hypot(*e1), np.hypot(*e2)), reverse=True)
        if a >= SUPPORT_MIN[0] and b >= SUPPORT_MIN[1]:
            out.append((o, float(c[:, 1].max()), c0, e1, e2))
    return out


def shoot(ctrl, lens, kind, x, z, support, h, pitch, yaw, roll, um=None, support_ids=None):
    """Render one view with the camera h above the support plane (height `support`: 0 = room floor, else a
    table top) and label it (floor and under masks of the module docstring; under from the UndersideMap um).
    Returns (rgb, floor 0/1, under 0/1, depth_mm, above_mm, info) at the output lens, or None if the view is
    rejected. above_mm (uint16): on floor pixels, height of the lowest surface above (mm, ABOVE_NONE where
    unknown or beyond 65 m); 0 elsewhere. It lets a metric use another clearance than CLEARANCE.
    info: agree (rule 1 vs 2 IoU), under_frac (share of the floor pixels that are under), and with plinths in
    the house (--plinths) info["plinth"]: the plinth pixels at the output lens (0/1), popped by the caller.
    """
    map_x, map_y, f, vfov = lens
    ctrl.step("RandomizeLighting", synchronized=False)
    ev = ctrl.step("UpdateThirdPartyCamera", thirdPartyCameraId=0, position=dict(x=x, y=support + h, z=z),
                   rotation=dict(x=pitch, y=yaw, z=roll), fieldOfView=vfov)
    rgb_r = ev.third_party_camera_frames[0][:, :, :3]  # RGBA -> RGB
    depth_r = ev.third_party_depth_frames[0]
    if (depth_r < 0.8).mean() > 0.6:  # staring at a wall / furniture
        return None
    seg = ev.third_party_instance_segmentation_frames[0][:, :, :3].astype(np.int32)
    seg_code = (seg[:, :, 0] << 16) | (seg[:, :, 1] << 8) | seg[:, :, 2]

    def codes(keep):
        return [(c[0] << 16) | (c[1] << 8) | c[2] for c, oid in ev.color_to_object_id.items() if keep(oid)]

    if kind == "floor":
        ids_r = np.isin(seg_code, codes(lambda oid: oid.lower().startswith("room")))
    else:
        ids_r = np.isin(seg_code, codes(lambda oid: oid in support_ids))
    # rule 1, |.| since the room floor below a table top is a support-plane-relative height of -0.5..-1 m
    geo_r = np.abs(height_above_floor(depth_r, f, support + h, pitch, roll) - support) < FLOOR_TOL
    if kind == "floor":
        union = (geo_r | ids_r).sum()
        agree = (geo_r & ids_r).sum() / union if union else 0.0
    else:  # the table instance also has legs and apron below the top: only check the top plane's pixels
        agree = (geo_r & ids_r).sum() / max(geo_r.sum(), 1)
    lab_r = geo_r & ids_r
    under_r = np.zeros_like(lab_r)
    above_r = np.zeros(lab_r.shape, np.uint16)
    info = dict(agree=float(agree), under_frac=0.0)
    if um is not None:  # a separate label, the floor label keeps these pixels
        wx, wz = world_xz(depth_r, f, x, z, yaw, pitch, roll)
        under_r = lab_r & um.lookup(wx, wz)
        info["under_frac"] = float(under_r.sum() / max(lab_r.sum(), 1))  # share of the floor pixels
        hgt = um.lookup_h(wx, wz)
        above_r = np.where(lab_r, np.where(np.isfinite(hgt), np.clip(hgt * 1000, 1, ABOVE_NONE - 1), ABOVE_NONE),
                           0).astype(np.uint16)
    floor = cv2.remap(lab_r.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    if floor.mean() < MIN_FLOOR[kind] or agree < MIN_AGREE:
        return None
    under_lab = cv2.remap(under_r.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    rgb = cv2.remap(rgb_r, map_x, map_y, cv2.INTER_LINEAR)
    depth_mm = cv2.remap(np.clip(depth_r * 1000, 0, 65535).astype(np.uint16), map_x, map_y, cv2.INTER_NEAREST)
    above_mm = cv2.remap(above_r, map_x, map_y, cv2.INTER_NEAREST)
    plinth_codes = codes(lambda oid: PLINTH_TAG in oid)
    if plinth_codes:
        info["plinth"] = cv2.remap(np.isin(seg_code, plinth_codes).astype(np.uint8), map_x, map_y,
                                   cv2.INTER_NEAREST)
    return rgb, floor, under_lab, depth_mm, above_mm, info


# Plinths (--plinths): skirting boards rendered as geometry along the room walls, so they stand where real ones
# do (walls only: not on doors, cabinets or bins, which a 2D augmentation without a wall mask could not tell
# apart) with the scene's own lighting, occlusion and contact. Each is an extra wall quad PLINTH_D in front of
# its wall, as high as the house's plinth height, cut at the door holes (with PLINTH_DOOR_M of casing). Its
# instance id holds "|plinth|": the floor rule makes it obstacle (it is not a room floor and rises above
# FLOOR_TOL), the underside map counts it as structure (an id starting with "wall"), and shoot() returns its
# pixels as a separate mask (<name>_plinth.png) so training can choose the style (wall colour, white, wood,
# the retextured floor's tiles) on the rendered shading. Neutral light grey: room for recolouring both ways.
PLINTH_H = (0.08, 0.20)   # m, one height per house
PLINTH_D = 0.01           # m in front of the wall (a ~1 cm thick board; its top face is not modelled)
PLINTH_DOOR_M = 0.03      # m kept free on each side of a door hole (the door casing)
PLINTH_MATERIAL = {"name": "PureWhite", "color": {"r": 0.8, "g": 0.8, "b": 0.8}}
PLINTH_TAG = "|plinth|"


def _in_polygon(x, z, poly):
    """Even-odd test of the point (x, z) in a polygon of {'x', 'z'} dicts."""
    inside = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if (a["z"] > z) != (b["z"] > z) and x < (b["x"] - a["x"]) * (z - a["z"]) / (b["z"] - a["z"]) + a["x"]:
            inside = not inside
    return inside


def door_segments(house):
    """World floor segments ((x0, z0), (x1, z1)) of the door holes (doors and doorways; their holes start at
    the floor), from the hole's x range along its wall0 (measured from the wall polygon's first point)."""
    walls = {w["id"]: w for w in house["walls"]}
    out = []
    for d in house["doors"]:
        w = walls.get(d["wall0"])
        if w is None:
            continue
        p0, p1 = np.array([w["polygon"][0]["x"], w["polygon"][0]["z"]]), np.array([w["polygon"][1]["x"],
                                                                                   w["polygon"][1]["z"]])
        u = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9)
        a, b = sorted((d["holePolygon"][0]["x"], d["holePolygon"][1]["x"]))
        out.append((tuple(p0 + a * u), tuple(p0 + b * u)))
    return out


def plinth_walls(house, height):
    """Plinth quads for every room wall (not exterior faces): the wall's floor line moved PLINTH_D into its
    room, minus the door holes on that line (+ PLINTH_DOOR_M each side), `height` m high. Returns wall dicts
    in the house JSON format (ids '<wall id>|plinth|<k>')."""
    rooms = {r["id"]: r["floorPolygon"] for r in house["rooms"]}
    doors = door_segments(house)
    out = []
    for w in house["walls"]:
        poly = rooms.get(w.get("roomId"))
        if poly is None or w["id"].startswith("wall|exterior"):
            continue
        p0 = np.array([w["polygon"][0]["x"], w["polygon"][0]["z"]])
        p1 = np.array([w["polygon"][1]["x"], w["polygon"][1]["z"]])
        length = np.hypot(*(p1 - p0))
        if length < 0.05:
            continue
        u = (p1 - p0) / length
        n = np.array([-u[1], u[0]])
        mid = (p0 + p1) / 2
        if not _in_polygon(*(mid + 0.05 * n), poly):   # the normal must point into the wall's own room
            n = -n
            if not _in_polygon(*(mid + 0.05 * n), poly):
                continue
        # free intervals along the wall: [0, length] minus the door holes lying on this wall's line
        cuts = []
        for q0, q1 in doors:
            t0, t1 = (np.dot(np.array(q) - p0, u) for q in (q0, q1))
            off = max(abs(u[0] * (q[1] - p0[1]) - u[1] * (q[0] - p0[0])) for q in (q0, q1))  # distance to the line
            if off < 0.02:
                cuts.append((min(t0, t1) - PLINTH_DOOR_M, max(t0, t1) + PLINTH_DOOR_M))
        spans, t = [], 0.0
        for a, b in sorted(cuts):
            if a > t:
                spans.append((t, min(a, length)))
            t = max(t, b)
        if t < length:
            spans.append((t, length))
        for k, (a, b) in enumerate(s for s in spans if s[1] - s[0] >= 0.05):
            s0, s1 = p0 + a * u + PLINTH_D * n, p0 + b * u + PLINTH_D * n
            pts = [(s0, 0.0), (s1, 0.0), (s0, height), (s1, height)]   # same corner order as ProcTHOR walls
            out.append({"id": f"{w['id']}{PLINTH_TAG}{k}", "roomId": w["roomId"],
                        "polygon": [{"x": float(p[0]), "y": float(y), "z": float(p[1])} for p, y in pts],
                        "material": dict(PLINTH_MATERIAL), "color": dict(PLINTH_MATERIAL["color"])})
    return out


def doorways_in_view(segments, x, z, yaw, max_d=4.0):
    """Door floor segments in the camera's yaw frame (gx right, gz forward, m; same frame as world_xz), those
    with a point within max_d of the camera: [[gx0, gz0, gx1, gz1], ...]. For training-time thresholds placed
    in the real doorways."""
    w = math.radians(yaw)
    out = []
    for (x0, z0), (x1, z1) in segments:
        pts = []
        for px, pz in ((x0, z0), (x1, z1)):
            dx, dz = px - x, pz - z
            pts += [dx * math.cos(w) - dz * math.sin(w), dx * math.sin(w) + dz * math.cos(w)]
        if min(math.hypot(pts[0], pts[1]), math.hypot(pts[2], pts[3])) <= max_d:
            out.append([round(v, 3) for v in pts])
    return out


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


def rest_on_floor(house, objects, tol=0.005):
    """Copy of the house with each clutter object moved vertically so the bottom of its box, as loaded in the
    scene (objects = metadata after reset), is on the floor (y = 0). The asset catalogue's box does not
    always match the loaded object: placed from it alone, some clutter floated (~10 cm) or sank into the floor
    (up to 0.45 m). Returns (house, number of objects moved)."""
    house = json.loads(json.dumps(house))
    low = {o["objectId"]: min(p[1] for p in o["axisAlignedBoundingBox"]["cornerPoints"]) for o in objects}
    moved = 0
    for h_obj in house["objects"]:
        y0 = low.get(h_obj["id"])
        if "|clutter|" in h_obj["id"] and y0 is not None and abs(y0) > tol:
            h_obj["position"]["y"] -= y0
            moved += 1
    return house, moved


def main():
    import prior  # here, so the geometry above can be imported (and tested) without AI2-THOR
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--houses", type=int, default=2000)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plans", action="store_true", help="also write h#####_plan.png (top-down QA plan)")
    ap.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"), default=(0, 1),
                    help="render only houses I, I+N, I+2N... of the list (N processes or machines in parallel, "
                         "same --out; stats go to stats_s<I>.jsonl, merge with: cat stats_s*.jsonl > stats.jsonl)")
    ap.add_argument("--only", type=int, nargs="+", help="render only these house indices (QA), instead of the list")
    ap.add_argument("--plinths", action="store_true",
                    help="add plinths along the room walls (plinth_walls), write <name>_plinth.png, and the door "
                         "floor segments of each view in stats.jsonl ('doorways', camera yaw frame)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    si, sn = args.shard
    stats_path = os.path.join(args.out, "stats.jsonl" if sn == 1 else f"stats_s{si}.jsonl")
    done = set()
    if os.path.exists(stats_path):
        done = {json.loads(line)["house"] for line in open(stats_path)}

    map_x, map_y, wr, hr, f, vfov = lens_maps(args.calib)
    lens = (map_x, map_y, f, vfov)
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
    for hi in (args.only or order[:args.houses])[si::sn]:
        if hi in done:
            continue
        rng = random.Random(args.seed * 100003 + hi)  # per house: reproducible and resumable
        try:
            ctrl.reset(scene=houses[hi])
            reach = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []
            if len(reach) < 10:
                continue
            house, placed = add_clutter(houses[hi], list(reach), by_type, rng)
            doors = []
            if args.plinths:  # own random stream: the clutter and the camera draws stay those of a run without
                plinth_h = random.Random(args.seed * 100003 + hi + 7919).uniform(*PLINTH_H)
                house["walls"] += plinth_walls(house, plinth_h)
                doors = door_segments(house)
            ctrl.reset(scene=house)
            house, moved = rest_on_floor(house, ctrl.last_event.metadata["objects"])
            if moved:
                ctrl.reset(scene=house)
            off = [min(p[1] for p in o["axisAlignedBoundingBox"]["cornerPoints"])
                   for o in ctrl.last_event.metadata["objects"] if "|clutter|" in o["objectId"]]
            clutter_off = max((abs(v) for v in off), default=0.0)  # check: ~0 after rest_on_floor
            ctrl.step("RandomizeMaterials")
            reach = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []  # around the clutter
            if not reach:
                continue
            objects = ctrl.last_event.metadata["objects"]
            tops = support_tops(objects)
            ctrl.step("AddThirdPartyCamera", position=dict(x=0, y=1, z=0), rotation=dict(x=0, y=0, z=0),
                      fieldOfView=vfov)  # id 0: the views
            ctrl.step("AddThirdPartyCamera", position=dict(x=0, y=-UNDER_D, z=0), rotation=dict(x=-90, y=0, z=0),
                      fieldOfView=UNDER_FOV)  # id 1: underside map
            um = measure_underside(ctrl, 1, reach, objects, wr, hr)
        except Exception as e:  # a broken house must not stop a long run
            print(f"house {hi}: setup failed ({e})", flush=True)
            continue

        shots = []  # (name, x, z, yaw) of the floor views, for the QA plan

        def save(kind, k, view, x, z, support, h, pitch, yaw, roll):
            rgb, floor, under, depth_mm, above_mm, info = view
            name = f"h{hi:05d}_{k}" if kind == "floor" else f"h{hi:05d}_t{k}"
            if args.plinths:
                plinth = info.pop("plinth", np.zeros_like(floor))
                cv2.imwrite(os.path.join(args.out, name + "_plinth.png"), plinth * 255)
                info["plinth_frac"] = float(plinth.mean())
                info["plinth_h"] = round(plinth_h, 3)
                info["doorways"] = doorways_in_view(doors, x, z, yaw) if kind == "floor" else []
            if kind == "floor":
                shots.append((name[-1], x, z, yaw))
            cv2.imwrite(os.path.join(args.out, name + ".jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(args.out, name + "_floor.png"), floor * 255)
            cv2.imwrite(os.path.join(args.out, name + "_under.png"), under * 255)
            cv2.imwrite(os.path.join(args.out, name + "_depth.png"), depth_mm)
            if kind == "floor":
                cv2.imwrite(os.path.join(args.out, name + "_above.png"), above_mm)
            stats_f.write(json.dumps(dict(name=name, house=hi, kind=kind, x=x, z=z, support=support, h=h,
                                          pitch=pitch, roll=roll, yaw=yaw, floor_frac=float(floor.mean()),
                                          n_clutter=len(placed), clutter_moved=moved,
                                          clutter_off=round(clutter_off, 4), **info)) + "\n")
            stats_f.flush()

        n_ok, tries = 0, 0
        while n_ok < args.views and tries < 5 * args.views:
            tries += 1
            pos = rng.choice(reach)
            h = rng.uniform(*H_RANGE)
            pitch = rng.uniform(0, 24) + rng.uniform(-5, 5)
            roll = rng.uniform(-3, 3)
            if placed and rng.random() < 0.7:
                cx, cz = rng.choice(placed)
                yaw = math.degrees(math.atan2(cx - pos["x"], cz - pos["z"])) + rng.uniform(-25, 25)
            else:
                yaw = rng.uniform(0, 360)
            view = shoot(ctrl, lens, "floor", pos["x"], pos["z"], 0.0, h, pitch, yaw, roll, um=um)
            if view:
                save("floor", n_ok, view, pos["x"], pos["z"], 0.0, h, pitch, yaw, roll)
                n_ok += 1
        n_tab = 0
        if tops and rng.random() < P_TABLE:
            for _ in range(10):
                o, top, c0, e1, e2 = rng.choice(tops)
                # random camera point on the top, ROBOT_R inside its edges (parallelogram coords s, t)
                m1, m2 = ROBOT_R / np.hypot(*e1), ROBOT_R / np.hypot(*e2)
                x, z = (c0 + rng.uniform(m1, 1 - m1) * e1 + rng.uniform(m2, 1 - m2) * e2).tolist()
                # the support surface = every support top at that height (counters are split in pieces)
                ids = {q["objectId"] for q, t, *_ in tops if abs(t - top) < FLOOR_TOL}
                h = rng.uniform(*H_RANGE)
                pitch = rng.uniform(0, 24) + rng.uniform(-5, 5)
                roll, yaw = rng.uniform(-3, 3), rng.uniform(0, 360)
                view = shoot(ctrl, lens, "table", x, z, top, h, pitch, yaw, roll, support_ids=ids)
                if view:
                    save("table", 0, view, x, z, top, h, pitch, yaw, roll)
                    n_tab = 1
                    break
        if args.plans:
            cv2.imwrite(os.path.join(args.out, f"h{hi:05d}_plan.png"), plan_image(um, reach, shots))
        print(f"house {hi}: {len(placed)} clutter, {n_ok} floor views, {n_tab} table views", flush=True)
    ctrl.stop()


if __name__ == "__main__":
    main()
