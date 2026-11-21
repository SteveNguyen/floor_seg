"""Pure-Python parts of the Infinigen data generator: camera geometry, labels, lens, room choice, stats.

No Blender here (tests run without it); floor_seg/infinigen_render.py (run with Infinigen's Python, which ships
Blender 4.2 as the bpy module) uses these. Same output format as the AI2-THOR sets (floor_seg/thor_render.py),
so floor_seg.train reads the result unchanged: <name>.jpg (640x360 through the calibrated Reachy Mini lens),
<name>_floor.png (255 floor / 0 not), <name>_depth.png (uint16 mm, optical-axis z) and stats.jsonl.

Conventions (identical to thor_render / floor_tex, so the training-time floor retexturing and structure
augmentations can use the stored pitch / roll / height): camera frame x right, y up, z forward; roll about the
optical axis, then pitch (positive looks down), then yaw about the world vertical. Blender world: z up.

Label rule (project decision): a pixel is floor if its 3D point lies within FLOOR_TOL (1.5 cm) of the floor plane
AND it belongs to the floor or to something flat lying on it (thin floor-level objects, cables, door threshold
strips); rugs and mats are floor by rule although they are 1-2 cm thick (their top up to RUG_TOP). Everything
else (walls, plinths / skirting boards, furniture, objects, windows, nothing hit) is not floor.
"""
import json
import math
import random

import cv2
import numpy as np

W, H = 640, 360
FLOOR_TOL = 0.015      # m, floor-plane tolerance of the height rule
RUG_TOP = 0.04         # m, a rug's top (1-2 cm thick + wrinkle displacement) is still floor
MIN_FLOOR = 0.15       # smallest floor fraction kept for a view (as thor_render floor views)
# object pass index codes (set on the Blender objects, read back from the IndexOB pass)
IDX_OTHER, IDX_FLOOR, IDX_RUG, IDX_FLAT = 0, 1, 2, 3

# Room types weighted toward ordinary homes. Infinigen 2.0 CLI generators; there is no corridor / hallway room
# generator, and the furnished house (several rooms, doors between them) is much slower: off by default.
ROOM_TYPES = {
    "livingroom": ("room_livingroom_rand", 0.30),
    "bedroom": ("room_bedroom_rand", 0.25),
    "kitchen": ("room_kitchen_rand", 0.18),
    "diningroom": ("room_diningroom_rand", 0.12),
    "bathroom": ("room_bathroom_rand", 0.15),
    "house": ("house_furnished_rand", 0.0),
}
# chance of repainting the walls with an ordinary paint colour, per room type (bathrooms and kitchens keep
# Infinigen's tiled / patterned walls more often: tiles are normal there)
P_PAINT = {"bathroom": 0.35, "kitchen": 0.6}
P_PAINT_DEFAULT = 0.9
# ordinary wall paints (linear RGB): white / off-white, cream, beige / greige, light grey, soft pastels
WALL_PALETTE = [
    ("white", 0.40, [(0.86, 0.86, 0.85), (0.83, 0.83, 0.84), (0.88, 0.87, 0.84)]),
    ("cream", 0.15, [(0.85, 0.80, 0.68), (0.88, 0.84, 0.72), (0.84, 0.79, 0.70)]),
    ("beige", 0.15, [(0.70, 0.62, 0.52), (0.66, 0.62, 0.56), (0.74, 0.68, 0.60)]),
    ("grey", 0.15, [(0.60, 0.61, 0.62), (0.50, 0.51, 0.52), (0.70, 0.70, 0.69)]),
    ("pastel", 0.15, [(0.62, 0.72, 0.78), (0.66, 0.76, 0.64), (0.82, 0.70, 0.68), (0.84, 0.80, 0.58)]),
]
# floor materials: Infinigen picks concrete / wood planks / carpet / tiles / layered mixes. Plain concrete is rare
# in homes: such a room is regenerated with this chance.
P_REJECT_CONCRETE = 0.85


def weighted(rng, items):
    """items: [(value, weight)] -> one value (random.Random rng)."""
    total = sum(w for _, w in items)
    x = rng.uniform(0, total)
    for v, w in items:
        x -= w
        if x <= 0:
            return v
    return items[-1][0]


def room_type(rng, weights=None):
    """A room type name (ROOM_TYPES key) drawn with the ordinary-home weights (or `weights`: name -> weight)."""
    w = weights or {k: v[1] for k, v in ROOM_TYPES.items()}
    return weighted(rng, [(k, x) for k, x in w.items() if x > 0])


def wall_paint(rng):
    """(palette name, linear RGB) of an ordinary wall paint."""
    name, colours = weighted(rng, [((n, c), w) for n, w, c in WALL_PALETTE])
    c = np.array(rng.choice(colours)) * rng.uniform(0.92, 1.05)
    return name, tuple(float(v) for v in np.clip(c, 0, 1))


def floor_kind(group_names):
    """Floor material family from the material's node-group names (Infinigen's procedural shaders)."""
    s = " ".join(group_names).lower()
    for kind, keys in (("carpet", ("carpet",)), ("tiles", ("tile",)), ("wood", ("wood", "plank")),
                       ("concrete", ("concrete",))):
        if any(k in s for k in keys):
            return kind
    return "other"


def camera_axes(yaw_deg, pitch_deg, roll_deg):
    """3x3 matrix: columns = world directions (Blender, z up) of the camera's x right, y up, z forward axes.

    roll about the optical axis, then pitch (positive looks down), then yaw (0 = world +x, counter-clockwise seen
    from above). Same as floor_tex.heights / floor_coords for the height and the forward distance."""
    r, p, w = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    roll = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    # (x1, y1, z) -> (right, up, forward)
    pitch = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    fwd = np.array([math.cos(w), math.sin(w), 0.0])
    right = np.array([math.sin(w), -math.cos(w), 0.0])
    basis = np.stack([right, np.array([0.0, 0.0, 1.0]), fwd], axis=1)  # (right, up, forward) -> world
    return basis @ pitch @ roll


def blender_rotation(axes):
    """Rotation matrix of a Blender camera object (local x right, y up, -z view) for camera_axes."""
    return np.stack([axes[:, 0], axes[:, 1], -axes[:, 2]], axis=1)


def _undistort(pts, K, D):
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.8 (Infinigen's env) has no criteria kwarg on undistortPoints
        return cv2.undistortPointsIter(pts, K, D, None, None, crit)
    return cv2.undistortPoints(pts, K, D, criteria=crit)


def lens_maps(calib_path, pixel_center=-0.5):
    """Output pixel (640x360, calibrated lens) -> pinhole render pixel maps; render size and focal length.

    Same as thor_render.lens_maps, plus pixel_center: Blender puts pixel u's centre at x = (u + 0.5 - w/2) / f,
    so a ray x lands at render coordinate x f + w/2 - 0.5 (cv2.remap samples pixel centres at integers)."""
    cal = json.load(open(calib_path))
    sx, sy = W / cal["image_resolution"][0], H / cal["image_resolution"][1]
    K = np.array(cal["K"]) * np.array([[sx], [sy], [1]])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], -1)[:, None, :]
    xn = _undistort(pts, K, np.array(cal["D"]))[:, 0]
    f = float(K[1, 1])
    wr = 2 * int(math.ceil(np.abs(xn[:, 0]).max() * f)) + 4
    hr = 2 * int(math.ceil(np.abs(xn[:, 1]).max() * f)) + 4
    map_x = (xn[:, 0] * f + wr / 2 + pixel_center).reshape(H, W).astype(np.float32)
    map_y = (xn[:, 1] * f + hr / 2 + pixel_center).reshape(H, W).astype(np.float32)
    return map_x, map_y, wr, hr, f


def pinhole_dirs(wr, hr, f, step=1):
    """Camera-frame ray directions (x right, y up, z forward = 1) of the pinhole render's pixel centres."""
    u = (np.arange(0, wr, step) + 0.5 - wr / 2) / f
    v = (np.arange(0, hr, step) + 0.5 - hr / 2) / f
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, -vv, np.ones_like(uu)], -1)


def labels_from_passes(pos, idx, cam_pos, axes, floor_z):
    """Floor mask and optical depth (m) of a render from its world-position and object-index passes.

    pos: H x W x 3 world points (Blender Position pass; all zero where nothing was hit); idx: H x W object pass
    index (IDX_* codes); cam_pos: camera centre; axes: camera_axes; floor_z: height of the floor plane."""
    hit = np.abs(pos).sum(-1) > 0
    rel = pos[..., 2] - floor_z
    idx = np.rint(idx).astype(np.int32)
    flat = ((idx == IDX_FLOOR) | (idx == IDX_FLAT)) & (np.abs(rel) < FLOOR_TOL)
    rug = (idx == IDX_RUG) & (rel > -FLOOR_TOL) & (rel < RUG_TOP)
    floor = hit & (flat | rug)
    depth = np.where(hit, (pos - np.asarray(cam_pos)) @ axes[:, 2], 0.0)
    return floor, np.clip(depth, 0, None)


def finish(rgb_r, floor_r, depth_r, maps):
    """Remap a pinhole render through the calibrated lens: (rgb uint8 HxWx3, floor 0/255 uint8, depth uint16 mm)."""
    map_x, map_y = maps[:2]
    rgb = cv2.remap(rgb_r, map_x, map_y, cv2.INTER_LINEAR)
    floor = cv2.remap(floor_r.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) * 255
    depth = cv2.remap(np.clip(depth_r * 1000, 0, 65535).astype(np.uint16), map_x, map_y, cv2.INTER_NEAREST)
    return rgb, floor.astype(np.uint8), depth


def fit_fisheye_poly(calib_path, sensor_mm=36.0, deg=4):
    """Fit Cycles' FISHEYE_LENS_POLYNOMIAL (angle = k0 + k1 r + ... + k4 r^4, r = sensor radius in mm from the
    image centre) to the calibrated lens at 640x360. Returns (coefficients k0..k4, rms px, max px): the pixel
    error of projecting each calibrated ray through the fitted model. The model is radially symmetric about the
    image centre, so a principal point off the centre (ours: ~16 px vertically at 640x360) cannot be matched."""
    cal = json.load(open(calib_path))
    sx, sy = W / cal["image_resolution"][0], H / cal["image_resolution"][1]
    K = np.array(cal["K"]) * np.array([[sx], [sy], [1]])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], -1)[:, None, :]
    xn = _undistort(pts, K, np.array(cal["D"]))[:, 0]
    theta = np.arctan(np.hypot(xn[:, 0], xn[:, 1]))
    du, dv = uu.ravel() + 0.5 - W / 2, vv.ravel() + 0.5 - H / 2
    mm = sensor_mm / W
    r_mm = np.hypot(du, dv) * mm
    # k0 = 0: the angle is zero at the centre; k1..k_deg by least squares
    k = np.concatenate([[0.0], np.linalg.lstsq(np.stack([r_mm ** i for i in range(1, deg + 1)], -1), theta,
                                               rcond=None)[0]])
    # project each true ray through the model: radius whose angle matches (dense table), true azimuth
    table_r = np.linspace(0, r_mm.max() * 1.3, 20000)
    table_t = np.polynomial.polynomial.polyval(table_r, k)
    order = np.argsort(table_t)
    r_model = np.interp(theta, table_t[order], table_r[order]) / mm
    az = np.arctan2(xn[:, 1], xn[:, 0])
    pu, pv = r_model * np.cos(az), r_model * np.sin(az)
    err = np.hypot(pu - du, pv - dv)
    return [float(c) for c in np.pad(k, (0, 5 - len(k)))], float(np.sqrt((err ** 2).mean())), float(err.max())


def stats_line(name, room, room_type_name, seed, pose, floor_frac, extra=None):
    """One stats.jsonl line in the thor sets' format (house = room index: the train / val split uses it)."""
    x, y, yaw, h, pitch, roll = pose
    s = dict(name=name, house=int(room), kind="floor", room_type=room_type_name, seed=int(seed),
             x=round(float(x), 3), y=round(float(y), 3), yaw=round(float(yaw), 2), h=round(float(h), 4),
             pitch=round(float(pitch), 3), roll=round(float(roll), 3), floor_frac=round(float(floor_frac), 4))
    s.update(extra or {})
    return json.dumps(s)


def sample_pose(rng, bounds, h_range=(0.25, 0.40), pitch_range=(5.0, 25.0), roll_max=3.0):
    """Random camera pose inside bounds (x0, x1, y0, y1): (x, y, yaw deg, h, pitch, roll)."""
    x0, x1, y0, y1 = bounds
    return (rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(-180, 180), rng.uniform(*h_range),
            rng.uniform(*pitch_range), rng.uniform(-roll_max, roll_max))


def yaw_toward(x, y, tx, ty):
    return math.degrees(math.atan2(ty - y, tx - x))


def new_rng(*keys):
    """A random.Random seeded from integers (stable across Python versions, unlike hash())."""
    s = 0
    for k in keys:
        s = (s * 1_000_003 + int(k)) % (2 ** 61 - 1)
    return random.Random(s)
