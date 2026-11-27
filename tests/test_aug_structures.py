"""Structure augmentations (floor_seg/aug_structures.py) on an analytic scene: a floor plane and a wall 2 m ahead,
seen by the calibrated lens from the robot's pose. Synthetic only, CPU."""
import math
import random
from dataclasses import replace

import cv2
import numpy as np
import pytest

from floor_seg import DEFAULT_CALIB
from floor_seg.aug_structures import UNDER, Geom, StructAug, StructAugCfg
from floor_seg.floor_tex import H, W, floor_coords, heights

CAM_H, PITCH, ROLL, WALL_Z = 0.27, 18.0, 1.5, 2.0
OFF = StructAugCfg(p_plinth=0.0, p_threshold=0.0, p_level=0.0, p_sun=0.0, p_specular=0.0, p_wall_tiles=0.0)


@pytest.fixture(scope="module")
def sa():
    return StructAug(OFF, DEFAULT_CALIB)


@pytest.fixture(scope="module")
def scene(sa):
    """(bgr, label, depth_mm): checker-tiled floor (label 1) up to a plain wall at gz = WALL_Z (label 0); a patch of
    'under' floor on the left (UNDER). Depth = the first of floor plane / wall plane along each pixel ray."""
    gx1, gz1 = floor_coords(sa.xn, sa.yn, 1.0, PITCH, ROLL)
    dh1 = heights(sa.xn, sa.yn, 1.0, PITCH, ROLL, 0.0)
    with np.errstate(divide="ignore"):
        tf = np.where(dh1 < 0, -CAM_H / dh1, np.inf)
        tw = np.where(gz1 > 0, WALL_Z / gz1, np.inf)
    t = np.minimum(tf, tw)
    floor = tf < tw
    gx, gz = gx1 * t, gz1 * t
    checker = ((np.floor(gx / 0.3) + np.floor(gz / 0.3)) % 2).astype(np.float32)
    bgr = np.where(floor[..., None], 90 + 60 * checker[..., None], 170).astype(np.uint8) * np.ones(3, np.uint8)
    label = floor.astype(np.uint8)
    label[floor & (gx < -0.6) & (gz < 1.2)] = UNDER
    depth = np.clip(t * 1000, 0, 65000).astype(np.uint16)
    return bgr, label, depth


def run(sa, scene, seed=0, **p):
    bgr, label, depth = scene
    sa.cfg = replace(OFF, **p)
    return sa.apply(bgr, label, depth, PITCH, ROLL, CAM_H, random.Random(seed))


def geom(sa, scene):
    return Geom(sa.xn, sa.yn, scene[2], PITCH, ROLL, CAM_H)


def test_projection_inverts_the_ray_geometry(sa, scene):
    g = geom(sa, scene)
    ys, xs = np.mgrid[20:H:40, 20:W:40]
    u, v = sa.project(g.gx[ys, xs], g.gz[ys, xs], g.hgt[ys, xs], g)
    assert np.nanmax(np.abs(u - xs)) < 0.05 and np.nanmax(np.abs(v - ys)) < 0.05


def test_reproject_no_points(sa, scene):
    # a level change whose floor beyond the edge is all hidden: no point to reproject (crashed v8 training)
    e = np.zeros(0)
    col, okf, oke = sa.reproject(scene[0].astype(np.float32) / 255, scene[1], geom(sa, scene), e, e, 1.0, (0.0, 1.0))
    assert col.shape == (0, 3) and okf.shape == oke.shape == (0,)


@pytest.mark.parametrize("style", StructAug.PLINTH_STYLES)
def test_plinth_is_on_the_wall_base_with_its_metric_height(sa, scene, style):
    bgr, label, _ = scene
    weights = tuple(float(s == style) for s in StructAug.PLINTH_STYLES)
    img, lab, info = run(sa, scene, p_plinth=1.0, plinth_style=weights)
    hp = info["plinth"]["h"]
    assert 0.08 <= hp <= 0.20 and info["plinth"]["style"] == style
    painted = (img != bgr).any(axis=2)
    hgt = geom(sa, scene).hgt
    assert painted.sum() > 1000
    assert (label[painted] == 0).all()                    # only wall pixels
    assert np.array_equal(lab, label)                     # obstacle stays obstacle; nothing else changes
    # the board up to hp, plus its joint line on the wall just above (<= 5 mm)
    assert hgt[painted].max() <= hp + 0.006 and hgt[painted].min() > -0.02
    # it covers the whole wall base up to hp, in metres (the wall spans the image width)
    wall_low = (label == 0) & (hgt < hp - 0.01) & (hgt > 0.005)
    assert painted[wall_low].mean() > 0.95
    mid = wall_low & (hgt > 0.01) & (hgt < hp - 0.012)    # the board's face: no edge / contact lines
    face = img[mid].astype(float).mean(axis=0)
    if style == "wall":      # blends with the wall (plain grey 170 in the scene): a shade off
        assert np.all(np.abs(face - 170) < 0.2 * 170)
    elif style == "white":
        assert face.min() > 190


WALL_BGR = (200, 150, 120)   # a bluish wall, so the "wall" style can be told from white / grey


def rendered_plinth_scene(sa, scene, hp=0.12):
    """The scene with a rendered plinth (thor_render --plinths) on the wall base up to hp: the board in the render's
    neutral grey (PLINTH_ALBEDO) under a light of 1.0, except its leftmost columns in a shadow (0.6); the wall
    above in WALL_BGR. Returns (bgr, label, depth, plinth mask, shadowed board mask)."""
    from floor_seg.aug_structures import PLINTH_ALBEDO

    bgr, label, depth = scene
    hgt = geom(sa, scene).hgt
    pm = (label == 0) & (hgt < hp) & (hgt > 0.0)
    cols = np.nonzero(pm.any(axis=0))[0]
    shadow = pm & (np.arange(W)[None, :] < cols.min() + 0.2 * (cols.max() - cols.min()))
    out = bgr.copy()
    out[(label == 0) & ~pm] = WALL_BGR
    out[pm] = round(PLINTH_ALBEDO * 255)
    out[shadow] = round(PLINTH_ALBEDO * 255 * 0.6)
    return out, label, depth, pm, shadow


def run_r(sa, scene3, pm, seed=0, src=None, doorways=None, **p):
    bgr, label, depth = scene3
    sa.cfg = replace(OFF, **p)
    return sa.apply(bgr, label, depth, PITCH, ROLL, CAM_H, random.Random(seed), plinth=pm.astype(np.uint8) * 255,
                    plinth_src=src, doorways=doorways)


@pytest.mark.parametrize("style", StructAug.PLINTH_STYLES)
def test_rendered_plinth_recoloured_on_its_own_shading(sa, scene, style):
    bgr, label, depth, pm, shadow = rendered_plinth_scene(sa, scene)
    weights = tuple(float(s == style) for s in StructAug.PLINTH_STYLES)
    img, lab, info = run_r(sa, (bgr, label, depth), pm, p_plinth_render=1.0, plinth_style=weights)
    p = info["plinth"]
    assert p["rendered"] and p["style"] == style and abs(p["h"] - 0.12) < 0.01
    assert np.array_equal(lab, label) and (lab[pm] == 0).all()        # obstacle, nothing else changes
    painted = (img != bgr).any(axis=2)
    above = np.zeros_like(pm)
    above[:-1] = pm[1:]                                               # the joint line: one row above the board
    assert painted[pm].mean() > 0.95 and not (painted & ~pm & ~above).any()
    hgt = geom(sa, scene).hgt
    face = pm & (hgt > 0.02) & (hgt < 0.09)                           # no top edge / contact line
    lit, dark = img[face & ~shadow].astype(float), img[face & shadow].astype(float)
    if style in ("wall", "white", "plain"):   # plain colours: the render's shadow is kept (light 0.6 vs 1.0)
        assert abs(dark.mean() / lit.mean() - 0.6) < 0.1
    if style == "wall":                       # the wall's colour (light 1.0 on most of the board), a shade off
        assert np.all(np.abs(lit.mean(axis=0) - WALL_BGR) < 0.2 * np.array(WALL_BGR))
    elif style == "white":
        assert lit.mean(axis=0).min() > 190


def test_rendered_plinth_light_read_before_retexturing(sa, scene):
    # floor retexturing (texture-swap negatives) may paint over the board: its light is read in plinth_src
    bgr, label, depth, pm, shadow = rendered_plinth_scene(sa, scene)
    over = bgr.copy()
    over[pm] = 40
    white = tuple(float(s == "white") for s in StructAug.PLINTH_STYLES)
    img, _, _ = run_r(sa, (over, label, depth), pm, src=bgr, p_plinth_render=1.0, plinth_style=white)
    assert img[pm & ~shadow].mean() > 190


def test_rendered_plinth_replaces_the_painted_one(sa, scene):
    bgr, label, depth, pm, _ = rendered_plinth_scene(sa, scene)
    img, _, info = run_r(sa, (bgr, label, depth), pm, p_plinth=1.0, p_wall_tiles=1.0, p_plinth_render=0.0)
    assert img is bgr and not any(info.values())   # a render with its own plinths gets no painted plinth / tiles


def test_threshold_in_a_real_doorway(sa, scene):
    bgr, label, _ = scene
    door = [[0.6, 1.4, -0.6, 1.4]]                  # a 1.2 m doorway across the view at 1.4 m (either direction)
    img, lab, info = run_r(sa, scene, np.zeros(label.shape, bool), doorways=door, p_door_threshold=1.0)
    t = info["threshold"]
    assert t["doorway"] and t["door"] and abs(t["len"] - 1.2) < 1e-6 and t["h"] <= 0.012
    painted = (img != bgr).any(axis=2)
    g = geom(sa, scene)
    assert painted.sum() > 100 and np.array_equal(lab, label) and (label[painted] == 1).all()   # floor stays floor
    # along the doorway's line, within the strip's depth; its top (h) seen past its far edge: floor points up to
    # h * gz / (cam_h - h) farther (parallax)
    reach = t["width"] / 2 + t["h"] * 1.5 / (CAM_H - t["h"]) + 0.01
    assert np.abs(g.gz[painted] - 1.4).max() < reach and np.abs(g.gx[painted]).max() < 0.65
    # doorways known: the strip goes in one or nowhere (never on open floor), whatever p_threshold says
    for doors in ([], [[0.6, -1.4, -0.6, -1.4]]):   # none in view / behind the camera
        img, _, info = run_r(sa, scene, np.zeros(label.shape, bool), doorways=doors, p_door_threshold=1.0,
                             p_threshold=1.0)
        assert img is bgr and not info.get("threshold")
    img, _, info = run_r(sa, scene, np.zeros(label.shape, bool), doorways=door, p_threshold=1.0)
    assert img is bgr and not info.get("threshold")  # p_door_threshold 0


def test_wall_tiles_up_to_their_height(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=4, p_wall_tiles=1.0, p_plinth=1.0)
    t = info["wall_tiles"]
    assert t and 0.5 <= t["h"] <= 2.0 and not info.get("plinth")   # tiles instead of a plinth
    painted = (img != bgr).any(axis=2)
    hgt = geom(sa, scene).hgt
    assert (label[painted] == 0).all() and np.array_equal(lab, label)   # wall pixels, obstacle
    assert hgt[painted].max() <= t["h"] + 1e-3
    wall_low = (label == 0) & (hgt < min(t["h"], 0.4) - 0.01) & (hgt > 0.005)
    assert painted[wall_low].mean() > 0.95


def test_defaults_match_the_v9a_plan():
    d = StructAugCfg()
    assert d.p_sun == 0.0 and d.p_specular == 0.0         # off: v8 called white wall bases floor
    # real steps only, in 5 % of views; flat flooring changes 3x as often, mostly with another flooring
    assert d.level_dz == (0.10, 0.20) and d.p_flat_other > 0.5
    assert d.p_level * (1 - d.p_level_flat) == pytest.approx(0.05)
    assert d.p_door == 1.0                                # strips / edges door-wide only
    assert d.p_plinth == 0.0 and d.p_wall_tiles == 0.0    # 2D plinths / tiles off (they land on doors, objects)
    assert d.plinth_h == (0.08, 0.20)


def test_flush_threshold_stays_floor(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=1, p_threshold=1.0, p_transition=0.0)
    t = info["threshold"]
    assert t and t["h"] <= 0.012
    painted = (img != bgr).any(axis=2)
    assert painted.sum() > 40
    assert np.array_equal(lab, label)                     # drivable: labels unchanged
    assert (label[painted] == 1).all()                    # painted on plain floor only (not under, not wall)


# steps at the low end of the default 10-20 cm: with the scene's camera 0.27 m high, a 20 cm raised floor is
# nearly at eye level and its top a sliver, too few pixels for the metric checks
LEVEL = dict(p_level=1.0, p_level_flat=0.0, p_level_other=0.0, thr_dist=(0.7, 0.9), level_dz=(0.10, 0.12))


def across(g, lv):
    """Per unit ray depth: across-edge coordinate (forward-ish) of every pixel ray, for the edge in info."""
    return -math.sin(lv["th"]) * g.gx1 + math.cos(lv["th"]) * g.gz1


def checker_match(img, g, t, sel):
    """Share of pixels sel whose brightness shows the scene's checker parity at the ray's point at depth t
    (bright 150, dark 90, scaled by the 0.85-1.1 'other room' light: threshold 113), cell interiors only."""
    gx, gz = t * g.gx1, t * g.gz1
    interior = (np.abs(np.mod(gx, 0.3) - 0.15) < 0.12) & (np.abs(np.mod(gz, 0.3) - 0.15) < 0.12)
    sel = sel & interior & (gz < 1.9) & ~((gx < -0.55) & (gz < 1.25))   # visible on the original floor, not under
    parity = ((np.floor(gx / 0.3) + np.floor(gz / 0.3)) % 2)[sel] > 0
    assert sel.sum() > 100
    return np.mean((img[..., 0][sel] > 113) == parity)


def test_step_up_riser_and_raised_floor_in_metres(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=0, p_level_up=1.0, **LEVEL)
    lv = info["level"]
    dz, ve = lv["dz"], lv["ve"]
    assert lv["up"] and 0.10 <= dz <= 0.20 and not lv["flat"]
    g = geom(sa, scene)
    dv = across(g, lv)
    painted = (img != bgr).any(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        te = ve / dv                                       # ray depth where it crosses the edge's vertical plane
    he = CAM_H + te * g.dh1                                # ... at this height
    # riser: rays crossing the edge plane between 0 and dz (and before the wall) are painted, obstacle
    riser = (dv > 0) & (he > 0.003) & (he < dz - 0.003) & (te < g.depth) & (label == 1)
    assert riser.sum() > 50 and painted[riser].all() and (lab[riser] == 0).all()
    # floor before the edge (beyond the riser's 6 mm bottom joint): untouched; floor beyond: obstacle
    vf = g.tf * dv
    before = (label == 1) & (vf < ve - 0.007)
    assert np.array_equal(img[before], bgr[before]) and (lab[before] == 1).all()
    assert (lab[(label != 0) & (vf > ve + 0.005)] == 0).all()
    # the raised top shows the floor texture of the point beneath it, at height dz: reprojection is metric
    top = (he > dz + 0.005) & (dv > 0) & (label == 1) & (vf > ve)
    tt = (dz - CAM_H) / np.where(g.dh1 < -1e-6, g.dh1, -1e-6)
    assert checker_match(img, g, tt, top) > 0.95


def test_step_down_lower_floor_slides_up(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=0, p_level_up=0.0, **LEVEL)
    lv = info["level"]
    dz, ve = lv["dz"], lv["ve"]
    assert not lv["up"] and -0.20 <= dz <= -0.10
    g = geom(sa, scene)
    dv = across(g, lv)
    vf = g.tf * dv
    beyond = (label == 1) & (vf > ve)
    assert (lab[beyond] == 0).all()                        # cliff: not drivable
    before = (label == 1) & (vf < ve - 0.026)              # beyond the lip (<= 2.5 cm): untouched
    assert np.array_equal(img[before], bgr[before]) and (lab[before] == 1).all()
    # rays meeting the floor beyond the edge now show the lower floor at dz, farther along the same ray
    td = (dz - CAM_H) / np.where(g.dh1 < -1e-6, g.dh1, -1e-6)
    far = beyond & (td * dv > ve + 0.09)                   # past the darker band along the riser
    assert checker_match(img, g, td, far) > 0.95
    assert checker_match(bgr, g, g.tf, far) > 0.95 and checker_match(img, g, g.tf, far) < 0.8  # really moved


@pytest.mark.parametrize("other", [0.0, 1.0])
def test_drivable_level_change_stays_floor(sa, scene, other):
    # flat flooring change, with the same floor or another flooring beyond the edge: floor all the same
    bgr, label, _ = scene
    for up in (0.0, 1.0):
        img, lab, info = run(sa, scene, seed=3, **dict(LEVEL, p_level_flat=1.0, p_level_up=up, p_flat_other=other))
        lv = info["level"]
        assert lv["flat"] and abs(lv["dz"]) <= 0.01 and lv["other"] == bool(other)
        painted = (img != bgr).any(axis=2)
        assert painted.sum() > 300 and np.array_equal(lab, label)
        assert (label[painted] == 1).all()


@pytest.fixture(scope="module")
def scene_occ(scene):
    """scene + a flat object 0.5 m in front of the camera over columns 260-380, hiding the floor behind it."""
    bgr, label, depth = (a.copy() for a in scene)
    bb = np.zeros(label.shape, bool)
    bb[:, 260:380] = depth[:, 260:380] > 500
    bgr[bb], label[bb], depth[bb] = 30, 0, 500
    return bgr, label, depth, bb


def test_level_change_respects_closer_objects(sa, scene_occ):
    bgr, label, depth, bb = scene_occ
    for up in (0.0, 1.0):
        for seed in range(4):
            img, lab, info = run(sa, (bgr, label, depth), seed=seed, **dict(LEVEL, p_level_up=up))
            assert info["level"]
            assert np.array_equal(img[bb], bgr[bb]) and np.array_equal(lab[bb], label[bb])


def test_light_keeps_labels_and_spares_under_and_walls(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=2, p_sun=1.0, p_specular=1.0)
    assert info["sun"] and info["specular"]
    painted = (img != bgr).any(axis=2)
    assert np.array_equal(lab, label) and painted.any()
    assert (label[painted] == 1).all()


def test_bright_step_beyond_the_line_labels_unchanged(sa, scene):
    bgr, label, _ = scene
    for seed in range(4):
        img, lab, info = run(sa, scene, seed=seed, p_bright_step=1.0)
        b = info["bright_step"]
        assert b and 1.3 <= b["gain"] <= 2.5
        assert np.array_equal(lab, label)                     # labels unchanged
        g = geom(sa, scene)
        across = -math.sin(b["th"]) * g.gx + math.cos(b["th"]) * g.gz
        lum0, lum1 = bgr.astype(np.float32).mean(2), img.astype(np.float32).mean(2)
        beyond = (g.valid & (across > b["ve"])).astype(np.uint8)
        # the soft edge (Gaussian, sigma <= 4 px: kernel radius 16 px) changes pixels up to 16 px across the line
        k = np.ones((41, 41), np.uint8)
        ring = (cv2.dilate(beyond, k) > 0) & ~(cv2.erode(beyond, k) > 0)
        far, near = (beyond > 0) & ~ring, (beyond == 0) & ~ring
        # beyond the line: brighter, floor and wall alike (not a "bright = floor" cue); before it: untouched
        for sel in (far & (label == 1), far & (label == 0)):
            if sel.sum() > 50:
                assert (lum1[sel] > lum0[sel] + 5).mean() > 0.95
        assert (far & (label == 1)).sum() > 300
        assert np.array_equal(img[near], bgr[near])


def test_bright_step_reuses_the_strip_edge(sa, scene):
    _, _, info = run(sa, scene, seed=1, p_threshold=1.0, p_transition=0.0, p_bright_step=1.0)
    t, b = info["threshold"], info["bright_step"]
    assert t and b and (b["th"], b["ve"]) == (t["th"], t["v1"])   # brighter beyond the strip's far side


def test_bright_step_off_draws_nothing(sa, scene):
    # p_bright_step = 0 (default) consumes no random number: the other augmentations come out as before
    full = dict(p_threshold=1.0, p_level=1.0, p_sun=1.0)
    a, b = run(sa, scene, seed=4, **full), run(sa, scene, seed=4, p_bright_step=0.0, **full)
    assert np.array_equal(a[0], b[0]) and "bright_step" not in b[2]


def test_under_pixels_never_painted(sa, scene):
    bgr, label, _ = scene
    under = label == UNDER
    for seed in range(15):
        img, lab, info = run(sa, scene, seed=seed, p_plinth=1.0, p_threshold=1.0, p_level=1.0, p_sun=1.0,
                             p_specular=1.0)
        assert (img[under] == bgr[under]).all()
        assert ((lab != label) <= ((label != 0) & (lab == 0))).all()  # only floor -> obstacle (level changes)
        lv = info.get("level")
        if not lv or lv["flat"]:          # 'under' only becomes obstacle beyond a non-drivable level edge
            assert (lab[under] == UNDER).all()


def test_deterministic_given_the_seed(sa, scene):
    full = dict(p_plinth=1.0, p_threshold=1.0, p_level=1.0, p_sun=1.0, p_specular=1.0)
    a, b, c = (run(sa, scene, seed=s, **full) for s in (5, 5, 6))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_off_by_default(renders):
    # train.py without --aug-structures builds no StructAug: the training pipeline is the one from before
    from floor_seg.train import build_parser, datasets, struct_aug

    args = build_parser().parse_args(["--data", str(renders), "--out", "x"])
    assert struct_aug(args) is None
    tr, _ = datasets([str(renders)], ftex=None, saug=struct_aug(args))
    assert all(d.saug is None for d in tr.datasets)
    on = build_parser().parse_args(["--data", "d", "--out", "x", "--aug-structures", "--aug-p-level", "0.5"])
    assert struct_aug(on).cfg.p_level == 0.5 and struct_aug(on).cfg.p_plinth == StructAugCfg().p_plinth
    assert struct_aug(on).cfg.p_bright_step == 0.0
    br = build_parser().parse_args(["--data", "d", "--out", "x", "--aug-structures", "--aug-p-bright-step", "0.3"])
    assert struct_aug(br).cfg.p_bright_step == 0.3
    # run A of the v9 plan: no plinth, no wall tiles (sun / specular off by default)
    a = struct_aug(build_parser().parse_args(["--data", "d", "--out", "x", "--aug-structures", "--aug-p-plinth", "0",
                                              "--aug-p-wall-tiles", "0"])).cfg
    assert (a.p_plinth, a.p_wall_tiles, a.p_sun, a.p_specular) == (0.0, 0.0, 0.0, 0.0) and a.p_threshold > 0


def test_zero_probabilities_are_identity(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene)
    assert img is bgr and lab is label and not any(info.values())


def test_hook_in_training_dataset(renders):
    # with the augmentation on, training items still come out (fake renders: lower half floor, plane depth)
    from floor_seg.train import datasets

    full = StructAugCfg(p_plinth=1.0, p_threshold=1.0, p_level=1.0, p_sun=1.0, p_specular=1.0)
    tr, va = datasets([str(renders)], saug=StructAug(full))
    x, y = tr[0]
    assert x.shape == (3, 192, 320) and y.shape == (192, 320)
    assert va.datasets[0].saug is None


def test_training_items_pass_rendered_plinths_and_doorways(renders, monkeypatch):
    # a thor_render --plinths set: <name>_plinth.png and 'doorways' in stats.jsonl reach StructAug.apply
    import json

    import cv2

    from floor_seg.train import build_parser, datasets, struct_aug

    stats = [json.loads(line) for line in (renders / "stats.jsonl").read_text().splitlines()]
    for s in stats:
        s["doorways"] = [[-0.5, 1.5, 0.5, 1.5]]
        pm = np.zeros((H, W), np.uint8)
        pm[H // 2 - 8:H // 2] = 255                   # just above the fake floor (lower half)
        cv2.imwrite(str(renders / f"{s['name']}_plinth.png"), pm)
    (renders / "stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    args = build_parser().parse_args(["--data", str(renders), "--out", "x", "--aug-structures",
                                      "--aug-p-plinth-render", "0.5", "--aug-p-door-threshold", "0.7"])
    saug = struct_aug(args)
    assert (saug.cfg.p_plinth_render, saug.cfg.p_door_threshold) == (0.5, 0.7)
    seen = []
    orig = saug.apply

    def spy(*a, **kw):
        seen.append(kw)
        return orig(*a, **kw)

    monkeypatch.setattr(saug, "apply", spy)
    tr, _ = datasets([str(renders)], saug=saug)
    x, _ = tr[0]
    assert x.shape == (3, 192, 320)
    kw = seen[0]
    assert kw["plinth"] is not None and kw["plinth"].max() == 255 and kw["plinth_src"] is not None
    assert kw["doorways"] == [[-0.5, 1.5, 0.5, 1.5]]


def test_under_value_matches_train():
    from floor_seg.train import UNDER as TRAIN_UNDER

    assert UNDER == TRAIN_UNDER
