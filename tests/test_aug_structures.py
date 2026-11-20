"""Structure augmentations (floor_seg/aug_structures.py) on an analytic scene: a floor plane and a wall 2 m ahead,
seen by the calibrated lens from the robot's pose. Synthetic only, CPU."""
import math
import random
from dataclasses import replace

import numpy as np
import pytest

from floor_seg import DEFAULT_CALIB
from floor_seg.aug_structures import UNDER, Geom, StructAug, StructAugCfg
from floor_seg.floor_tex import H, W, floor_coords, heights

CAM_H, PITCH, ROLL, WALL_Z = 0.27, 18.0, 1.5, 2.0
OFF = StructAugCfg(p_plinth=0.0, p_threshold=0.0, p_level=0.0, p_sun=0.0, p_specular=0.0)


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


@pytest.mark.parametrize("plain", [0.0, 1.0])
def test_plinth_is_on_the_wall_base_with_its_metric_height(sa, scene, plain):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, p_plinth=1.0, p_plinth_plain=plain)
    hp = info["plinth"]["h"]
    painted = (img != bgr).any(axis=2)
    hgt = geom(sa, scene).hgt
    assert painted.sum() > 1000
    assert (label[painted] == 0).all()                    # only wall pixels
    assert np.array_equal(lab, label)                     # obstacle stays obstacle; nothing else changes
    assert hgt[painted].max() <= hp + 1e-3 and hgt[painted].min() > -0.02
    # it covers the whole wall base up to hp, in metres (the wall spans the image width)
    wall_low = (label == 0) & (hgt < hp - 0.01) & (hgt > 0.005)
    assert painted[wall_low].mean() > 0.95
    assert info["plinth"]["plain"] == bool(plain)


def test_flush_threshold_stays_floor(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=1, p_threshold=1.0, p_transition=0.0)
    t = info["threshold"]
    assert t and t["h"] <= 0.012
    painted = (img != bgr).any(axis=2)
    assert painted.sum() > 40
    assert np.array_equal(lab, label)                     # drivable: labels unchanged
    assert (label[painted] == 1).all()                    # painted on plain floor only (not under, not wall)


LEVEL = dict(p_level=1.0, p_level_flat=0.0, p_level_other=0.0, thr_dist=(0.7, 0.9))


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
    assert lv["up"] and 0.02 <= dz <= 0.18 and not lv["flat"]
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
    assert not lv["up"] and -0.18 <= dz <= -0.02
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


def test_drivable_level_change_stays_floor(sa, scene):
    bgr, label, _ = scene
    for up in (0.0, 1.0):
        img, lab, info = run(sa, scene, seed=3, **dict(LEVEL, p_level_flat=1.0, p_level_up=up))
        lv = info["level"]
        assert lv["flat"] and abs(lv["dz"]) <= 0.01
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


def test_under_value_matches_train():
    from floor_seg.train import UNDER as TRAIN_UNDER

    assert UNDER == TRAIN_UNDER
