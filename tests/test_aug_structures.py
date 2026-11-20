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
OFF = StructAugCfg(p_plinth=0.0, p_threshold=0.0, p_sun=0.0, p_specular=0.0)


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
    img, lab, info = run(sa, scene, seed=1, p_threshold=1.0, p_step=0.0, p_transition=0.0)
    t = info["threshold"]
    assert t and not t["raised"] and t["h"] <= 0.012
    painted = (img != bgr).any(axis=2)
    assert painted.sum() > 40
    assert np.array_equal(lab, label)                     # drivable: labels unchanged
    assert (label[painted] == 1).all()                    # painted on plain floor only (not under, not wall)


def test_raised_step_is_obstacle_with_its_front_face(sa, scene):
    bgr, label, _ = scene
    for seed in range(20):                                # a placement off the 'under' patch
        img, lab, info = run(sa, scene, seed=seed, p_threshold=1.0, p_step=1.0)
        if info.get("threshold"):
            break
    t = info["threshold"]
    assert t["raised"] and t["h"] >= 0.025
    g = geom(sa, scene)
    new_obst = (label != 0) & (lab == 0)
    assert new_obst.any() and ((lab != label) <= new_obst).all()      # only floor -> obstacle changes
    assert (img != bgr).any(axis=2)[new_obst].all()                   # every relabelled pixel is painted
    # floor points under the box footprint are covered; covered floor lies behind its front edge (v >= v0)
    a, n = (math.cos(t["th"]), math.sin(t["th"])), (-math.sin(t["th"]), math.cos(t["th"]))
    u = a[0] * g.gx + a[1] * g.gz
    v = n[0] * g.gx + n[1] * g.gz
    foot = (label == 1) & (np.abs(u - t["uc"]) < t["len"] / 2 - 0.02) & (v > t["v0"] + 0.005) & (v < t["v1"] - 0.005)
    assert foot.any() and (lab[foot] == 0).all()
    assert (v[new_obst & (label == 1)] > t["v0"] - 0.01).all()
    # the front face and top hide floor beyond the footprint: more pixels than the footprint alone
    inside = (label == 1) & (np.abs(u - t["uc"]) < t["len"] / 2) & (v >= t["v0"]) & (v <= t["v1"])
    assert new_obst.sum() > inside.sum()
    assert (lab[label == UNDER] == UNDER).all()


def test_light_keeps_labels_and_spares_under_and_walls(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene, seed=2, p_sun=1.0, p_specular=1.0)
    assert info["sun"] and info["specular"]
    painted = (img != bgr).any(axis=2)
    assert np.array_equal(lab, label) and painted.any()
    assert (label[painted] == 1).all()


def test_under_pixels_never_painted_or_relabelled(sa, scene):
    bgr, label, _ = scene
    under = label == UNDER
    for seed in range(15):
        img, lab, _ = run(sa, scene, seed=seed, p_plinth=1.0, p_threshold=1.0, p_sun=1.0, p_specular=1.0)
        assert (lab[under] == UNDER).all() and (img[under] == bgr[under]).all()
        assert ((lab != label) <= ((label != 0) & (lab == 0))).all()  # only floor -> obstacle (raised steps)


def test_deterministic_given_the_seed(sa, scene):
    full = dict(p_plinth=1.0, p_threshold=1.0, p_sun=1.0, p_specular=1.0)
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
    on = build_parser().parse_args(["--data", "d", "--out", "x", "--aug-structures", "--aug-p-step", "0.5"])
    assert struct_aug(on).cfg.p_step == 0.5 and struct_aug(on).cfg.p_plinth == StructAugCfg().p_plinth


def test_zero_probabilities_are_identity(sa, scene):
    bgr, label, _ = scene
    img, lab, info = run(sa, scene)
    assert img is bgr and lab is label and not any(info.values())


def test_hook_in_training_dataset(renders):
    # with the augmentation on, training items still come out (fake renders: lower half floor, plane depth)
    from floor_seg.train import datasets

    full = StructAugCfg(p_plinth=1.0, p_threshold=1.0, p_sun=1.0, p_specular=1.0)
    tr, va = datasets([str(renders)], saug=StructAug(full))
    x, y = tr[0]
    assert x.shape == (3, 192, 320) and y.shape == (192, 320)
    assert va.datasets[0].saug is None


def test_under_value_matches_train():
    from floor_seg.train import UNDER as TRAIN_UNDER

    assert UNDER == TRAIN_UNDER
