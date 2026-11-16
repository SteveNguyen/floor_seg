"""Mask / label conventions of the training data and the floor retexturing augmentation."""
import random

import cv2
import numpy as np

from floor_seg import DEFAULT_CALIB
from floor_seg.floor_tex import FloorTex, floor_coords, lens_rays
from floor_seg.model import IN_H, IN_W
from floor_seg.train import IGNORE, Renders


def test_training_labels_are_0_1(renders):
    ds = Renders(str(renders), ["h00003_0"], train=False)
    x, y = ds[0]
    assert x.shape == (3, IN_H, IN_W) and y.shape == (IN_H, IN_W)
    assert set(np.unique(y.numpy())) == {0, 1}  # stored 0/255 -> 0 not floor, 1 floor
    assert y[-1].min() == 1 and y[0].max() == 0


def test_augmentation_only_adds_ignore(renders):
    random.seed(0)
    ds = Renders(str(renders), ["h00003_0"] * 20, train=True)
    for i in range(20):
        _, y = ds[i]
        assert set(np.unique(y.numpy())) <= {0, 1, IGNORE}  # IGNORE only from rotation corners


def test_floor_coords_flat_floor():
    # camera 0.3 m above the floor, pitch 20 deg down, no roll: depth of floor points from the ray geometry,
    # then floor_coords must give a forward distance of h / tan(pitch + ray angle) for the image centre column
    xn, yn = lens_rays(DEFAULT_CALIB)
    assert xn.shape == (360, 640)
    h, pitch = 0.3, 20.0
    p = np.radians(pitch)
    v = np.arange(200, 360)
    u = np.full_like(v, 320)
    ray_down = np.arctan(yn[v, u])  # angle of the ray below the optical axis (approximately, small xn)
    dist = h / np.tan(p + ray_down)  # forward distance of the floor point
    depth = dist * np.cos(p) + h * np.sin(p)  # optical z of that point
    gx, gz = floor_coords(xn[v, u], yn[v, u], depth, pitch, 0.0)
    np.testing.assert_allclose(gz, dist, rtol=0.02)
    assert np.abs(gx).max() < 0.05


def test_texture_swap_negatives_touch_only_low_obstacle_surfaces(tmp_path, monkeypatch):
    # with the camera height, obstacle surfaces (split at depth edges) reaching below NEG_MAX_H may get a floor
    # texture; floor pixels and surfaces entirely above NEG_MAX_H never change (labels never change at all)
    import floor_seg.floor_tex as ftm
    monkeypatch.setattr(ftm, "P_NEG", 1.0)
    monkeypatch.setattr(ftm, "P_RETEX", 0.0)
    monkeypatch.setattr(ftm, "P_RUG", 0.0)
    monkeypatch.setattr(ftm, "P_SUN", 0.0)
    monkeypatch.setattr(ftm, "P_GLARE", 0.0)
    (tmp_path / "floor").mkdir()
    cv2.imwrite(str(tmp_path / "floor" / "t.jpg"), np.random.default_rng(1).integers(0, 256, (64, 64, 3), np.uint8))
    ft = FloorTex(str(tmp_path), DEFAULT_CALIB)
    bgr = np.full((360, 640, 3), 128, np.uint8)
    floor = np.zeros((360, 640), np.uint8)
    floor[200:] = 1
    depth = np.full((360, 640), 1500, np.uint16)  # rows 100-199: near surface reaching low
    depth[:100] = 4000                            # rows 0-99: far surface, all of it high (depth edge between)
    hgt = ftm.heights(ft.xn, ft.yn, depth / 1000.0, 0.0, 0.0, 0.3)
    assert hgt[:100].min() > ftm.NEG_MAX_H and hgt[101:200].min() < ftm.NEG_MAX_H
    out = ft.apply(bgr, floor, depth, 0.0, 0.0, random.Random(0), cam_h=0.3)
    diff = np.abs(out.astype(int) - bgr).max(axis=2)
    assert diff[:100].max() <= 1 and diff[200:].max() <= 1
    assert (diff[101:200] > 1).mean() > 0.5


def test_retexturing_keeps_non_floor_pixels(tmp_path):
    # labels never change; pixels outside the floor mask must stay as rendered (+-1 for float rounding)
    for g in ("floor", "rug"):
        (tmp_path / g).mkdir()
        cv2.imwrite(str(tmp_path / g / "t.jpg"), np.random.default_rng(1).integers(0, 256, (64, 64, 3), np.uint8))
    ft = FloorTex(str(tmp_path), DEFAULT_CALIB)
    bgr = np.random.default_rng(2).integers(0, 256, (360, 640, 3), np.uint8)
    floor = np.zeros((360, 640), np.uint8)
    floor[200:] = 1
    depth = np.full((360, 640), 1500, np.uint16)
    changed = False
    for seed in range(5):
        out = ft.apply(bgr, floor, depth, 15.0, 0.0, random.Random(seed))
        assert np.abs(out[:200].astype(int) - bgr[:200]).max() <= 1
        changed |= bool(np.abs(out[200:].astype(int) - bgr[200:]).max() > 1)
    assert changed
