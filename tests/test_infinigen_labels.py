"""floor_seg.infinigen_labels (the Blender-free parts of the Infinigen generator): camera conventions match the
thor sets', labels by the height rule + object identity, lens remap, room choice, stats format."""
import json
import math
import random

import numpy as np
import pytest
from floor_seg import DEFAULT_CALIB
from floor_seg import infinigen_labels as L
from floor_seg.floor_tex import floor_coords, heights, lens_rays


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 15, 0), (37, 22, -2.5), (-120, 5, 3)])
def test_camera_axes_match_the_floor_tex_conventions(yaw, pitch, roll):
    """World height and forward distance of a camera ray agree with floor_tex.heights / floor_coords, so the stored
    pitch / roll / h drive the training-time floor retexturing exactly as for the thor sets."""
    rng = np.random.default_rng(0)
    xn, yn, d, h = rng.uniform(-1, 1, 50), rng.uniform(-0.6, 0.6, 50), rng.uniform(0.5, 4, 50), 0.31
    v = np.stack([xn * d, -yn * d, d], -1)  # camera frame: x right, y up, z forward
    w = v @ L.camera_axes(yaw, pitch, roll).T
    assert np.allclose(h + w[:, 2], heights(xn, yn, d, pitch, roll, h), atol=1e-9)
    gx, gz = floor_coords(xn, yn, d, pitch, roll)
    fwd = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    right = np.array([math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))])
    assert np.allclose(w[:, :2] @ fwd, gz, atol=1e-9) and np.allclose(w[:, :2] @ right, gx, atol=1e-9)


def test_blender_rotation_is_a_rotation_looking_forward():
    axes = L.camera_axes(30, 18, 2)
    R = L.blender_rotation(axes)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ np.array([0, 0, -1.0]), axes[:, 2])  # a Blender camera looks down its local -z
    assert axes[2, 2] < 0  # pitched down: the optical axis points below the horizon


def test_labels_height_rule_with_rugs_and_flat_objects():
    floor_z, cam = -0.005, np.array([1.0, 2.0, 0.3])
    axes = L.camera_axes(0, 20, 0)
    pts = {  # (world z, object index) -> expected floor
        "floor": (floor_z + 0.002, L.IDX_FLOOR, True),
        "floor_too_high": (floor_z + 0.03, L.IDX_FLOOR, False),     # e.g. a bump in a floor mesh: height rule
        "rug_top": (floor_z + 0.018, L.IDX_RUG, True),             # 1-2 cm thick rugs are floor by rule
        "rug_far_above": (floor_z + 0.2, L.IDX_RUG, False),
        "cable": (floor_z + 0.006, L.IDX_FLAT, True),
        "object_on_floor": (floor_z + 0.005, L.IDX_OTHER, False),   # bottom of a box / plinth: not floor
        "wall": (floor_z + 1.0, L.IDX_OTHER, False),
    }
    pos = np.array([[[2.0, 2.0, z] for z, _, _ in pts.values()]])
    idx = np.array([[i for _, i, _ in pts.values()]], np.float32)
    pos = np.concatenate([pos, np.zeros((1, 1, 3))], 1)            # + one pixel that hit nothing
    idx = np.concatenate([idx, np.array([[L.IDX_FLOOR]], np.float32)], 1)
    floor, depth = L.labels_from_passes(pos, idx, cam, axes, floor_z)
    assert floor[0].tolist() == [e for _, _, e in pts.values()] + [False]
    p = np.array([2.0, 2.0, floor_z + 0.002])
    assert depth[0, 0] == pytest.approx((p - cam) @ axes[:, 2]) and depth[0, -1] == 0


def test_lens_maps_match_thor_render_and_rays_match_floor_tex():
    pytest.importorskip("cv2")
    from floor_seg.thor_render import lens_maps as thor_maps
    mx, my, wr, hr, f = L.lens_maps(DEFAULT_CALIB, pixel_center=0.0)
    tx, ty, twr, thr, tf, _ = thor_maps(DEFAULT_CALIB)
    assert (wr, hr) == (twr, thr) and f == pytest.approx(tf)
    assert np.allclose(mx, tx, atol=1e-4) and np.allclose(my, ty, atol=1e-4)
    # Blender pixel centres: the render pixel sampled for each output pixel looks along that pixel's lens ray
    mx, my, wr, hr, f = L.lens_maps(DEFAULT_CALIB)
    xn, yn = lens_rays(DEFAULT_CALIB)
    assert np.allclose((mx + 0.5 - wr / 2) / f, xn, atol=1e-5) and np.allclose((my + 0.5 - hr / 2) / f, yn, atol=1e-5)
    d = L.pinhole_dirs(wr, hr, f)
    assert d.shape == (hr, wr, 3) and np.allclose(d[0, 0, :2], [(0.5 - wr / 2) / f, -(0.5 - hr / 2) / f])


def test_finish_output_format():
    maps = L.lens_maps(DEFAULT_CALIB)
    wr, hr = maps[2], maps[3]
    rgb_r = np.full((hr, wr, 3), 100, np.uint8)
    floor_r = np.zeros((hr, wr), bool)
    floor_r[hr // 2:] = True
    rgb, floor, depth, under = L.finish(rgb_r, floor_r, np.full((hr, wr), 1.234), maps, under_r=floor_r)
    assert under.dtype == np.uint8 and (under == floor).all()
    assert L.finish(rgb_r, floor_r, np.full((hr, wr), 1.0), maps)[3] is None
    assert rgb.shape == (L.H, L.W, 3) and floor.dtype == np.uint8 and set(np.unique(floor)) <= {0, 255}
    assert depth.dtype == np.uint16 and depth[L.H // 2, L.W // 2] == 1234
    assert floor[-1].all() and not floor[0].any()


def room_heights():
    """A 3 x 2 m room raster (2 cm cells): open floor (clear), a bed (underside 0.2 m), a slatted shelf (0.3 m slats
    4 cm apart), a box standing on the floor (contact), and a 30 x 30 cm free pocket enclosed by a ring of chair
    stretchers (0.1 m). Returns (bounds, heights, cell xy)."""
    bounds = (0.0, 0.0, 3.0, 2.0)
    xs, ys = L.raster_xy(*bounds)
    h = np.full(xs.shape, L.CLEAR_H, np.float32)
    h[(xs > 0.2) & (xs < 1.0) & (ys > 0.2) & (ys < 1.6)] = 0.2                       # bed
    slats = (xs > 1.4) & (xs < 2.0) & (ys > 1.4) & (ys < 1.9)
    h[slats & ((np.round(ys / 0.02) % 4) < 2)] = 0.3                                   # 4 cm slats, 4 cm gaps
    h[(xs > 1.2) & (xs < 1.5) & (ys > 0.2) & (ys < 0.5)] = 0.0                         # box on the floor
    ring = (xs > 2.2) & (xs < 2.7) & (ys > 0.2) & (ys < 0.7)
    inner = (xs > 2.3) & (xs < 2.6) & (ys > 0.3) & (ys < 0.6)
    h[ring & ~inner] = 0.1                                                             # stretchers around a pocket
    return bounds, h, xs, ys


def test_underside_map_thor_rules():
    bounds, h, xs, ys = room_heights()
    um = L.underside_map(*bounds, h)
    at = lambda x, y: bool(um.lookup(np.array([x]), np.array([y]))[0])  # noqa: E731
    assert at(0.6, 0.9)                     # under the bed
    assert not at(2.0, 1.0) and not at(0.1, 0.1)   # open floor
    assert at(1.7, 1.65) and at(1.7, 1.63)  # slats and the 4 cm gaps between them (closed)
    assert not at(1.35, 0.35)               # box standing on the floor: contact, not under
    assert at(2.45, 0.45)                   # free pocket enclosed by stretchers (< 0.25 m^2): unreachable, under
    with pytest.raises(ValueError):
        L.underside_map(*bounds, h[:-1])


def test_under_mask_is_a_subset_of_floor():
    bounds, h, _, _ = room_heights()
    um = L.underside_map(*bounds, h)
    pos = np.array([[[0.6, 0.9, 0.0], [0.6, 0.9, 0.0], [2.0, 1.0, 0.0], [5.0, 5.0, 0.0]]])  # last: outside the room
    floor = np.array([[True, False, True, True]])
    assert L.under_mask(pos, floor, um).tolist() == [[True, False, False, False]]


def test_heavy_under_views_capped_per_room():
    assert L.keep_view(0.2, 5, 5)                                       # not heavy: always kept
    assert L.keep_view(0.9, 0, 5) and L.keep_view(0.9, 1, 5) and not L.keep_view(0.9, 2, 5)  # ceil(5 / 3) = 2
    assert L.keep_view(0.9, 0, 3) and not L.keep_view(0.9, 1, 3)


def test_fisheye_polynomial_cannot_match_our_lens():
    """Why the generator renders a pinhole and remaps: the radial polynomial is centred, our principal point is
    not (~16 px at 640x360)."""
    k, rms, worst = L.fit_fisheye_poly(DEFAULT_CALIB)
    assert len(k) == 5 and rms > 1.0 and worst > rms


def test_room_types_and_seeds_are_deterministic():
    a = [L.room_type(L.new_rng(0, r)) for r in range(300)]
    assert a == [L.room_type(L.new_rng(0, r)) for r in range(300)]
    assert "house" not in a and set(a) == {"livingroom", "bedroom", "kitchen", "diningroom", "bathroom"}
    assert a.count("livingroom") > a.count("diningroom")
    assert L.room_type(random.Random(1), {"house": 1.0}) == "house"
    name, rgb = L.wall_paint(random.Random(3))
    assert name in {n for n, _, _ in L.WALL_PALETTE} and all(0 <= c <= 1 for c in rgb)


def test_floor_kind_from_node_groups():
    assert L.floor_kind(["_carpet_cell_randomizer", "carpet_weaver"]) == "carpet"
    assert L.floor_kind(["wood_planks", "coord_warp"]) == "wood"
    assert L.floor_kind(["brick_concrete"]) == "concrete"
    assert L.floor_kind(["tile_indoor_ground", "concrete_layer"]) == "tiles"
    assert L.floor_kind([]) == "other"


def test_reject_carpet_in_wet_rooms_and_concrete_mostly():
    assert L.p_reject("carpet", "bathroom") == 1.0
    assert L.p_reject("carpet", "kitchen") == 1.0
    assert L.p_reject("carpet", "bedroom") == L.P_REJECT_CARPET
    assert L.p_reject("concrete", "livingroom") == L.P_REJECT_CONCRETE
    assert L.p_reject("wood", "bathroom") == 0.0
    assert L.p_reject("tiles", "livingroom") == 0.0


def test_stats_line_is_what_train_reads():
    s = json.loads(L.stats_line("r00012_3", 12, "bedroom", 99, (1, 2, 30, 0.3, 15, -1), 0.4, {"cables": 1}))
    assert s["name"] == "r00012_3" and s["house"] == 12 and s["kind"] == "floor"
    assert (s["pitch"], s["roll"], s["h"]) == (15, -1, 0.3) and s["cables"] == 1
