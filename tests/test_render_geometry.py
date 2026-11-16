"""Renderer geometry used by the labels (no AI2-THOR needed): world projection, underside map, table tops."""
import itertools

import numpy as np

from floor_seg.thor_render import UndersideMap, height_above_floor, support_tops, world_xz


def box(otype, x, z, y):
    """Fake THOR object: axis-aligned box over x = (x0, x1), z = (z0, z1), y = (y0, y1)."""
    corners = [list(p) for p in itertools.product(x, y, z)]  # (x, y, z) like THOR cornerPoints
    return {"objectType": otype, "objectId": f"{otype}|{x[0]}|{z[0]}", "objectOrientedBoundingBox": None,
            "axisAlignedBoundingBox": {"cornerPoints": corners}}


def test_world_xz_yaw_convention():
    # camera at (x 1, z 2) facing +x (yaw 90, Unity: positive yaw turns right from +z), level
    depth = np.full((101, 201), 3.0, np.float32)
    wx, wz = world_xz(depth, 100.0, 1.0, 2.0, 90.0, 0.0, 0.0)
    assert abs(wx[50, 100] - 4.0) < 0.05 and abs(wz[50, 100] - 2.0) < 0.05   # straight ahead
    # 50 px right of the centre at f = 100: 1.5 m to the camera's right, which is -z when facing +x
    assert abs(wx[50, 150] - 4.0) < 0.05 and abs(wz[50, 150] - 0.5) < 0.05


def test_world_xz_pitch_brings_the_point_closer():
    # looking 30 deg down, the image centre at depth 2 is 2 cos 30 ahead horizontally
    depth = np.full((101, 201), 2.0, np.float32)
    wx, wz = world_xz(depth, 100.0, 0.0, 0.0, 0.0, 30.0, 0.0)
    assert abs(wz[50, 100] - 2 * np.cos(np.radians(30))) < 0.03 and abs(wx[50, 100]) < 0.03


def test_looking_up_camera_geometry():
    # the underside camera: 10 m below the floor at (x 1, z 2), pitch -90 (looking up), depth 10.5 everywhere
    depth = np.full((100, 200), 10.5, np.float32)                # even sizes, as rendered: centre = (50, 100)
    hgt = height_above_floor(depth, 100.0, -10.0, -90.0, 0.0)
    wx, wz = world_xz(depth, 100.0, 1.0, 2.0, 0.0, -90.0, 0.0)
    assert np.allclose(hgt, 0.5, atol=1e-4)                      # a surface 0.5 m above the floor
    assert abs(wx[50, 100] - 1.0) < 1e-3 and abs(wz[50, 100] - 2.0) < 1e-3
    assert wz[99, 100] > 2.0 and wx[50, 199] > 1.0              # image down = +z, image right = +x


def surface(um, x, z, height):
    """Dense samples of a horizontal surface over the rectangle x = (x0, x1), z = (z0, z1)."""
    xx, zz = np.meshgrid(np.arange(*x, 0.01), np.arange(*z, 0.01))
    um.add(xx.ravel(), zz.ravel(), np.full(xx.size, height))


def test_underside_map_lowest_surface_decides():
    um = UndersideMap(0, 0, 10, 2)
    surface(um, (0, 10), (0, 2), 2.8)        # ceiling everywhere
    surface(um, (0, 2), (0, 1), 0.25)        # bed slats: under
    surface(um, (3, 4), (0, 1), 1.5)         # loft bed: drivable under
    surface(um, (5, 6), (0, 1), 0.72)        # dining table: drivable under
    surface(um, (7, 7.5), (0, 0.5), 0.42)    # chair seat: under
    surface(um, (8, 8.2), (0, 0.2), 0.0)     # box standing on the floor: contact, the floor is covered, not under
    surface(um, (8.6, 9.0), (1, 1.5), 0.25)  # ...but a 25 cm overhang (bed frame) is
    pts = np.array([[1, 0.5], [3.5, 0.5], [5.5, 0.5], [7.2, 0.2], [8.1, 0.1], [9.5, 0.5], [8.8, 1.2]])
    um.finish()
    assert um.lookup(pts[:, 0], pts[:, 1]).tolist() == [True, False, False, True, False, False, True]
    assert not um.unknown[1:-1, 1:-1].any()


def test_slatted_shelf_is_under_but_a_robot_wide_gap_is_not():
    um = UndersideMap(0, 0, 3, 1)
    surface(um, (0, 3), (0, 1), 2.8)
    for x in np.arange(0.2, 1.0, 0.08):      # slats 4 cm wide every 8 cm, 0.15 m high: 4 cm gaps
        surface(um, (x, x + 0.04), (0.2, 0.8), 0.15)
    surface(um, (1.5, 1.8), (0.2, 0.8), 0.15)
    surface(um, (2.1, 2.4), (0.2, 0.8), 0.15)  # 30 cm gap between these two
    um.finish()
    gaps = np.arange(0.25, 0.95, 0.08) + 0.01   # floor points between slats
    assert um.lookup(gaps, np.full(len(gaps), 0.5)).all()
    assert not um.lookup(np.array([1.95]), np.array([0.5]))[0]


def test_enclosed_small_pocket_is_under_but_a_walled_room_is_not():
    um = UndersideMap(0, 0, 6, 3)
    surface(um, (0, 6), (0, 3), 2.8)
    xx, zz = np.meshgrid(np.arange(0, 6, 0.01), np.arange(0, 3, 0.01))
    r = np.hypot(xx - 1.0, zz - 1.0)
    ring = (r > 0.17) & (r < 0.2)             # bar stool foot ring, 0.25 m high, ~0.35 m inside
    um.add(xx[ring], zz[ring], np.full(ring.sum(), 0.25))
    wall = ((np.abs(xx - 3.0) < 0.05) | (np.abs(xx - 5.5) < 0.05)) & (zz > 0.2) & (zz < 2.8) | \
           ((np.abs(zz - 0.2) < 0.05) | (np.abs(zz - 2.8) < 0.05)) & (xx > 3.0) & (xx < 5.5)
    um.add(xx[wall], zz[wall], np.full(wall.sum(), 0.0))   # 2.5 x 2.6 m walled room
    um.finish()
    assert um.lookup(np.array([1.0, 4.2, 1.6]), np.array([1.0, 1.5, 1.0])).tolist() == [True, False, False]


def test_house_bounds_cover_far_furniture():
    from floor_seg.thor_render import house_bounds

    reach = [{"x": 0.0, "z": 0.0}, {"x": 0.5, "z": 0.25}]         # few reachable cells
    objs = [box("Stool", (4.0, 4.4), (3.0, 3.4), (0, 0.7))]        # furniture 4 m away
    x0, z0, x1, z1 = house_bounds(reach, objs)
    assert x0 <= -0.5 and z0 <= -0.5 and x1 >= 4.4 and z1 >= 3.4


def test_rest_on_floor_moves_floating_and_sunken_clutter_only():
    from floor_seg.thor_render import rest_on_floor

    def obj(oid, y):
        return {"objectId": oid, "axisAlignedBoundingBox": {"cornerPoints": [[0, y, 0], [1, y + 0.3, 1]]}}

    house = {"objects": [{"id": "Box|clutter|0", "position": {"x": 0, "y": 0.5, "z": 0}},
                         {"id": "Pot|clutter|1", "position": {"x": 0, "y": 0.1, "z": 0}},
                         {"id": "Book|clutter|2", "position": {"x": 0, "y": 0.2, "z": 0}},
                         {"id": "Bed|3|0", "position": {"x": 0, "y": 0.4, "z": 0}}]}
    objects = [obj("Box|clutter|0", 0.1), obj("Pot|clutter|1", -0.45), obj("Book|clutter|2", 0.002),
               obj("Bed|3|0", 0.2)]
    new, moved = rest_on_floor(house, objects)
    ys = [o["position"]["y"] for o in new["objects"]]
    assert moved == 2 and np.allclose(ys, [0.4, 0.55, 0.2, 0.4])
    assert house["objects"][0]["position"]["y"] == 0.5  # input left as is


def test_underside_map_fills_small_gaps_and_keeps_big_ones_under():
    um = UndersideMap(0, 0, 4, 2)
    surface(um, (0, 4), (0, 2), 2.8)
    um.h[40:42, 40:42] = np.inf              # 4 cm hole (shadow behind a lower surface): filled
    um.h[:, 150:] = np.inf                   # 0.5 m strip never seen: unknown, counted as under
    um.finish()
    assert np.isfinite(um.h[40:42, 40:42]).all()
    assert um.lookup(np.array([0.81, 3.5]), np.array([0.81, 1.0])).tolist() == [False, True]


def test_support_tops_only_large_ones():
    objs = [box("DiningTable", (0, 1.2), (0, 0.8), (0, 0.76)), box("SideTable", (3, 3.4), (0, 0.4), (0, 0.6)),
            box("Bed", (5, 7), (0, 2), (0, 0.5))]
    tops = support_tops(objs)
    assert [o["objectType"] for o, *_ in tops] == ["DiningTable"]
    o, top, c0, e1, e2 = tops[0]
    assert abs(top - 0.76) < 1e-9 and np.allclose(c0, [0, 0]) and np.allclose(e1 + e2, [1.2, 0.8])
