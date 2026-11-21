"""Renderer geometry used by the labels (no AI2-THOR needed): world projection, underside map, table tops."""
import itertools

import numpy as np

from floor_seg.thor_render import (PLINTH_D, PLINTH_DOOR_M, PLINTH_TAG, UndersideMap, door_segments,
                                   doorways_in_view, height_above_floor, plinth_walls, support_tops, world_xz)


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
    # heights for metrics at another clearance (_above.png): the lowest surface; inf outside the raster
    hgt = um.lookup_h(np.r_[pts[:, 0], 20.0], np.r_[pts[:, 1], 0.5])
    np.testing.assert_allclose(hgt[:7], [0.25, 1.5, 0.72, 0.42, 0.0, 2.8, 0.25], atol=1e-6)
    assert np.isinf(hgt[7])


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


def square_house():
    """ProcTHOR-like house: one 4 x 4 m room (x, z in 0..4), its 4 walls, the exterior face of the z = 0 wall,
    and a door 1.0-2.0 m along that wall (hole x measured from the wall polygon's first point)."""
    def wall(wid, room, a, b):
        return {"id": wid, "roomId": room, "material": {"name": "PureWhite"},
                "polygon": [{"x": a[0], "y": 0, "z": a[1]}, {"x": b[0], "y": 0, "z": b[1]},
                            {"x": a[0], "y": 2.5, "z": a[1]}, {"x": b[0], "y": 2.5, "z": b[1]}]}
    corners = [(0, 0), (4, 0), (4, 4), (0, 4)]
    walls = [wall(f"wall|1|{a}|{b}", "room|1", a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
    walls.append(wall("wall|exterior|0|0|4|0", "exterior", (0, 0), (4, 0)))
    door = {"id": "door|1", "wall0": walls[0]["id"], "wall1": "wall|exterior|0|0|4|0", "assetId": "Doorway_1",
            "holePolygon": [{"x": 1.0, "y": 0, "z": 0}, {"x": 2.0, "y": 2.1, "z": 0}]}
    room = {"id": "room|1", "floorPolygon": [{"x": x, "y": 0, "z": z} for x, z in corners]}
    return {"rooms": [room], "walls": walls, "doors": [door], "windows": [], "objects": []}


def test_plinths_follow_room_walls_inside_and_skip_doors():
    house = square_house()
    assert door_segments(house) == [((1.0, 0.0), (2.0, 0.0))]
    pl = plinth_walls(house, 0.12)
    assert all(PLINTH_TAG in w["id"] and w["roomId"] == "room|1" for w in pl)
    assert not any(w["id"].startswith("wall|exterior") for w in pl)   # no plinth on the outside face
    # the door wall is cut in two (door + casing), the three others are whole: 5 boards
    assert len(pl) == 5
    for w in pl:
        ys = sorted(p["y"] for p in w["polygon"])
        assert ys == [0.0, 0.0, 0.12, 0.12]
    door_wall = [w for w in pl if w["id"].startswith("wall|1|(0, 0)")]
    spans = sorted((min(p["x"] for p in w["polygon"]), max(p["x"] for p in w["polygon"])) for w in door_wall)
    assert np.allclose(spans, [(0.0, 1.0 - PLINTH_DOOR_M), (2.0 + PLINTH_DOOR_M, 4.0)])
    # PLINTH_D in front of the wall, into the room (z = 0 wall: +z; x = 4 wall: -x)
    assert all(abs(p["z"] - PLINTH_D) < 1e-9 for w in door_wall for p in w["polygon"])
    east = [w for w in pl if w["id"].startswith("wall|1|(4, 0)")]
    assert all(abs(p["x"] - (4 - PLINTH_D)) < 1e-9 for w in east for p in w["polygon"])


def test_doorways_in_the_camera_yaw_frame():
    seg = [((-0.5, 2.0), (0.5, 2.0))]
    # yaw 0 looks along +z: the doorway is 2 m ahead, from 0.5 m left to 0.5 m right
    assert doorways_in_view(seg, 0.0, 0.0, 0.0) == [[-0.5, 2.0, 0.5, 2.0]]
    # yaw 90 looks along +x, right is -z (world_xz): a doorway 2 m along +x, across z = -0.5 .. 0.5
    assert doorways_in_view([((2.0, -0.5), (2.0, 0.5))], 0.0, 0.0, 90.0) == [[0.5, 2.0, -0.5, 2.0]]
    assert doorways_in_view(seg, 0.0, -10.0, 0.0) == []                  # farther than max_d
