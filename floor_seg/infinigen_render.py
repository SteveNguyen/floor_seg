"""Render low robot-camera views of one Infinigen scene.blend and write floor_seg training data.

Runs with Infinigen's Python (Blender 4.2 as the bpy module), not floor_seg's env; driven by
scripts/infinigen_gen.py, which generates the room first. Can be run by hand:

    PYTHONPATH=<floor_seg repo> <infinigen>/.venv/bin/python -m floor_seg.infinigen_render scene.blend OUT \
        --room 12 --room-type livingroom --seed 1234 --views 5

Per scene:
- objects get an object pass index (floor, rug, flat floor-level objects; infinigen_labels.IDX_*); the floor plane
  height is the floor object's top.
- "ordinary home" bias: with --tex (ambientCG textures, as for training), each floor is kept or re-materialled
  as tiles (photo or floor_tex.procedural_tiles) or wood (infinigen_labels.floor_plan / floor_look): carpet and
  concrete replaced where they used to be rejected, some wood floors tiled. Without --tex, a room whose floor is
  plain concrete or carpet is rejected (exit code 3, the driver regenerates) with infinigen_labels.p_reject.
  Walls are repainted with an ordinary paint (white, cream, beige, grey, soft pastel) with P_PAINT per room type, and the skirting board then mostly white or the wall colour.
- optional: cables lying on the floor (--cables) and flush door threshold strips (--thresholds), both floor.
- optional (--cross-door, houses): views across one doorway into the next room, its two floors made different
  (tiles on one side), threshold strips on the real doorways; --far-bright: the far room lit brighter.
- K cameras: collision-free (rays around the robot's footprint), 0.25-0.40 m above the floor, pitch 5-25 deg
  down, roll +-3 deg; yaw toward furniture / objects 40 %, toward a wall 30 %, random 30 %; a ray grid checks
  that enough floor is visible before rendering.
- render: Cycles on the GPU (OptiX, else CUDA / HIP / oneAPI, else CPU), denoised, a pinhole image large enough
  for the lens, then remapped through the calibrated Reachy Mini lens (infinigen_labels.lens_maps), like
  thor_render. A fitted fisheye polynomial cannot match our lens (principal point off-centre, see
  infinigen_labels.fit_fisheye_poly), so the remap is used.
- under: floor with something lower than 0.45 m above it, from upward ray casts on thor_render's UndersideMap
  raster (measure_underside); views with under > 40 % of the floor are capped per room (infinigen_labels.keep_view).
Outputs (in OUT): <name>.jpg, <name>_floor.png, <name>_under.png, <name>_depth.png with name = r<room:05d>_<k>, and one stats line
per view in OUT/rooms/r<room:05d>.jsonl (the driver merges them into OUT/stats.jsonl).
Exit codes: 0 done, 3 room rejected (regenerate), 4 no usable view.
"""
import argparse
import math
import os
import sys
import time

import bpy
import cv2
import numpy as np
import OpenEXR
from mathutils import Matrix, Vector

from floor_seg import DEFAULT_CALIB
from floor_seg import infinigen_labels as L

ROBOT_R = 0.22          # m, free radius around the camera position (base ~0.15 m + margin)
NAME_SKIP = ("door_affordance_collider", "clearance")  # helper volumes, never rendered as obstacles


def world_bbox(obj, dg):
    ev = obj.evaluated_get(dg)
    pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    return (min(p.x for p in pts), max(p.x for p in pts), min(p.y for p in pts), max(p.y for p in pts),
            min(p.z for p in pts), max(p.z for p in pts))


def eval_materials(obj, dg):
    ev = obj.evaluated_get(dg)
    mats = list(getattr(ev.data, "materials", []) or [])
    return [m for m in mats if m is not None]


def node_groups(mat):
    if not mat or not mat.node_tree:
        return []
    return [n.node_tree.name for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeGroup" and n.node_tree]


def new_node_material(name):
    """A new, empty node material. Infinigen's own materials are never edited: editing one in place changes the
    original but not the evaluated copy the render uses (smoke test, room 1002), and clearing its group nodes
    trips Blender's user counts. New materials go into the objects' slots instead (swap_materials)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    return m


def swap_materials(objs, mapping):
    """Put mapping[old name] in every material slot of objs that holds the old material."""
    for o in objs:
        for slot in o.material_slots:
            if slot.material is not None and slot.material.name in mapping:
                slot.material = mapping[slot.material.name]


def paint_material(name, rgb, rng):
    """A new plain matte paint material (subtle roller texture)."""
    mat = new_node_material(name)
    nt = mat.node_tree
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.55, 0.85)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = rng.uniform(150, 400)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = rng.uniform(0.02, 0.06)
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def plain_material(name, rgb, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def texture_floor_material(name, img_path, size_m, roughness, rot_deg, rng):
    """A new floor material: a colour image laid on the floor in world coordinates (x, y; one repeat = size_m),
    rotated by rot_deg, a bump from the image's brightness (grout lines sink) and the given roughness."""
    mat = new_node_material(name)
    nt = mat.node_tree
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (rng.uniform(0, 1), rng.uniform(0, 1), 0)
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(rot_deg))
    mp.inputs["Scale"].default_value = (1 / size_m, 1 / size_m, 1)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(img_path)
    tex.extension = "REPEAT"
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = rng.uniform(0.3, 0.8)
    bump.inputs["Distance"].default_value = 0.001   # m: brightness 0..1 = 1 mm of relief (grout, texture)
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(geo.outputs["Position"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
    nt.links.new(bw.outputs["Val"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def refloor(floors, kinds, room_type, tex_root, tmp_prefix, dg, rng, force=None):
    """Re-material each floor object with infinigen_labels.floor_plan (kinds: its floor family). Returns the
    final family per floor ("carpet", "tiles:floor", "tiles:procedural", "wood:wood", ...) and the temporary
    procedural texture files (deleted by the caller after the renders). force: floor object name -> plan
    ("tiles" / "wood") for floors that must get that plan and a material of their own (--cross-door)."""
    import glob

    from floor_seg.floor_tex import procedural_tiles

    force = force or {}
    final, tmp, done = [], [], set()
    os.makedirs(os.path.dirname(tmp_prefix), exist_ok=True)
    for i, (o, fkind) in enumerate(zip(floors, kinds)):
        plan = force.get(o.name) or L.floor_plan(fkind, room_type, rng)
        if plan == "keep":
            final.append(fkind)
            continue
        group, size, rough, rot = L.floor_look(plan, rng)
        if group == "procedural":
            img, size = procedural_tiles(rng)
            path = f"{tmp_prefix}_tiles{i}.png"
            cv2.imwrite(path, img)
            tmp.append(path)
        else:
            path = rng.choice(sorted(glob.glob(os.path.join(tex_root, group, "*.jpg"))))
        if o.name in force:   # its own material in all its slots (a material shared with the other room's floor
            mat = texture_floor_material(f"floor_{plan}_{i}", path, size, rough, rot, rng)   # stays there)
            for slot in o.material_slots:
                slot.material = mat
            final.append(f"{plan}:{group}")
            continue
        mapping = {m.name: texture_floor_material(f"floor_{plan}_{i}", path, size, rough, rot, rng)
                   for m in eval_materials(o, dg) if m.name not in done}   # shared by two floors: first plan wins
        swap_materials(floors, mapping)
        done |= set(mapping)
        final.append(f"{plan}:{group}")
    return final, tmp


def floor_at(scene, dg, x, y, floor_z, names):
    """Name of the floor object (one of names) at (x, y): a short ray UP from 0.3 m below the floor plane (nothing
    stands below the floor, unlike a ray down that may meet a door leaf or furniture), or None."""
    hit = scene.ray_cast(dg, Vector((x, y, floor_z - 0.3)), Vector((0, 0, 1)), distance=0.31)
    name = hit[4].original.name if hit[0] and hit[4] is not None else None
    return name if name in names else None


def find_doorways(scene, dg, floor_z, names):
    """Doorways between two rooms of a house, from Infinigen's door_affordance_collider helper boxes (hidden in the
    render, one per doorway, centred on the doorway's centre line in the wall; local x = the wall normal, 1.6 m
    long; local y = along the wall, the door width; Infinigen 2 house_walls_rand). The two rooms' floors meet on
    that centre line (smoke test, house seed 777). Returns [(centre, normal, along, width, floor_a, floor_b)]
    (2-D Vectors, m; floor_a on the -normal side, floor_b on the +normal side), only doorways with a floor of
    `names` on both sides, two different floor objects."""
    out = []
    for o in scene.objects:
        if "door_affordance_collider" not in o.name:
            continue
        mw = o.matrix_world
        c, n, a = mw.translation.to_2d(), mw.col[0].to_2d().normalized(), mw.col[1].to_2d().normalized()
        sides = [floor_at(scene, dg, *(c + n * s), floor_z, names) for s in (-0.3, 0.3)]
        if None not in sides and sides[0] != sides[1]:
            out.append((c, n, a, o.dimensions.y, *sides))
    return out


def setup_gpu(scene):
    """Cycles on the first available GPU backend, else CPU. Returns the device description."""
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except (TypeError, ValueError):
            continue
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"{backend}: " + ", ".join(d.name for d in gpus)
    scene.cycles.device = "CPU"
    return "CPU"


def read_exr(path):
    ch = OpenEXR.File(path).parts[0].channels

    def get(suffix):
        key = next(k for k in ch if k.endswith(suffix))
        return np.asarray(ch[key].pixels, np.float32)

    pos = np.stack([get("Position.X"), get("Position.Y"), get("Position.Z")], -1)
    idx = get("IndexOB.X") if any(k.endswith("IndexOB.X") for k in ch) else get("IndexOB.V")
    return pos, idx


STRUCT_PREFIX = ("mesh_from_corners", "room_", "plain_wall", "wall_", "cutout", "window", "door", "skirting",
                 "ceiling")


def measure_underside(scene, dg, bounds, floor_z, structure, step=1):
    """Cell heights for infinigen_labels.underside_map over bounds (x0, y0, x1, y1), from the geometry: per 2 cm
    cell (every `step`-th, the rest copied from the nearest measured cell)
    - contact (0): a ray down from CONTACT_M above the floor hits something other than the floor / rug / flat
      floor-level objects before the floor: an object stands there (the floor is covered);
    - else a ray straight up from 1 mm above the floor: the height of the first surface it hits below
      CLEARANCE + 5 cm, skipping floor-level things (rug tops, cables, strips); structure (walls, doors, windows,
      skirting, ceiling) and nothing hit count as clear (CLEAR_H), as thor_render treats them from below."""
    xs, ys = L.raster_xy(*bounds)
    heights = np.full(xs.shape, L.CLEAR_H, np.float32)
    up, down, top = Vector((0, 0, 1)), Vector((0, 0, -1)), floor_z + L.CLEARANCE + 0.05
    floorish = (L.IDX_FLOOR, L.IDX_RUG, L.IDX_FLAT)
    for i in range(0, xs.shape[0], step):
        for j in range(0, xs.shape[1], step):
            x, y = float(xs[i, j]), float(ys[i, j])
            hit = scene.ray_cast(dg, Vector((x, y, floor_z + L.CONTACT_M)), down, distance=L.CONTACT_M + 0.004)
            if hit[0] and hit[4] is not None and hit[4].original.pass_index not in floorish:
                heights[i, j] = 0.0
                continue
            o = Vector((x, y, floor_z + 0.001))
            for _ in range(4):
                hit = scene.ray_cast(dg, o, up, distance=max(top - o.z, 1e-4))
                if not hit[0] or hit[4] is None:
                    break
                ob = hit[4].original
                if ob.pass_index in floorish:  # a rug top, a cable: look further up
                    o = hit[1] + up * 0.001
                    continue
                if ob not in structure and not ob.name.startswith(STRUCT_PREFIX):
                    heights[i, j] = hit[1].z - floor_z
                break
    if step > 1:  # nearest measured cell
        ii = np.minimum((np.arange(xs.shape[0]) + step // 2) // step * step, (xs.shape[0] - 1) // step * step)
        jj = np.minimum((np.arange(xs.shape[1]) + step // 2) // step * step, (xs.shape[1] - 1) // step * step)
        heights = heights[ii][:, jj]
    return heights


def read_png_bgr(path, w, h):
    """The display-referred render Blender saved, read back through Blender (cv2.imread fails on it inside the
    Blender process: two libpng builds clash). uint8 BGR, top row first."""
    img = bpy.data.images.load(path)
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    rgba = px.reshape(h, w, 4)[::-1]
    return np.ascontiguousarray((np.clip(rgba[..., 2::-1], 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blend")
    ap.add_argument("out")
    ap.add_argument("--room", type=int, required=True)
    ap.add_argument("--room-type", default="livingroom")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--cables", type=float, default=0.0, help="chance per room of 1-3 cables on the floor")
    ap.add_argument("--thresholds", type=float, default=0.0, help="chance per door of a flush threshold strip")
    ap.add_argument("--tex", help="ambientCG texture root (data/textures/ambientcg): re-material floors instead of "
                                    "rejecting rooms (infinigen_labels.floor_plan)")
    ap.add_argument("--no-reject", action="store_true", help="never reject a room (smoke tests)")
    ap.add_argument("--overlay", action="store_true", help="also write <name>_overlay.jpg (photo | label)")
    ap.add_argument("--cpu-threads", type=int, default=0, help="CPU render threads (0 = auto)")
    ap.add_argument("--no-under", action="store_true", help="no _under.png (no upward ray casts)")
    ap.add_argument("--under-step", type=int, default=1,
                    help="measure every n-th 2 cm cell of the under map (the rest: nearest measured cell)")
    ap.add_argument("--cross-door", type=float, default=0.0,
                    help="chance per view of a view across a doorway into the next room (houses; needs --tex): "
                         "camera 0.5-2.5 m before one doorway, facing it; that doorway's two floors forced "
                         "different (tiles on one side, wood / other tiles beyond); thresholds on the real doorways")
    ap.add_argument("--far-bright", type=float, default=0.0,
                    help="with --cross-door: chance per cross-door view that the room beyond is lit brighter "
                         "(its lamps x 2.5-5 and a ceiling area light)")
    args = ap.parse_args(sys.argv[1:])
    if args.cross_door and not args.tex:
        ap.error("--cross-door needs --tex (the two floors are re-materialled)")
    rng = L.new_rng(args.seed, args.room)
    t0 = time.time()

    bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    meshes = [o for o in scene.objects if o.type in ("MESH", "CURVE") and not o.hide_render
              and not any(s in o.name for s in NAME_SKIP)]
    for o in scene.objects:
        if any(s in o.name for s in NAME_SKIP):
            o.hide_render = True
    boxes = {o.name: world_bbox(o, dg) for o in meshes}

    # -- object classes: floor (flat, at floor level), rugs, the rest
    flat = [o for o in meshes if boxes[o.name][5] - boxes[o.name][4] < 0.025 and boxes[o.name][5] < 0.03
            and (boxes[o.name][1] - boxes[o.name][0]) * (boxes[o.name][3] - boxes[o.name][2]) > 0.05]
    rugs = [o for o in meshes if o.name.lower().startswith(("rug", "mat", "bath_mat", "doormat"))]
    floors = sorted([o for o in flat if o not in rugs],
                    key=lambda o: -(boxes[o.name][1] - boxes[o.name][0]) * (boxes[o.name][3] - boxes[o.name][2]))
    if not floors:
        print("no floor object found", flush=True)
        sys.exit(4)
    main_floor = floors[0]
    floor_z = boxes[main_floor.name][5]
    for o in meshes:
        o.pass_index = L.IDX_OTHER
    for o in floors:
        o.pass_index = L.IDX_FLOOR
    for o in rugs:
        o.pass_index = L.IDX_RUG
    # cameras and the under map over all floors at the main floor's level (a house: several rooms)
    level = [o for o in floors if abs(boxes[o.name][5] - floor_z) < 0.02]
    fx0, fx1 = min(boxes[o.name][0] for o in level), max(boxes[o.name][1] for o in level)
    fy0, fy1 = min(boxes[o.name][2] for o in level), max(boxes[o.name][3] for o in level)

    # -- cross-door views (--cross-door): the house's doorways, one of them the target of those views. Its floors:
    # tiles on one side (the side the camera is on 75 % of the time), wood or other tiles beyond.
    doorways, door, force = [], None, {}
    if args.cross_door > 0:
        doorways = find_doorways(scene, dg, floor_z, {o.name for o in level})
        # the helper boxes are hidden in the render but not in scene.ray_cast: they would block cameras near a
        # doorway and the ray-grid preview through it (1.6 m deep): removed
        for o in [o for o in scene.objects if "door_affordance_collider" in o.name]:
            bpy.data.objects.remove(o, do_unlink=True)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        if doorways:
            door = rng.choice(doorways)
            tiles_side = rng.choice(door[4:6])
            other_side = door[5] if tiles_side == door[4] else door[4]
            force = {tiles_side: "tiles", other_side: L.weighted(rng, [("wood", 0.6), ("tiles", 0.4)])}

    # -- ordinary-home bias: floor family, wall paint, skirting colour
    groups = [[g for m in eval_materials(o, dg) for g in node_groups(m)] for o in level]
    kinds = [L.floor_kind(g) for g in groups]
    fkind = kinds[0]
    print(f"floors: {len(level)}, main {fkind}, node groups {sorted(set(groups[0]))}", flush=True)
    tmp_tex = []
    if args.tex:
        finals, tmp_tex = refloor(level, kinds, args.room_type, args.tex,
                                  os.path.join(args.out, "rooms", f"r{args.room:05d}"), dg, rng, force)
        floor_final = finals[0]
        final_of = {o.name: f for o, f in zip(level, finals)}
    else:
        if not args.no_reject and rng.random() < L.p_reject(fkind, args.room_type):
            print(f"reject: {fkind} floor", flush=True)
            sys.exit(3)
        floor_final = fkind
    walls = [o for o in meshes if boxes[o.name][4] < 0.05 and boxes[o.name][5] > 1.8
             and min(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) < 0.25
             and max(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) > 1.2]
    wall_mats = {m.name: m for o in walls for m in eval_materials(o, dg) if m.node_tree}
    paint = None
    if wall_mats and rng.random() < L.P_PAINT.get(args.room_type, L.P_PAINT_DEFAULT):
        paint = L.wall_paint(rng)
        accent = L.wall_paint(rng) if rng.random() < 0.2 else paint
        swap_materials(walls, {m: paint_material(f"wall_paint_{i}", (paint if i == 0 else accent)[1], rng)
                               for i, m in enumerate(sorted(wall_mats))})
        skirts = [o for o in meshes if o.name.startswith("skirting")]
        style = L.weighted(rng, [("white", 0.5), ("wall", 0.3), ("keep", 0.2)])
        if style != "keep":
            rgb = (0.85, 0.85, 0.84) if style == "white" else tuple(c * 0.9 for c in paint[1])
            swap_materials(skirts, {m.name: paint_material(f"skirt_paint_{i}", rgb, rng)
                                    for i, m in enumerate({m.name: m for o in skirts
                                                           for m in eval_materials(o, dg)}.values())})
    # the render must see the new materials (a material set by geometry nodes would ignore the slots): check
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    seen = {m.name for o in level for m in eval_materials(o, dg)}
    if args.tex and floor_final != fkind and not all(n.startswith("floor_") for n in seen):
        print(f"WARNING: floor material swap not effective, the floor renders with {sorted(seen)}", flush=True)

    # -- optional cables (floor by rule: lying on it, < 1 cm high)
    n_cables = rng.randint(1, 3) if rng.random() < args.cables else 0
    for i in range(n_cables):
        cu = bpy.data.curves.new(f"cable_{i}", "CURVE")
        cu.dimensions = "3D"
        r = rng.uniform(0.002, 0.004)
        cu.bevel_depth, cu.bevel_resolution = r, 3
        sp = cu.splines.new("BEZIER")
        n = rng.randint(4, 7)
        sp.bezier_points.add(n - 1)
        x, y, a = rng.uniform(fx0 + 0.3, fx1 - 0.3), rng.uniform(fy0 + 0.3, fy1 - 0.3), rng.uniform(0, 6.3)
        for p in sp.bezier_points:
            p.co = (x, y, floor_z + r)
            p.handle_left_type = p.handle_right_type = "AUTO"
            a += rng.uniform(-1.0, 1.0)
            step = rng.uniform(0.3, 0.8)
            x = min(max(x + step * math.cos(a), fx0 + 0.05), fx1 - 0.05)
            y = min(max(y + step * math.sin(a), fy0 + 0.05), fy1 - 0.05)
        ob = bpy.data.objects.new(f"cable_{i}", cu)
        col = rng.choice([(0.02, 0.02, 0.02), (0.8, 0.8, 0.78), (0.3, 0.3, 0.3)])
        ob.data.materials.append(plain_material(f"cable_mat_{i}", col, 0.4))
        ob.pass_index = L.IDX_FLAT
        scene.collection.objects.link(ob)
    # -- optional flush door threshold strips (floor): doors = thin, door-high, door-wide objects at the floor
    n_thr = 0
    doors = [o for o in meshes if boxes[o.name][4] < 0.03 and 1.9 < boxes[o.name][5] < 2.4
             and min(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) < 0.25
             and 0.6 < max(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) < 1.4]
    seen = set()
    # with --cross-door the real doorways are known: strips on their centre line (where the two floors meet),
    # along the wall, door-wide; the door-leaf boxes below are not used then (an open leaf's box is not the doorway)
    for c, _, a, width, _, _ in doorways:
        if rng.random() >= args.thresholds:
            continue
        depth, height = rng.uniform(0.04, 0.10), rng.uniform(0.004, 0.012)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(c.x, c.y, floor_z + height / 2))
        st = bpy.context.active_object
        st.name = f"threshold_{n_thr}"
        st.rotation_euler = (0.0, 0.0, math.atan2(a.y, a.x))   # local x along the doorway
        st.scale = (width, depth, height)
        col = rng.choice([(0.35, 0.22, 0.12), (0.6, 0.6, 0.6), (0.7, 0.7, 0.68), (0.15, 0.12, 0.1)])
        st.data.materials.append(plain_material(f"threshold_mat_{n_thr}", col, rng.uniform(0.2, 0.6)))
        st.pass_index = L.IDX_FLAT
        n_thr += 1
    for o in [] if doorways else doors:
        x0, x1, y0, y1 = boxes[o.name][:4]
        key = (round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1))
        if key in seen or rng.random() >= args.thresholds:
            continue
        seen.add(key)
        along_x = (x1 - x0) > (y1 - y0)
        width, depth, height = max(x1 - x0, y1 - y0), rng.uniform(0.04, 0.10), rng.uniform(0.004, 0.012)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=((x0 + x1) / 2, (y0 + y1) / 2, floor_z + height / 2))
        st = bpy.context.active_object
        st.name = f"threshold_{n_thr}"
        st.scale = (width, depth, height) if along_x else (depth, width, height)
        col = rng.choice([(0.35, 0.22, 0.12), (0.6, 0.6, 0.6), (0.7, 0.7, 0.68), (0.15, 0.12, 0.1)])
        st.data.materials.append(plain_material(f"threshold_mat_{n_thr}", col, rng.uniform(0.2, 0.6)))
        st.pass_index = L.IDX_FLAT
        n_thr += 1

    # -- camera, render settings, passes
    map_x, map_y, wr, hr, f = L.lens_maps(args.calib)
    cam_data = bpy.data.cameras.new("robot_cam")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = 2 * math.atan(wr / 2 / f)
    cam_data.clip_start = 0.01
    cam = bpy.data.objects.new("robot_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    device = setup_gpu(scene)
    scene.cycles.samples = args.samples
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.film_exposure = 2.0 * rng.uniform(0.7, 1.4)   # Infinigen's renders use 2.0
    if args.cpu_threads:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.cpu_threads
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = wr, hr, 100
    vl = scene.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    dg = bpy.context.evaluated_depsgraph_get()
    um, t_under = None, time.time()
    if not args.no_under:  # once per room, after the cables / strips were added
        bounds = (fx0, fy0, fx1, fy1)
        um = L.underside_map(*bounds, measure_underside(scene, dg, bounds, floor_z, set(walls), args.under_step))
    t_under = time.time() - t_under

    # targets for "look at furniture": other objects standing on the floor, 0.1-1.5 m tall
    targets = [((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) for o, b in ((o, boxes[o.name]) for o in meshes)
               if o.pass_index == L.IDX_OTHER and b[4] < 0.1 and 0.1 < b[5] < 1.5 and o not in walls]
    grid = L.pinhole_dirs(wr, hr, f, step=max(wr // 32, 1))
    # --far-bright: the lamps of each of the target doorway's two rooms (floor below the lamp; not the sun, not
    # the window portals) and one ceiling area light moved into the far room per view, off (energy 0) otherwise
    lamps, far_light = {}, None
    if door is not None and args.far_bright > 0:
        for o in scene.objects:
            if o.type == "LIGHT" and o.data.type != "SUN" and not getattr(o.data.cycles, "is_portal", False):
                fl = floor_at(scene, dg, o.matrix_world.translation.x, o.matrix_world.translation.y, floor_z,
                              set(door[4:6]))
                if fl:
                    lamps.setdefault(fl, []).append((o, o.data.energy))
        ld = bpy.data.lights.new("far_light", "AREA")
        ld.size, ld.energy, ld.color = 1.5, 0.0, (0.95, 0.97, 1.0)   # emits down (-z), daylight-ish
        far_light = bpy.data.objects.new("far_light", ld)
        scene.collection.objects.link(far_light)
    os.makedirs(os.path.join(args.out, "rooms"), exist_ok=True)
    lines, k, tries, n_heavy = [], 0, 0, 0
    print(f"room {args.room} {args.room_type}: floor {fkind} -> {floor_final}, paint {paint and paint[0]}, "
          f"cables {n_cables}, "
          f"thresholds {n_thr}, render {wr}x{hr}, {device}, under map {t_under:.1f} s"
          f"{f' ({um.grid.mean():.0%} of the room)' if um else ''}, setup {time.time() - t0:.1f} s"
          f"{f', doorways {len(doorways)}, cross-door floors {force}' if args.cross_door else ''}", flush=True)
    while k < args.views and tries < 40 * args.views:
        tries += 1
        cross = door is not None and rng.random() < args.cross_door
        x, y, yaw, h, pitch, roll = L.sample_pose(rng, (fx0 + 0.3, fx1 - 0.3, fy0 + 0.3, fy1 - 0.3))
        if cross:   # robot 0.5-2.5 m before the doorway, roughly facing it (height, pitch, roll as sampled)
            mode = "door"
            c, nrm, a, width, fa, fb = door
            near = tiles_side if rng.random() < 0.75 else other_side
            far = fb if near == fa else fa
            nn = nrm if near == fa else -nrm            # from the near room into the far room
            dist = rng.uniform(0.5, 2.5)
            p = c - nn * dist + a * (rng.uniform(-1, 1) * min(0.4, 0.3 * dist))   # up to 0.4 m off the axis
            tgt = c + a * (rng.uniform(-0.25, 0.25) * width)                       # a point of the doorway
            x, y = p.x, p.y
            yaw = L.yaw_toward(x, y, tgt.x, tgt.y) + rng.uniform(-10, 10)
            if floor_at(scene, dg, x, y, floor_z, {near}) is None:   # the near room is shorter than dist
                continue
        else:
            mode = L.weighted(rng, [("object", 0.4), ("wall", 0.3), ("random", 0.3)])
        if mode == "object" and targets:
            tx, ty = rng.choice(targets)
            if 0.6 < math.hypot(tx - x, ty - y) < 4.0:
                yaw = L.yaw_toward(x, y, tx, ty) + rng.uniform(-15, 15)
        cpos = Vector((x, y, floor_z + h))
        # free space: the robot's footprint at the floor (5 cm up) and at the camera, and floor right below
        down = scene.ray_cast(dg, cpos, Vector((0, 0, -1)))
        if not down[0] or down[4] is None or down[4].original.pass_index not in (L.IDX_FLOOR, L.IDX_RUG) \
                or abs(down[1].z - floor_z) > 0.05:
            continue
        blocked = False
        for zz in (floor_z + 0.05, cpos.z):
            for a in range(0, 360, 30):
                d = Vector((math.cos(math.radians(a)), math.sin(math.radians(a)), 0))
                hit = scene.ray_cast(dg, Vector((x, y, zz)), d, distance=ROBOT_R)
                if hit[0]:
                    blocked = True
                    break
            if blocked:
                break
        # headroom: nothing above the robot lower than the under clearance (else the camera sits under a table
        # or bed, where the robot cannot be); checked at the centre and on the footprint ring
        top = floor_z + max(L.CLEARANCE, h + 0.1)
        for a in [None] + list(range(0, 360, 90)):
            px, py = (x, y) if a is None else (x + ROBOT_R * math.cos(math.radians(a)),
                                               y + ROBOT_R * math.sin(math.radians(a)))
            if scene.ray_cast(dg, Vector((px, py, floor_z + 0.05)), Vector((0, 0, 1)),
                              distance=top - floor_z - 0.05)[0]:
                blocked = True
                break
        if blocked:
            continue
        if mode == "wall":  # face the nearest wall within 0.5-2.5 m
            best = None
            for a in range(0, 360, 15):
                d = Vector((math.cos(math.radians(a)), math.sin(math.radians(a)), 0))
                hit = scene.ray_cast(dg, cpos, d, distance=2.5)
                if hit[0] and hit[4] is not None and hit[4].original in walls and (best is None or (hit[1] - cpos).length < best[0]):
                    best = ((hit[1] - cpos).length, a)
            if best and best[0] > 0.5:
                yaw = best[1] + rng.uniform(-30, 30)
        axes = L.camera_axes(yaw, pitch, roll)
        # ray-grid preview: enough floor, not staring at a wall
        n_floor = n_near = n = n_far = 0
        for d in grid.reshape(-1, 3)[::3]:
            w = Vector(axes @ d)
            hit = scene.ray_cast(dg, cpos, w.normalized(), distance=30.0)
            n += 1
            if not hit[0]:
                continue
            ob = hit[4].original if hit[4] is not None else None
            if ob is not None and ob.pass_index in (L.IDX_FLOOR, L.IDX_RUG, L.IDX_FLAT) and hit[1].z - floor_z < L.RUG_TOP:
                n_floor += 1
                n_far += cross and (hit[1].to_2d() - c) @ nn > 0.05   # floor beyond the doorway's centre line
            if (hit[1] - cpos).length * (Vector(axes[:, 2]) @ w.normalized()) < 0.8:
                n_near += 1
        if n_floor < 0.12 * n or n_near > 0.6 * n:
            continue
        if cross and n_far < 0.03 * n:   # the next room's floor hardly visible (door leaf, furniture, angle)
            continue
        gain = 0.0
        if cross and far_light is not None and rng.random() < args.far_bright:   # the far room brighter
            gain = rng.uniform(2.5, 5.0)
            for o, e in lamps.get(far, []):
                o.data.energy = e * gain
            # plus a ceiling area light 1 m into the far room (rooms lit only through windows / skylights have
            # no lamp to scale): (gain - 1) x the room's lamp power, at least 20 W
            q = c + nn * 1.0
            up = scene.ray_cast(dg, Vector((q.x, q.y, floor_z + 0.5)), Vector((0, 0, 1)), distance=4.0)
            far_light.location = (q.x, q.y, (up[1].z if up[0] else floor_z + 2.4) - 0.05)
            far_light.data.energy = (gain - 1) * max(sum(e for _, e in lamps.get(far, [])), 20.0)
        cam.matrix_world = Matrix.Translation(cpos) @ Matrix(L.blender_rotation(axes).tolist()).to_4x4()
        name = f"r{args.room:05d}_{k}"
        exr = os.path.join(args.out, "rooms", name + ".exr")
        png = os.path.join(args.out, "rooms", name + "_render.png")
        t1 = time.time()
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = exr
        bpy.ops.render.render(write_still=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        bpy.data.images["Render Result"].save_render(filepath=png)   # with the scene's view transform
        if gain:   # lights back as they were
            for o, e in lamps.get(far, []):
                o.data.energy = e
            far_light.data.energy = 0.0
        pos, idx = read_exr(exr)
        rgb_r = read_png_bgr(png, wr, hr)
        pos = pos.reshape(hr, wr, 3)
        floor_r, depth_r = L.labels_from_passes(pos, idx.reshape(hr, wr), cpos, axes, floor_z)
        lum_ratio = None
        if cross:   # QA: mean brightness of the far floor / the near floor (> 0.1 m from the centre line)
            acr = (pos[..., 0] - c.x) * nn.x + (pos[..., 1] - c.y) * nn.y
            lum = rgb_r.mean(axis=2)
            fm, nm = floor_r & (acr > 0.1), floor_r & (acr < -0.1)
            if fm.sum() > 100 and nm.sum() > 100:
                lum_ratio = round(float(lum[fm].mean() / max(lum[nm].mean(), 1.0)), 3)
        under_r = L.under_mask(pos, floor_r, um) if um is not None else None
        os.remove(exr)
        os.remove(png)
        rgb, floor, depth, under = L.finish(rgb_r, floor_r, depth_r, (map_x, map_y), under_r)
        frac = float((floor > 0).mean())
        ufrac = float((under > 0).sum() / max((floor > 0).sum(), 1)) if under is not None else 0.0
        if frac < L.MIN_FLOOR:
            print(f"  {name}: floor {frac:.2f} < {L.MIN_FLOOR}, dropped ({time.time() - t1:.1f} s)", flush=True)
            continue
        if not L.keep_view(ufrac, n_heavy, args.views):
            print(f"  {name}: under {ufrac:.2f} of the floor, heavy views capped, dropped", flush=True)
            continue
        n_heavy += ufrac > L.UNDER_HEAVY
        cv2.imwrite(os.path.join(args.out, name + ".jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(args.out, name + "_floor.png"), floor)
        cv2.imwrite(os.path.join(args.out, name + "_depth.png"), depth)
        if under is not None:
            cv2.imwrite(os.path.join(args.out, name + "_under.png"), under)
        if args.overlay:  # green floor, blue under (a subset of floor), red not floor
            ov = rgb.copy()
            ov[floor > 0] = (0.5 * ov[floor > 0] + (0, 110, 0)).astype(np.uint8)
            ov[floor == 0] = (0.5 * ov[floor == 0] + (0, 0, 120)).astype(np.uint8)
            if under is not None:
                ov[under > 0] = (0.5 * rgb[under > 0] + (130, 40, 0)).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"), np.hstack([rgb, ov]))
        extra = dict(floor_material=floor_final, floor_infinigen=fkind, wall_paint=paint and paint[0],
                     cables=n_cables, thresholds=n_thr,
                     aim=mode, under_frac=round(ufrac, 4), render_s=round(time.time() - t1, 1))
        if cross:   # near / far floor families, far-room lamp gain (0: off), measured far / near floor brightness
            extra.update(near_floor=final_of[near], far_floor=final_of[far], far_gain=round(gain, 2),
                         far_near_lum=lum_ratio)
        lines.append(L.stats_line(name, args.room, args.room_type, args.seed, (x, y, yaw, h, pitch, roll), frac,
                                  extra))
        print(f"  {name}: floor {frac:.2f}, aim {mode}, {time.time() - t1:.1f} s"
              f"{f', far gain {gain:.1f}, far/near floor brightness {lum_ratio}' if cross else ''}", flush=True)
        k += 1
    for path in tmp_tex:
        os.remove(path)
    with open(os.path.join(args.out, "rooms", f"r{args.room:05d}.jsonl"), "w") as fo:
        fo.write("".join(s + "\n" for s in lines))
    print(f"room {args.room}: {k} views, {tries} poses tried, {time.time() - t0:.1f} s", flush=True)
    sys.exit(0 if k else 4)


if __name__ == "__main__":
    main()
