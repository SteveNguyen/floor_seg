"""Render low robot-camera views of one Infinigen scene.blend and write floor_seg training data.

Runs with Infinigen's Python (Blender 4.2 as the bpy module), not floor_seg's env; driven by
scripts/infinigen_gen.py, which generates the room first. Can be run by hand:

    PYTHONPATH=<floor_seg repo> <infinigen>/.venv/bin/python -m floor_seg.infinigen_render scene.blend OUT \
        --room 12 --room-type livingroom --seed 1234 --views 5

Per scene:
- objects get an object pass index (floor, rug, flat floor-level objects; infinigen_labels.IDX_*); the floor plane
  height is the floor object's top.
- "ordinary home" bias: a room whose floor is plain concrete is rejected (exit code 3, the driver regenerates)
  with infinigen_labels.P_REJECT_CONCRETE; walls are repainted with an ordinary paint (white, cream, beige, grey,
  soft pastel) with P_PAINT per room type, and the skirting board then mostly white or the wall colour.
- optional: cables lying on the floor (--cables) and flush door threshold strips (--thresholds), both floor.
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


def paint_material(mat, rgb, rng):
    """Replace a material's nodes with a plain matte paint (subtle roller texture)."""
    nt = mat.node_tree
    nt.nodes.clear()
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


def plain_material(name, rgb, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return m


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
    ap.add_argument("--no-reject", action="store_true", help="never reject a room (smoke tests)")
    ap.add_argument("--overlay", action="store_true", help="also write <name>_overlay.jpg (photo | label)")
    ap.add_argument("--cpu-threads", type=int, default=0, help="CPU render threads (0 = auto)")
    ap.add_argument("--no-under", action="store_true", help="no _under.png (no upward ray casts)")
    ap.add_argument("--under-step", type=int, default=1,
                    help="measure every n-th 2 cm cell of the under map (the rest: nearest measured cell)")
    args = ap.parse_args(sys.argv[1:])
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
    fx0, fx1, fy0, fy1 = boxes[main_floor.name][:4]

    # -- ordinary-home bias: floor family, wall paint, skirting colour
    fkind = L.floor_kind([g for m in eval_materials(main_floor, dg) for g in node_groups(m)])
    if fkind == "concrete" and not args.no_reject and rng.random() < L.P_REJECT_CONCRETE:
        print("reject: concrete floor", flush=True)
        sys.exit(3)
    walls = [o for o in meshes if boxes[o.name][4] < 0.05 and boxes[o.name][5] > 1.8
             and min(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) < 0.25
             and max(boxes[o.name][1] - boxes[o.name][0], boxes[o.name][3] - boxes[o.name][2]) > 1.2]
    wall_mats = {m.name: m for o in walls for m in eval_materials(o, dg) if m.node_tree}
    paint = None
    if wall_mats and rng.random() < L.P_PAINT.get(args.room_type, L.P_PAINT_DEFAULT):
        paint = L.wall_paint(rng)
        accent = L.wall_paint(rng) if rng.random() < 0.2 else paint
        for i, m in enumerate(sorted(wall_mats.values(), key=lambda m: m.name)):
            paint_material(m, (paint if i == 0 else accent)[1], rng)
        skirt_mats = {m.name: m for o in meshes if o.name.startswith("skirting") for m in eval_materials(o, dg)
                      if m.node_tree}
        style = L.weighted(rng, [("white", 0.5), ("wall", 0.3), ("keep", 0.2)])
        for m in skirt_mats.values():
            if style == "white":
                paint_material(m, (0.85, 0.85, 0.84), rng)
            elif style == "wall":
                paint_material(m, tuple(c * 0.9 for c in paint[1]), rng)

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
    for o in doors:
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
    os.makedirs(os.path.join(args.out, "rooms"), exist_ok=True)
    lines, k, tries, n_heavy = [], 0, 0, 0
    print(f"room {args.room} {args.room_type}: floor {fkind}, paint {paint and paint[0]}, cables {n_cables}, "
          f"thresholds {n_thr}, render {wr}x{hr}, {device}, under map {t_under:.1f} s"
          f"{f' ({um.grid.mean():.0%} of the room)' if um else ''}, setup {time.time() - t0:.1f} s", flush=True)
    while k < args.views and tries < 40 * args.views:
        tries += 1
        x, y, yaw, h, pitch, roll = L.sample_pose(rng, (fx0 + 0.3, fx1 - 0.3, fy0 + 0.3, fy1 - 0.3))
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
        n_floor = n_near = n = 0
        for d in grid.reshape(-1, 3)[::3]:
            w = Vector(axes @ d)
            hit = scene.ray_cast(dg, cpos, w.normalized(), distance=30.0)
            n += 1
            if not hit[0]:
                continue
            ob = hit[4].original if hit[4] is not None else None
            if ob is not None and ob.pass_index in (L.IDX_FLOOR, L.IDX_RUG, L.IDX_FLAT) and hit[1].z - floor_z < L.RUG_TOP:
                n_floor += 1
            if (hit[1] - cpos).length * (Vector(axes[:, 2]) @ w.normalized()) < 0.8:
                n_near += 1
        if n_floor < 0.12 * n or n_near > 0.6 * n:
            continue
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
        pos, idx = read_exr(exr)
        rgb_r = read_png_bgr(png, wr, hr)
        pos = pos.reshape(hr, wr, 3)
        floor_r, depth_r = L.labels_from_passes(pos, idx.reshape(hr, wr), cpos, axes, floor_z)
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
        extra = dict(floor_material=fkind, wall_paint=paint and paint[0], cables=n_cables, thresholds=n_thr,
                     aim=mode, under_frac=round(ufrac, 4), render_s=round(time.time() - t1, 1))
        lines.append(L.stats_line(name, args.room, args.room_type, args.seed, (x, y, yaw, h, pitch, roll), frac,
                                  extra))
        print(f"  {name}: floor {frac:.2f}, aim {mode}, {time.time() - t1:.1f} s", flush=True)
        k += 1
    with open(os.path.join(args.out, "rooms", f"r{args.room:05d}.jsonl"), "w") as fo:
        fo.write("".join(s + "\n" for s in lines))
    print(f"room {args.room}: {k} views, {tries} poses tried, {time.time() - t0:.1f} s", flush=True)
    sys.exit(0 if k else 4)


if __name__ == "__main__":
    main()
