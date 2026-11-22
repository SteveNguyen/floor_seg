"""Debug: why re-materialling an Infinigen floor (infinigen_render.refloor) does not change the render.

    PYTHONPATH=. <infinigen>/.venv/bin/python scripts/infinigen_debug_floor.py <work>/scene.blend <tex_root>

Prints, for the floor objects infinigen_render picks (flat, at floor level): material slots (link: object or
data), data / evaluated materials, modifiers (geometry nodes, with any Set Material nodes); then the material
on the face a ray from above actually hits (evaluated mesh, face material_index), before and after
texture_floor_material is applied to the evaluated materials. Same for one wall (the paint).
"""
import sys

import bpy
from mathutils import Vector

from floor_seg.infinigen_render import NAME_SKIP, eval_materials, node_groups, texture_floor_material, world_bbox

blend, tex_root = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
meshes = [o for o in scene.objects if o.type in ("MESH", "CURVE") and not o.hide_render
          and not any(s in o.name for s in NAME_SKIP)]
boxes = {o.name: world_bbox(o, dg) for o in meshes}
flat = [o for o in meshes if boxes[o.name][5] - boxes[o.name][4] < 0.025 and boxes[o.name][5] < 0.03
        and (boxes[o.name][1] - boxes[o.name][0]) * (boxes[o.name][3] - boxes[o.name][2]) > 0.05]
floors = sorted(flat, key=lambda o: -(boxes[o.name][1] - boxes[o.name][0]) * (boxes[o.name][3] - boxes[o.name][2]))
print("flat floor-level objects:", [o.name for o in floors[:6]])
# any object reaching the floor that is big and not flat (a floor slab thicker than 2.5 cm?)
slabs = [o for o in meshes if boxes[o.name][5] < 0.05 and (boxes[o.name][1] - boxes[o.name][0]) > 2
         and (boxes[o.name][3] - boxes[o.name][2]) > 2]
print("big objects with top < 5 cm:", [(o.name, [round(v, 3) for v in boxes[o.name][4:]]) for o in slabs])


def describe(o):
    ev = o.evaluated_get(dg)
    print(f"--- {o.name} type {o.type} data {o.data.name if o.data else None}")
    print("  slots:", [(s.link, s.material.name if s.material else None) for s in o.material_slots])
    print("  data materials:", [m.name if m else None for m in getattr(o.data, "materials", [])])
    print("  evaluated materials:", [m.name for m in eval_materials(o, dg)], "groups",
          sorted({g for m in eval_materials(o, dg) for g in node_groups(m)})[:8])
    for md in o.modifiers:
        setm = []
        if md.type == "NODES" and md.node_group:
            setm = [(n.name, n.inputs["Material"].default_value.name if n.inputs["Material"].default_value else None)
                    for n in md.node_group.nodes if n.bl_idname == "GeometryNodeSetMaterial"]
        print("  modifier:", md.type, md.name, getattr(md.node_group, "name", None), "SetMaterial:", setm)


def hit_material(o):
    """Material on the face a ray from 1 m above the object's centre hits (evaluated mesh)."""
    b = boxes[o.name]
    hit = scene.ray_cast(dg, Vector(((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 1.0)), Vector((0, 0, -1)))
    if not hit[0]:
        return None
    ob = hit[4]
    me = ob.data
    poly = me.polygons[hit[3]] if hit[3] < len(me.polygons) else None
    mats = list(me.materials)
    m = mats[poly.material_index] if poly is not None and poly.material_index < len(mats) else None
    return ob.original.name, (m.name if m else None), (len(m.node_tree.nodes) if m and m.node_tree else None)


for o in floors[:2]:
    describe(o)
    print("  ray hit (object, material, n nodes) before:", hit_material(o))
o = floors[0]
import glob  # noqa: E402
import random  # noqa: E402

path = sorted(glob.glob(tex_root + "/floor/*.jpg"))[0]
for m in eval_materials(o, dg):
    if m.node_tree:
        texture_floor_material(m, path, 1.0, 0.2, 0.0, random.Random(0))
        print("  textured", m.name, "->", len(m.node_tree.nodes), "nodes")
dg = bpy.context.evaluated_depsgraph_get()
dg.update()
print("  ray hit after:", hit_material(o))
print("  evaluated materials after:", [(m.name, len(m.node_tree.nodes)) for m in eval_materials(o, dg)])
