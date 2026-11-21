# Infinigen Indoors renders for floor_seg (sim pilot, docs/SIM_PLAN.md)

Photoreal procedural rooms (Infinigen 2.0, BSD-3, no third-party assets) rendered with Blender Cycles from the
robot's viewpoint, labelled by the project's height rule, in the same format as the AI2-THOR sets: `floor_seg.train`
reads the folder unchanged.

Code:
- `floor_seg/infinigen_labels.py`: Blender-free parts (camera conventions, labels, lens remap, room types,
  stats); tested in `tests/test_infinigen_labels.py`.
- `floor_seg/infinigen_render.py`: runs with Infinigen's Python (Blender 4.2 as `bpy`) on one generated room:
  ordinary-home bias, cameras, render, labels.
- `scripts/infinigen_gen.py`: the driver (rooms, seeds, shards, resume, merge).

## What a set contains

Per view `r<room:05d>_<k>`: `.jpg` (640x360 through the calibrated Reachy Mini lens), `_floor.png` (255 floor /
0 not), `_depth.png` (uint16 mm, optical-axis z); `stats.jsonl` with `name`, `house` (= room index: rooms with
index % 10 == 0 are the validation split), `kind` = floor, camera `h`, `pitch`, `roll`, `yaw`, `x`, `y`,
`floor_frac`, `room_type`, `seed`, `floor_material`, `wall_paint`, `cables`, `thresholds`, `aim`, `render_s`.
No `_under.png` yet (training with `--under` treats it as unknown, as for thor_v1).

- **Labels**: floor = within 1.5 cm of the floor plane (world-position pass) AND the floor object or something
  flat lying on it (thin floor-level objects, cables, threshold strips; object-index pass). Rugs are floor by rule
  although they are 1-2 cm thick (their top up to 4 cm, wrinkles included). Walls, skirting boards, furniture,
  objects and anything above are not floor.
- **Camera**: 0.25-0.40 m above the floor, pitch 5-25 deg down, roll +-3 deg, collision-free (rays around a 0.22 m
  footprint at 5 cm and at camera height); aimed at furniture / objects 40 %, at a wall 30 %, random 30 %; a ray
  grid rejects poses with < 12 % floor or > 60 % within 0.8 m before rendering; renders with < 15 % floor are
  dropped (as thor_render).
- **Lens**: a pinhole image (680x410, f = 410.6 px) rendered and remapped through the calibrated lens (same as
  thor_render, with Blender's half-pixel centre). Cycles' fisheye polynomial was fitted to the calibration but
  cannot match it (rms 19 px, max 31 px at 640x360: the model is centred, our principal point is ~16 px off).
- **Ordinary homes** (knobs in `floor_seg/infinigen_labels.py`):
  - room types: living 30 %, bedroom 25 %, kitchen 18 %, dining 12 %, bathroom 15 %; Infinigen 2.0 has no
    corridor room; `house_furnished_rand` (several rooms with doors) is available as `house` with weight 0
    (much slower, untested here): `--room-weights livingroom=1 ... house=0.2`.
  - floors: Infinigen draws concrete / wood planks / carpet / indoor tiles / layered mixes; a room whose floor is
    plain concrete is regenerated with probability 0.85.
  - walls: Infinigen's wall materials include bricks, hexagonal concrete, tiles, wood planks and patterned paint;
    the walls are repainted with an ordinary matte paint (white / off-white 40 %, cream 15 %, beige 15 %, light
    grey 15 %, soft pastel 15 %; an accent wall 20 %) in 90 % of rooms (kitchens 60 %, bathrooms 35 %: tiles are
    normal there); the skirting board then becomes white 50 %, the wall colour 30 %, unchanged 20 %. Some wall
    materials have their relief baked into the mesh at generation: it stays under the paint (textured plaster).
  - Infinigen's own knobs are code-level (`pf.control.choice` weights in `infinigen2/shaders/functionality_lists.py`:
    `wall_material_rand`, `floor_material_rand`, `skirt_material_rand`); the CLI only takes a generator and a seed,
    so the bias is applied after generation.
- **Optional additions** (driver flags, defaults below): `--cables 0.3` (chance per room of 1-3 cables, 2-4 mm
  radius, lying on the floor: floor), `--thresholds 0.5` (chance per door of a flush strip 4-12 mm high, 4-10 cm
  deep, across the door: floor).

## Install on a GPU machine

Linux with an NVIDIA GPU (OptiX / CUDA; HIP and oneAPI are tried too, else CPU), ~3 GB of disk for Infinigen.

```bash
git clone https://github.com/princeton-vl/infinigen.git ~/infinigen
git -C ~/infinigen checkout 53033dd55c7191cbc436be6ffbb63c026935a509   # the commit this was tested with
cd ~/infinigen && uv sync          # its own Python 3.11 env with Blender 4.2 as a module (~2.3 GB)
~/infinigen/.venv/bin/python -c "import bpy; p = bpy.context.preferences.addons['cycles'].preferences; \
    p.compute_device_type = 'OPTIX'; p.get_devices(); print([d.name for d in p.devices])"   # GPU check
git clone <floor_seg repo> ~/floor_seg && git -C ~/floor_seg checkout infinigen
```

## Generate the pilot set (~1,000 images: 200 rooms x 5 views)

One process per GPU, sharded by room index; re-running skips finished rooms:

```bash
cd ~/floor_seg
for i in 0 1; do   # N GPUs: --shard i N
  CUDA_VISIBLE_DEVICES=$i nohup ~/infinigen/.venv/bin/python scripts/infinigen_gen.py run \
      --infinigen ~/infinigen --out data/synth/infinigen_v1 --rooms 200 --views 5 --shard $i 2 \
      > infinigen_v1_$i.log 2>&1 &
done
~/infinigen/.venv/bin/python scripts/infinigen_gen.py merge data/synth/infinigen_v1   # stats.jsonl
```

QA: add `--overlay` to write `<name>_overlay.jpg` (photo | label: green floor, red not floor).

Timings (smoke test, RTX 4070 Laptop shared with another render job, 64 samples + denoising, 680x410 pinhole):
generating a room 28-29 s (+ ~2 s setup), 12-15 s per rendered view, plus a few seconds of collision / floor
ray checks; ~1.5-2 min per room of 5 views, so ~5-7 h for 200 rooms on one such GPU (faster on a desktop / server
GPU, divided by the number of GPUs). Output ~0.23 MB per view (~230 MB for 1,000 views); the generated .blend
files (~70 MB each) are deleted after rendering unless `--keep-blend`.

Copy back: `rsync -a --exclude work/ server:~/floor_seg/data/synth/infinigen_v1 data/synth/`.

## Train with it

```bash
python -m floor_seg.train --data data/synth/thor_v1 data/synth/infinigen_v1 --kinds floor \
    --out models/floor_inf_v1 --tex data/textures/ambientcg --under
```

Floor retexturing (`--tex`) applies to Infinigen views too (it uses the stored depth / pitch / roll / height);
their floors are already varied, so a run without `--tex` is worth comparing. Score on the real test set:
`python -m floor_seg.eval_real --data data/real/test_v2/labels --groups ... models/floor_inf_v1/best.pt`.

## Limitations / TODO

- No `_under.png` (floor with something < 0.45 m above it): to add with an upward ray / depth render as in
  thor_render if the pilot is worth scaling.
- Single rooms: no corridors, doorways between rooms only in `house` scenes (weight 0 by default).
- Rug detection is by object name (`rug*`); other mats Infinigen may add under other names are labelled by the
  height rule only.
- Lighting is Infinigen's (sky + sun through the windows, ceiling lights); film exposure 2.0 x 0.7-1.4.
- Infinigen's one-command render with its own ground truth crashed after the RGB image on the laptop; the
  render here is our own (Cycles directly), so it is not affected.
