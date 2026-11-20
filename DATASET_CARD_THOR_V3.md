---
license: other
pretty_name: Reachy Mini synthetic floor segmentation, thor_v3 (internal)
task_categories:
  - image-segmentation
tags:
  - robotics
  - reachy-mini
  - floor-segmentation
  - synthetic
  - procthor
  - ai2thor
---

# Reachy Mini synthetic floor segmentation: thor_v3 (internal, private)

> This card describes a private dataset that is not downloadable yet: it documents how the development models
> were trained. A public synthetic dataset is planned (`DATASET_CARD.md`) once the licence check is done; the
> renders can be reproduced with `floor_seg/thor_render.py`.

Internal training set for the Reachy Mini floor segmentation model. Synthetic views of furnished, cluttered
ProcTHOR houses seen from the Reachy Mini head camera close to the floor, with two per-pixel masks per view:
**floor** and **under** (floor the robot cannot drive to because something lower than 0.45 m is above it).
All images are rendered: no photos of real homes or people.

Private, not for redistribution: the AI2-THOR / ProcTHOR redistribution check for a public release is still
open. The planned public dataset is `pollen-robotics/reachy-mini-floor-synth` (see `DATASET_CARD.md` in the
floor_seg repo).

## Content

About 10,000 floor views and 1,000-1,500 table-top views of up to 2,000 ProcTHOR-10K houses (5 floor views per
house, plus at most one table view). Exact counts: `stats.jsonl` (one line per view).

Split by house, as in `floor_seg/train.py`: house id % 10 == 0 is validation (unseen houses), the rest train.

### Files

| File | Content |
|---|---|
| `h0.tar` ... `h9.tar` | the views, packed by thousands of the house id: `h<d>.tar` holds houses d000-d999 (`h0<d>*` files; the Hub allows 10,000 files per folder) |
| `stats.jsonl` | one JSON line per view (fields below) |
| `reachy_mini_camera.json` | lens calibration the views were rendered with (1280 x 720, OpenCV rational model; at 640 x 360 use K / 2) |

Per view `<name>` = `h<house:05d>_<k>` (floor view k) or `h<house:05d>_t0` (table view), inside the tars:

| File | Type | Meaning |
|---|---|---|
| `<name>.jpg` | JPEG, RGB 640 x 360 | rendered view through the Reachy Mini lens model (distorted like the real camera) |
| `<name>_floor.png` | PNG, 8-bit 640 x 360 | 255 = floor, 0 = not floor |
| `<name>_under.png` | PNG, 8-bit 640 x 360 | 255 = floor pixel with something lower than 0.45 m above it (a subset of floor); all 0 in table views |
| `<name>_depth.png` | PNG, 16-bit 640 x 360 | optical-axis depth z in mm (not ray length), 0 = no value |

`stats.jsonl` fields:

| Field | Meaning |
|---|---|
| `name`, `house` | view name, ProcTHOR-10K train-split house index |
| `kind` | `floor` (camera above the room floor) or `table` (camera above a table / counter top) |
| `x`, `z`, `yaw` | camera position (m) and heading (deg) in the house frame (Unity: y up, left-handed) |
| `support`, `h` | height of the surface the robot stands on (0 = room floor), camera height above it (0.25-0.55 m) |
| `pitch`, `roll` | camera pitch (positive = looking down, about -5 to 29 deg) and roll (-3 to 3 deg) |
| `floor_frac` | fraction of floor pixels |
| `under_frac` | fraction of the floor pixels that are under |
| `agree` | rules 1-2 agreement (quality check, >= 0.85; see below) |
| `n_clutter`, `clutter_moved`, `clutter_off` | clutter objects added; how many were moved onto the floor; largest remaining distance of a clutter object's bottom from the floor (m, ~0) |

```bash
hf download pollen-robotics/reachy-mini-floor-synth-dev --repo-type dataset --local-dir thor_v3_tar
mkdir -p thor_v3 && for t in thor_v3_tar/h*.tar; do tar -xf "$t" -C thor_v3; done
cp thor_v3_tar/stats.jsonl thor_v3/
```

## How it was made

`floor_seg/thor_render.py`, floor_seg repo branch `thor-v3-labels` (rendering code as of commit `35ae8a7`), seed 0,
`--houses 2000 --views 5`, rendered in 6 parallel shards (`--shard I 6`), AI2-THOR 5.0 CloudRendering.

Per house: clutter objects from the AI2-THOR asset set (boxes, pillows, toys, shoes, bins, plants, stools,
...) are dropped on free floor, then moved so that their bottom rests on the floor (as loaded in the scene);
materials are randomized. Per view: random lighting, camera 0.25-0.55 m above the support surface, pitch 0-24
deg plus +-5 deg base tilt, roll +-3 deg, pointed at clutter 70 % of the time. A larger pinhole image is
rendered and remapped through the calibrated lens (RGB bilinear, masks and depth nearest).

**floor:** a pixel is floor if (1) its 3D point is within 1 cm of the plane the robot stands on AND (2) it
belongs to that surface in the instance segmentation (a room floor; in table views, the table top). Views where
(1) and (2) disagree, with too little floor, or mostly facing a wall are dropped.

**under:** floor pixels with a surface lower than 0.45 m above them (Reachy Mini on its base is <= 0.40 m tall,
plus 5 cm). Measured from the geometry, not from object boxes: a depth render from 10 m below the floor looking
up (the floor mesh is invisible from below) gives the height of the lowest surface above every floor point
(2 cm grid over the whole house). Then: gaps narrower than ~10 cm between under cells are closed (bed slats,
shelf slats), free pockets smaller than 0.25 m^2 fully enclosed by under cells become under (inside a bar
stool's foot ring, chair stretchers), and cells never measured count as under. Not under: cells whose lowest
surface is below 3 cm (an object standing on the floor covers it; this keeps rims off object bases) and walls,
doors and windows (structure, not overhangs). So the floor under a bed, sofa,
chair seat or low table is under; under a dining table, desk or loft bed it is not (only around their legs).

No noise, blur or colour jitter is baked in (training-time augmentation, as are floor retexturing and the
texture-swap negatives of `floor_seg/floor_tex.py`).

## Differences from earlier renders

- thor_v1: no under mask; clutter was placed from the asset catalogue's box, and many objects floated or sank
  into the floor (up to 0.45 m).
- thor_v2: under was removed from the floor mask, using object boxes (wrong for loft beds, tall tables, meshes
  larger or smaller than their box). A model trained on it missed much more real open floor; superseded.

## Limitations

- Synthetic only: ProcTHOR rooms, AI2-THOR assets and lighting; all floors are wood (hence retexturing).
- No rugs, cables or door thresholds; no people or pets.
- Single-sided meshes are invisible from below, so a surface modelled as a single upward-facing plane would not
  make under.
- Table views: some look washed out (lighting); not reviewed in detail yet.
- One camera model (one Reachy Mini unit's calibration) and one height range.

## License and attribution

- Internal use; no redistribution until the AI2-THOR / ProcTHOR licence check is done.
- Scenes: ProcTHOR-10K and AI2-THOR, Allen Institute for AI. Cite:
  Deitke et al., "ProcTHOR: Large-Scale Embodied AI Using Procedural Generation", NeurIPS 2022;
  Kolve et al., "AI2-THOR: An Interactive 3D Environment for Visual AI", 2017.
