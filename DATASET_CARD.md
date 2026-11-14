---
license: apache-2.0
pretty_name: Reachy Mini synthetic floor segmentation
task_categories:
  - image-segmentation
tags:
  - robotics
  - reachy-mini
  - floor-segmentation
  - synthetic
  - procthor
  - ai2thor
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
---

# Reachy Mini synthetic floor segmentation

Synthetic views of furnished, cluttered houses seen from the Reachy Mini head camera, close to the floor,
with a per-pixel **floor** mask and metric **depth**. Used to train
[pollen-robotics/reachy-mini-floor-seg](https://huggingface.co/pollen-robotics/reachy-mini-floor-seg).
All images are rendered; there are no photos of real homes or people in this dataset.

## Content

9,929 views of 1,996 ProcTHOR-10K houses (about 5 views per house), rendered with AI2-THOR 5.0.

| Split | Views | Houses | Rule |
|---|---|---|---|
| train | 9,011 | 1,812 | house id % 10 != 0 |
| validation | 918 | 184 | house id % 10 == 0 (unseen houses) |

Columns:

| Column | Type | Meaning |
|---|---|---|
| `image` | JPEG, RGB 640 x 360 | rendered view, through the Reachy Mini lens model (distorted like the real camera) |
| `floor` | PNG, 8-bit 640 x 360 | 255 = floor, 0 = not floor (no ignore label) |
| `depth` | PNG, 16-bit 640 x 360 | optical-axis depth z in mm (not ray length), 0 = no value |
| `name` | string | `h<house:05d>_<view>` |
| `house` | int | ProcTHOR-10K train-split house index |
| `view` | int | view index within the house |
| `camera_height_m` | float | camera height above the floor, m (0.25-0.55) |
| `pitch_deg` | float | camera pitch, positive = looking down (about -5 to 29 deg) |
| `roll_deg` | float | camera roll about the optical axis (-3 to 3 deg) |
| `yaw_deg` | float | camera heading in the house frame |
| `floor_frac` | float | fraction of floor pixels |
| `agree` | float | IoU between the geometric and instance floor masks (quality check, >= 0.85) |
| `n_clutter` | int | number of clutter objects added to the house |

`reachy_mini_camera.json`: the lens calibration (1280 x 720, OpenCV rational model) the views were rendered with;
at 640 x 360 use K / 2. With depth, pitch and roll it puts every pixel back in 3D (see
`floor_seg/floor_tex.py:floor_coords` for the conventions).

```python
from datasets import load_dataset
ds = load_dataset("pollen-robotics/reachy-mini-floor-synth", split="validation")
ex = ds[0]   # ex["image"], ex["floor"], ex["depth"] are PIL images
```

## How it was made

`floor_seg/thor_render.py` (random seed 0): per house, clutter objects from the AI2-THOR asset set (boxes,
pillows, toys, shoes, bins, plants, stools, ...) are dropped on free floor, materials are randomized; per view,
random lighting, a camera 0.25-0.55 m above the floor, pitch 0-24 deg plus +-5 deg base tilt, roll +-3 deg,
pointed at clutter 70 % of the time. A larger pinhole image is rendered and remapped through the calibrated
lens (RGB bilinear, mask and depth nearest).

**Floor label rule:** a pixel is floor if its 3D point is less than 1 cm above the floor AND it belongs to a
room floor surface in the instance segmentation (removes depth seen through glass). Views where the two masks
disagree (IoU < 0.85), with less than 15 % floor, or mostly facing a wall are dropped.
Project convention for real data (same rule): anything > 1 cm above the floor is an obstacle; cables, door
thresholds, rug edges and rug surfaces are floor.

No noise, blur or colour jitter is baked in: they are training-time augmentations. Floors are as rendered
(ProcTHOR floors are wood); the released model adds floor retexturing at training time with the ambientCG
textures listed in `textures/NOTICE.md` (when the `textures/` folder is included) or in
`floor_seg/assets/ambientcg_textures.md`.

## Limitations

- Synthetic only: ProcTHOR rooms, AI2-THOR assets and lighting; all floors are wood (hence retexturing).
- No rugs, cables or door thresholds in the renders; no people or pets.
- Overhangs (table tops, chair seats) are "not floor" only where they hide the floor.
- One camera model (Reachy Mini, one unit's calibration) and one height range.

## License and attribution

- This dataset: Apache-2.0.
- Scenes: ProcTHOR-10K (Apache-2.0) and AI2-THOR (Apache-2.0), Allen Institute for AI. Cite:
  Deitke et al., "ProcTHOR: Large-Scale Embodied AI Using Procedural Generation", NeurIPS 2022;
  Kolve et al., "AI2-THOR: An Interactive 3D Environment for Visual AI", 2017.
- Textures (if included): ambientCG, CC0 1.0 (public domain).
