---
license: apache-2.0
tags: [image-segmentation, robotics, onnx]
---

# floor_seg development models

Development models for the Reachy Mini floor segmentation (code: the `floor_seg` repo), public at
[SteveNguyen/reachy-mini-floor-seg-dev](https://huggingface.co/SteveNguyen/reachy-mini-floor-seg-dev): working
versions, not a release. Each folder holds `best.pt` (state dict, chosen by synthetic val floor IoU),
`last.pt`, `floor.onnx` (export of `best.pt`) and `metrics.jsonl` (one line per epoch).

**Current choice for navigation: `v7_floor_under`**, floor cutoff 0.7, under cutoff 0.5.

## Render sets (training data, `floor_seg.thor_render`, ProcTHOR-10K houses)

| Set | Views | Content |
|---|---|---|
| thor_v1 | 9,929 | low camera views; floor / not floor |
| thor_v2 | | + table-top views and "floor under low furniture = not floor". Abandoned: the model then doubted plain open floor |
| thor_v3 | | new scenes, clutter resting on the floor, table-top views, + `_under.png`: floor with something < 0.45 m above it, from an upward depth render of the scene |

Validation = houses with index % 10 == 0, never trained on.

## Models (LR-ASPP / MobileNetV3, input 320 x 192)

| Folder | Data | Outputs | Synthetic val floor IoU | Real floor IoU, cutoff 0.5 / 0.7 |
|---|---|---|---|---|
| B (`floor_v3_neg`, local only) | thor_v1 | floor | 0.947 (thor_v1 val) | **0.911** / **0.896** |
| `v5_under` | thor_v3 | floor + under | 0.933 | 0.822 / 0.773 |
| `v6_mix_small` | thor_v1 + thor_v3 | floor + under | 0.940 | 0.870 / 0.829 |
| `v6_mix_large` (MobileNetV3-Large, 3.2 M) | thor_v1 + thor_v3 | floor + under | 0.953 | 0.875 / 0.839 |
| **`v7_floor_under`** | thor_v1 + thor_v3 floor views (no table views) | floor + under | 0.946 | 0.889 / 0.877 |
| `v7_floor_only` | same as v7_floor_under | floor | 0.951 | 0.895 / 0.873 |

All trained 30 epochs with floor retexturing (`--tex`). Synthetic val sets differ (v5 / v6 include table views),
so compare that column within a row group only. Real = 50 hand-labelled home frames (private, never published).

- **Table views hurt real floor** (v6 -> v7): they label visible floor below the table as not floor.
- **v7 vs B on real frames**: same false floor; the remaining ~0.02 IoU gap is mostly two frames (sunlit dark
  shiny tiles, a slatted wooden bath mat). The under head costs ~0.005.
- **Closed-loop sim** (the Python prototype's v3 behaviour, 16 houses, same seed): v7 drives 4.19 m mean (B 3.43), sees more
  floor (0.548 vs 0.511) and passes the bed frame B collided with. Using the under output as obstacles changed
  nothing measurable in 16 episodes.
- **Second real test set** (75 hand-labelled frames from the robot's own runs, private, grouped by difficulty;
  `floor_seg.eval_real --groups`): v7_floor_under floor IoU 0.946 / 0.932 at cutoff 0.5 / 0.7; weakest on wall
  plinths (14 % false floor at 0.5) and on doorsteps (5-13 % missed floor), the targets of v8.

## The "under" output

Three logit channels: 0-1 floor (softmax over these two only), 2 under (sigmoid, meaningful on floor pixels).
Drivable = floor > 0.7 and not under > 0.5.

| v7_floor_under on thor_v3 val floor views (892) | Cutoff 0.3 | Cutoff 0.5 |
|---|---|---|
| Strict IoU / precision / recall | 0.402 / 0.59 / 0.56 | 0.358 / 0.73 / 0.41 |
| Lenient IoU / precision (a false alarm under something < 1 m is not counted) | 0.437 / 0.67 | 0.373 / 0.80 |

Most disagreement is outlines (within 6 px of the label edge), and small tables (top 0.45-1 m) called under.
Not yet checked on real frames (no under labels there).

## Training command (v7_floor_under)

```bash
python -m floor_seg.train --data data/synth/thor_v1 data/synth/thor_v3 --kinds floor \
    --out models/floor_v7_floor_under --tex data/textures/ambientcg --under
```

## v8 (`floor_v8_struct`, local only): v7 + structure augmentations, worse than v7

v7 recipe plus `--aug-structures` with the augmentation defaults of the time (commit `a44ccd3`): plinths on wall
bases (70 % the floor's own tiles, 5-12 cm, obstacle), flush threshold strips (floor), level changes (the floor
beyond an edge 2-18 cm higher or lower: obstacle; 20 % within 1 cm: floor), sun streaks / specular highlights
(floor). Probabilities plinth 0.3, threshold 0.3, level 0.3, sun 0.25, specular 0.2. Synthetic val floor IoU 0.9365.

Real test set (75 frames, cutoff 0.5), v7 -> v8, floor IoU (false floor / missed floor):

| Group (frames) | v7 | v8 |
|---|---|---|
| all (75) | 0.946 (2.7 / 2.9 %) | 0.930 (3.2 / 4.1 %) |
| doorstep (7) | 0.932 (1.8 / 5.0 %) | **0.955** (1.8 / 2.7 %) |
| blocked (5) | 0.931 (1.0 / 5.7 %) | **0.957** (1.7 / 2.2 %) |
| doorstep_rough (3) | 0.869 (0.5 / 12.7 %) | 0.749 (0.3 / 24.8 %) |
| plinth (3) | 0.743 (14.2 / 3.3 %) | 0.713 (16.1 / 4.3 %) |
| ordinary (15) | 0.973 (2.2 / 0.9 %) | 0.954 (3.8 / 1.5 %) |
| passage (26) | 0.962 (2.3 / 1.8 %) | 0.951 (2.6 / 2.7 %) |

Lessons (side-by-side images of the frames that changed most):
- **Threshold strips work**: doorsteps v7 refused are accepted (missed floor 2-6x lower on those frames).
- **Small level changes taught a shortcut.** A 2-8 cm step is hardly visible in one image from 0.3 m high, so
  the model learned "another flooring past a line across the floor = obstacle" and refused flat doorways
  (terracotta beyond a white-tile threshold: IoU 0.97 -> 0.46). Real homes have drivable doorsteps or real steps,
  hardly anything in between.
- **Sun / specular** most likely taught "bright, washed-out = floor": v8 calls white plinths and wall bases floor.
- **The plinth augmentation did not match real plinths** (mostly painted white or wall-coloured boards 8-20 cm,
  rarely tiles): no gain on real plinths.

## v9a (`floor_v9a_steps`, local only): steps and door-wide thresholds only, worse than v7

Augmentations revised after v8 (current defaults of [floor_seg/aug_structures.py](floor_seg/aug_structures.py)):
real steps only (10-20 cm, obstacle) in 5 % of views; flat flooring changes (<= 1 cm, floor) 3x as often, 70 %
with another flooring beyond; threshold strips and level edges door-wide only (room-wide ones in open floor
looked like random bars); sun / specular off. Plinths and tiled wall bases are off: painted in 2D they also land
on doors, cabinets and bins (the renders have no wall mask). The 2D plinth styles (wall colour, white, wood,
floor tiles, grey / black; lit top edge, joint line, contact line) stay available (`--aug-p-plinth`).
Next (v9b): plinths and doorway thresholds rendered in 3D in `thor_render` (walls and doors are known there).
Preview: `python -m floor_seg.aug_preview --data data/synth/thor_v3 --n 12 --all-only --out p.jpg`.

```bash
python -m floor_seg.train --data data/synth/thor_v1 data/synth/thor_v3 --kinds floor \
    --out models/floor_v9a_steps --tex data/textures/ambientcg --under --aug-structures
```

Synthetic val floor IoU 0.9430 (not comparable with v8: other augmentations in val). Real test set, cutoff 0.5:

| Group (frames) | v7 | v9a |
|---|---|---|
| all (75) | **0.946** (2.7 / 2.9 %) | 0.921 (3.5 / 4.7 %) |
| doorstep (7) | 0.932 (1.8 / 5.0 %) | **0.945** (2.1 / 3.4 %) |
| blocked (5) | 0.931 (1.0 / 5.7 %) | **0.955** (2.4 / 1.7 %) |
| doorstep_rough (3) | 0.869 (0.5 / 12.7 %) | **0.918** (1.1 / 7.0 %) |
| plinth (3) | **0.743** (14.2 / 3.3 %) | 0.677 (20.3 / 3.2 %) |
| ordinary (15) | **0.973** (2.2 / 0.9 %) | 0.965 (2.9 / 1.2 %) |
| passage (26) | **0.962** (2.3 / 1.8 %) | 0.904 (2.6 / 7.5 %) |

19 frames worse by > 0.02 IoU, 7 better. Doorsteps and door frames improve (no longer refused). The loss is one dim,
motion-blurred tiled hallway (5 frames of one run): v9a calls large patches of the plain far floor non-floor
(IoU 0.99 -> 0.37); v8 did not fail there, so the cause is not isolated (candidates: frequent flooring changes,
steps whose lower floor keeps the same texture, or run-to-run variance). Plinths get worse with no plinth
augmentation (floor-matching tiled plinths read as floor).

Diagnosis of the hallway failure (no retraining): v9a is unsure there rather than confidently wrong (missed floor
mostly at p 0.1-0.5, diffuse over the far floor, no sharp edge at a grout line). Smoothing the speckled tile
texture (median 9 px) or blurring the frame restores v9a (IoU 0.37 -> 0.99 on the worst frame); brightening makes
it worse. Previewing steps with the same texture beyond (p_level_other 0.7 of steps) shows the likely source:
step-downs of 10-20 cm often leave the image almost unchanged (uniform floor, edge barely visible) while most of
the floor is relabelled obstacle, so the model learns "far floor may be obstacle, no visible cue needed", and its
far-floor confidence then hangs on texture. Fix to try: keep only steps with visual evidence (another flooring
beyond, a visible nosing or riser), or drop same-texture step-downs; v9b runs with steps off.

## v9b (`floor_v9b_plinth`, local only): rendered plinths and real doorways, steps off; about equal to v7

thor_v4 = the thor_v3 houses and cameras (same seed) rendered with `thor_render --plinths`: a 1 cm board, 8-20 cm
high (one height per house), along every room wall, cut at the doors, labelled obstacle by the floor rule;
`<name>_plinth.png` its pixels; stats.jsonl `doorways` (door floor segments in view). Training recolours the
board on its rendered shading (style mix: wall colour 40 %, white 30 %, wood 15 %, the retextured floor's tiles
10 %, grey / black 5 %; lit top edge, joint, contact line) and puts the threshold strips in real doorways
(`--aug-p-door-threshold`, 0.5), never on open floor. thor_v1 has no plinths: about half the views without.
Steps / flat flooring changes and 2D strips off (after the v9a diagnosis), no sun / specular, no painted plinths.
Validation (unaugmented) keeps the rendered boards neutral grey.

```bash
python -m floor_seg.thor_render --out data/synth/thor_v4 --houses 2000 --views 5 --plinths   # AI2-THOR env
python -m floor_seg.train --data data/synth/thor_v1 data/synth/thor_v4 --kinds floor \
    --out models/floor_v9b_plinth --tex data/textures/ambientcg --under --aug-structures \
    --aug-p-level 0 --aug-p-threshold 0 --workers 16
```

Synthetic val floor IoU 0.9420. Real test set, cutoff 0.5:

| Group (frames) | v7 | v9b |
|---|---|---|
| all (75) | 0.946 (2.7 / 2.9 %) | 0.952 (2.0 / 3.0 %) |
| plinth (3) | 0.743 (14.2 / 3.3 %) | 0.825 (7.6 / 4.2 %) |
| passage (26) | 0.962 (2.3 / 1.8 %) | 0.964 (1.7 / 2.0 %) |
| doorstep (7) | 0.932 (1.8 / 5.0 %) | 0.958 (0.6 / 3.5 %) |
| ordinary (15) | 0.973 (2.2 / 0.9 %) | 0.971 (2.1 / 1.2 %) |
| doorstep_rough (3) | 0.869 (0.5 / 12.7 %) | 0.872 (0.3 / 12.5 %) |
| blocked (5) | 0.931 (1.0 / 5.7 %) | 0.928 (2.1 / 4.8 %) |

Not a measurable gain. Paired over frames (bootstrap 95 % interval of the pooled-IoU difference): +0.006
[-0.001, +0.013], 12 frames better / 7 worse by > 0.02 (sign test p 0.36); without the 3 plinth frames +0.003
[-0.003, +0.008]. Plinth false floor halves, but on 3 frames, and thor_v4 also replaces thor_v3 (renders differ),
so the gain is not cleanly the plinths'. The hallway failure of v9a is gone (passage 0.964). v7 stays the
default; the test set cannot resolve differences below about 1 point.
