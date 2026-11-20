# Auto-labelling real robot frames (plan, not started)

Goal: use a lot of real indoor frames recorded by the robot for training, without hand labelling, by spending
compute on big "teacher" models offline. The labels must follow the project rule: anything more than 1-2 cm
above the floor is an obstacle; rugs, cables, drivable thresholds and doorsteps are floor; wall plinths are
obstacle; floor beyond a level change of more than 2 cm is obstacle. The rule is about height, not meaning,
which no off-the-shelf model gives on its own.

Real home frames stay private (consent, no publication): see `docs/dataset/` and the README.

## Candidate teachers

Model names as known mid-2026, to be checked by a literature pass before starting.

| Approach | Examples | Strengths | Weaknesses |
|---|---|---|---|
| A. Text-prompted segmentation | SAM 3 ("floor", "rug", "doorstep"), Grounded-SAM 2 | object classes, rugs, plinths as wall, clean edges | semantic, not height: the next room's floor is "floor" even across a step; doorsteps ambiguous |
| B. Closed-set segmenters | OneFormer / Mask2Former (ADE20K: floor, rug, stairs, step), DINOv3 + segmentation head | robust | same semantic-versus-height mismatch; other label conventions |
| C. Single-image depth + floor plane | Depth Anything 3, MoGe-2, Metric3D v2, surface normals | matches the height rule; the robot's known camera height and pitch fix the scale and the plane | 1-2 cm accuracy only near (< ~1.5 m) |
| D. 3D reconstruction of the whole run | VGGT, MASt3R-SfM, MapAnything on the sequence, seeded with odometry | real multi-view 3D: steps, level changes, plinths measured; labels consistent across frames | heaviest; fails on blur and texture-less walls; moving people |
| E. Where the robot drove | the "driven-over" masks (traversability learning, e.g. Wild Visual Navigation) | certain positives, free (doorsteps crossed = floor) | positives only; biased toward easy floor |

## Proposed approach: agreement of several teachers

- **Floor**: driven over (E), or semantic floor (A/B) lying on the measured floor plane (C/D) within 2 cm.
- **Obstacle**: semantic wall / furniture / object, or geometry more than 2 cm off the floor plane.
- **Ignore (255)** wherever the teachers disagree: training skips those pixels, so guesses are never taught.
- Train FloorNet on these pseudo-labels together with the synthetic data.

## First step (when we get there)

Pilot on the hand-labelled real test set (`data/real/test_v2/labels`, 75 frames, private): run A (SAM 3,
"floor" prompt) and C (depth + known floor plane), alone and combined, and score them against the hand labels
with `floor_seg.eval_real` per frame group (ordinary, passage, doorstep, plinth, ...). Target: near-perfect
agreement on ordinary scenes, and a measured behaviour on doorsteps and plinths. Needs a GPU.

Before that: a literature pass (auto-labelling for floor / traversability segmentation, current state of the art).

The test set itself is never used for training: it is what measures both the teachers and the trained models.
