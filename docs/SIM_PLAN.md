# Photoreal household renders: pilot plan (Infinigen Indoors + Blender Cycles)

Status: planned. Context: the training data are AI2-THOR / ProcTHOR renders (Unity, simple lighting, all floors
wood) plus 2D augmentations; real-frame generalisation is the bottleneck (v8: augmentations that do not match
reality teach shortcuts, see DEV_MODELS.md).

## Why Infinigen Indoors + Cycles (survey summary, Nov 2026)

- Photoreal (path traced), fully procedural rooms: no third-party assets, BSD-3 licence, so a derived dataset is
  publishable and commercial use is fine. The other rich household sources are non-commercial (HSSD, 3D-FRONT,
  Structured3D, Replica, HM3D / Matterport, Aria Synthetic Environments) or locked to their simulator
  (BEHAVIOR-1K assets).
- Content: plinths (skirting boards), rugs, door casings, tile / wood / carpet floors. Missing: cables, door
  thresholds (to add procedurally).
- Labels from the height rule directly: the world-position pass gives each pixel's height above the floor;
  floor = h < 1.5 cm (rugs, cables, thresholds floor; plinths obstacle). Works with any lens.
- Lens: Cycles' polynomial fisheye camera, fitted to the Reachy Mini calibration.
- NVIDIA Isaac Sim: renderer and Replicator are good (Apache 2.0, free), but no ready-made homes, fisheye models
  with reported bugs, RTX >= 16 GB. The same Infinigen scenes export to Isaac if physics is ever needed.
- Evidence for real fine-tuning: NVIDIA's floor-segmentation case went from ~52 % (sim only) to 96 % mIoU with 44
  real images. A small private set of real training frames (separate from the test set) is likely the largest
  single lever.

## Pilot (2-3 days of work; render batches need a go)

1. Install Infinigen, generate and render one test room.
2. Camera: 0.25-0.40 m above the floor, pitch 5-25 deg down, roll +-3 deg; Cycles polynomial fisheye fitted to
   `floor_seg/assets/reachy_mini_camera.json` (check: < 1 px reprojection error against the OpenCV model).
3. Labels by the height rule from the position pass; `_under.png` from the geometry as in thor_render if cheap.
   Add procedural cables (4-8 mm) and door thresholds (5-15 mm); randomise lighting (day / night, lamps).
4. ~1,000 images: 20-30 rooms x 30-50 views. Measure time per room and per frame.
5. Train ProcTHOR only (v9 recipe), ProcTHOR + Infinigen, Infinigen only; score on the real test set
   (`floor_seg.eval_real --groups`, per group: ordinary, passage, doorstep, plinth).

Decision after the pilot: scale up Infinigen (thousands of rooms, server), or stay with ProcTHOR + real frames.

## Not verified yet

Plinths in every room type by default; Cycles time per frame; rug thickness vs the 1.5 cm rule; how the fisheye
fit behaves at the image corners. Full survey with sources: kept outside the repo (scratch report).
