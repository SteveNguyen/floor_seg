---
license: apache-2.0
tags: [image-segmentation, robotics, onnx]
---

# floor_seg development models

Private working models for the Reachy Mini floor segmentation (code: the `floor_seg` repo). Not released:
the public model is [pollen-robotics/reachy-mini-floor-seg](https://huggingface.co/pollen-robotics/reachy-mini-floor-seg)
(planned, the v2 recipe). Each folder holds `best.pt` (state dict, chosen by synthetic val floor IoU),
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
- **Closed-loop sim** (hb_nav v3 behaviour, 16 houses, same seed): v7 drives 4.19 m mean (B 3.43), sees more
  floor (0.548 vs 0.511) and passes the bed frame B collided with. Using the under output as obstacles changed
  nothing measurable in 16 episodes.

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
