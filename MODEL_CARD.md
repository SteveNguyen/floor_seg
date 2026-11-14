---
license: apache-2.0
library_name: floor_seg
pipeline_tag: image-segmentation
tags:
  - robotics
  - reachy-mini
  - floor-segmentation
  - free-space
  - onnx
  - lraspp
  - mobilenetv3
datasets:
  - pollen-robotics/reachy-mini-floor-synth
---

# Reachy Mini floor segmentation (FloorNet v2)

Per-pixel **drivable floor vs. not floor** for the Reachy Mini head camera, small enough to run on the
robot's Raspberry Pi CM4 at ~9 Hz (fp32, ONNX Runtime, 1 thread). Trained only on synthetic renders.

- Architecture: LR-ASPP head on a MobileNetV3-Small backbone (torchvision), 2 classes, 1.08 M parameters.
- Input: 1 x 3 x 192 x 320, RGB, scaled to 0..1 then ImageNet-normalized
  (mean 0.485 0.456 0.406, std 0.229 0.224 0.225); resize the camera frame with area interpolation.
- Output: 1 x 2 x 192 x 320 logits (channel 1 = floor). **Decision: floor where softmax[1] > 0.8**
  (equivalently `logit[1] - logit[0] > ln 4`).
- Files: `model.safetensors` + `config.json` (PyTorch, `floor_seg.model.FloorNet`), `floor.onnx`
  (fp32, opset 17, input `input`, output `logits`), `metrics.jsonl` (synthetic validation per epoch).
- Code: `floor_seg` Python package (training, rendering, evaluation; repository URL TODO).

## Label definition

A pixel is **floor** if its 3D point is at most 1 cm above the floor plane. Anything higher is an obstacle.
Cables, door thresholds, rug edges and rug surfaces count as floor. Walls, furniture, objects and everything
above the floor are "not floor". The model says where the floor is visible; it does not see overhangs
(table tops, chair seats, sofa edges) as obstacles at their height, only where things touch the floor.

## Training data

- [pollen-robotics/reachy-mini-floor-synth](https://huggingface.co/datasets/pollen-robotics/reachy-mini-floor-synth):
  9,929 views of 1,996 ProcTHOR-10K houses rendered with AI2-THOR from the robot's viewpoint (camera 0.25-0.55 m
  above the floor) through the calibrated Reachy Mini lens, with clutter objects dropped on the floor.
  Train 9,011 views (1,812 houses), validation 918 views (184 houses, house id % 10 == 0).
- Training-time floor retexturing (ProcTHOR floors are all wood): photo textures from ambientCG (CC0),
  procedural tiles, rugs, sun patches and glare, laid on the floor plane in perspective using the stored depth.
- Camera-like augmentation: colour / gamma, noise, defocus and motion blur, JPEG, zoom-crop, flip, roll +-15 deg.
- 30 epochs, AdamW, one-cycle LR 1e-3, batch 32, mixed precision; best epoch by validation floor IoU.
- Backbone initialised from torchvision's ImageNet-1k MobileNetV3-Small weights (see Licensing notes).

## Results

Metrics are pooled over pixels. **False floor** = obstacle pixels predicted as floor / obstacle pixels
(the error that drives the robot into things). **Missed floor** = floor pixels predicted as not floor / floor pixels.

| Evaluation set | Threshold | Floor IoU | False floor | Missed floor |
|---|---|---|---|---|
| Synthetic validation (918 views, unseen houses, no retexturing) | 0.5 | 0.948 | 1.7 % | 2.8 % |
| Real home frames (50, private, hand-reviewed labels) | 0.5 | 0.91 | 7.6 % | 3.5 % |
| Real home frames (50, private, hand-reviewed labels) | **0.8** | **0.89** | **3.0 %** | **9.0 %** |
| Real, v1 = same renders without floor retexturing | 0.5 | 0.73 | 3.5 % | 25 % |

The real test set (50 frames from several homes, Reachy Mini camera, labels from SAM2 regions + manual review)
is private and not released. Threshold 0.8 trades missed floor (robot more cautious) for fewer false floor.

On-robot cost (Raspberry Pi CM4, live 1280x720 camera, fp32 ONNX Runtime CPU, intra-op spinning off):

| Threads | Rate | Inference (median) | Total incl. resize + normalize | CPU |
|---|---|---|---|---|
| 1 | 8.7 Hz | 89 ms | 107 ms | ~1.1 cores |
| 2 | 10 Hz (camera-limited) | 68 ms | 86 ms | ~1.5 cores |

Leave ONNX Runtime's thread spinning off (`session.intra_op.allow_spinning = 0`): it costs ~0.45 core extra.

## Usage

```python
import cv2, numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(hf_hub_download("pollen-robotics/reachy-mini-floor-seg", "floor.onnx"),
                            providers=["CPUExecutionProvider"])
bgr = cv2.imread("frame.jpg")
x = cv2.resize(bgr, (320, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
z = sess.run(None, {"input": x})[0][0]
floor = (z[1] - z[0]) > np.log(0.8 / 0.2)     # 192 x 320 bool, softmax[1] > 0.8
```

PyTorch: `from floor_seg.model import FloorNet; model = FloorNet.from_pretrained("pollen-robotics/reachy-mini-floor-seg", pretrained_backbone=False).eval()`
(or `floor_seg.model.load_torch(repo_id)`).

## Intended use

Near-field free-space detection for a small indoor robot (Reachy Mini, on the floor or on a mobile base) in
homes: which floor pixels in front of the robot are free, to feed a local obstacle map or a safety filter.
Not a safety-certified component: use it with a planner safety margin and other safeguards.

## Limitations

- **int8 post-training quantization breaks the model** (ONNX Runtime static QDQ, per-channel; tried min-max,
  entropy, percentile calibration and keeping layer groups in fp32): real-set IoU 0.68 / false floor 26 %
  vs 0.89 / 3.0 % in fp32. Use fp32; int8 would need quantization-aware training.
- **Flat obstacles 1-3 cm high are the weak case** (board edges, bath mats on duckboards): often called floor.
- **Overhangs are not represented**: table tops, chair seats and sofa edges are only "not floor" where they hide
  the floor; their height is not modelled.
- Far floor seen at grazing angles (> 1-1.5 m) tends to be marked not floor (over-cautious).
- Trained on synthetic data only; real failure cases seen before retexturing were dark blue tiles, rugs, sun
  patches and reflections. Evaluated on only 50 real frames from a few homes.
- Tuned to the Reachy Mini camera (lens, 16:9, viewpoint 0.25-0.55 m above the floor, camera pitch -5 to 29 deg).
  Other cameras or viewpoints are out of distribution.

## Licensing notes

- Code and weights: Apache-2.0.
- Training data: ProcTHOR-10K houses and AI2-THOR (Apache-2.0, Allen Institute for AI); ambientCG textures (CC0).
- **Open point (not decided):** the backbone was initialised from torchvision's ImageNet-1k pretrained
  MobileNetV3-Small weights. torchvision code is BSD-3-Clause, but ImageNet images come with
  non-commercial research terms, and whether weights trained on them inherit those terms is a known grey area
  for commercial use. Alternative if needed: retrain from scratch or from a backbone pretrained on
  commercially licensed data.

## Citation / attribution

- Deitke et al., "ProcTHOR: Large-Scale Embodied AI Using Procedural Generation", NeurIPS 2022.
- Kolve et al., "AI2-THOR: An Interactive 3D Environment for Visual AI", 2017.
- Howard et al., "Searching for MobileNetV3", ICCV 2019 (MobileNetV3 and LR-ASPP).
- ambientCG (https://ambientcg.com), CC0 textures.
