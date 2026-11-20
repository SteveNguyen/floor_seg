# floor_seg

Floor segmentation for the Reachy Mini head camera: which pixels are drivable floor. A 1.08 M-parameter
LR-ASPP / MobileNetV3-Small network, trained only on synthetic renders of ProcTHOR houses, that runs on the
robot's CM4 at ~9 Hz in fp32 with ONNX Runtime.

- Model: `pollen-robotics/reachy-mini-floor-seg` on the Hugging Face Hub (planned, not published yet) — card:
  [MODEL_CARD.md](MODEL_CARD.md)
- Dataset: `pollen-robotics/reachy-mini-floor-synth` on the Hugging Face Hub (planned, not published yet: the
  AI2-THOR / ProcTHOR redistribution check is open) — card: [DATASET_CARD.md](DATASET_CARD.md)
- Development models (working versions, floor + "under" output; versions, data and scores):
  [DEV_MODELS.md](DEV_MODELS.md)

**Label rule:** anything more than 1 cm above the floor is an obstacle; cables, door thresholds, rug edges and
rug surfaces are floor. From the thor_v3 renders on, each view has a second mask, `_under.png`: floor pixels
with something lower than 0.45 m above them (bed, sofa, chair seat, low table; the robot is <= 0.40 m tall),
measured from the scene geometry (a depth render from below the floor), not from object boxes. The floor mask
keeps them: mixing "under" into the floor label made the model doubt plain open floor (thor_v2 experiment). In
views from a table top, the top is floor and the room floor below is not (see
[floor_seg/thor_render.py](floor_seg/thor_render.py)).

## Results (v2: ProcTHOR renders + floor retexturing)

| Evaluation set | Threshold | Floor IoU | False floor | Missed floor |
|---|---|---|---|---|
| Synthetic validation (918 views, unseen houses) | 0.5 | 0.948 | 1.7 % | 2.8 % |
| Real home frames (50, private) | 0.5 | 0.91 | 7.6 % | 3.5 % |
| Real home frames (50, private) | **0.8** | **0.89** | **3.0 %** | **9.0 %** |

False floor = obstacle pixels predicted floor / obstacle pixels; missed floor = floor pixels predicted not floor /
floor pixels. Use threshold **0.8**. On the CM4 (fp32, 1 thread, ORT spinning off): 8.7 Hz, 89 ms inference,
~1.1 cores. Details and limitations: [MODEL_CARD.md](MODEL_CARD.md).

## Install

```bash
uv sync                   # inference, training, evaluation, HF export
uv sync --extra label     # + SAM2 (transformers) for floor_seg.label_real
uv run pytest
```

Rendering needs its own Python 3.11 env (AI2-THOR 5), see [floor_seg/thor_render.py](floor_seg/thor_render.py).

## Inference

ONNX Runtime only (no torch; this is what runs on the robot):

```python
import cv2, numpy as np, onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # spinning wastes ~0.45 core
sess = ort.InferenceSession("floor.onnx", so, providers=["CPUExecutionProvider"])

bgr = cv2.imread("frame.jpg")                                          # any size, 16:9
x = cv2.resize(bgr, (320, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
z = sess.run(None, {"input": x})[0][0]                                 # 2 x 192 x 320 logits
floor = (z[1] - z[0]) > np.log(0.8 / 0.2)                              # softmax[1] > 0.8
```

A model trained with `--under` has a third channel: `under = 1 / (1 + np.exp(-z[2])) > 0.5` (floor with something
low above it); the floor test above is unchanged. See [DEV_MODELS.md](DEV_MODELS.md).

PyTorch:

```python
import cv2, torch
from floor_seg.model import THRESHOLD, load_torch, preprocess

model = load_torch("models/floor_v2")   # a local folder, a best.pt state dict, or a Hub repo id once published
rgb = cv2.imread("frame.jpg")[:, :, ::-1].copy()
with torch.no_grad():
    prob = torch.softmax(model(torch.from_numpy(preprocess(rgb))), 1)[0, 1]   # 192 x 320
floor = prob > THRESHOLD
```

Command line (mask + overlay per image; `--model` takes `.onnx`, `.pt`, a folder or a Hub repo id):

```bash
python -m floor_seg.predict --model floor.onnx --out out/ frames/*.jpg
```

On the robot: [examples/floor_live.py](examples/floor_live.py) runs the ONNX model on the live camera and
reports timing / CPU (it produced the CM4 numbers above).

## Data generation

```bash
# renders: <name>.jpg, <name>_floor.png (0/255), <name>_depth.png (uint16 mm), stats.jsonl
PYTHONPATH=. .venv-thor/bin/python -m floor_seg.thor_render --out data/synth/thor_v1 --houses 2000 --views 5
```

The lens defaults to the bundled Reachy Mini calibration
([floor_seg/assets/reachy_mini_camera.json](floor_seg/assets/reachy_mini_camera.json)).
Floor textures for retexturing: ambientCG CC0 1K-JPG colour maps listed in
[floor_seg/assets/ambientcg_textures.md](floor_seg/assets/ambientcg_textures.md), stored as
`data/textures/ambientcg/<group>/<AssetId>.jpg` with groups `floor terrazzo marble stone concrete wood rug`.

## Training and evaluation

```bash
# v2 recipe (needs a CUDA GPU): writes best.pt, last.pt, metrics.jsonl, floor.onnx
python -m floor_seg.train --data data/synth/thor_v1 --out models/floor_v2 --tex data/textures/ambientcg

# real labelled frames (<name>.jpg + <name>_floor.png: 0 obstacle, 1 floor, 255 ignore)
python -m floor_seg.eval_real --data data/real/test_v1 --thr 0.5 0.8 models/floor_v2/best.pt models/floor_v2/floor.onnx

# semi-automatic labels for real frames (SAM2 regions pre-labelled by a model, then reviewed by hand)
python -m floor_seg.label_real prelabel --model models/floor_v2/best.pt --out data/real/new_set recordings/*/spot*
python -m floor_seg.label_real apply --out data/real/new_set

python -m floor_seg.export_onnx --model models/floor_v2/best.pt --out floor.onnx
```

Real frames from homes are private: keep them out of any public dataset.

### Labelling real frames

```bash
# 1. SAM2 regions + model pre-labels: <name>.jpg, <name>_regions.png (uint16 ids), <name>_init.json (GPU)
python -m floor_seg.label_real prelabel --model models/floor_v2/best.pt --out data/real/new_set recordings/*/spot*
# 2. click-to-label page on http://127.0.0.1:8766 (local, offline): writes edits.json + reviewed.json
python -m floor_seg.label_page data/real/new_set
# 3. <name>_floor.png (0 obstacle, 1 floor, 255 ignore) for the reviewed frames (also the page's "apply" button)
python -m floor_seg.label_real apply --out data/real/new_set --reviewed-only
# 4. score
python -m floor_seg.eval_real --data data/real/new_set --thr 0.5 0.8 models/floor_v2/best.pt
```

On the page: click a region to cycle floor (green) -> obstacle (red) -> ignore (grey); 1/2/3 set the region under
the cursor; A all floor, Shift+A all obstacle; Z undo; R raw image; D driven-over hint (`<name>_driven.png`, blue);
V toggle reviewed; Enter reviewed + save + next; arrows / Space browse; S save. Anything > 1 cm above the floor is
an obstacle; rugs, cables, thresholds and doorsteps the robot can drive over are floor. Hand polygons (`"poly"`)
in edits.json are kept and applied last. eval_real scores every `_floor.png` in the folder, so delete stale ones
from an earlier apply without `--reviewed-only`.

## Hugging Face folders (local only)

```bash
python scripts/build_hf_model.py --ckpt models/floor_v2/best.pt --onnx models/floor_v2/floor.onnx \
    --metrics models/floor_v2/metrics.jsonl --out hf/model          # safetensors + config + onnx + card, self-checked
python scripts/build_hf_dataset.py data/synth/thor_v1 --out hf/dataset --textures data/textures/ambientcg
```

Neither script uploads anything.

## Real home floors dataset (public, consent-based; tools only)

Homes recorded with the robot by phone teleoperation (protocol: [docs/dataset/RECORDING.md](docs/dataset/RECORDING.md),
consent form: [docs/dataset/CONSENT.md](docs/dataset/CONSENT.md)). Keep STAGING outside the repo.

```bash
python -m floor_seg.dataset.import_run RUN_DIR STAGING --session home03   # Navette run -> subsampled session
python -m floor_seg.dataset.privacy STAGING/home03                        # face blur (YuNet), re-encode without EXIF
python -m floor_seg.dataset.review STAGING/home03                         # owner deletes frames (local page)
python -m floor_seg.dataset.consent STAGING/home03 --given y --date 2026-11-20 --role owner --form-ref "..."
python -m floor_seg.dataset.export STAGING/home03 STAGING/home04 --val home04 --out hf/real_floor   # no upload
```

Export refuses sessions without the privacy pass, the owner's review or written consent. Details:
[floor_seg/dataset/__init__.py](floor_seg/dataset/__init__.py).

## Known limitations

- **int8 post-training quantization breaks the model** (real IoU 0.68 / false floor 26 % vs 0.89 / 3.0 % fp32):
  deploy fp32. Quantization-aware training would be needed for int8; no quantization script is provided.
- Flat obstacles 1-3 cm high are often called floor; overhangs are not represented; far floor at grazing
  angles is marked not floor (cautious).
- Backbone initialised from torchvision ImageNet-1k weights: see the licensing note in [MODEL_CARD.md](MODEL_CARD.md).

## Layout

```
floor_seg/model.py        FloorNet (+ Hub mixin), preprocess, load_torch
floor_seg/infer.py        ONNX / torch loader, floor probability at image size
floor_seg/train.py        training on renders (augmentation, house split, ONNX export)
floor_seg/floor_tex.py    floor retexturing augmentation (needs depth + pose + lens)
floor_seg/aug_structures.py  plinths, threshold strips, level changes, sun streaks (train --aug-structures)
floor_seg/aug_preview.py  preview grid of those augmentations with labels
floor_seg/thor_render.py  ProcTHOR / AI2-THOR renderer
floor_seg/eval_real.py    scores on real labelled frames
floor_seg/label_real.py   SAM2-assisted labelling of real frames
floor_seg/label_page.py   click-to-label review page for label_real output
floor_seg/predict.py      mask + overlay CLI
floor_seg/export_onnx.py  ONNX export
floor_seg/dataset/        real home dataset: import, privacy, review, consent, export
scripts/                  local HF model / dataset folders
examples/floor_live.py    on-robot benchmark (CM4)
```

## License

Apache-2.0 (code and model). Data sources: ProcTHOR-10K and AI2-THOR (Apache-2.0), ambientCG (CC0).
