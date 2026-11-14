"""Write a local Hugging Face model-repo folder from a training run (nothing is uploaded).

    python scripts/build_hf_model.py --ckpt models/floor_v2/best.pt --onnx models/floor_v2/floor.onnx \
        --metrics models/floor_v2/metrics.jsonl --out hf/model

Output: model.safetensors + config.json (FloorNet.save_pretrained), floor.onnx (copied as is),
metrics.jsonl (synthetic validation per epoch, if given), README.md (= MODEL_CARD.md of this repo).
Checks that the saved folder reloads to the same outputs as the checkpoint, and that the ONNX file
agrees with it, before reporting success.
"""
import argparse
import os
import shutil

import numpy as np
import onnxruntime as ort
import torch

from floor_seg.model import IN_H, IN_W, FloorNet, load_torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="state dict saved by floor_seg.train (best.pt)")
    ap.add_argument("--onnx", required=True, help="floor.onnx exported from the same checkpoint")
    ap.add_argument("--metrics", help="metrics.jsonl of the run (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = load_torch(args.ckpt)  # pretrained_backbone=False: stored as such in config.json
    model.save_pretrained(args.out)
    shutil.copy(args.onnx, os.path.join(args.out, "floor.onnx"))
    if args.metrics:
        shutil.copy(args.metrics, os.path.join(args.out, "metrics.jsonl"))
    shutil.copy(os.path.join(REPO, "MODEL_CARD.md"), os.path.join(args.out, "README.md"))  # replaces the mixin's card

    # same weights, same outputs: checkpoint vs reloaded folder (exact), checkpoint vs ONNX (float tolerance)
    x = torch.randn(2, 3, IN_H, IN_W, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(x)
        back = FloorNet.from_pretrained(args.out, strict=True).eval()(x)
    sess = ort.InferenceSession(os.path.join(args.out, "floor.onnx"), providers=["CPUExecutionProvider"])
    onnx_out = np.concatenate([sess.run(None, {"input": x[i:i + 1].numpy()})[0] for i in range(len(x))])
    d_hub = (ref - back).abs().max().item()
    d_onnx = float(np.abs(ref.numpy() - onnx_out).max())
    print(f"max |ckpt - from_pretrained| {d_hub:.3g}, max |ckpt - onnx| {d_onnx:.3g}")
    assert d_hub == 0, "reloaded model differs from the checkpoint"
    assert d_onnx < 1e-3, "floor.onnx does not match the checkpoint"
    print("wrote", args.out, sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
