"""Export a trained FloorNet to ONNX (fp32, opset 17, fixed 1 x 3 x 192 x 320 input "input", output "logits").

    python -m floor_seg.export_onnx --model models/floor_v2/best.pt --out floor.onnx

Model: .pt state dict, save_pretrained folder or Hub repo id. floor_seg.train calls export() at the end.
Do not post-training-quantize the result to int8: it breaks this model (README, known limitations).
"""
import argparse

import torch

from floor_seg.model import IN_H, IN_W, load_torch


def export(model, path):
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, IN_H, IN_W), path,
                      input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(load_torch(args.model), args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
