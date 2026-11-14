"""Inference helpers shared by eval_real and predict: one call signature for PyTorch and ONNX Runtime models."""
import cv2
import numpy as np
import torch

from floor_seg.model import load_torch, preprocess


class OnnxModel:
    """ONNX Runtime CPU session with the FloorNet call signature (fp32 export from floor_seg.export_onnx)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        return torch.from_numpy(self.sess.run(None, {"input": x.numpy()})[0])


def load_model(path):
    """.onnx -> ONNX Runtime; .pt state dict, save_pretrained folder or Hub repo id -> PyTorch FloorNet."""
    return OnnxModel(path) if path.endswith(".onnx") else load_torch(path)


def floor_prob(model, bgr):
    """Floor probability (float32, 0..1) for every pixel of a BGR image, at the image's own size."""
    with torch.no_grad():
        logits = model(torch.from_numpy(preprocess(np.ascontiguousarray(bgr[:, :, ::-1]))))[0]
    prob = torch.softmax(logits, 0)[1].numpy()
    return cv2.resize(prob, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
