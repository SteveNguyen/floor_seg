"""FloorNet shapes, preprocessing, Hub save/load and ONNX export (CPU only, no ImageNet download)."""
import json

import numpy as np
import onnxruntime as ort
import torch

from floor_seg.export_onnx import export
from floor_seg.infer import OnnxModel, floor_prob
from floor_seg.model import IN_H, IN_W, MEAN, STD, FloorNet, load_torch, preprocess


def small_model():
    torch.manual_seed(0)
    return FloorNet(pretrained_backbone=False).eval()


def test_forward_shape_and_size():
    model = small_model()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, IN_H, IN_W))
    assert out.shape == (2, 2, IN_H, IN_W)
    n = sum(p.numel() for p in model.parameters())
    assert 1.0e6 < n < 1.1e6  # 1.08 M parameters


def test_preprocess():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    x = preprocess(rgb)
    assert x.shape == (1, 3, IN_H, IN_W) and x.dtype == np.float32
    # channel order stays RGB, ImageNet normalization
    np.testing.assert_allclose(x[0, :, 0, 0], (np.array([1, 0, 0], np.float32) - MEAN) / STD, rtol=1e-6)


def test_hub_round_trip(tmp_path):
    model = small_model()
    model.save_pretrained(tmp_path)
    assert json.load(open(tmp_path / "config.json")) == {"pretrained_backbone": False, "under": False}
    assert (tmp_path / "model.safetensors").exists()
    back = load_torch(str(tmp_path))
    x = torch.randn(1, 3, IN_H, IN_W)
    with torch.no_grad():
        assert torch.equal(model(x), back(x))


def test_state_dict_checkpoint(tmp_path):
    model = small_model()
    torch.save(model.state_dict(), tmp_path / "best.pt")  # what floor_seg.train writes
    back = load_torch(str(tmp_path / "best.pt"))
    x = torch.randn(1, 3, IN_H, IN_W)
    with torch.no_grad():
        assert torch.equal(model(x), back(x))


def test_onnx_export_matches_torch(tmp_path):
    model = small_model()
    path = str(tmp_path / "floor.onnx")
    export(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"] and [o.name for o in sess.get_outputs()] == ["logits"]
    bgr = np.random.default_rng(0).integers(0, 256, (360, 640, 3), np.uint8)
    p_torch, p_onnx = floor_prob(model, bgr), floor_prob(OnnxModel(path), bgr)
    assert p_torch.shape == (360, 640)
    np.testing.assert_allclose(p_torch, p_onnx, atol=1e-4)


def test_under_model_three_channels_onnx_and_pt(tmp_path):
    from floor_seg.infer import under_prob

    torch.manual_seed(0)
    model = FloorNet(pretrained_backbone=False, under=True).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, IN_H, IN_W)).shape == (1, 3, IN_H, IN_W)
    torch.save(model.state_dict(), tmp_path / "best.pt")
    back = load_torch(str(tmp_path / "best.pt"))  # head size read from the state dict
    path = str(tmp_path / "floor.onnx")
    export(model, path)
    bgr = np.random.default_rng(0).integers(0, 256, (360, 640, 3), np.uint8)
    for m in (back, OnnxModel(path)):
        np.testing.assert_allclose(floor_prob(m, bgr), floor_prob(model, bgr), atol=1e-4)
        np.testing.assert_allclose(under_prob(m, bgr), under_prob(model, bgr), atol=1e-4)
    assert under_prob(small_model(), bgr) is None  # floor-only model


def test_logit_threshold_equals_softmax_threshold():
    # robot-side shortcut (examples/floor_live.py): softmax(z)[1] > thr  <=>  z1 - z0 > log(thr / (1 - thr))
    z = torch.randn(2, 1000) * 5
    thr = 0.8
    assert torch.equal(torch.softmax(z, 0)[1] > thr, (z[1] - z[0]) > np.log(thr / (1 - thr)))
