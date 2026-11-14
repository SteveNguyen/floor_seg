"""Floor segmentation model: LR-ASPP on a MobileNetV3-Small backbone (torchvision), 2 classes, 1.08 M parameters.

Chosen for the Reachy Mini's CM4: fp32 with ONNX Runtime, 1 thread, runs at ~9 Hz (see README).
Input: N x 3 x H x W, RGB, ImageNet-normalized, H x W = 192 x 320. Output: N x 2 x H x W logits
(0 = not floor, 1 = floor). Recommended decision: floor where softmax(logits)[1] > 0.8.

Hugging Face Hub: FloorNet.save_pretrained(dir) writes model.safetensors + config.json,
FloorNet.from_pretrained(repo_id or dir) loads them (huggingface_hub.PyTorchModelHubMixin).
"""
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.lraspp import LRASPP

IN_W, IN_H = 320, 192
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
THRESHOLD = 0.8  # floor probability above which a pixel is floor (trades missed floor for fewer false floor)


def lraspp_small(pretrained_backbone=True, num_classes=2):
    """Same construction as torchvision's lraspp_mobilenet_v3_large, on the small backbone."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = mobilenet_v3_small(weights=weights, dilated=True).features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos, high_pos = stage_indices[-4], stage_indices[-1]
    low_ch, high_ch = backbone[low_pos].out_channels, backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})
    return LRASPP(backbone, low_ch, high_ch, num_classes)


class FloorNet(torch.nn.Module, PyTorchModelHubMixin, library_name="floor_seg", license="apache-2.0",
               pipeline_tag="image-segmentation", repo_url="https://huggingface.co/pollen-robotics/reachy-mini-floor-seg"):
    """Returns the logits tensor only (torchvision returns a dict), for training and ONNX export.

    pretrained_backbone: start from torchvision's ImageNet-1k MobileNetV3-Small weights (training only;
    loading a trained model never needs them). The mixin stores this argument in config.json.
    """

    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.m = lraspp_small(pretrained_backbone)

    def forward(self, x):
        return self.m(x)["out"]


def load_torch(path):
    """FloorNet in eval mode from a .pt state dict, a local save_pretrained folder or a Hub repo id."""
    if path.endswith(".pt"):
        model = FloorNet(pretrained_backbone=False)
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:  # pretrained_backbone=False overrides config.json: no ImageNet download, all weights come from the file
        model = FloorNet.from_pretrained(path, pretrained_backbone=False, strict=True)
    return model.eval()


def preprocess(rgb):
    """H x W x 3 uint8 RGB (any size) -> 1 x 3 x IN_H x IN_W float32, as used in training."""
    import cv2

    x = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].copy()
