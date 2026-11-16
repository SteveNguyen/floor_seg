"""Training labels for the 3-channel (under) model: encoding and loss masking (CPU, no data needed)."""
import numpy as np
import torch

from floor_seg.train import FLOOR, FLOOR_NA, IGNORE, NOT_FLOOR, UNDER, encode_labels, seg_loss


def test_encode_labels():
    floor = np.array([[0, 1, 1]], np.uint8)
    under = np.array([[1, 0, 1]], np.uint8)  # 'under' on an obstacle pixel is meaningless: stays NOT_FLOOR
    assert encode_labels(floor, under).tolist() == [[NOT_FLOOR, FLOOR, UNDER]]
    assert encode_labels(floor, None).tolist() == [[NOT_FLOOR, FLOOR_NA, FLOOR_NA]]


def test_under_loss_only_on_labelled_floor():
    lab = torch.tensor([[[NOT_FLOOR, FLOOR, UNDER, FLOOR_NA, IGNORE]]])
    z = torch.zeros(1, 3, 1, 5, requires_grad=True)
    seg_loss(z, lab).backward()
    g = z.grad[0, 2, 0]
    assert g[1] != 0 and g[2] != 0                      # FLOOR and UNDER pixels train the under channel
    assert g[0] == 0 and g[3] == 0 and g[4] == 0       # obstacle, unknown, ignored: no under gradient
    assert z.grad[0, :2, 0, 4].abs().sum() == 0        # ignored pixel: no floor gradient either
    assert z.grad[0, :2, 0, 3].abs().sum() > 0         # FLOOR_NA still trains floor


def test_two_channel_loss_unchanged():
    # a floor-only model with 0/1 labels: plain cross-entropy, as before the under output
    lab = torch.tensor([[[0, 1, 1, IGNORE]]])
    z = torch.randn(1, 2, 1, 4)
    ref = torch.nn.functional.cross_entropy(z, lab, ignore_index=IGNORE)
    assert torch.allclose(seg_loss(z, lab), ref)
