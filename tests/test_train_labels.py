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


def test_mixed_folders_keep_their_own_labels(tmp_path):
    # thor_v1-like folder (no _under.png) + thor_v3-like folder (same view names, with _under.png): each view is
    # read from its own folder, v1 views are FLOOR_NA, v3 views carry UNDER; split by house in both folders
    import cv2
    from conftest import make_renders
    from floor_seg.train import datasets

    v1, v3 = tmp_path / "v1", tmp_path / "v3"
    v1.mkdir()
    v3.mkdir()
    make_renders(v1)
    make_renders(v3)
    for name in ("h00003_0", "h00003_1", "h00010_0", "h00010_1"):
        u = np.zeros((360, 640), np.uint8)
        u[300:] = 255  # bottom rows of the floor are under
        cv2.imwrite(str(v3 / f"{name}_under.png"), u)
    tr, va = datasets([str(v1), str(v3)], under=True)
    assert len(tr) == 4 and len(va) == 4  # house 3 -> train, house 10 -> val, in each folder
    labs = [set(np.unique(va[i][1].numpy())) for i in range(4)]
    assert labs[0] == labs[1] == {NOT_FLOOR, FLOOR_NA}
    assert labs[2] == labs[3] == {NOT_FLOOR, FLOOR, UNDER}
    # --repeat: the second folder's training views come 3 times per epoch, validation unchanged
    tr, va = datasets([str(v1), str(v3)], under=True, repeat=[1, 3])
    assert len(tr) == 2 + 3 * 2 and len(va) == 4


def test_two_channel_loss_unchanged():
    # a floor-only model with 0/1 labels: plain cross-entropy, as before the under output
    lab = torch.tensor([[[0, 1, 1, IGNORE]]])
    z = torch.randn(1, 2, 1, 4)
    ref = torch.nn.functional.cross_entropy(z, lab, ignore_index=IGNORE)
    assert torch.allclose(seg_loss(z, lab), ref)
