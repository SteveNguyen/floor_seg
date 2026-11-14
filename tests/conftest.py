"""A tiny fake render folder in the thor_render.py output format, shared by the tests."""
import json

import cv2
import numpy as np
import pytest

W, H = 640, 360


def make_renders(root, houses=(3, 10), views=2):
    """Views h<house>_<k>: random image, floor = lower half (0/255), depth in mm; stats.jsonl like thor_render."""
    rng = np.random.default_rng(0)
    with open(root / "stats.jsonl", "w") as f:
        for house in houses:
            for k in range(views):
                name = f"h{house:05d}_{k}"
                cv2.imwrite(str(root / f"{name}.jpg"), rng.integers(0, 256, (H, W, 3), np.uint8))
                floor = np.zeros((H, W), np.uint8)
                floor[H // 2:] = 255
                cv2.imwrite(str(root / f"{name}_floor.png"), floor)
                depth = np.full((H, W), 3000, np.uint16)
                depth[H // 2:] = np.linspace(3000, 400, H - H // 2, dtype=np.uint16)[:, None]
                cv2.imwrite(str(root / f"{name}_depth.png"), depth)
                f.write(json.dumps(dict(name=name, house=house, h=0.3, pitch=10.0 + k, roll=-1.0, yaw=45.0,
                                        floor_frac=0.5, agree=0.95, n_clutter=7)) + "\n")
    return root


@pytest.fixture
def renders(tmp_path):
    d = tmp_path / "renders"
    d.mkdir()
    return make_renders(d)
