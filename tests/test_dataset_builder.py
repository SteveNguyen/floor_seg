"""scripts/build_hf_dataset.py on a fake render folder: splits, columns, byte-exact images, mask / depth types."""
import importlib.util
import os

import numpy as np
from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_hf_dataset", os.path.join(REPO, "scripts", "build_hf_dataset.py"))
bhd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhd)


def test_build_and_load(renders, tmp_path):
    out = tmp_path / "hf_dataset"
    counts = bhd.build(str(renders), str(out), shard_size=1)
    assert counts == {"train": 2, "validation": 2}  # house 3 -> train, house 10 -> validation
    assert len(list((out / "data").glob("train-*.parquet"))) == 2  # shard_size 1
    assert (out / "README.md").exists() and (out / "reachy_mini_camera.json").exists()

    ds = load_dataset(str(out))  # uses the README's configs, like the Hub does
    assert ds["train"].num_rows == 2 and ds["validation"].num_rows == 2
    assert set(ds["validation"]["house"]) == {10}
    ex = ds["train"][0]
    assert ex["name"] == "h00003_0" and ex["view"] == 0 and abs(ex["camera_height_m"] - 0.3) < 1e-6
    assert ex["image"].size == (640, 360) and ex["image"].mode == "RGB"
    floor = np.array(ex["floor"])
    assert set(np.unique(floor)) == {0, 255} and floor[-1].min() == 255 and floor[0].max() == 0
    depth = np.array(ex["depth"])
    assert depth.dtype in (np.uint16, np.int32) and depth.max() == 3000 and depth[-1, 0] == 400


def test_images_stored_byte_exact(renders, tmp_path):
    out = tmp_path / "hf_dataset"
    bhd.build(str(renders), str(out))
    ds = load_dataset(str(out), split="train").cast_column("image", bhd.Image(decode=False))
    assert ds[0]["image"]["bytes"] == open(renders / "h00003_0.jpg", "rb").read()


def test_limit(renders, tmp_path):
    counts = bhd.build(str(renders), str(tmp_path / "o"), limit=3)
    assert counts == {"train": 2, "validation": 1}
