"""Convert a render folder (floor_seg/thor_render.py output) into a local Hugging Face dataset folder (nothing uploaded).

    python scripts/build_hf_dataset.py data/synth/thor_v1 --out hf/dataset [--limit 50] [--textures data/textures/ambientcg]

Output:
    data/{train,validation}-NNNNN-of-MMMMM.parquet   one row per view, split by house (house % 10 == 0 -> validation,
                                                      the same split as floor_seg/train.py)
    README.md                                        dataset card (= DATASET_CARD.md of this repo)
    reachy_mini_camera.json                          lens calibration the views were rendered with
    textures/                                        only with --textures: copy of the CC0 texture folder

Columns: image (JPEG, RGB 640x360), floor (PNG, 255 floor / 0 not floor), depth (PNG uint16, optical z in mm),
name, house, view, camera_height_m, pitch_deg, roll_deg, yaw_deg, floor_frac, agree, n_clutter.
Files are stored byte for byte (no re-encoding). --limit takes the first N views of stats.jsonl (dry runs).
"""
import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image, Value

from floor_seg import DEFAULT_CALIB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = Features({
    "image": Image(), "floor": Image(), "depth": Image(),
    "name": Value("string"), "house": Value("int32"), "view": Value("int32"),
    "camera_height_m": Value("float32"), "pitch_deg": Value("float32"), "roll_deg": Value("float32"),
    "yaw_deg": Value("float32"), "floor_frac": Value("float32"), "agree": Value("float32"),
    "n_clutter": Value("int32"),
})


def split_of(house):
    return "validation" if house % 10 == 0 else "train"


def row(root, s):
    """One dataset row from a stats.jsonl record (field names of floor_seg/thor_render.py)."""
    def file(suffix):
        path = os.path.join(root, s["name"] + suffix)
        return {"bytes": open(path, "rb").read(), "path": os.path.basename(path)}

    return {"image": file(".jpg"), "floor": file("_floor.png"), "depth": file("_depth.png"),
            "name": s["name"], "house": s["house"], "view": int(s["name"].rsplit("_", 1)[1]),
            "camera_height_m": s["h"], "pitch_deg": s["pitch"], "roll_deg": s["roll"], "yaw_deg": s["yaw"],
            "floor_frac": s["floor_frac"], "agree": s["agree"], "n_clutter": s["n_clutter"]}


def build(root, out, limit=None, shard_size=1000, textures=None):
    stats = [json.loads(line) for line in open(os.path.join(root, "stats.jsonl"))][:limit]
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    schema = FEATURES.arrow_schema  # carries the HF feature types (Image columns) in the Parquet metadata
    counts = {}
    for split in ("train", "validation"):
        recs = [s for s in stats if split_of(s["house"]) == split]
        n_shards = max(1, -(-len(recs) // shard_size))
        for k in range(n_shards):
            rows = [FEATURES.encode_example(row(root, s)) for s in recs[k * shard_size:(k + 1) * shard_size]]
            pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                           os.path.join(out, "data", f"{split}-{k:05d}-of-{n_shards:05d}.parquet"))
        counts[split] = len(recs)
    shutil.copy(os.path.join(REPO, "DATASET_CARD.md"), os.path.join(out, "README.md"))
    shutil.copy(DEFAULT_CALIB, os.path.join(out, "reachy_mini_camera.json"))
    if textures:
        shutil.copytree(textures, os.path.join(out, "textures"), dirs_exist_ok=True)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders", help="render folder with stats.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N views (dry run)")
    ap.add_argument("--shard-size", type=int, default=1000, help="views per Parquet file (~150 MB)")
    ap.add_argument("--textures", help="also copy this texture folder (CC0) into the dataset")
    args = ap.parse_args()
    counts = build(args.renders, args.out, args.limit, args.shard_size, args.textures)
    print(f"wrote {args.out}: " + ", ".join(f"{k} {v} views" for k, v in counts.items()))


if __name__ == "__main__":
    main()
