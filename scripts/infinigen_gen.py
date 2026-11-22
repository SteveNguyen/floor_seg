"""Generate a floor_seg training set from Infinigen Indoors rooms (see docs/INFINIGEN.md).

    ~/infinigen/.venv/bin/python scripts/infinigen_gen.py run --infinigen ~/infinigen \
        --out data/synth/infinigen_v1 --rooms 200 --views 5 --shard 0 2   # one process per GPU (--shard I N)
    ~/infinigen/.venv/bin/python scripts/infinigen_gen.py merge data/synth/infinigen_v1   # stats.jsonl

Run it with Infinigen's Python (it has numpy and OpenCV; floor_seg's own env is not needed). It runs two
programs of the Infinigen clone:
1. `<infinigen>/.venv/bin/infinigen <room generator> --seed S --output WORK --save_blend` (the room, ~0.5-1 min)
2. `<infinigen>/.venv/bin/python -m floor_seg.infinigen_render WORK/scene.blend OUT ...` (views, labels)
Room types are drawn per room index with the ordinary-home weights of floor_seg/infinigen_labels.py (15 %
furnished houses: several rooms with doors). With --tex, floors are re-materialled (tiles, wood) where Infinigen's
pick is unusual; without, such a room is rejected (exit code 3) and regenerated with the next seed (up to
--attempts). Resumable: a room
with OUT/rooms/r<room>.done is skipped. Rooms with index % 10 == 0 are the validation split in training.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from floor_seg import (
    infinigen_labels as L,
)


def merge(out):
    """OUT/stats.jsonl = all finished rooms' lines, in room order (atomic replace: safe with running shards)."""
    lines = []
    for p in sorted(glob.glob(os.path.join(out, "rooms", "r*.jsonl"))):
        if os.path.exists(p[:-len(".jsonl")] + ".done"):
            lines += [s for s in open(p).read().splitlines() if s]
    tmp = os.path.join(out, f".stats.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write("".join(s + "\n" for s in lines))
    os.replace(tmp, os.path.join(out, "stats.jsonl"))
    return len(lines)


def run(args):
    infinigen = os.path.abspath(os.path.expanduser(args.infinigen))
    cli, py = os.path.join(infinigen, ".venv/bin/infinigen"), os.path.join(infinigen, ".venv/bin/python")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "rooms"), exist_ok=True)
    work_root = os.path.abspath(args.work or os.path.join(out, "work"))
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    weights = None
    if args.room_weights:
        weights = {k: float(v) for k, v in (kv.split("=") for kv in args.room_weights)}
    i, n = args.shard
    t_start = time.time()
    for room in range(args.first, args.first + args.rooms):
        if room % n != i:
            continue
        done = os.path.join(out, "rooms", f"r{room:05d}.done")
        if os.path.exists(done):
            continue
        rtype = L.room_type(L.new_rng(args.seed, room), weights)
        gen = L.ROOM_TYPES[rtype][0]
        status = "failed"
        for attempt in range(args.attempts):
            seed = (args.seed * 1_000_003 + room * 16 + attempt) % (2 ** 31)
            work = os.path.join(work_root, f"r{room:05d}_{attempt}")
            shutil.rmtree(work, ignore_errors=True)
            t0 = time.time()
            gcmd = [cli, gen, "--seed", str(seed), "--output", work, "--save_blend"]
            if args.cpu_threads:
                gcmd += ["--cpu_threads", str(args.cpu_threads)]
            g = subprocess.run(gcmd, cwd=infinigen, env=env, capture_output=True, text=True, timeout=args.timeout)
            t_gen = time.time() - t0
            blend = os.path.join(work, "scene.blend")
            if g.returncode != 0 or not os.path.exists(blend):
                print(f"room {room} {rtype} seed {seed}: generation failed ({g.returncode}, {t_gen:.0f} s)\n"
                      f"{g.stderr[-2000:]}", flush=True)
                status = "generation failed"
                continue
            cmd = [py, "-m", "floor_seg.infinigen_render", blend, out, "--room", str(room), "--room-type", rtype,
                   "--seed", str(seed), "--views", str(args.views), "--samples", str(args.samples),
                   "--cables", str(args.cables), "--thresholds", str(args.thresholds)]
            if args.tex:
                cmd += ["--tex", os.path.abspath(os.path.expanduser(args.tex))]
            if args.overlay:
                cmd.append("--overlay")
            if args.cpu_threads:
                cmd += ["--cpu-threads", str(args.cpu_threads)]
            if args.no_under:
                cmd.append("--no-under")
            cmd += ["--under-step", str(args.under_step)]
            t1 = time.time()
            r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=args.timeout)
            t_render = time.time() - t1
            log = [s for s in r.stdout.splitlines() if s.startswith(("room", "  r", "reject", "no floor", "floors", "WARNING"))]
            print("\n".join(log) or r.stdout[-1500:], flush=True)
            if not args.keep_blend:
                shutil.rmtree(work, ignore_errors=True)
            if r.returncode == 3:
                status = "rejected"
                continue
            if r.returncode != 0:
                print(f"room {room}: render exit {r.returncode}\n{r.stderr[-2000:]}", flush=True)
                status = f"render exit {r.returncode}"
                break
            status = "ok"
            print(f"room {room} {rtype} seed {seed}: generate {t_gen:.0f} s, render {t_render:.0f} s", flush=True)
            break
        with open(done, "w") as f:
            json.dump(dict(room=room, room_type=rtype, status=status), f)
    print(f"shard {i}/{n} finished in {time.time() - t_start:.0f} s; stats.jsonl: {merge(out)} views", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--infinigen", required=True, help="Infinigen clone (with its .venv from `uv sync`)")
    r.add_argument("--out", required=True)
    r.add_argument("--work", help="scratch folder for the generated .blend files (default OUT/work)")
    r.add_argument("--rooms", type=int, default=200)
    r.add_argument("--first", type=int, default=0, help="first room index (to extend a set)")
    r.add_argument("--views", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"), default=(0, 1))
    r.add_argument("--samples", type=int, default=64)
    r.add_argument("--cables", type=float, default=0.3, help="chance per room of 1-3 cables on the floor")
    r.add_argument("--thresholds", type=float, default=0.5, help="chance per door of a flush threshold strip")
    r.add_argument("--tex", help="ambientCG texture root (data/textures/ambientcg): floors re-materialled (tiles, "
                                   "wood) instead of rooms rejected (infinigen_labels.floor_plan)")
    r.add_argument("--room-weights", nargs="+", metavar="TYPE=W",
                   help="override the room-type weights, e.g. livingroom=1 bedroom=1 house=0.2")
    r.add_argument("--attempts", type=int, default=3)
    r.add_argument("--timeout", type=int, default=3600, help="s per generation / render call")
    r.add_argument("--cpu-threads", type=int, default=0,
                   help="CPU threads per Blender process (generation and render; 0 = all): set it when several "
                        "shards share one machine")
    r.add_argument("--no-under", action="store_true", help="no _under.png")
    r.add_argument("--under-step", type=int, default=1, help="measure every n-th 2 cm cell of the under map")
    r.add_argument("--overlay", action="store_true", help="also write <name>_overlay.jpg (QA)")
    r.add_argument("--keep-blend", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "merge":
        print(merge(os.path.abspath(args.out)), "views")
    else:
        run(args)


if __name__ == "__main__":
    main()
