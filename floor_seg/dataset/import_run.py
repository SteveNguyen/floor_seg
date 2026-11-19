"""Import a Navette run folder (Reachy Mini on the Poussah base) into a staged dataset session.

    python -m floor_seg.dataset.import_run RUN_DIR STAGING --session home03 [--min-move 0.10 --min-turn 10 --max-gap 2]

Reads (never modifies) the run folder written by navette.runtime.loop and the reachy_poussah adapter:
    frames/NNNNNN.jpg   every camera frame (640 x 360)
    steps.csv           one row per navigator step: t (s since the first step), frame, x, y, heading_deg, cam_h,
                        pitch_down_deg, v, w, state, ctrl, ...
    meta.json           "robot": image_resolution, K, D of the saved frames (the adapter's calibration, scaled)
Only frames with a steps.csv row are imported: the others have no camera height / pitch / pose (IMU not ready,
no odometry). Values are taken as Navette logged them, nothing is recomputed.

Subsampling: a frame is kept if, since the last kept frame, the robot moved >= --min-move m, or turned >= --min-turn
deg, or --max-gap s passed (the first frame is always kept).

Privacy: no wall-clock time is copied (frames.csv arrival times, meta started_wall, the run folder name), only times
relative to the first step; the session id is chosen by the user ("home03": no address, no names). Images are
copied byte for byte here; floor_seg.dataset.privacy blurs faces and re-encodes them without metadata.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil

import cv2
import numpy as np

from floor_seg.dataset import save_frames, save_session

SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")  # e.g. home03: short, no spaces, nothing personal
MOVING_V = 0.05   # m/s: navette.runtime.loop.MOVING_V, the speed from which the robot counts as moving
TURNING_W = 0.1   # rad/s: below this a still robot counts as still (the arc controller's small corrections)


def motion_of(v, w):
    """'moving' (translating, possibly on an arc), 'turning' (in place) or 'still', from the navigator output."""
    if abs(v) >= MOVING_V:
        return "moving"
    return "turning" if abs(w) >= TURNING_W else "still"


def subsample(rows, min_move, min_turn_deg, max_gap):
    """Indices of the rows to keep (rows in time order, dicts with t, x, y, heading_deg as floats)."""
    keep, last = [], None
    for i, r in enumerate(rows):
        if last is not None:
            moved = math.hypot(r["x"] - last["x"], r["y"] - last["y"])
            # heading difference wrapped to [-180, 180)
            turned = abs((r["heading_deg"] - last["heading_deg"] + 180.0) % 360.0 - 180.0)
            if moved < min_move and turned < min_turn_deg and r["t"] - last["t"] < max_gap:
                continue
        keep.append(i)
        last = r
    return keep


def read_steps(run_dir):
    """steps.csv rows as dicts with the needed columns parsed, in time order."""
    rows = []
    with open(os.path.join(run_dir, "steps.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({"t": float(r["t"]), "frame": int(r["frame"]), "x": float(r["x"]), "y": float(r["y"]),
                         "heading_deg": float(r["heading_deg"]), "cam_h": float(r["cam_h"]),
                         "pitch_down_deg": float(r["pitch_down_deg"]), "v": float(r["v"]), "w": float(r["w"]),
                         "state": r.get("state", ""), "ctrl": r.get("ctrl", "")})
    rows.sort(key=lambda r: r["t"])
    return rows


def intrinsics(meta, calib_path, image_wh):
    """{"image_resolution", "K", "D"} at the frames' resolution: meta.json "robot" (what the adapter used, already
    scaled to the saved frames), else the calibration file given (scaled if its resolution differs)."""
    robot = meta.get("robot", {})
    if calib_path is None and all(k in robot for k in ("image_resolution", "K", "D")):
        calib = {k: robot[k] for k in ("image_resolution", "K", "D")}
    elif calib_path is not None:
        with open(calib_path) as f:
            calib = json.load(f)
    else:
        raise SystemExit("no intrinsics in meta.json 'robot': pass --calib (json with image_resolution, K, D)")
    s = image_wh[0] / calib["image_resolution"][0]
    if abs(image_wh[1] / calib["image_resolution"][1] - s) > 1e-6:
        raise SystemExit(f"calibration {calib['image_resolution']} and frames {image_wh} differ in aspect ratio")
    K = np.array(calib["K"], float)
    K[:2] *= s  # fx, cx, fy, cy scale with the image; distortion does not
    return {"image_resolution": list(image_wh), "K": K.tolist(), "D": list(calib["D"])}


def import_run(run_dir, staging, session_id, min_move=0.10, min_turn=10.0, max_gap=2.0, calib_path=None):
    if not SESSION_RE.match(session_id):
        raise SystemExit(f"session id {session_id!r}: use lowercase letters, digits, - and _ (e.g. home03)")
    out = os.path.join(staging, session_id)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: choose another session id or delete it")
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    rows = [r for r in read_steps(run_dir)
            if os.path.exists(os.path.join(run_dir, "frames", f"{r['frame']:06d}.jpg"))]
    if not rows:
        raise SystemExit(f"{run_dir}: no steps with a saved frame")
    keep = subsample(rows, min_move, min_turn, max_gap)

    first = cv2.imread(os.path.join(run_dir, "frames", f"{rows[keep[0]]['frame']:06d}.jpg"))
    calib = intrinsics(meta, calib_path, (first.shape[1], first.shape[0]))

    os.makedirs(os.path.join(out, "frames"))
    t_start = rows[0]["t"]
    frames = []
    for k, i in enumerate(keep):
        r = rows[i]
        name = f"{session_id}_{k:05d}"
        shutil.copyfile(os.path.join(run_dir, "frames", f"{r['frame']:06d}.jpg"),
                        os.path.join(out, "frames", name + ".jpg"))
        frames.append({"name": name, "t_s": round(r["t"] - t_start, 3), "cam_h_m": r["cam_h"],
                       "pitch_down_deg": r["pitch_down_deg"], "roll_deg": None,
                       "x_m": r["x"], "y_m": r["y"], "heading_deg": r["heading_deg"], "v": r["v"], "w": r["w"],
                       "motion": motion_of(r["v"], r["w"]), "nav_state": r["state"], "ctrl": r["ctrl"]})
    save_frames(out, frames)
    with open(os.path.join(out, "calib.json"), "w") as f:
        json.dump(calib, f, indent=2)
    n_in = len(os.listdir(os.path.join(run_dir, "frames")))
    save_session(out, {"session_id": session_id, "frames_in": n_in, "frames_with_step": len(rows),
                       "frames_kept": len(frames),
                       "subsample": {"min_move_m": min_move, "min_turn_deg": min_turn, "max_gap_s": max_gap},
                       "privacy": None, "review": None, "consent": None})
    return out, n_in, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Navette run folder (read only)")
    ap.add_argument("staging", help="staging folder (outside the repo); the session goes into STAGING/<session>")
    ap.add_argument("--session", required=True, help="session id, e.g. home03 (no address, no names)")
    ap.add_argument("--min-move", type=float, default=0.10, help="m since the last kept frame")
    ap.add_argument("--min-turn", type=float, default=10.0, help="deg since the last kept frame")
    ap.add_argument("--max-gap", type=float, default=2.0, help="s since the last kept frame")
    ap.add_argument("--calib", help="intrinsics json, only if meta.json has none")
    args = ap.parse_args()
    out, n_in, n_kept = import_run(args.run_dir, args.staging, args.session, args.min_move, args.min_turn,
                                   args.max_gap, args.calib)
    print(f"{out}: {n_in} frames in, {n_kept} kept")


if __name__ == "__main__":
    main()
