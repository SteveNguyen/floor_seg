"""Tools for the public real-home floor dataset: Navette run -> staged session -> privacy -> owner review ->
consent -> export (a local Hugging Face folder; nothing is uploaded).

    python -m floor_seg.dataset.import_run RUN_DIR STAGING --session home03
    python -m floor_seg.dataset.privacy STAGING/home03
    python -m floor_seg.dataset.review STAGING/home03          # the home owner deletes frames in a browser page
    python -m floor_seg.dataset.consent STAGING/home03 --given y --date 2026-11-20 --role owner --form-ref "..."
    python -m floor_seg.dataset.export STAGING/home03 STAGING/home04 --val home04 --out hf/real_floor

A session folder (STAGING/<session id>, keep STAGING outside the repo):
    session.json    session id, counts, import parameters, and the status of each step: "privacy", "review",
                    "consent" (null until done)
    calib.json      camera intrinsics of the frames {"image_resolution": [W, H], "K": 3x3, "D": OpenCV distortion}
    frames/         <session>_<k:05d>.jpg, kept frames in time order
    frames.jsonl    one record per frame (FRAME_FIELDS), in time order
    review_log.jsonl  one line per applied review: number of frames deleted / kept (no content)
"""
import json
import os

# Per-frame record of frames.jsonl. Times are seconds since the session start (no wall clock); the pose is the
# odometry Navette had (x, y m; heading deg, the camera's heading in the odometry frame); cam_h_m and
# pitch_down_deg are what Navette computed from the IMU 'up' for that frame (pitch = optical axis below the
# horizon); roll_deg is None: Navette does not log it yet. v, w: the navigator's output at that step (m/s, rad/s);
# motion: "still" / "moving" / "turning" from them; nav_state, ctrl: behaviour and controller states.
FRAME_FIELDS = ["name", "t_s", "cam_h_m", "pitch_down_deg", "roll_deg", "x_m", "y_m", "heading_deg", "v", "w",
                "motion", "nav_state", "ctrl"]


def load_session(session_dir):
    with open(os.path.join(session_dir, "session.json")) as f:
        return json.load(f)


def save_session(session_dir, session):
    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump(session, f, indent=2)


def load_frames(session_dir):
    with open(os.path.join(session_dir, "frames.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_frames(session_dir, frames):
    with open(os.path.join(session_dir, "frames.jsonl"), "w") as f:
        for r in frames:
            f.write(json.dumps(r) + "\n")


def frame_path(session_dir, name):
    return os.path.join(session_dir, "frames", name + ".jpg")
