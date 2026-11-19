"""Export reviewed, consented sessions into a local dataset folder (Hugging Face ImageFolder layout). No upload.

    python -m floor_seg.dataset.export STAGING/home03 STAGING/home04 --val home04 --out hf/real_floor

Refuses (and lists why) any session without: the privacy pass, a completed owner review, written consent given.

Output:
    README.md                     dataset card (floor_seg/dataset/dataset_card.md + a table of the sessions)
    calibration/<session>.json    camera intrinsics of that session's frames (image_resolution, K, D)
    <split>/images/<name>.jpg     frames (byte for byte from the session: blurred, without metadata)
    <split>/metadata.jsonl        one row per frame: file_name (relative to the split folder), session, calibration,
                                  and the frame fields of floor_seg.dataset.FRAME_FIELDS
Splits are by session (a home is in one split only): --val sessions -> validation, the others -> train.
"""
import argparse
import json
import os
import shutil

from floor_seg.dataset import FRAME_FIELDS, frame_path, load_frames, load_session

CARD = os.path.join(os.path.dirname(__file__), "dataset_card.md")


def problems(session_dir):
    """Why this session may not be exported (empty: it may)."""
    s = load_session(session_dir)
    out = []
    if not (s.get("privacy") or {}).get("done"):
        out.append("no privacy pass")
    if not (s.get("review") or {}).get("done"):
        out.append("no completed owner review")
    if not (s.get("consent") or {}).get("given"):
        out.append("no written consent recorded" if s.get("consent") is None else "consent refused")
    for r in load_frames(session_dir):
        if not os.path.exists(frame_path(session_dir, r["name"])):
            out.append(f"missing image {r['name']}.jpg")
            break
    return out


def export(session_dirs, out, val=()):
    refused = {d: p for d in session_dirs if (p := problems(d))}
    if refused:
        raise SystemExit("refused, nothing written:\n" + "\n".join(f"  {d}: {', '.join(p)}"
                                                                   for d, p in refused.items()))
    ids = [load_session(d)["session_id"] for d in session_dirs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate session ids: {ids}")
    if set(val) - set(ids):
        raise SystemExit(f"--val sessions not given: {sorted(set(val) - set(ids))}")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")

    os.makedirs(os.path.join(out, "calibration"), exist_ok=True)
    counts = []
    rows = {"train": [], "validation": []}
    for d, sid in zip(session_dirs, ids):
        split = "validation" if sid in val else "train"
        calib = f"calibration/{sid}.json"
        shutil.copyfile(os.path.join(d, "calib.json"), os.path.join(out, calib))
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        frames = load_frames(d)
        for r in frames:
            shutil.copyfile(frame_path(d, r["name"]), os.path.join(out, split, "images", r["name"] + ".jpg"))
            row = {"file_name": f"images/{r['name']}.jpg", "session": sid, "calibration": calib}
            row.update({k: r[k] for k in FRAME_FIELDS if k != "name"})  # not "faces": no hint of who was there
            rows[split].append(row)
        counts.append((sid, split, len(frames)))
    for split, rs in rows.items():
        if rs:
            with open(os.path.join(out, split, "metadata.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")

    table = "| Session | Split | Frames |\n|---|---|---|\n" + "".join(f"| {s} | {p} | {n} |\n" for s, p, n in counts)
    with open(CARD) as f:
        card = f.read().replace("<!-- SESSIONS -->", table.rstrip("\n"))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(card)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="staged session folders")
    ap.add_argument("--out", required=True, help="dataset folder to create (must not exist or be empty)")
    ap.add_argument("--val", nargs="*", default=[], help="session ids for the validation split")
    args = ap.parse_args()
    counts = export(args.sessions, args.out, args.val)
    print(f"wrote {args.out}: " + ", ".join(f"{s} ({p}, {n} frames)" for s, p, n in counts))


if __name__ == "__main__":
    main()
