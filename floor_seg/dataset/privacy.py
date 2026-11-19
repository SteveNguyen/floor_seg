"""Privacy pass on a staged session: blur faces, re-encode every frame without metadata (EXIF, comments).

    python -m floor_seg.dataset.privacy STAGING/home03 [--model-dir DIR] [--score 0.6]

Face detector: YuNet, face_detection_yunet_2023mar.onnx from OpenCV Zoo
(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), run with cv2.FaceDetectorYN.
Licence verified 2026-11-19: "All files in this directory are licensed under MIT License" (directory README),
LICENSE = MIT License, Copyright (c) 2020 Shiqi Yu. Downloaded once (227 kB, sha256 pinned below) into
--model-dir, default $XDG_CACHE_HOME/floor_seg/yunet (~/.cache/floor_seg/yunet), outside the repo.

Each detected face box is enlarged by 30 % (each side, about its centre) and blurred with a strong Gaussian
(sigma = half the box size: nothing recognisable is left). The score threshold is lower than YuNet's usual 0.9:
a missed face costs more than a blurred patch of wall. Each frame record gets "faces" (number blurred) so the
owner's review (floor_seg.dataset.review) shows those frames first. Faces the detector misses (profile, far, at
the lens edge) are for the review to catch. People's bodies are not detected (v1).
"""
import argparse
import hashlib
import os
import urllib.request

import cv2
import numpy as np

from floor_seg.dataset import frame_path, load_frames, load_session, save_frames, save_session

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_NAME
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_LICENCE = "MIT (OpenCV Zoo, Copyright (c) 2020 Shiqi Yu)"
ENLARGE = 1.3      # face box scale before blurring
JPEG_QUALITY = 95  # re-encoding quality (the source frames are already JPEG)


def default_model_dir():
    return os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "floor_seg", "yunet")


def ensure_model(model_dir):
    """Path of the YuNet ONNX file, downloaded on first use; refuses a file whose hash differs."""
    path = os.path.join(model_dir, MODEL_NAME)
    if not os.path.exists(path):
        os.makedirs(model_dir, exist_ok=True)
        print(f"downloading {MODEL_URL} -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path + ".part")
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"{path}: sha256 {digest} is not the pinned {MODEL_SHA256}; delete it and retry")
    return path


def yunet_detector(model_path, score=0.6):
    """detect(bgr) -> list of (x, y, w, h) face boxes in pixels."""
    det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=score)

    def detect(bgr):
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = det.detect(bgr)
        return [] if faces is None else [tuple(float(v) for v in f[:4]) for f in faces]

    return detect


def blur_boxes(bgr, boxes):
    """Blur each box enlarged by ENLARGE about its centre (clipped to the image), in place."""
    H, W = bgr.shape[:2]
    for x, y, w, h in boxes:
        cx, cy, hw, hh = x + w / 2, y + h / 2, ENLARGE * w / 2, ENLARGE * h / 2
        x0, y0 = max(0, int(np.floor(cx - hw))), max(0, int(np.floor(cy - hh)))
        x1, y1 = min(W, int(np.ceil(cx + hw))), min(H, int(np.ceil(cy + hh)))
        if x1 <= x0 or y1 <= y0:
            continue
        # blur computed from the box only (reflected borders): nothing of the face leaks in or survives
        sigma = 0.5 * max(x1 - x0, y1 - y0)
        bgr[y0:y1, x0:x1] = cv2.GaussianBlur(bgr[y0:y1, x0:x1], (0, 0), sigma)
    return bgr


def run(session_dir, detect, detector_info=None):
    """Blur faces and re-encode every frame of the session. Returns (frames, frames with faces, faces)."""
    session = load_session(session_dir)
    if session.get("privacy"):
        raise SystemExit(f"{session_dir}: privacy pass already done (re-encoding again only loses quality)")
    frames = load_frames(session_dir)
    n_frames_faces = n_faces = 0
    for r in frames:
        path = frame_path(session_dir, r["name"])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        boxes = detect(bgr)
        blur_boxes(bgr, boxes)
        # cv2.imwrite writes a bare JFIF file: no EXIF, no comment, nothing from the source file but pixels
        if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise SystemExit(f"could not write {path}")
        r["faces"] = len(boxes)
        n_frames_faces += bool(boxes)
        n_faces += len(boxes)
    save_frames(session_dir, frames)
    session["privacy"] = {"done": True, "detector": detector_info, "frames_with_faces": n_frames_faces,
                          "faces": n_faces, "blur_enlarge": ENLARGE}
    session["review"] = None  # the images changed: any earlier review no longer applies
    save_session(session_dir, session)
    return len(frames), n_frames_faces, n_faces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="staged session folder")
    ap.add_argument("--model-dir", default=default_model_dir(), help="YuNet cache folder (outside the repo)")
    ap.add_argument("--score", type=float, default=0.6, help="YuNet face score threshold")
    args = ap.parse_args()
    detect = yunet_detector(ensure_model(args.model_dir), args.score)
    info = {"model": MODEL_NAME, "sha256": MODEL_SHA256, "licence": MODEL_LICENCE, "score": args.score}
    n, nf, faces = run(args.session, detect, info)
    print(f"{args.session}: {n} frames re-encoded, {nf} with faces ({faces} faces blurred)")


if __name__ == "__main__":
    main()
