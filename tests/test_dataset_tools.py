"""floor_seg.dataset: import, privacy, review, consent, export on a tiny synthetic Navette run folder."""
import json
import os
import threading
import urllib.request

import cv2
import numpy as np
import pytest

from floor_seg.dataset import consent, export, import_run, load_frames, load_session, privacy, review

W, H = 160, 90  # synthetic frames (calibration in meta is at 640 x 360: K scales by 1/4)
STARTED_WALL = "2026-01-02T03:04:05.678901"
# (frame, t, x, y, heading_deg, v, w): kept = 0 (first), 2 (moved 0.11 m), 4 (turned 12 deg), 5 (2.1 s later)
STEPS = [(0, 5.0, 0.00, 0.0, 0.0, 0.0, 0.0),
         (1, 5.1, 0.05, 0.0, 0.0, 0.15, 0.0),
         (2, 5.2, 0.11, 0.0, 0.0, 0.15, 0.0),
         (3, 5.3, 0.11, 0.0, 5.0, 0.0, 0.5),
         (4, 5.4, 0.11, 0.0, 12.0, 0.0, 0.5),
         (5, 7.5, 0.11, 0.0, 12.0, 0.0, 0.0),
         (6, 7.6, 0.12, 0.0, 12.0, 0.0, 0.0),
         (8, 7.7, 0.50, 0.0, 12.0, 0.0, 0.0)]  # frame 8 has no image: skipped
K640 = [[300.0, 0.0, 320.0], [0.0, 300.0, 180.0], [0.0, 0.0, 1.0]]


def with_exif(jpg_bytes):
    """The JPEG with a fake EXIF APP1 segment right after SOI (as a phone or camera would write)."""
    payload = b"Exif\x00\x00" + b"MM\x00*" + b"FAKE-CAMERA-SERIAL" * 4
    return jpg_bytes[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + jpg_bytes[2:]


def make_run(root):
    run = root / "navette-20260102-030405"
    (run / "frames").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(8):  # frames 0-7; frame 7 has no step row
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 256, (H, W, 3), np.uint8))
        (run / "frames" / f"{i:06d}.jpg").write_bytes(with_exif(buf.tobytes()))
    (run / "frames.csv").write_text("idx,t\n" + "".join(f"{i},{1889.5 + 0.1 * i:.6f}\n" for i in range(8)))
    lines = ["t,frame,x,y,heading_deg,cam_h,pitch_down_deg,v,w,v_req,w_req,state,seg_ms,scan_ms,nav_ms,ctrl,run"]
    for f, t, x, y, h, v, w in STEPS:
        lines.append(f"{t:.3f},{f},{x:.3f},{y:.3f},{h:.1f},0.381,15.2,{v:.3f},{w:.3f},{v:.3f},{w:.3f},"
                     f"TELEOP,10.0,1.0,1.0,DRIVING,1")
    (run / "steps.csv").write_text("\n".join(lines) + "\n")
    meta = {"clock": "time.monotonic", "started_wall": STARTED_WALL,
            "config": {"adapter": {"calib": "/home/someone/calib.json"}},
            "robot": {"image_resolution": [640, 360], "K": K640, "D": [0.1, -0.01, 0.0, 0.0, 0.0]}}
    (run / "meta.json").write_text(json.dumps(meta))
    return run


@pytest.fixture
def session(tmp_path):
    run = make_run(tmp_path)
    out, n_in, n_kept = import_run.import_run(str(run), str(tmp_path / "staging"), "home01")
    return out


def no_faces(bgr):
    return []


def finish(session_dir, tmp_path, given=True):
    privacy.run(session_dir, no_faces)
    review.apply_review(session_dir, [])
    consent.record(session_dir, given, "2026-01-03", "owner", "binder A", str(tmp_path / "register.jsonl"))


# ---------------------------------------------------------------- import


def test_subsample_rules():
    rows = [dict(t=t, x=x, y=0.0, heading_deg=h) for _, t, x, _, h, _, _ in STEPS]
    assert import_run.subsample(rows, 0.10, 10.0, 2.0) == [0, 2, 4, 5, 7]
    # heading wraps: 179 -> -179 is a 2 deg turn
    rows = [dict(t=0.0, x=0.0, y=0.0, heading_deg=179.0), dict(t=0.1, x=0.0, y=0.0, heading_deg=-179.0)]
    assert import_run.subsample(rows, 0.10, 10.0, 2.0) == [0]


def test_import_metadata(session):
    s = load_session(session)
    assert s["session_id"] == "home01" and s["frames_in"] == 8 and s["frames_kept"] == 4
    frames = load_frames(session)
    assert [f["name"] for f in frames] == [f"home01_{k:05d}" for k in range(4)]
    assert [f["t_s"] for f in frames] == [0.0, 0.2, 0.4, 2.5]  # relative to the first step
    f = frames[2]  # source frame 4: turning in place
    assert (f["x_m"], f["heading_deg"], f["cam_h_m"], f["pitch_down_deg"]) == (0.11, 12.0, 0.381, 15.2)
    assert f["roll_deg"] is None
    assert [f["motion"] for f in frames] == ["still", "moving", "turning", "still"]
    assert frames[1]["nav_state"] == "TELEOP" and frames[1]["ctrl"] == "DRIVING"
    calib = json.load(open(os.path.join(session, "calib.json")))
    assert calib["image_resolution"] == [W, H]
    np.testing.assert_allclose(calib["K"], [[75, 0, 80], [0, 75, 45], [0, 0, 1]])  # 640 x 360 calib / 4
    assert len(os.listdir(os.path.join(session, "frames"))) == 4


def test_import_strips_wall_clock(session):
    text = "".join(open(os.path.join(dp, f), errors="ignore").read()
                   for dp, _, fs in os.walk(session) for f in fs if not f.endswith(".jpg"))
    for leak in (STARTED_WALL, "2026-01-02", "1889.", "navette-2026", "/home/someone"):
        assert leak not in text


def test_import_calib_file_and_bad_session_id(tmp_path):
    run = make_run(tmp_path)
    meta = json.loads((run / "meta.json").read_text())
    del meta["robot"]
    (run / "meta.json").write_text(json.dumps(meta))
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"image_resolution": [1280, 720], "K": K640, "D": [0.0] * 5}))
    out, _, _ = import_run.import_run(str(run), str(tmp_path / "st"), "home02", calib_path=str(calib))
    np.testing.assert_allclose(json.load(open(os.path.join(out, "calib.json")))["K"][0], [37.5, 0, 40])
    with pytest.raises(SystemExit):
        import_run.import_run(str(run), str(tmp_path / "st"), "John's flat", calib_path=str(calib))


# ---------------------------------------------------------------- privacy


def test_privacy_blurs_detected_faces_and_strips_exif(session):
    first = load_frames(session)[0]["name"]
    path = os.path.join(session, "frames", first + ".jpg")
    assert b"Exif" in open(path, "rb").read()  # import copies the source bytes
    before = cv2.imread(path).astype(float)
    box = (60, 30, 20, 20)

    def detect(bgr):  # mocked detector: one face on the first frame only
        return [box] if np.allclose(bgr.astype(float), before) else []

    n, nf, faces = privacy.run(session, detect, {"model": "mock"})
    assert (n, nf, faces) == (4, 1, 1)
    after = cv2.imread(path).astype(float)
    # inside the box enlarged by 30 % (57-83, 27-53): blurred, noise texture gone; well outside: unchanged
    assert after[28:52, 58:82].std(axis=(0, 1)).mean() < 0.3 * before[28:52, 58:82].std(axis=(0, 1)).mean()
    assert np.abs(after[:, :40] - before[:, :40]).mean() < 12  # JPEG re-encoding noise only
    frames = load_frames(session)
    assert [f["faces"] for f in frames] == [1, 0, 0, 0]
    for f in frames:
        assert b"Exif" not in open(os.path.join(session, "frames", f["name"] + ".jpg"), "rb").read()
    s = load_session(session)
    assert s["privacy"]["frames_with_faces"] == 1 and s["review"] is None
    with pytest.raises(SystemExit):  # not twice
        privacy.run(session, detect)


def test_blur_box_clipped_at_border():
    img = np.random.default_rng(1).integers(0, 256, (90, 160, 3), np.uint8)
    out = privacy.blur_boxes(img.copy(), [(-5, -5, 20, 20)])
    assert not np.array_equal(out[:10, :10], img[:10, :10]) and np.array_equal(out[40:], img[40:])


# ---------------------------------------------------------------- review


def test_review_requires_privacy(session):
    with pytest.raises(ValueError):
        review.apply_review(session, [])


def test_review_delete_applies(session):
    privacy.run(session, no_faces)
    names = [f["name"] for f in load_frames(session)]
    assert review.apply_review(session, names[1:3]) == (2, 2)
    assert [f["name"] for f in load_frames(session)] == [names[0], names[3]]
    assert sorted(os.listdir(os.path.join(session, "frames"))) == [names[0] + ".jpg", names[3] + ".jpg"]
    log = [json.loads(line) for line in open(os.path.join(session, "review_log.jsonl"))]
    assert log == [{"deleted": 2, "kept": 2}]
    assert load_session(session)["review"]["done"]
    with pytest.raises(ValueError):
        review.apply_review(session, ["home01_99999"])


def test_review_page_faces_first_and_apply(session):
    privacy.run(session, lambda bgr: [])
    frames = load_frames(session)
    frames[2]["faces"] = 1
    review.save_frames(session, frames)
    done = []
    server = review.make_server(session, port=0, on_done=lambda d, k: done.append((d, k)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(url + "/").read().decode()
        assert page.index("home01_00002") < page.index("home01_00000")  # the face frame first
        assert urllib.request.urlopen(url + "/frames/home01_00001.jpg").read()[:2] == b"\xff\xd8"
        with pytest.raises(urllib.error.HTTPError):  # only frames of the session are served
            urllib.request.urlopen(url + "/frames/../session.json")
        req = urllib.request.Request(url + "/apply", json.dumps({"delete": ["home01_00002"]}).encode(),
                                     {"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read()) == {"deleted": 1, "kept": 3}
    finally:
        server.shutdown()
        server.server_close()
    assert done == [(1, 3)]


# ---------------------------------------------------------------- consent and export


def test_consent_register_outside_session(session, tmp_path):
    with pytest.raises(ValueError):
        consent.record(session, True, "2026-01-03", "owner", "x", os.path.join(session, "register.jsonl"))
    reg = tmp_path / "register.jsonl"
    consent.record(session, True, "2026-01-03", "tenant", "binder A", str(reg))
    assert load_session(session)["consent"] == {"given": True, "date": "2026-01-03", "signer_role": "tenant"}
    assert "binder A" not in open(os.path.join(session, "session.json")).read()
    assert json.loads(reg.read_text())["form_ref"] == "binder A"


def test_export_refuses_without_consent_or_review(session, tmp_path):
    out = str(tmp_path / "ds")
    privacy.run(session, no_faces)
    with pytest.raises(SystemExit, match="no completed owner review"):
        export.export([session], out)
    review.apply_review(session, [])
    with pytest.raises(SystemExit, match="no written consent"):
        export.export([session], out)
    consent.record(session, False, "2026-01-03", "owner", "refused", str(tmp_path / "register.jsonl"))
    with pytest.raises(SystemExit, match="consent refused"):
        export.export([session], out)
    assert not os.path.exists(out)


def test_export_layout(tmp_path):
    run = make_run(tmp_path)
    staging = tmp_path / "staging"
    a, _, _ = import_run.import_run(str(run), str(staging), "home01")
    b, _, _ = import_run.import_run(str(run), str(staging), "home02")
    for s in (a, b):
        finish(s, tmp_path)
    out = tmp_path / "ds"
    assert export.export([a, b], str(out), val=["home02"]) == [("home01", "train", 4), ("home02", "validation", 4)]
    assert sorted(os.listdir(out)) == ["README.md", "calibration", "train", "validation"]
    assert sorted(os.listdir(out / "calibration")) == ["home01.json", "home02.json"]
    rows = [json.loads(line) for line in open(out / "train" / "metadata.jsonl")]
    assert len(rows) == 4 and all((out / "train" / r["file_name"]).exists() for r in rows)
    assert set(rows[0]) == {"file_name", "session", "calibration", "t_s", "cam_h_m", "pitch_down_deg", "roll_deg",
                            "x_m", "y_m", "heading_deg", "v", "w", "motion", "nav_state", "ctrl"}
    card = (out / "README.md").read_text()
    assert "license: cc-by-4.0" in card and "| home02 | validation | 4 |" in card
    with pytest.raises(SystemExit):  # never into a non-empty folder
        export.export([a], str(out))
