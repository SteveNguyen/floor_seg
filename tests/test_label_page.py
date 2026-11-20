"""floor_seg.label_page on a tiny synthetic label_real prelabel folder: HTTP handlers, save merge, apply."""
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from floor_seg import label_page

W, H = 640, 360  # label_real works at this fixed size
# region ids: 1 = bottom half, 2 = top-left, 3 = top-right, 0 = one-pixel seam row between top and bottom
SEAM = H // 2 - 1
POLY = {"label": "obstacle", "pts": [[0, 300], [20, 300], [20, 320], [0, 320]]}  # hand polygon in region 1


def regions():
    reg = np.ones((H, W), np.uint16)
    reg[:SEAM, :W // 2] = 2
    reg[:SEAM, W // 2:] = 3
    reg[SEAM] = 0
    return reg


@pytest.fixture
def data(tmp_path):
    """Frames a and b as prelabel writes them, plus an edits.json written by hand earlier."""
    rng = np.random.default_rng(0)
    (tmp_path / "review").mkdir()
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), rng.integers(0, 256, (H, W, 3), np.uint8))
        cv2.imwrite(str(tmp_path / f"{name}_regions.png"), regions())
        (tmp_path / f"{name}_init.json").write_text(json.dumps({"1": 1, "2": 0, "3": 1}))
    driven = np.zeros((H, W), np.uint8)
    driven[250:, 100:200] = 1
    cv2.imwrite(str(tmp_path / "a_driven.png"), driven)
    (tmp_path / "edits.json").write_text(json.dumps({"a": {"poly": [POLY], "note": "kept"}, "b": {"obstacle": [1]}}))
    return tmp_path


@pytest.fixture
def url(data):
    server = label_page.make_server(str(data), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def post(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_frames(url):
    assert b"<canvas>" in get(url + "/")
    assert json.loads(get(url + "/api/frames")) == [{"name": "a", "reviewed": False}, {"name": "b", "reviewed": False}]
    a = json.loads(get(url + "/api/frame/a"))
    assert (a["w"], a["h"], a["driven"], a["poly"]) == (W, H, True, [POLY])
    assert a["labels"] == {"1": "floor", "2": "obstacle", "3": "floor"}
    assert json.loads(get(url + "/api/frame/b"))["labels"] == {"1": "obstacle", "2": "obstacle", "3": "floor"}
    reg = np.frombuffer(get(url + "/regions/a.bin"), "<u2").reshape(H, W)
    assert (reg == regions()).all()  # full 16-bit ids, not cut to 8 bits
    drv = np.frombuffer(get(url + "/driven/a.bin"), np.uint8).reshape(H, W)
    assert drv.sum() == 110 * 100
    assert get(url + "/img/a.jpg")[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", ["/img/../edits.json", "/img/zz.jpg", "/api/frame/zz", "/driven/b.bin", "/x"])
def test_not_found(url, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(url + path)
    assert e.value.code == 404


def test_save_merges(url, data):
    labels = {"1": "floor", "2": "floor", "3": "ignore"}
    code, j = post(url + "/api/save", {"frames": {"a": {"labels": labels, "reviewed": True}}})
    assert (code, j) == (200, {"reviewed": ["a"]})
    edits = json.loads((data / "edits.json").read_text())
    # only the differences from the pre-labels; hand keys of a kept; frame b untouched
    assert edits == {"a": {"poly": [POLY], "note": "kept", "floor": [2], "ignore": [3]}, "b": {"obstacle": [1]}}
    assert json.loads((data / "reviewed.json").read_text()) == ["a"]
    assert json.loads(get(url + "/api/frames"))[0] == {"name": "a", "reviewed": True}
    assert json.loads(get(url + "/api/frame/a"))["labels"] == labels
    # back to the pre-labels for b: its entry goes away; a is still reviewed
    back = {"1": "floor", "2": "obstacle", "3": "floor"}
    assert post(url + "/api/save", {"frames": {"b": {"labels": back, "reviewed": False}}}) == (200, {"reviewed": ["a"]})
    assert "b" not in json.loads((data / "edits.json").read_text())


def test_save_polygons_from_the_page(url, data):
    """The page sends the frame's whole polygon list: it replaces the old one; an empty list removes it."""
    labels = {"1": "floor", "2": "obstacle", "3": "floor"}  # = pre-labels
    new = {"label": "ignore", "pts": [[10.4, 10], [50, 10], [50, 40.6]]}
    assert post(url + "/api/save", {"frames": {"a": {"labels": labels, "poly": [new]}}})[0] == 200
    a = json.loads((data / "edits.json").read_text())["a"]
    assert a == {"note": "kept", "poly": [{"label": "ignore", "pts": [[10, 10], [50, 10], [50, 41]]}]}
    post(url + "/api/save", {"frames": {"a": {"labels": labels, "poly": []}}})
    assert json.loads((data / "edits.json").read_text())["a"] == {"note": "kept"}


def test_save_folds_all(data):
    """An "all" reset written by hand becomes explicit lists (same labels for label_real apply)."""
    (data / "edits.json").write_text(json.dumps({"a": {"all": "obstacle", "floor": [3]}}))
    labels = label_page.current_labels({"1": 1, "2": 0, "3": 1}, {"all": "obstacle", "floor": [3]})
    assert labels == {"1": "obstacle", "2": "obstacle", "3": "floor"}
    label_page.save(str(data), {"a": {"labels": labels, "reviewed": False}})
    assert json.loads((data / "edits.json").read_text()) == {"a": {"obstacle": [1]}}


@pytest.mark.parametrize("frames", [{"zz": {"labels": {}}},                                    # unknown frame
                                    {"a": {"labels": {"1": "floor"}}},                         # missing regions
                                    {"a": {"labels": {"1": "floor", "2": "floor", "3": "x"}}},   # bad label
                                    {"a": {"labels": {"1": "floor", "2": "floor", "3": "floor"},  # 2-point polygon
                                           "poly": [{"label": "floor", "pts": [[0, 0], [1, 1]]}]}},
                                    {"a": {"labels": {"1": "floor", "2": "floor", "3": "floor"},  # bad polygon label
                                           "poly": [{"label": "x", "pts": [[0, 0], [1, 1], [2, 0]]}]}}])
def test_save_rejects(url, data, frames):
    before = (data / "edits.json").read_text()
    code, j = post(url + "/api/save", {"frames": frames})
    assert code == 400 and j["error"]
    assert (data / "edits.json").read_text() == before
    assert not (data / "reviewed.json").exists()


def test_apply_reviewed(url, data):
    labels = {"1": "floor", "2": "floor", "3": "ignore"}
    post(url + "/api/save", {"frames": {"a": {"labels": labels, "reviewed": True}}})
    assert post(url + "/api/apply") == (200, {"written": 1})
    lab = cv2.imread(str(data / "a_floor.png"), cv2.IMREAD_UNCHANGED)
    expect = np.ones((H, W), np.uint8)       # regions 1 and 2 floor
    expect[:SEAM, W // 2:] = 255             # region 3 ignore
    expect[SEAM] = 255                       # seam (id 0) always ignore
    expect[300:321, 0:21] = 0                # hand polygon, applied last (cv2.fillPoly includes the border)
    assert (lab == expect).all()
    assert not (data / "b_floor.png").exists()  # not reviewed: no label written
