"""Click-to-label review page for floor_seg.label_real output (replaces writing edits.json by hand).

    python -m floor_seg.label_page data/real/new_set [--port 8766]
    # then open http://127.0.0.1:8766 (served on this computer only, works offline)

Reads what `label_real prelabel` wrote per frame: <name>.jpg (640x360), <name>_regions.png (uint16 SAM2 region
ids, 0 = thin uncovered seam, always ignore) and <name>_init.json ({id: 1 floor / 0 obstacle} from the model).
Optional hint: <name>_driven.png (non-zero = floor the robot later drove over) is outlined in blue.

Regions are tinted by their current label: floor green, obstacle red, ignore grey; region outlines yellow.
Mouse: click a region to cycle floor -> obstacle -> ignore.
Keys: 1 / 2 / 3 set the region under the cursor to floor / obstacle / ignore; A sets every region floor,
Shift+A every region obstacle; Z undo; R raw image on/off; D driven hint on/off; V toggle "reviewed";
Enter mark reviewed, save and go to the next frame; Right / Space next, Left / Shift+Space previous; S save.
Hand polygon (a patch SAM2 merged into a bigger region): P, click the corners (also on the raw image), then 1 / 2 / 3
closes it as floor / obstacle / ignore; Backspace removes the last corner, Esc cancels; X deletes the polygon under
the cursor. Polygons are saved in the frame's "poly" list and applied last by `label_real apply`.

Label rule (project decision): anything > 1 cm above the floor is an obstacle; rugs, cables, door thresholds and
doorsteps the robot can drive over are floor.

Saves DATA_DIR/edits.json in the format `label_real apply` reads, as the difference from the pre-labels
(lists "floor" / "obstacle" / "ignore"; an "all" reset is folded into the lists), merged into the existing file:
other frames and other keys of a frame (hand "poly" polygons) are kept. Reviewed frames are listed in
DATA_DIR/reviewed.json. "Apply" runs `label_real apply` on the reviewed frames only, so unreviewed model guesses
never become <name>_floor.png labels (same as `python -m floor_seg.label_real apply --out DIR --reviewed-only`).
"""
import argparse
import glob
import html
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

IGNORE = 255                                            # same as floor_seg.label_real (not imported: it loads torch)
LABELS = {"floor": 1, "obstacle": 0, "ignore": IGNORE}  # edits.json list name -> label png value
REVIEWED = "reviewed.json"                              # sorted list of reviewed frame names


def frame_names(data_dir):
    """Frames prelabel wrote, in the order `label_real apply` walks them."""
    return sorted(os.path.basename(p)[:-len("_init.json")] for p in glob.glob(os.path.join(data_dir, "*_init.json")))


def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def write_json_atomic(path, obj):
    """Temp file in the same folder + os.replace: a crash mid-write never leaves a truncated edits.json."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def current_labels(init, edit):
    """{id: "floor"|"obstacle"|"ignore"} after the frame's edits; mirrors label_real.apply (polygons aside)."""
    lab = {r: v for r, v in init.items()}
    if edit.get("all") in ("floor", "obstacle"):
        lab = {r: LABELS[edit["all"]] for r in lab}
    for key, val in LABELS.items():
        for r in edit.get(key, []):
            lab[str(r)] = val
    names = {v: k for k, v in LABELS.items()}
    return {r: names[v] for r, v in lab.items()}


def edit_entry(init, labels, old):
    """edits.json entry for a frame: regions whose label differs from the pre-label. Keys the page does not edit
    ("poly", anything hand-written) are carried over from the old entry; "all" is dropped (folded into the lists)."""
    e = {k: v for k, v in old.items() if k not in ("all", *LABELS)}
    for key, val in LABELS.items():
        ids = sorted(int(r) for r, lab in labels.items() if lab == key and init[r] != val)
        if ids:
            e[key] = ids
    return e


def check_polys(name, polys):
    """Hand polygons as label_real apply reads them: [{"label": floor|obstacle|ignore, "pts": [[x, y], ...]}],
    at least 3 points, pixel coords rounded to int. Raises ValueError otherwise."""
    out = []
    for p in polys:
        pts = p.get("pts") if isinstance(p, dict) else None
        if not isinstance(pts, list) or p.get("label") not in LABELS or len(pts) < 3 or \
                not all(isinstance(q, list) and len(q) == 2 and all(isinstance(c, (int, float)) for c in q)
                        for q in pts):
            raise ValueError(f"{name}: bad polygon {p!r}")
        out.append({"label": p["label"], "pts": [[round(x), round(y)] for x, y in pts]})
    return out


def save(data_dir, frames):
    """Merge {name: {"labels": {id: label}, "reviewed": bool}} into edits.json and reviewed.json.
    Raises ValueError (nothing written) on an unknown frame, a missing / unknown region id or label."""
    known = set(frame_names(data_dir))
    edits_path = os.path.join(data_dir, "edits.json")
    edits = load_json(edits_path, {})
    reviewed = set(load_json(os.path.join(data_dir, REVIEWED), []))
    for name, f in frames.items():
        if name not in known:
            raise ValueError(f"unknown frame {name!r}")
        init = load_json(os.path.join(data_dir, name + "_init.json"), {})
        labels = {str(r): v for r, v in f["labels"].items()}
        if set(labels) != set(init) or not set(labels.values()) <= set(LABELS):
            raise ValueError(f"{name}: labels must give floor/obstacle/ignore for every region")
        e = edit_entry(init, labels, edits.get(name, {}))
        if "poly" in f:   # the page sends the frame's full polygon list (drawn or deleted there); absent: kept
            e.pop("poly", None)
            if f["poly"]:
                e["poly"] = check_polys(name, f["poly"])
        if e:
            edits[name] = e
        else:
            edits.pop(name, None)  # back to the pre-labels: no entry needed
        (reviewed.add if f.get("reviewed") else reviewed.discard)(name)
    write_json_atomic(edits_path, edits)
    write_json_atomic(os.path.join(data_dir, REVIEWED), sorted(reviewed))
    return sorted(reviewed)


def frame_info(data_dir, name):
    init = load_json(os.path.join(data_dir, name + "_init.json"), {})
    edit = load_json(os.path.join(data_dir, "edits.json"), {}).get(name, {})
    h, w = cv2.imread(os.path.join(data_dir, name + "_regions.png"), cv2.IMREAD_UNCHANGED).shape
    return {"name": name, "w": w, "h": h, "init": init, "labels": current_labels(init, edit),
            "poly": edit.get("poly", []), "reviewed": name in load_json(os.path.join(data_dir, REVIEWED), []),
            "driven": os.path.exists(os.path.join(data_dir, name + "_driven.png"))}


def raw_regions(data_dir, name):
    """Region ids as raw little-endian uint16 bytes: a PNG decoded by the browser would be cut to 8 bits."""
    return cv2.imread(os.path.join(data_dir, name + "_regions.png"), cv2.IMREAD_UNCHANGED).astype("<u2").tobytes()


def raw_driven(data_dir, name, shape):
    m = cv2.imread(os.path.join(data_dir, name + "_driven.png"), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != shape:
        m = cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8).tobytes()


def apply_reviewed(data_dir):
    """`label_real apply` on the reviewed frames; imported here because label_real loads torch."""
    from floor_seg import label_real
    names = load_json(os.path.join(data_dir, REVIEWED), [])
    return label_real.apply(argparse.Namespace(out=data_dir, names=set(names)))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Labels __TITLE__</title>
<style>
body { font-family: sans-serif; margin: 0; background: #222; color: #eee; }
header { position: sticky; top: 0; background: #fff; color: #222; padding: 8px 16px; border-bottom: 1px solid #ccc;
  z-index: 1; }
header button, header select { font-size: 0.95em; padding: 4px 10px; }
#rev.on { background: #43a047; color: #fff; }
#pbtn.on { background: #d500f9; color: #fff; }
#dirty { color: #e53935; font-weight: bold; }
#stage { position: relative; margin: 8px auto; }
#stage img, #stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
#stage canvas { image-rendering: pixelated; cursor: crosshair; }
#msg { text-align: center; }
</style></head><body>
<header>
  <b>__TITLE__</b> &mdash; <select id="sel"></select> <span id="pos"></span> &middot;
  reviewed <b id="prog"></b> &middot; region <span id="hov">-</span>
  <button id="rev">reviewed (V)</button> <button id="raw">raw (R)</button> <button id="hint">driven (D)</button>
  <button id="pbtn">polygon (P)</button>
  <button id="save">save (S)</button> <button id="apply">apply reviewed</button> <span id="dirty"></span>
  <div><small>Click: floor &rarr; obstacle &rarr; ignore. 1/2/3: region under cursor floor/obstacle/ignore.
  A: all floor, Shift+A: all obstacle. Z: undo. Enter: reviewed + save + next. &rarr;/Space next, &larr;/Shift+Space
  previous. <b>Polygon</b> (for a patch merged into a bigger region): P, click the corners, then 1/2/3 closes it as
  floor/obstacle/ignore (applied on top of the regions); Backspace last corner, Esc cancel; X deletes the polygon
  under the cursor. <span style="color:#2e7d32">floor</span>, <span style="color:#c62828">obstacle</span>, grey ignore,
  <span style="color:#1e88e5">blue: driven over</span>. Rule: &gt; 1 cm above the floor = obstacle; rugs, cables,
  thresholds, drivable doorsteps = floor.</small></div>
</header>
<div id="stage"><img alt=""><canvas></canvas></div>
<div id="msg"></div>
<script>
const NAMES = ["floor", "obstacle", "ignore"];             // label codes 0 / 1 / 2 in the page
const COLORS = [[0, 200, 0], [220, 0, 0], [128, 128, 128]];
const stage = document.getElementById("stage"), img = stage.querySelector("img"), cv = stage.querySelector("canvas");
const ctx = cv.getContext("2d");
let FRAMES = [];          // [{name, reviewed}] from the server
const cache = new Map();  // name -> loaded frame state (kept while browsing so unsaved changes survive)
let cur = null, idx = 0, hover = 0, showRaw = false, showHint = true;
let drawPts = null, mouse = null;  // polygon being drawn (image px corners, null = not drawing); cursor in image px

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

// one frame: region ids (Uint16Array), label code per id, precomputed outline masks
async function load(name) {
  if (cache.has(name)) return cache.get(name);
  const info = await getJSON("/api/frame/" + name);
  const reg = new Uint16Array(await (await fetch("/regions/" + name + ".bin")).arrayBuffer());
  const ids = Object.keys(info.init).map(Number);
  let maxId = 0;  // a loop, not Math.max(...reg): spreading 230k values overflows the argument limit
  for (const r of reg) if (r > maxId) maxId = r;
  const lab = new Uint8Array(maxId + 1).fill(2);  // id 0 (seams) stays ignore
  for (const r of ids) lab[r] = NAMES.indexOf(info.labels[r]);
  const {w, h} = info;
  const edge = new Uint8Array(w * h), drv = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    edge[i] = (x < w - 1 && reg[i + 1] !== reg[i]) || (y < h - 1 && reg[i + w] !== reg[i]) ? 1 : 0;
  }
  if (info.driven) {  // blue outline of the driven-over mask
    const d = new Uint8Array(await (await fetch("/driven/" + name + ".bin")).arrayBuffer());
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      drv[i] = (x < w - 1 && d[i + 1] !== d[i]) || (y < h - 1 && d[i + w] !== d[i]) ? 1 : 0;
    }
  }
  const f = {name, w, h, reg, lab, ids, edge, drv, poly: info.poly, reviewed: info.reviewed, dirty: false,
             undo: []};
  cache.set(name, f);
  return f;
}

function draw() {
  if (!cur) return;
  const {w, h, reg, lab, edge, drv} = cur;
  const im = ctx.createImageData(w, h), d = im.data;  // all transparent: what raw mode shows over the photo
  for (let i = 0; i < w * h && !showRaw; i++) {
    const r = reg[i], c = COLORS[lab[r]], o = 4 * i;
    if (showHint && drv[i]) { d[o] = 30; d[o + 1] = 136; d[o + 2] = 229; d[o + 3] = 255; continue; }
    if (edge[i]) { d[o] = 255; d[o + 1] = 255; d[o + 2] = r && r === hover ? 255 : 0; d[o + 3] = 255; continue; }
    d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
    d[o + 3] = r && r === hover ? 150 : 95;  // the hovered region a bit stronger (target of 1/2/3)
  }
  ctx.putImageData(im, 0, 0);
  for (const p of showRaw ? [] : cur.poly) {  // hand polygons: applied last by label_real, filled with their label
    const c = COLORS[NAMES.indexOf(p.label)];
    ctx.fillStyle = "rgba(" + c + ",0.55)";
    ctx.strokeStyle = "rgb(" + c + ")";
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    p.pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  if (drawPts) {  // polygon being drawn: corners so far, then a line to the cursor
    ctx.strokeStyle = "rgb(255,0,255)";
    ctx.setLineDash([]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    drawPts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    if (mouse && drawPts.length) ctx.lineTo(mouse[0], mouse[1]);
    ctx.stroke();
    ctx.fillStyle = "rgb(255,0,255)";
    for (const [x, y] of drawPts) ctx.fillRect(x - 2, y - 2, 4, 4);
  }
  document.getElementById("pbtn").classList.toggle("on", !!drawPts);
  status();
}

function status() {
  const n = FRAMES.filter(f => f.reviewed).length;
  document.getElementById("prog").textContent = n + " / " + FRAMES.length;
  document.getElementById("pos").textContent = (idx + 1) + " / " + FRAMES.length;
  document.getElementById("rev").classList.toggle("on", !!(cur && cur.reviewed));
  document.getElementById("hov").textContent = hover ? hover + " " + NAMES[cur.lab[hover]] : "-";
  const dirty = [...cache.values()].filter(f => f.dirty).length;
  document.getElementById("dirty").textContent = dirty ? dirty + " frame(s) not saved" : "";
  const sel = document.getElementById("sel");
  FRAMES.forEach((f, i) => { sel.options[i].textContent = (f.reviewed ? "\\u2713 " : "\\u2003 ") + f.name; });
  sel.selectedIndex = idx;
}

function fit() {  // largest size that fits under the header, keeping the frame's aspect ratio
  if (!cur) return;
  const top = document.querySelector("header").offsetHeight + 16;
  const wpx = Math.min(window.innerWidth - 16, (window.innerHeight - top) * cur.w / cur.h);
  stage.style.width = wpx + "px";
  stage.style.height = wpx * cur.h / cur.w + "px";
}

async function go(i) {
  idx = (i + FRAMES.length) % FRAMES.length;
  try {
    cur = await load(FRAMES[idx].name);
  } catch (e) { document.getElementById("msg").textContent = "Error: " + e.message; return; }
  cv.width = cur.w; cv.height = cur.h;
  img.src = "/img/" + cur.name + ".jpg";
  hover = 0; drawPts = null;
  fit();
  draw();
}

function change(fn) {  // every edit goes through here: undo snapshot, dirty flag, redraw
  cur.undo.push({lab: cur.lab.slice(), reviewed: cur.reviewed, poly: cur.poly.slice()});
  fn();
  cur.dirty = true;
  FRAMES[idx].reviewed = cur.reviewed;
  draw();
}
function setRegion(r, code) { if (r && cur.ids.includes(r)) change(() => { cur.lab[r] = code; }); }
function setAll(code) { change(() => { for (const r of cur.ids) cur.lab[r] = code; }); }
function undo() {
  const u = cur.undo.pop();
  if (!u) return;
  cur.lab = u.lab; cur.reviewed = u.reviewed; cur.poly = u.poly; cur.dirty = true;
  FRAMES[idx].reviewed = cur.reviewed;
  draw();
}

async function save() {
  const frames = {};
  for (const f of cache.values()) if (f.dirty) {
    const labels = {};
    for (const r of f.ids) labels[r] = NAMES[f.lab[r]];
    frames[f.name] = {labels, reviewed: f.reviewed, poly: f.poly};
  }
  if (!Object.keys(frames).length) return true;
  try {
    const j = await getJSON("/api/save", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify({frames})});
    const done = new Set(j.reviewed);
    for (const f of FRAMES) f.reviewed = done.has(f.name);
    for (const n of Object.keys(frames)) cache.get(n).dirty = false;
    document.getElementById("msg").textContent = "saved " + Object.keys(frames).length + " frame(s)";
    status();
    return true;
  } catch (e) { document.getElementById("msg").textContent = "Save error: " + e.message; return false; }
}

async function applyReviewed() {
  if (!(await save())) return;
  if (!confirm("Write <name>_floor.png labels for the " + FRAMES.filter(f => f.reviewed).length +
               " reviewed frame(s)?")) return;
  try {
    const j = await getJSON("/api/apply", {method: "POST"});
    document.getElementById("msg").textContent = "apply: wrote " + j.written + " label png(s)";
  } catch (e) { document.getElementById("msg").textContent = "Apply error: " + e.message; }
}

function pixAt(ev) {  // mouse position in image pixels
  const b = cv.getBoundingClientRect();
  return [Math.floor((ev.clientX - b.left) * cur.w / b.width), Math.floor((ev.clientY - b.top) * cur.h / b.height)];
}
function regionAt(ev) {
  const [x, y] = pixAt(ev);
  return x >= 0 && y >= 0 && x < cur.w && y < cur.h ? cur.reg[y * cur.w + x] : 0;
}
function inPoly([x, y], pts) {  // even-odd ray casting
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function closePoly(code) {  // the drawn polygon becomes a hand polygon with this label
  if (drawPts.length < 3) return;
  const pts = drawPts;
  drawPts = null;
  change(() => { cur.poly = [...cur.poly, {label: NAMES[code], pts}]; });
}
function deletePolyAt(p) {  // the last-drawn polygon under the cursor (it is the one on top)
  for (let k = cur.poly.length - 1; k >= 0; k--)
    if (inPoly(p, cur.poly[k].pts)) { change(() => { cur.poly = cur.poly.filter((_, i) => i !== k); }); return; }
}
cv.addEventListener("mousemove", ev => {
  const r = regionAt(ev);
  mouse = pixAt(ev);
  if (r !== hover || drawPts) { hover = r; draw(); }
});
cv.addEventListener("mouseleave", () => { hover = 0; mouse = null; draw(); });
cv.addEventListener("click", ev => {
  if (drawPts) { drawPts.push(pixAt(ev)); draw(); return; }  // drawing: a click adds a corner
  const r = regionAt(ev);
  if (r && !showRaw) setRegion(r, (cur.lab[r] + 1) % 3);  // raw image: no relabelling by accident
});
document.getElementById("pbtn").onclick = () => { drawPts = drawPts ? null : []; draw(); };
document.getElementById("sel").onchange = e => go(e.target.selectedIndex);
document.getElementById("rev").onclick = () => change(() => { cur.reviewed = !cur.reviewed; });
document.getElementById("raw").onclick = () => { showRaw = !showRaw; draw(); };
document.getElementById("hint").onclick = () => { showHint = !showHint; draw(); };
document.getElementById("save").onclick = save;
document.getElementById("apply").onclick = applyReviewed;
window.addEventListener("resize", fit);
window.addEventListener("beforeunload", e => { if ([...cache.values()].some(f => f.dirty)) e.preventDefault(); });
document.addEventListener("keydown", async e => {
  if (!cur || e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (drawPts) {  // drawing a polygon: only these keys; 1/2/3 close it
    if (k === "1" || k === "2" || k === "3") closePoly(Number(k) - 1);
    else if (k === "Backspace") { drawPts.pop(); draw(); }
    else if (k === "Escape" || k === "p" || k === "P") { drawPts = null; draw(); }
    else return;
    e.preventDefault();
    return;
  }
  if (k === "p" || k === "P") { drawPts = []; draw(); }
  else if ((k === "x" || k === "X") && mouse) deletePolyAt(mouse);
  else if (k === "ArrowRight" || (k === " " && !e.shiftKey)) go(idx + 1);
  else if (k === "ArrowLeft" || (k === " " && e.shiftKey)) go(idx - 1);
  else if ((k === "1" || k === "2" || k === "3") && !showRaw) setRegion(hover, Number(k) - 1);
  else if (k === "a") setAll(0);
  else if (k === "A") setAll(1);
  else if (k === "z" || k === "Z") undo();
  else if (k === "s" || k === "S") save();
  else if (k === "r" || k === "R") { showRaw = !showRaw; draw(); }
  else if (k === "d" || k === "D") { showHint = !showHint; draw(); }
  else if (k === "v" || k === "V") change(() => { cur.reviewed = !cur.reviewed; });
  else if (k === "Enter") {
    if (!cur.reviewed) change(() => { cur.reviewed = true; });
    if (await save()) go(idx + 1);
  } else return;
  e.preventDefault();
});
(async () => {
  FRAMES = await getJSON("/api/frames");
  const sel = document.getElementById("sel");
  for (const f of FRAMES) sel.add(new Option(f.name));
  if (!FRAMES.length) { document.getElementById("msg").textContent = "no *_init.json frames in this folder"; return; }
  const first = FRAMES.findIndex(f => !f.reviewed);  // resume at the first unreviewed frame
  go(first < 0 ? 0 : first);
})();
</script></body></html>
"""


def make_server(data_dir, port=8766):
    """The labelling HTTP server on 127.0.0.1 (port 0: any free port)."""
    data_dir = os.path.abspath(data_dir)
    lock = threading.Lock()  # one writer at a time on edits.json / reviewed.json / label pngs

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: no request log on the terminal
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code, obj):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def route(self, prefix, suffix):
            """Frame name from /<prefix><name><suffix>, only if it is a known frame (no path traversal)."""
            if self.path.startswith(prefix) and self.path.endswith(suffix):
                name = self.path[len(prefix):len(self.path) - len(suffix)]
                if name in frame_names(data_dir):
                    return name
            return None

        def do_GET(self):
            if self.path == "/":
                page = PAGE.replace("__TITLE__", html.escape(os.path.basename(data_dir)))
                return self.send(200, page.encode(), "text/html; charset=utf-8")
            if self.path == "/api/frames":
                done = set(load_json(os.path.join(data_dir, REVIEWED), []))
                return self.send_json(200, [{"name": n, "reviewed": n in done} for n in frame_names(data_dir)])
            if name := self.route("/api/frame/", ""):
                return self.send_json(200, frame_info(data_dir, name))
            if name := self.route("/img/", ".jpg"):
                with open(os.path.join(data_dir, name + ".jpg"), "rb") as f:
                    return self.send(200, f.read(), "image/jpeg")
            if name := self.route("/regions/", ".bin"):
                return self.send(200, raw_regions(data_dir, name), "application/octet-stream")
            name = self.route("/driven/", ".bin")
            if name and os.path.exists(os.path.join(data_dir, name + "_driven.png")):
                shape = cv2.imread(os.path.join(data_dir, name + "_regions.png"), cv2.IMREAD_UNCHANGED).shape
                return self.send(200, raw_driven(data_dir, name, shape), "application/octet-stream")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                with lock:
                    if self.path == "/api/save":
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                        return self.send_json(200, {"reviewed": save(data_dir, body.get("frames", {}))})
                    if self.path == "/api/apply":
                        return self.send_json(200, {"written": apply_reviewed(data_dir)})
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self.send_json(400, {"error": str(e)})
            self.send(404, b"not found", "text/plain")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="label_real prelabel output folder (<name>.jpg, _regions.png, _init.json)")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    if not frame_names(args.data):
        raise SystemExit(f"no *_init.json in {args.data}: run `python -m floor_seg.label_real prelabel` first")
    server = make_server(args.data, args.port)
    print(f"label page: http://127.0.0.1:{server.server_address[1]}  (Ctrl+C to quit; save with S first)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
