"""Owner review of a staged session: a local page to delete frames before anything leaves the laptop.

    python -m floor_seg.dataset.review STAGING/home03 [--port 8765]
    # then open http://127.0.0.1:8765 (served on this computer only, works offline)

The page shows every frame as a thumbnail, frames where faces were found (and blurred) first, then in time order.
Click a frame (or Space / X on the selected one) to mark it for deletion; arrows move the selection, Enter shows the
frame large, Esc closes it; "Apply" (or Shift+Enter) deletes the marked frames from the session and ends the
review. The privacy pass (floor_seg.dataset.privacy) must have run first.

The review log (review_log.jsonl in the session) records how many frames were deleted and kept, nothing about
their content. Export (floor_seg.dataset.export) requires a completed review.
"""
import argparse
import html
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from floor_seg.dataset import frame_path, load_frames, load_session, save_frames, save_session


def apply_review(session_dir, delete):
    """Delete the named frames (images and records), log the counts, mark the review done. Returns (deleted, kept)."""
    session = load_session(session_dir)
    if not (session.get("privacy") or {}).get("done"):
        raise ValueError("run the privacy pass (floor_seg.dataset.privacy) before the review")
    frames = load_frames(session_dir)
    names = {r["name"] for r in frames}
    delete = set(delete)
    unknown = delete - names
    if unknown:
        raise ValueError(f"unknown frames: {sorted(unknown)[:5]}")
    for name in delete:
        os.remove(frame_path(session_dir, name))
    kept = [r for r in frames if r["name"] not in delete]
    save_frames(session_dir, kept)
    with open(os.path.join(session_dir, "review_log.jsonl"), "a") as f:
        f.write(json.dumps({"deleted": len(delete), "kept": len(kept)}) + "\n")
    session["review"] = {"done": True, "deleted": len(delete) + (session.get("review") or {}).get("deleted", 0),
                         "kept": len(kept)}
    save_session(session_dir, session)
    return len(delete), len(kept)


def review_order(frames):
    """Frames with faces first (most faces first), then time order."""
    return sorted(frames, key=lambda r: (-r.get("faces", 0), r["t_s"]))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Review %(session)s</title>
<style>
body { font-family: sans-serif; margin: 0; background: #f4f4f4; color: #222; }
header { position: sticky; top: 0; background: #fff; padding: 10px 16px; border-bottom: 1px solid #ccc; z-index: 1; }
header button { font-size: 1em; padding: 6px 14px; }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px; padding: 12px; }
.cell { position: relative; border: 3px solid transparent; background: #fff; cursor: pointer; padding: 0; }
.cell img { width: 100%%; display: block; }
.cell .tag { position: absolute; left: 4px; top: 4px; background: #fff; font-size: 12px; padding: 1px 4px; }
.cell .face { background: #ffd54f; }
.cell.sel { border-color: #1e88e5; }
.cell.del { border-color: #e53935; }
.cell.del img { opacity: 0.3; }
.cell.del::after { content: "DELETE"; position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; color: #e53935; font-weight: bold; font-size: 24px; }
#big { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 2;
  align-items: center; justify-content: center; }
#big img { max-width: 96vw; max-height: 96vh; }
</style></head><body>
<header>
  <b>Session %(session)s</b> &mdash; <span id="count"></span>
  <button id="apply">Apply: delete marked frames</button>
  <div><small>Click or Space / X: mark or unmark for deletion. Arrows: move. Enter: view large (Esc closes).
  Shift+Enter: apply. Frames with a yellow tag had faces (already blurred): check them first.</small></div>
</header>
<div id="grid"></div>
<div id="big"><img alt=""></div>
<script>
const FRAMES = %(frames)s;
const grid = document.getElementById("grid"), big = document.getElementById("big");
const marked = new Set();
let sel = 0;
FRAMES.forEach((f, i) => {
  const b = document.createElement("button");
  b.className = "cell";
  b.dataset.i = i;
  b.innerHTML = '<img loading="lazy" alt="' + f.name + '" src="/frames/' + f.name + '.jpg">' +
    '<span class="tag' + (f.faces ? ' face' : '') + '">' + (i + 1) + (f.faces ? ' &middot; faces: ' + f.faces : '') + '</span>';
  b.onclick = () => { select(i); toggle(i); };
  grid.appendChild(b);
});
const cells = grid.children;
function update() {
  document.getElementById("count").textContent = FRAMES.length + " frames, " + marked.size + " marked for deletion";
}
function select(i) {
  if (!cells.length) return;
  sel = Math.max(0, Math.min(cells.length - 1, i));
  for (const c of cells) c.classList.remove("sel");
  cells[sel].classList.add("sel");
  cells[sel].focus();
}
function toggle(i) {
  const n = FRAMES[i].name;
  marked.has(n) ? marked.delete(n) : marked.add(n);
  cells[i].classList.toggle("del", marked.has(n));
  update();
}
function columns() {  // cells on the first row of the grid
  let n = 0;
  for (const c of cells) { if (c.offsetTop !== cells[0].offsetTop) break; n++; }
  return Math.max(1, n);
}
async function apply() {
  if (!confirm("Delete " + marked.size + " frame(s) and finish the review?")) return;
  const r = await fetch("/apply", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify({"delete": [...marked]})});
  const j = await r.json();
  document.body.innerHTML = r.ok
    ? "<h2>Review saved: " + j.deleted + " frame(s) deleted, " + j.kept + " kept. You can close this page.</h2>"
    : "<h2>Error: " + j.error + "</h2>";
}
document.getElementById("apply").onclick = apply;
document.addEventListener("keydown", e => {
  if (big.style.display === "flex") {
    if (e.key === "Escape" || e.key === "Enter") { big.style.display = "none"; e.preventDefault(); }
    return;
  }
  const k = e.key;
  if (k === "ArrowRight") select(sel + 1);
  else if (k === "ArrowLeft") select(sel - 1);
  else if (k === "ArrowDown") select(sel + columns());
  else if (k === "ArrowUp") select(sel - columns());
  else if (k === " " || k === "x" || k === "X") toggle(sel);
  else if (k === "Enter" && e.shiftKey) apply();
  else if (k === "Enter" && cells.length) {
    big.querySelector("img").src = "/frames/" + FRAMES[sel].name + ".jpg";
    big.style.display = "flex";
  } else return;
  e.preventDefault();
});
update();
select(0);
</script></body></html>
"""


def make_server(session_dir, port=8765, on_done=None):
    """The review HTTP server on 127.0.0.1 (port 0: any free port). on_done(deleted, kept) after an apply."""
    session_dir = os.path.abspath(session_dir)
    sid = load_session(session_dir)["session_id"]

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

        def do_GET(self):
            frames = load_frames(session_dir)
            if self.path == "/":
                order = [{"name": r["name"], "faces": r.get("faces", 0)} for r in review_order(frames)]
                page = PAGE % {"session": html.escape(sid), "frames": json.dumps(order)}
                return self.send(200, page.encode(), "text/html; charset=utf-8")
            # only frames of the session are served (the name is checked against the list: no path traversal)
            name = self.path[len("/frames/"):-len(".jpg")] if self.path.startswith("/frames/") else None
            if name and self.path.endswith(".jpg") and name in {r["name"] for r in frames}:
                with open(frame_path(session_dir, name), "rb") as f:
                    return self.send(200, f.read(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/apply":
                return self.send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                deleted, kept = apply_review(session_dir, body.get("delete", []))
            except (ValueError, KeyError) as e:
                return self.send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            self.send(200, json.dumps({"deleted": deleted, "kept": kept}).encode(), "application/json")
            if on_done is not None:
                on_done(deleted, kept)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="staged session folder (privacy pass done)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not (load_session(args.session).get("privacy") or {}).get("done"):
        raise SystemExit("run the privacy pass (python -m floor_seg.dataset.privacy) first")

    def done(deleted, kept):
        print(f"review applied: {deleted} deleted, {kept} kept")
        threading.Thread(target=server.shutdown).start()  # shutdown() blocks: not from the handler's thread

    server = make_server(args.session, args.port, done)
    print(f"review page: http://127.0.0.1:{server.server_address[1]}  (Ctrl+C to quit without applying)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("quit, nothing applied")
    server.server_close()


if __name__ == "__main__":
    main()
