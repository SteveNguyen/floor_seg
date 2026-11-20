"""Structure augmentations for real-robot failure cases, painted on existing ProcTHOR renders at training time.

Three failures seen on the robot (camera ~0.27 m high, pitched 15-20 deg down) that the renders never show:
  1. plinths / skirting boards made of the floor's own tiles read as floor (false floor: drives into walls);
  2. door thresholds / floor-transition strips read as obstacles (the robot refuses doors);
  3. sun streaks and specular highlights on glossy tiles read as obstacles.

Geometry: every render has <name>_depth.png (optical z, mm) and stats.jsonl gives the camera height h (m), pitch
and roll (deg); with the calibrated lens (floor_tex.lens_rays) each pixel's ray is known, so each pixel's 3D point
in the camera-yaw frame is (gx right, gz forward, height above the floor), all linear in the stored depth
(floor_tex.floor_coords / heights). Everything below is placed in metres with that geometry; no re-rendering.

  plinth     wall bases are found from the render: non-floor pixels just above a floor pixel whose 3D point
             touches the floor (< contact_h), followed upward while the surface stays vertical (horizontal
             distance changes much less than height, pixel to pixel), and kept only if that vertical surface rises
             >= wall_min_h or leaves the image (a wall or a tall cabinet, not a box). The band of those pixels
             below a metric height (plinth_h) is painted with the image's own floor texture (a top-down patch
             of the floor near the camera, cut by inverse perspective and laid along the wall in metres: same
             tiles, same scale) or, with p_plinth_plain, a plain skirting colour; thin top edge and bottom joint.
             Label: obstacle (it was a wall pixel already: no label changes).
  threshold  a strip of another material (wood, metal, stone, tile, dark vinyl), thr_width deep, at 0.5-2.5 m
             ahead, roughly across the view (+-thr_angle), door-wide (p_door) or spanning the room (then with
             p_transition the floor beyond it gets another tiling, as at a room transition). Door openings are
             not detected (renders store no door geometry); a random placement on visible floor teaches the
             same appearance cue. The strip is a box ray-traced against every pixel ray with the stored depth
             as z-buffer: flush / low (flush_h, <= 1.2 cm) keeps the floor label (drivable, label rule); with
             p_step a raised step (step_h, >= 2.5 cm) shows its front face and is labelled obstacle, so the
             model has to tell them apart by geometry, not by "different material across the floor".
  light      sun streaks: window light on the floor in metres (panes with mullion shadows, or blind slats),
             multiplicative gain + additive warm light so even dark tiles saturate; specular highlights: window
             reflections on glossy tiles, image-space elongated quads from the far floor toward the camera,
             screen-blended to white. Labels unchanged. floor_tex (with --tex) already has one hard-edged sun
             parallelogram and a soft additive glare blob; these add the saturated, striped and reflection-shaped
             cases that failed on dark shiny tiles, and work without --tex.

Labels: only pixels painted as a raised step change (-> obstacle). Floor paint goes only on floor pixels that
are not 'under' (train.UNDER), so the under channel stays consistent (no strip, plinth or light under a bed);
steps whose box would cover 'under' pixels are rejected. Deterministic given the random.Random passed in.

    sa = StructAug(StructAugCfg(), floor_seg.DEFAULT_CALIB)
    bgr, label, info = sa.apply(bgr, label, depth_mm, pitch_deg, roll_deg, cam_h_m, random.Random(seed))
"""
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np

from floor_seg import DEFAULT_CALIB
from floor_seg.floor_tex import H, W, FloorTex, floor_coords, heights, lens_rays, procedural_tiles, pyramid

UNDER = 2  # = floor_seg.train.UNDER (not imported: train imports this module, and pulls in torch)


@dataclass
class StructAugCfg:
    # plinth / skirting board on wall bases (labelled obstacle)
    p_plinth: float = 0.3
    plinth_h: tuple = (0.05, 0.12)       # m, plinth height
    p_plinth_plain: float = 0.3          # plain skirting colour instead of the floor's own texture
    wall_min_h: float = 0.4              # m, a vertical surface rising this high (or out of the image) is a wall
    contact_h: float = 0.03              # m, a non-floor pixel this close to the floor touches it
    # threshold / doorstep strip across the floor
    p_threshold: float = 0.3
    thr_dist: tuple = (0.5, 2.5)         # m ahead (gz) of a point of the strip
    thr_width: tuple = (0.03, 0.10)      # m, strip depth on the floor
    thr_angle: float = 25.0              # deg, strip direction around "across the view"
    p_door: float = 0.5                  # door-wide strip, else it spans the room
    door_len: tuple = (0.7, 1.0)         # m
    flush_h: tuple = (0.0, 0.012)        # m, flush / low threshold: floor (robot drives over it)
    p_step: float = 0.35                 # raised step instead: obstacle
    step_h: tuple = (0.025, 0.06)        # m
    p_transition: float = 0.3            # room-spanning strip: other flooring beyond it
    # light on the floor (labels unchanged)
    p_sun: float = 0.25
    p_specular: float = 0.2


def lens_K_D(calib_path):
    """K (scaled to the W x H render) and distortion of the calibrated lens, as in floor_tex.lens_rays."""
    cal = json.load(open(calib_path))
    sx, sy = W / cal["image_resolution"][0], H / cal["image_resolution"][1]
    return np.array(cal["K"]) * np.array([[sx], [sy], [1]]), np.array(cal["D"], np.float64)


def strip_texture(rng, kinds=("wood", "metal", "stone", "tile", "dark")):
    """Threshold strip material: (1024 px BGR uint8 texture, its size in m). Procedural, no texture files."""
    kind = rng.choice(kinds)
    if kind == "tile":
        return procedural_tiles(rng)
    g = np.random.default_rng(rng.randint(0, 2**31))
    n = 1024
    if kind == "wood":   # grain: 1-D noise along x, stretched along y, plus fine noise
        base = np.array(rng.choice([(0.25, 0.4, 0.6), (0.15, 0.25, 0.4), (0.4, 0.55, 0.7), (0.1, 0.15, 0.25)]))
        grain = cv2.GaussianBlur(g.normal(0, 1, (1, n)).astype(np.float32), (0, 0), 3).repeat(n, 0)
        tex = base * (1 + 0.15 * grain + 0.04 * g.normal(0, 1, (n, n)))[..., None]
        size = rng.uniform(0.3, 0.6)
    elif kind == "metal":  # brushed aluminium / brass
        base = np.array(rng.choice([(0.7, 0.7, 0.72), (0.5, 0.5, 0.52), (0.35, 0.6, 0.75)]))
        brush = cv2.GaussianBlur(g.normal(0, 1, (n, n)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=0.8)
        tex = base * (1 + 0.6 * brush)[..., None]
        size = rng.uniform(0.15, 0.3)
    elif kind == "stone":  # marble / granite: multi-scale noise
        base = np.array(rng.choice([(0.75, 0.75, 0.75), (0.6, 0.65, 0.7), (0.35, 0.35, 0.37), (0.85, 0.85, 0.8)]))
        nz = sum(cv2.GaussianBlur(g.normal(0, 1, (n, n)).astype(np.float32), (0, 0), s) * s ** 0.5
                 for s in (1, 4, 16))
        tex = base * (1 + 0.08 * nz)[..., None]
        size = rng.uniform(0.3, 0.6)
    else:  # dark rubber / vinyl
        tex = np.full((n, n, 3), rng.uniform(0.06, 0.2)) * (1 + 0.05 * g.normal(0, 1, (n, n)))[..., None]
        size = 0.3
    return (np.clip(tex, 0, 1) * 255).astype(np.uint8), size


def _slab(d, lo, hi):
    """Ray parameter interval where lo <= t * d <= hi (per pixel); rays parallel to the slab: all or nothing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / d, hi / d
    par = np.abs(d) < 1e-9
    inside = lo <= 0 <= hi
    tmin = np.where(par, -np.inf if inside else np.inf, np.minimum(t1, t2))
    tmax = np.where(par, np.inf if inside else -np.inf, np.maximum(t1, t2))
    return tmin, tmax


def floor_shade(img, free):
    """Floor shading (light falloff, shadows): blurred floor luminance relative to its median (as floor_tex)."""
    mf = free.astype(np.float32)
    lb = cv2.GaussianBlur(img.mean(axis=2) * mf, (0, 0), 4) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 4), 1e-3)
    return np.clip(lb / max(np.median(lb[free]), 1e-3), 0.15, 2.5)[..., None]


class Geom:
    """Per-pixel ray geometry of one render. A pixel's ray at optical depth t reaches the 3D point
    (t * gx1, t * gz1) on the floor plane axes, at height cam_h + t * dh1 (camera-yaw frame, floor_tex conventions)."""

    def __init__(self, xn, yn, depth_mm, pitch, roll, cam_h):
        self.pitch, self.roll, self.cam_h = pitch, roll, cam_h
        self.gx1, self.gz1 = floor_coords(xn, yn, 1.0, pitch, roll)
        self.dh1 = heights(xn, yn, 1.0, pitch, roll, 0.0)
        self.valid = depth_mm > 0
        self.depth = depth_mm.astype(np.float32) / 1000
        self.gx, self.gz = self.gx1 * self.depth, self.gz1 * self.depth
        self.hgt = cam_h + self.dh1 * self.depth
        # where the ray meets the floor plane (rays at or above the horizon: far away, clipped)
        with np.errstate(divide="ignore"):
            self.tf = np.where(self.dh1 < -1e-6, -cam_h / self.dh1, 50.0).clip(0, 50).astype(np.float32)


class StructAug:
    # FloorTex.sample (mip-mapped texture lookup at metric coordinates, random rotation / offset) uses no
    # FloorTex state: reused as is
    sample = FloorTex.sample

    def __init__(self, cfg=None, calib_path=DEFAULT_CALIB):
        self.cfg = cfg or StructAugCfg()
        self.xn, self.yn = lens_rays(calib_path)
        self.K, self.D = lens_K_D(calib_path)

    def project(self, gx, gz, hgt, geo):
        """Pixel coords (u, v) of 3D points given in the camera-yaw frame: the inverse of Geom; nan behind."""
        p, r = math.radians(geo.pitch), math.radians(geo.roll)
        hp = hgt - geo.cam_h
        z = gz * math.cos(p) - hp * math.sin(p)          # undo the pitch (floor_tex.floor_coords / heights)
        y1 = gz * math.sin(p) + hp * math.cos(p)
        x = gx * math.cos(r) + y1 * math.sin(r)          # undo the roll
        y = -gx * math.sin(r) + y1 * math.cos(r)
        pts = np.stack([x, -y, z], -1).reshape(-1, 1, 3).astype(np.float64)   # optical frame: y down
        uv = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(gx.shape + (2,))
        uv[z <= 0.05] = np.nan
        return uv[..., 0], uv[..., 1]

    def apply(self, bgr, label, depth_mm, pitch, roll, cam_h, rng):
        """bgr: H x W x 3 uint8 render (after floor_tex if used); label: H x W uint8, 0 = obstacle, any other
        value floor (train.encode_labels, or 0/1). Returns (bgr, label, info): new arrays only when something was
        painted; info lists what was painted (preview, tests)."""
        c, info = self.cfg, {}
        geo = Geom(self.xn, self.yn, depth_mm, pitch, roll, cam_h)
        free = geo.valid & (label != 0) & (label != UNDER)   # floor that may be painted
        if free.mean() < 0.05:
            return bgr, label, info
        img, lab = bgr.astype(np.float32) / 255, label.copy()
        # decisions drawn up front: the probability of each augmentation does not depend on the others
        do = [rng.random() < p for p in (c.p_plinth, c.p_threshold, c.p_sun, c.p_specular)]
        if do[0]:
            info["plinth"] = self.plinth(img, lab, geo, rng)
        if do[1]:
            info["threshold"] = self.threshold(img, lab, geo, free, rng)
            free &= lab != 0                                  # a raised step is no longer floor
        if do[2]:
            info["sun"] = self.sun(img, free, geo, rng)
        if do[3]:
            info["specular"] = self.specular(img, free, rng)
        if not any(info.values()):
            return bgr, label, info
        return (np.clip(img, 0, 1) * 255).astype(np.uint8), lab, info

    # ---------------------------------------------------------------- plinth
    def wall_base(self, lab, geo):
        """Non-floor pixels on vertical surfaces standing on the floor and rising like a wall. Returns
        (mask, start) where start[y, x] is the image row of the floor contact of the pixel's vertical run."""
        c = self.cfg
        nf = geo.valid & (lab == 0)
        floor = lab != 0
        r = np.hypot(geo.gx, geo.gz)
        hgt = geo.hgt
        # run start: a non-floor pixel right above a floor pixel, whose 3D point touches the floor
        start = np.zeros((H, W), bool)
        start[:-1] = nf[:-1] & floor[1:] & (np.abs(hgt[:-1]) < c.contact_h)
        run = np.full((H, W), -1, np.int32)
        cur = np.full(W, -1, np.int32)
        cols = np.arange(W)
        for y in range(H - 2, -1, -1):
            # walk up each column while the surface stays vertical: the horizontal distance stays within
            # 1 cm + 1 % (depth quantization, the slight slant of an image column) + half the height climbed
            # of the run's floor contact. Compared to the contact, not pixel to pixel: 1 mm depth steps at 3 m
            # are as large as the per-pixel height step. A horizontal top (box, table) fails at once.
            c0 = np.maximum(cur, 0)
            r0, h0 = r[c0, cols], hgt[c0, cols]
            vert = nf[y] & (hgt[y] >= hgt[y + 1] - 0.002) & \
                (np.abs(r[y] - r0) < 0.01 + 0.01 * r0 + 0.5 * (hgt[y] - h0))
            cur = np.where((cur >= 0) & vert, cur, -1)
            cur = np.where(start[y], y, cur)
            run[y] = cur
        m = run >= 0
        key = run * W + np.arange(W)[None, :]   # one key per (start row, column) run
        top_h = np.full(H * W, -np.inf)
        np.maximum.at(top_h, key[m], hgt[m])
        top_row = np.full(H * W, H)
        np.minimum.at(top_row, key[m], np.broadcast_to(np.arange(H)[:, None], (H, W))[m])
        wall = (top_h >= c.wall_min_h) | (top_row == 0)   # rises high enough, or leaves the image
        return m & wall[np.where(m, key, 0)], run

    def floor_patch(self, img, free, geo, rng, ppm=256.0, size=0.5):
        """Top-down patch (size m square, ppm px/m) of the visible floor near the camera, by inverse
        perspective, with its lighting flattened: the floor's own texture at a known metric scale. None if no
        clean square of floor is found."""
        cand = free & (geo.gz > 0.3) & (geo.gz < 1.5)
        ys, xs = np.nonzero(cand)
        if len(ys) < 100:
            return None
        n = int(size * ppm)
        off = (np.arange(n, dtype=np.float32) + 0.5) / ppm - size / 2
        for _ in range(6):
            i = rng.randrange(len(ys))
            px, pz = np.meshgrid(geo.gx[ys[i], xs[i]] + off, geo.gz[ys[i], xs[i]] - off)  # rows: far -> near
            u, v = self.project(px, pz, np.zeros_like(px), geo)
            ui, vi = np.nan_to_num(np.round(u), nan=-1).astype(int), np.nan_to_num(np.round(v), nan=-1).astype(int)
            ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ok[ok] = free[vi[ok], ui[ok]]
            if ok.mean() < 0.95:
                continue
            patch = cv2.remap(img, np.nan_to_num(u, nan=-1).astype(np.float32),
                              np.nan_to_num(v, nan=-1).astype(np.float32), cv2.INTER_LINEAR)
            patch[~ok] = np.median(patch[ok], axis=0)
            lum = patch.mean(axis=2)
            flat = cv2.GaussianBlur(lum, (0, 0), n / 6) / max(lum.mean(), 1e-3)  # shadows, light falloff
            return patch / np.maximum(flat, 0.2)[..., None]
        return None

    def plinth(self, img, lab, geo, rng):
        c = self.cfg
        hp = rng.uniform(*c.plinth_h)
        base, run = self.wall_base(lab, geo)
        band = base & (geo.hgt <= hp) & (geo.hgt > -0.02)
        if band.sum() < 200:
            return None
        free = geo.valid & (lab != 0) & (lab != UNDER)
        patch = None if rng.random() < c.p_plinth_plain else self.floor_patch(img, free, geo, rng)
        # along-wall coordinate (m) per column: arc length of the floor-contact line, from column to column
        # (restarting at jumps between walls), so the texture keeps its metric scale on oblique walls
        cols = np.nonzero(band.any(axis=0))[0]
        row0 = run[H - 1 - np.argmax(band[::-1, cols], axis=0), cols]   # contact row of each column's lowest band px
        cx, cz = geo.gx[row0, cols], geo.gz[row0, cols]
        step = np.hypot(np.diff(cx), np.diff(cz))
        step = np.where(step > 0.15, rng.uniform(0, 1), step)
        s_col = np.zeros(W, np.float32)
        s_col[cols] = np.concatenate([[0], np.cumsum(step)]) + rng.uniform(0, 10)
        s = np.broadcast_to(s_col[None, :], (H, W))
        if patch is not None:   # floor tiles laid on the wall, mirrored at the patch borders (no seams)
            ppm, n = 256.0, patch.shape[0]
            def mirror(t):
                t = np.mod(t, 2 * n)
                return np.where(t >= n, 2 * n - 1 - t, t).astype(np.float32)
            a = mirror(s * ppm)
            b = mirror((hp - geo.hgt) * ppm + rng.uniform(0, n))
            col = cv2.remap(patch, a, b, cv2.INTER_LINEAR) * rng.uniform(0.6, 1.0)  # vertical face: less light
            edge_gain = rng.choice([0.55, 1.3])         # grout / cut tile edge, darker or lighter
        else:   # plain skirting: white, grey, wood, black, or the floor's own median colour
            colour = rng.choice([(0.85, 0.87, 0.88), (0.6, 0.6, 0.6), (0.2, 0.35, 0.5), (0.1, 0.1, 0.1), None])
            colour = np.median(img[free], axis=0) if colour is None or not free.any() else np.array(colour)
            g = np.random.default_rng(rng.randint(0, 2**31))
            col = colour.astype(np.float32) * (1 + 0.03 * g.normal(0, 1, (H, W, 1))).astype(np.float32)
            edge_gain = 1.3                              # lit top bevel
        # the wall's own shading (light gradient, shadows of furniture) carried onto the plinth
        bf = band.astype(np.float32)
        lb = cv2.GaussianBlur(img.mean(axis=2) * bf, (0, 0), 4) / np.maximum(cv2.GaussianBlur(bf, (0, 0), 4), 1e-3)
        shade = np.clip(lb / max(np.median(lb[band]), 1e-3), 0.3, 2.0)[..., None]
        col = col * shade
        top = band & ((geo.hgt > hp - rng.uniform(0.003, 0.008)) | ~np.roll(band, 1, axis=0))
        col[top] *= edge_gain
        col[band & (geo.hgt < 0.004)] *= 0.6             # joint / shadow line at the floor
        img[band] = col[band]
        lab[band] = 0                                    # already obstacle (wall pixels): explicit
        return {"h": hp, "plain": patch is None, "px": int(band.sum())}

    # ---------------------------------------------------------------- threshold strip
    def threshold(self, img, lab, geo, free, rng):
        c = self.cfg
        cand = free & (geo.gz > c.thr_dist[0]) & (geo.gz < c.thr_dist[1])
        ys, xs = np.nonzero(cand)
        if len(ys) < 50:
            return None
        raised = rng.random() < c.p_step
        top = max(rng.uniform(*(c.step_h if raised else c.flush_h)), 1e-3)
        width = rng.uniform(*c.thr_width)
        door = rng.random() < c.p_door
        length = rng.uniform(*c.door_len) if door else 8.0
        for _ in range(6):
            i = rng.randrange(len(ys))
            cx, cz = geo.gx[ys[i], xs[i]], geo.gz[ys[i], xs[i]]
            th = math.radians(rng.uniform(-c.thr_angle, c.thr_angle))
            a, nrm = (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))   # along, across (forward-ish)
            uc = a[0] * cx + a[1] * cz
            v0 = nrm[0] * cx + nrm[1] * cz - width / 2
            v1 = v0 + width
            du = a[0] * geo.gx1 + a[1] * geo.gz1          # along / across coordinate per unit ray depth
            dv = nrm[0] * geo.gx1 + nrm[1] * geo.gz1
            # ray / box intersection (slabs: height, across, along); the box is entered on the face whose
            # slab is entered last
            ts = [_slab(geo.dh1, -geo.cam_h, top - geo.cam_h), _slab(dv, v0, v1),
                  _slab(du, uc - length / 2, uc + length / 2)]
            tmin = np.stack([t[0] for t in ts])
            tin, tout = tmin.max(0), np.min(np.stack([t[1] for t in ts]), 0)
            face = tmin.argmax(0)                         # 0 top, 1 front / back, 2 ends
            # z-buffer against the render (tolerance: the flush top is within mm of the floor depth)
            hit = (tin <= tout) & (tin > 0) & geo.valid & (tin < geo.depth * 1.005 + 0.01)
            if (hit & free).sum() < 40:
                continue
            if raised:
                # no step over 'under' floor, and mostly on visible floor (not inside a sofa)
                if (hit & (lab == UNDER)).any() or (hit & free).sum() < 0.6 * hit.sum():
                    continue
                paint = hit
            else:
                paint = hit & free
            break
        else:
            return None
        t = np.where(hit, tin, geo.tf)                    # off the box: the ray's floor point (smooth mip levels)
        u, v, h = t * du, t * dv, geo.cam_h + t * geo.dh1
        ca = np.where(face == 2, v, u)                    # texture coords: (along, across) on top,
        cb = np.where(face == 0, v, h)                    # (along, height) on the front, (across, height) at ends
        tex, size = strip_texture(rng)
        col = self.sample(pyramid(tex), ca, cb, size, rng)
        col *= np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32) * rng.uniform(0.6, 1.1)
        col[face == 1] *= rng.uniform(0.4, 0.75)          # vertical faces get less light
        col[face == 2] *= 0.6
        j = 0.004
        col[(face == 0) & ((np.abs(v - v0) < j) | (np.abs(v - v1) < j))] *= 0.5   # joints with the floor
        shade = floor_shade(img, free)
        img[paint] = (col * shade)[paint]
        info = {"raised": raised, "h": top, "width": width, "door": door, "th": th, "uc": uc, "v0": v0,
                "v1": v1, "len": length, "px": int(paint.sum())}
        if raised:
            lab[paint] = 0                                # > 2 cm: an obstacle, front face and top
        elif not door and rng.random() < c.p_transition:  # other flooring beyond a room-spanning strip
            beyond = free & ~paint & (geo.tf * dv > v1)
            if beyond.any():
                tex2, size2 = strip_texture(rng, ("tile", "tile", "wood", "stone"))   # floorings
                col2 = self.sample(pyramid(tex2), geo.tf * du, geo.tf * dv, size2, rng) * rng.uniform(0.5, 1.1)
                img[beyond] = (col2 * shade)[beyond]
                info["transition_px"] = int(beyond.sum())
        return info

    # ---------------------------------------------------------------- light
    def sun(self, img, free, geo, rng):
        """Window light on the floor (m): a sheared rectangle split into panes by mullion shadows, or into thin
        slats (blinds); overexposed (gain + additive warm light: dark tiles saturate too); soft edges."""
        ys, xs = np.nonzero(free)
        if len(ys) == 0:
            return None
        i = rng.randrange(len(ys))
        cx, cz = geo.gx[ys[i], xs[i]], geo.gz[ys[i], xs[i]]
        th, shear = rng.uniform(0, math.pi), rng.uniform(-0.8, 0.8)
        u = math.cos(th) * (geo.gx - cx) - math.sin(th) * (geo.gz - cz)
        v = math.sin(th) * (geo.gx - cx) + math.cos(th) * (geo.gz - cz)
        length, width = rng.uniform(0.4, 2.0), rng.uniform(0.2, 1.0)
        w = u + shear * v + width / 2                     # 0 .. width across the patch
        inside = (np.abs(v) < length / 2) & (w > 0) & (w < width)
        blinds = rng.random() < 0.3
        if blinds:      # slats across the patch length
            period = rng.uniform(0.03, 0.1)
            lit = np.mod(v / period, 1) < rng.uniform(0.35, 0.7)
        else:           # 1-4 panes, mullions 3-8 cm
            k = rng.choice([1, 2, 3, 4])
            pane = width / k
            lit = np.mod(w, pane) > min(rng.uniform(0.03, 0.08), pane / 3)
        mask = (inside & lit & free).astype(np.float32)
        soft = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.7, 3.0)) * free
        warm = np.array([rng.uniform(0.75, 0.95), rng.uniform(0.9, 1.0), 1.0], np.float32)  # BGR
        gain = rng.uniform(1.8, 4.0)
        img *= 1 + (gain * warm - 1) * soft[..., None]
        img += rng.uniform(0.05, 0.4) * warm * soft[..., None]
        return {"blinds": blinds, "gain": gain, "px": int((soft > 0.5).sum())}

    def specular(self, img, free, rng):
        """Reflections of a window on glossy tiles: 1-3 bright quads side by side, from the far floor (top of the
        floor in that column) toward the camera, wider and fading as they come closer; screen-blended (to white)."""
        cols = np.nonzero(free.any(axis=0))[0]
        if len(cols) == 0:
            return None
        blob = np.zeros((H, W), np.float32)
        x0, w0, gap = int(rng.choice(cols)), rng.randint(8, 50), rng.randint(3, 12)
        for k in range(rng.choice([1, 1, 2, 3])):
            xc = x0 + k * (w0 + gap)
            if xc >= W or not free[:, xc].any():
                break
            ytop = int(np.argmax(free[:, xc])) + rng.randint(0, 10)
            n = rng.randint(30, 170)
            wb = w0 * rng.uniform(1.0, 1.8)
            one = np.zeros((H, W), np.float32)
            quad = np.array([[xc - w0 / 2, ytop], [xc + w0 / 2, ytop], [xc + wb / 2, ytop + n],
                             [xc - wb / 2, ytop + n]], np.int32)
            cv2.fillConvexPoly(one, quad, 1.0)
            fade = np.clip(1 - (np.arange(H) - ytop) / n * (1 - rng.uniform(0.1, 0.7)), 0, 1)
            blob = np.maximum(blob, one * fade[:, None].astype(np.float32))
        blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(2, 8)) * free * rng.uniform(0.5, 1.0)
        img += (1 - np.clip(img, 0, 1)) * blob[..., None]
        return {"px": int((blob > 0.3).sum())} if blob.max() > 0 else None
