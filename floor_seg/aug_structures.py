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
             below a metric height (plinth_h, 8-20 cm) is painted as a skirting board, one style per image
             (plinth_style): the wall's own colour a shade off, white, wood grain along the board, the image's
             own floor tiles (a top-down patch of the floor near the camera, cut by inverse perspective and laid
             along the wall in metres: same tiles, same scale), or plain grey / black. The board stands ~1 cm
             proud of the wall: lit top edge, dark joint line on the wall above it, dark contact line at the floor.
             Label: obstacle (it was a wall pixel already: no label changes).
  wall tiles same wall bases, tiled (bathroom, kitchen) from the floor up to wall_tile_h with procedural tiles
             unlike the floor's; instead of a plinth. Label: obstacle (unchanged).
  threshold  a strip of another material (wood, metal, stone, tile, dark vinyl), thr_width deep, at 0.5-2.5 m
             ahead, roughly across the view (+-thr_angle), door-wide (p_door) or spanning the room (then with
             p_transition the floor beyond it gets another tiling, as at a room transition). Door openings are
             not detected (renders store no door geometry); a random placement on visible floor teaches the
             same appearance cue. The strip is a box ray-traced against every pixel ray with the stored depth
             as z-buffer; flush / low (flush_h, <= 1.2 cm): keeps the floor label (drivable, label rule).
             (A raised box strip lying on the floor was dropped: it has hardly any real equivalent; the real
             "not drivable" case is a level change, below.)
  level      a straight edge across the floor (same placement as the strips), beyond which the floor is dz
             higher (step up) or lower (step down), room-wide (a door-wide one, without door jambs in the
             render, looks like a platform / trench in the room; across a rendered doorway the room-wide edge
             gives the real picture). Labels follow the robot's rule
             (only floor at its own level is drivable): a real step, |dz| 10-20 cm -> riser and the floor beyond
             are obstacle; with p_level_flat (half of them) |dz| <= 1 cm, a thin edge line only, everything stays
             floor, mostly with another flooring beyond (p_flat_other), so the model has to learn the edge
             geometry, not "an edge or a new flooring across the floor = stop". Nothing in between: a 2-8 cm step
             is rare in homes and hardly visible in one image (v8 trained on them and refused flat doorways).
             Step up: a box (edge .. 30 m beyond, height dz) ray-traced with the stored depth as z-buffer:
             objects closer than its surface keep occluding it; it covers the bottom dz of whatever stands
             beyond (walls then start on the raised floor, as they would). Faces: the riser (vertical, with a
             lit nosing) and the raised top.
             Step down: rays that met visible floor beyond the edge continue to the lower plane (farther: the
             floor slides up in the image); the riser faces away (invisible), the near edge shows as a lip, the
             lower floor is darker along the riser (occlusion).
             Floor appearance on the shifted plane = the original image at the same (gx, gz) on the original
             plane, i.e. a plane-to-plane reprojection (Geom -> project -> remap); where that point is hidden or
             off-image: the floor patch below (same texture, metric scale); with p_level_other another flooring.
             Approximation: walls and objects standing beyond the edge are not moved: on a step up their base is
             covered by the raised floor; on a step down a lower-floor point that projects onto a wall / object
             standing beyond the edge takes that surface's colour (it is extruded down), labelled obstacle anyway.
  light      sun streaks: window light on the floor in metres (panes with mullion shadows, or blind slats),
             multiplicative gain + additive warm light so even dark tiles saturate; specular highlights: window
             reflections on glossy tiles, image-space elongated quads from the far floor toward the camera,
             screen-blended to white. Labels unchanged. floor_tex (with --tex) already has one hard-edged sun
             parallelogram and a soft additive glare blob; these add the saturated, striped and reflection-shaped
             cases that failed on dark shiny tiles, and work without --tex. Off by default (p_sun = p_specular =
             0): with them, v8 called bright white wall bases and plinths floor on real frames.

Labels: only level changes (real steps) change labels (floor -> obstacle). 'under' pixels (train.UNDER) are
never painted (no strip, plinth, step or light under a bed); beyond a non-drivable level edge they become obstacle
(unreachable floor), unpainted. Deterministic given the random.Random passed in.

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
    # plinth / skirting board on wall bases (labelled obstacle). Most home walls have one: a thin strip (~1 cm
    # deep) 8-20 cm tall, painted to blend with the wall or white, sometimes bare wood, rarely tiles matching the
    # floor (v8 had 70 % floor tiles and 5-12 cm: it did not help on real plinths)
    p_plinth: float = 0.6
    plinth_h: tuple = (0.08, 0.20)       # m, plinth height
    # style weights: wall colour (a shade off), white, wood, the floor's own tiles, other plain (grey / black)
    plinth_style: tuple = (0.4, 0.3, 0.15, 0.1, 0.05)
    wall_min_h: float = 0.4              # m, a vertical surface rising this high (or out of the image) is a wall
    contact_h: float = 0.03              # m, a non-floor pixel this close to the floor touches it
    # tiled wall base (bathroom, kitchen): tiles unlike the floor's up to wall_tile_h (labelled obstacle); a wall
    # gets either this or a plinth
    p_wall_tiles: float = 0.1
    wall_tile_h: tuple = (0.5, 2.0)      # m (above the image top: the whole visible wall)
    # threshold / doorstep strip across the floor (labelled floor)
    p_threshold: float = 0.3
    thr_dist: tuple = (0.5, 2.5)         # m ahead (gz) of a point of the strip / level edge
    thr_width: tuple = (0.03, 0.10)      # m, strip depth on the floor
    thr_angle: float = 25.0              # deg, strip / edge direction around "across the view"
    p_door: float = 0.5                  # door-wide strip / level change, else it spans the room
    door_len: tuple = (0.7, 1.0)         # m
    flush_h: tuple = (0.0, 0.012)        # m, flush / low threshold: floor (robot drives over it)
    p_transition: float = 0.3            # room-spanning strip: other flooring beyond it
    # level change: the floor beyond a straight edge is higher or lower (placement as the strips). Real homes have
    # drivable doorsteps or real steps, hardly anything in between: steps are 10-20 cm (a riser or a drop edge the
    # image shows). v8 used 2-18 cm, mostly with no visible cue, and learned the shortcut "other flooring past a
    # line = obstacle" (it refused flat doorways). Flat flooring changes, labelled floor, are now as frequent as
    # steps and mostly show another flooring, so a texture change alone no longer means obstacle.
    p_level: float = 0.3
    p_level_up: float = 0.5              # step up, else step down
    level_dz: tuple = (0.10, 0.20)       # m, |dz| of a step: riser and floor beyond are obstacle
    p_level_flat: float = 0.5            # drivable level change instead: |dz| <= flat_dz, stays floor
    flat_dz: float = 0.01                # m
    p_level_other: float = 0.3           # step: other flooring beyond the edge
    p_flat_other: float = 0.7            # flat change: other flooring beyond the edge
    # light on the floor (labels unchanged). Off by default: with them on, v8 called bright white wall bases and
    # plinths floor on real frames. Kept for experiments (train.py --aug-p-sun / --aug-p-specular).
    p_sun: float = 0.0
    p_specular: float = 0.0


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


def _mirror(t, n):
    """Texture coordinate t (px) mirrored into 0 .. n-1: tiles a patch without seams."""
    t = np.mod(t, 2 * n)
    return np.where(t >= n, 2 * n - 1 - t, t).astype(np.float32)


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
        do = [rng.random() < p for p in (c.p_plinth, c.p_threshold, c.p_level, c.p_sun, c.p_specular,
                                         c.p_wall_tiles)]
        if do[5]:     # a tiled wall base has no plinth on top of it
            info["wall_tiles"] = self.wall_tiles(img, lab, geo, rng)
        elif do[0]:
            info["plinth"] = self.plinth(img, lab, geo, rng)
        if do[1]:
            info["threshold"] = self.threshold(img, lab, geo, free, rng)
        if do[2]:
            info["level"] = self.level(img, lab, geo, free, rng)
            free &= lab != 0                                  # floor beyond a step is no longer floor
        if do[3]:
            info["sun"] = self.sun(img, free, geo, rng)
        if do[4]:
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

    @staticmethod
    def along_wall(band, run, geo, rng):
        """Along-wall coordinate (m) of every pixel of a wall-base band: arc length of the floor-contact line from
        column to column (restarting at jumps between walls), so a texture keeps its metric scale on oblique
        walls. H x W, constant per column."""
        cols = np.nonzero(band.any(axis=0))[0]
        row0 = run[H - 1 - np.argmax(band[::-1, cols], axis=0), cols]   # contact row of each column's lowest band px
        cx, cz = geo.gx[row0, cols], geo.gz[row0, cols]
        step = np.hypot(np.diff(cx), np.diff(cz))
        step = np.where(step > 0.15, rng.uniform(0, 1), step)
        s_col = np.zeros(W, np.float32)
        s_col[cols] = np.concatenate([[0], np.cumsum(step)]) + rng.uniform(0, 10)
        return np.broadcast_to(s_col[None, :], (H, W))

    @staticmethod
    def wall_shade(img, band):
        """The wall's own shading (light gradient, furniture shadows) over a band, relative to its median."""
        bf = band.astype(np.float32)
        lb = cv2.GaussianBlur(img.mean(axis=2) * bf, (0, 0), 4) / np.maximum(cv2.GaussianBlur(bf, (0, 0), 4), 1e-3)
        return np.clip(lb / max(np.median(lb[band]), 1e-3), 0.3, 2.0)[..., None]

    PLINTH_STYLES = ("wall", "white", "wood", "tiles", "plain")

    def plinth(self, img, lab, geo, rng):
        """A skirting board on the wall bases in view: the band of wall pixels below hp (metres), one style for
        the image. The board stands ~1 cm proud of the wall, which shows as a thin lit top edge, a dark joint line
        on the wall just above it, and a dark contact line at the floor (the 1 cm offset itself is below a pixel
        at these distances: the band is painted in place)."""
        c = self.cfg
        hp = rng.uniform(*c.plinth_h)
        base, run = self.wall_base(lab, geo)
        band = base & (geo.hgt <= hp) & (geo.hgt > -0.02)
        if band.sum() < 200:
            return None
        style = rng.choices(self.PLINTH_STYLES, weights=c.plinth_style)[0]
        free = geo.valid & (lab != 0) & (lab != UNDER)
        g = np.random.default_rng(rng.randint(0, 2**31))
        s = self.along_wall(band, run, geo, rng)
        patch = self.floor_patch(img, free, geo, rng) if style == "tiles" else None
        if style == "tiles" and patch is None:
            style = "white"                              # no clean floor patch to copy: a white board instead
        if style == "tiles":   # floor tiles laid on the wall, mirrored at the patch borders (no seams)
            ppm, n = 256.0, patch.shape[0]
            a = _mirror(s * ppm, n)
            b = _mirror((hp - geo.hgt) * ppm + rng.uniform(0, n), n)
            col = cv2.remap(patch, a, b, cv2.INTER_LINEAR) * rng.uniform(0.6, 1.0)  # vertical face: less light
        elif style == "wood":  # wood grain running along the board (texture rows = height, columns = along)
            tex, size = strip_texture(rng, ("wood",))
            ppm = 1024 / size
            col = cv2.remap(tex.astype(np.float32) / 255, ((geo.hgt + rng.uniform(0, 1)) * ppm).astype(np.float32),
                            (s * ppm).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        else:
            if style == "wall":   # painted to blend with the wall: the wall's colour just above, a shade off
                above = base & (geo.hgt > hp + 0.02) & (geo.hgt < hp + 0.3)
                colour = np.median(img[above] if above.sum() > 50 else img[base & ~band], axis=0)
                colour = colour * rng.uniform(0.85, 1.12)
            elif style == "white":
                colour = np.array([rng.uniform(0.82, 0.95)] * 3) * [rng.uniform(0.96, 1.0), 1.0, rng.uniform(0.96, 1.0)]
            else:                 # grey / black / dark brown
                colour = np.array(rng.choice([(0.55, 0.55, 0.55), (0.1, 0.1, 0.1), (0.15, 0.22, 0.3)]))
            # raw wall pixels already carry the light: no extra shading on a colour taken from them
            col = np.asarray(colour, np.float32) * (1 + 0.02 * g.normal(0, 1, (H, W, 1))).astype(np.float32)
        if style != "wall":
            col = col * self.wall_shade(img, band)
        e = rng.uniform(0.003, 0.008)                    # visible depth of the board's top, m (1 cm, foreshortened)
        top = band & ((geo.hgt > hp - e) | ~np.roll(band, 1, axis=0))
        col[top] *= rng.uniform(1.15, 1.35)              # lit top edge
        col[band & (geo.hgt < 0.004)] *= 0.6             # contact line at the floor
        img[band] = col[band]
        joint = base & ~band & (geo.hgt > hp) & (geo.hgt < hp + rng.uniform(0.002, 0.005))
        img[joint] *= rng.uniform(0.6, 0.8)              # shadow / caulk line where the board meets the wall
        lab[band] = 0                                    # already obstacle (wall pixels): explicit
        return {"h": hp, "style": style, "px": int(band.sum())}

    def wall_tiles(self, img, lab, geo, rng):
        """Tiled wall base (bathroom, kitchen): procedural tiles unlike the floor's, from the floor up to a random
        height on the wall bases in view, in metres; a dark grout line at the floor. Labelled obstacle."""
        c = self.cfg
        ht = rng.uniform(*c.wall_tile_h)
        base, run = self.wall_base(lab, geo)
        band = base & (geo.hgt <= ht) & (geo.hgt > -0.02)
        if band.sum() < 200:
            return None
        s = self.along_wall(band, run, geo, rng)
        tex, size = procedural_tiles(rng)
        ppm = 1024 / size
        col = cv2.remap(tex.astype(np.float32) / 255, (s * ppm).astype(np.float32),
                        ((ht - geo.hgt) * ppm + rng.uniform(0, 1024)).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
        col = col * self.wall_shade(img, band) * rng.uniform(0.7, 1.0)   # vertical face: less light
        col[band & (geo.hgt < 0.004)] *= 0.6
        img[band] = col[band]
        lab[band] = 0
        return {"h": ht, "px": int(band.sum())}

    # ---------------------------------------------------------------- threshold strip / level change
    def edge(self, geo, ys, xs, rng):
        """A straight line on the floor through a random candidate pixel's floor point, within +-thr_angle of
        square-on to the view. Returns (uc, ve, th, du, dv): along / across (forward-ish) coordinates of that
        point, the angle, and the along / across coordinate of every pixel ray per unit optical depth."""
        i = rng.randrange(len(ys))
        cx, cz = geo.gx[ys[i], xs[i]], geo.gz[ys[i], xs[i]]
        th = math.radians(rng.uniform(-self.cfg.thr_angle, self.cfg.thr_angle))
        a, nrm = (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))
        return (a[0] * cx + a[1] * cz, nrm[0] * cx + nrm[1] * cz, th,
                a[0] * geo.gx1 + a[1] * geo.gz1, nrm[0] * geo.gx1 + nrm[1] * geo.gz1)

    @staticmethod
    def box(geo, du, dv, uc, length, v0, v1, top):
        """Ray / box intersection, box = along uc +- length/2, across v0 .. v1, height 0 .. top (slabs). Returns
        (tin, face, hit): entry depth, face entered (0 top, 1 across faces = front / back, 2 ends: the slab
        entered last) and hit = entered before the render's own surface (stored depth as z-buffer; tolerance:
        a flush top is within mm of the floor depth)."""
        ts = [_slab(geo.dh1, -geo.cam_h, top - geo.cam_h), _slab(dv, v0, v1),
              _slab(du, uc - length / 2, uc + length / 2)]
        tmin = np.stack([t[0] for t in ts])
        tin, tout = tmin.max(0), np.min(np.stack([t[1] for t in ts]), 0)
        hit = (tin <= tout) & (tin > 0) & geo.valid & (tin < geo.depth * 1.005 + 0.01)
        return tin, tmin.argmax(0), hit

    def reproject(self, img0, lab0, geo, gx, gz, extrude_v=None, nrm=None):
        """Colours of floor points (gx, gz: 1-D, metres) as the ORIGINAL image shows them on the original floor
        plane: plane-to-plane reprojection. Returns (colour N x 3, floor N, extruded N). floor: the point is
        visible floor there (its pixel is floor and the pixel's 3D point is that point, not an occluder).
        extruded (only with extrude_v, step down): the pixel shows a wall / object standing beyond the edge (its
        across coordinate, along nrm, > extrude_v): the lower floor point is hidden by that surface's downward
        extension, so it takes its colour."""
        if len(gx) == 0:   # no point (a level change whose far floor is all hidden): cv2.projectPoints returns None
            return np.zeros((0, 3), np.float32), np.zeros(0, bool), np.zeros(0, bool)
        u, v = self.project(gx, gz, np.zeros_like(gx), geo)
        inside = np.isfinite(u) & np.isfinite(v) & (u > -0.5) & (u < W - 0.5) & (v > -0.5) & (v < H - 0.5)
        ui = np.clip(np.nan_to_num(np.round(u)), 0, W - 1).astype(int)
        vi = np.clip(np.nan_to_num(np.round(v)), 0, H - 1).astype(int)
        sx, sz = geo.gx[vi, ui], geo.gz[vi, ui]
        same = np.hypot(sx - gx, sz - gz) < 0.03 + 0.03 * np.hypot(gx, gz)
        floor = inside & geo.valid[vi, ui] & (lab0[vi, ui] != 0) & (lab0[vi, ui] != UNDER) & same
        extruded = np.zeros_like(floor)
        if extrude_v is not None:
            extruded = inside & geo.valid[vi, ui] & (lab0[vi, ui] == 0) & (nrm[0] * sx + nrm[1] * sz > extrude_v)
        # remap maps are 2-D and < 32767 per side: the N points as rows of 1024, padded
        n, k = len(gx), 1024
        pad = -n % k
        mx = np.pad(np.nan_to_num(u, nan=-1).astype(np.float32), (0, pad)).reshape(-1, k)
        my = np.pad(np.nan_to_num(v, nan=-1).astype(np.float32), (0, pad)).reshape(-1, k)
        col = cv2.remap(img0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)[:n]
        return col, floor, extruded

    def level(self, img, lab, geo, free, rng):
        c = self.cfg
        cand = free & (geo.gz > c.thr_dist[0]) & (geo.gz < c.thr_dist[1])
        ys, xs = np.nonzero(cand)
        if len(ys) < 50:
            return None
        flat = rng.random() < c.p_level_flat
        up = rng.random() < c.p_level_up
        other = rng.random() < (c.p_flat_other if flat else c.p_level_other)  # other flooring beyond the edge
        dz = rng.uniform(0.002, c.flat_dz) if flat else rng.uniform(*c.level_dz)
        dz = dz if up else -dz
        # Room-wide edges only: a door-wide one, with no door jambs beside it in the render, comes out as a
        # platform / trench in the middle of the room (seen in the preview), a shape that does not exist. Where
        # the render shows a doorway, a room-wide edge across it gives the real picture (walls stay walls).
        door, length = False, 8.0
        for _ in range(8):
            uc, ve, th, du, dv = self.edge(geo, ys, xs, rng)
            if up:      # the raised floor: a box from the edge to 30 m beyond, height dz
                tin, face, region = self.box(geo, du, dv, uc, length, ve, ve + 30.0, dz)
                t = np.where(region, tin, geo.tf)
            else:       # rays meeting visible floor beyond the edge go down to the lower plane
                region = free & (geo.tf * dv > ve) & (np.abs(geo.tf * du - uc) < length / 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    td = np.where(geo.dh1 < -1e-6, (dz - geo.cam_h) / geo.dh1, 50.0)
                    ud = td * du
                    # door-wide opening: rays leaving its width meet its side walls first
                    ts = (uc + np.sign(ud - uc) * length / 2) / du
                side = np.abs(ud - uc) > length / 2
                t = np.where(side, ts, td)
                face = np.where(side, 2, 0)
            # drivable: floor stays floor, nothing else touched; 'under' floor is never painted
            paint = region & free if flat else region & (lab != UNDER)
            if (paint & free).sum() >= 300:              # enough visible change
                break
        else:
            return None
        nrm = (-math.sin(th), math.cos(th))
        img0, lab0 = img.copy(), lab.copy()
        u, v, h = t * du, t * dv, geo.cam_h + t * geo.dh1  # 3D point of each ray on the new surface
        gx, gz = t * geo.gx1, t * geo.gz1
        top = paint & (face == 0)                          # the floor beyond the edge, at its new height
        vert = paint & (face != 0)                         # riser (up) / ends (up) / opening sides (down)
        # floor beyond the edge: reprojection of the original floor, or another flooring
        patch = self.floor_patch(img0, free, geo, rng)
        col, okf, oke = self.reproject(img0, lab0, geo, gx[top], gz[top], None if up else ve, None if up else nrm)
        use = oke | (okf & (not other))   # extruded walls / objects always; reprojected floor unless other flooring
        if not use.all():   # other flooring / fallback, sampled at the metric (along, across) coordinates
            if other:
                tex, size = strip_texture(rng, ("tile", "tile", "wood", "stone"))
                alt = self.sample(pyramid(tex), u, v, size, rng)[top] * rng.uniform(0.5, 1.1)
            elif patch is not None:       # hidden or off-image: the floor's own texture, metric scale
                n = patch.shape[0]
                alt = cv2.remap(patch, _mirror(u * 256.0, n), _mirror(v * 256.0, n), cv2.INTER_LINEAR)[top]
            else:
                alt = np.broadcast_to(np.median(img0[free], axis=0), (int(top.sum()), 3))
            col = np.where(use[:, None], col, alt)
        col = col * rng.uniform(0.85, 1.1)                 # the other room is lit a bit differently
        if not up:   # lower floor darker along the riser it lies against (ambient occlusion)
            col = col * (0.55 + 0.45 * np.clip((v[top] - ve) / 0.08, 0, 1))[:, None]
        img[top] = col
        # vertical faces: the floor's own material (tiled step) or a plain colour, less lit than the floor
        if patch is not None and rng.random() < 0.6:
            n = patch.shape[0]
            vcol = cv2.remap(patch, _mirror(np.where(face == 2, v, u) * 256.0, n), _mirror(h * 256.0, n),
                             cv2.INTER_LINEAR)
        else:
            colour = rng.choice([(0.85, 0.87, 0.88), (0.6, 0.6, 0.6), (0.2, 0.35, 0.5), (0.1, 0.1, 0.1), None])
            colour = np.median(img0[free], axis=0) if colour is None else np.array(colour, np.float32)
            vcol = np.broadcast_to(colour.astype(np.float32), (H, W, 3)).copy()
        vcol *= rng.uniform(0.45, 0.8)
        vcol[face == 2] *= 0.6
        if up:       # lit nosing along the riser's top edge
            vcol[(face == 1) & (h > dz - min(0.01, 0.25 * dz))] *= 1.35
        img[vert] = vcol[vert]
        # the edge on our side: a lip (step down, flush) or the riser's bottom joint (step up), floor label
        uf, vf = geo.tf * du, geo.tf * dv
        lip = rng.uniform(0.01, 0.025) if not up else 0.006
        near = free & ~paint & (vf > ve - lip) & (vf <= ve) & (np.abs(uf - uc) < length / 2)
        img[near] *= 0.6 if up else rng.choice([0.6, 1.35])
        if not flat:
            lab[paint] = 0               # riser + floor at another level: not drivable (obstacle / cliff)
            # 'under' floor beyond the edge (under a bed in the other room): left unpainted (approximation: it
            # stays at the old height in the image) but it is unreachable now: obstacle, not floor / under
            lab[(lab == UNDER) & (vf > ve) & (np.abs(uf - uc) < length / 2)] = 0
        return {"up": up, "dz": dz, "flat": flat, "door": door, "th": th, "uc": uc, "ve": ve, "len": length,
                "other": other, "px": int(paint.sum())}

    def threshold(self, img, lab, geo, free, rng):
        c = self.cfg
        cand = free & (geo.gz > c.thr_dist[0]) & (geo.gz < c.thr_dist[1])
        ys, xs = np.nonzero(cand)
        if len(ys) < 50:
            return None
        top = max(rng.uniform(*c.flush_h), 1e-3)
        width = rng.uniform(*c.thr_width)
        door = rng.random() < c.p_door
        length = rng.uniform(*c.door_len) if door else 8.0
        for _ in range(6):
            uc, ve, th, du, dv = self.edge(geo, ys, xs, rng)
            v0, v1 = ve - width / 2, ve + width / 2
            tin, face, hit = self.box(geo, du, dv, uc, length, v0, v1, top)
            paint = hit & free                            # only on plain floor: the label stays floor
            if paint.sum() >= 40:
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
        info = {"h": top, "width": width, "door": door, "th": th, "uc": uc, "v0": v0, "v1": v1, "len": length,
                "px": int(paint.sum())}
        if not door and rng.random() < c.p_transition:  # other flooring beyond a room-spanning strip
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
