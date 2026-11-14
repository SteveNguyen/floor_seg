"""Floor retexturing augmentation for ProcTHOR renders (whose floors are all wood).

Every floor pixel of a render is put back on the floor plane (stored optical depth + camera pitch/roll +
the calibrated lens), so a texture can be laid on the floor in the right perspective and scale, then
multiplied by the render's own floor shading (light falloff, object shadows) and pasted on floor pixels
only (objects keep occluding it). Labels do not change: rugs and sun patches are floor (rule: <= 1 cm).

Variations (independent draws): whole floor retextured (ambientCG CC0 photos or procedural tiles),
0-2 rugs, a hard-edged sun patch, soft glare from a window on a glossy floor.

    ft = FloorTex("data/textures/ambientcg", floor_seg.DEFAULT_CALIB)
    bgr = ft.apply(bgr, floor01, depth_mm, pitch_deg, roll_deg, random.Random())

Texture root layout: <root>/<group>/*.jpg, groups floor, terrazzo, marble, stone, concrete, wood, rug
(list of the ambientCG assets used: floor_seg/assets/ambientcg_textures.md).
"""
import glob
import json
import math
import os

import cv2
import numpy as np

W, H = 640, 360           # render size (floor_seg/thor_render.py)
P_RETEX, P_RUG, P_SUN, P_GLARE = 0.6, 0.35, 0.25, 0.2
P_PROCEDURAL = 0.3        # share of retextured floors drawn as procedural tiles instead of photos
LEVELS = 5                # texture mip levels (1024 .. 64 px)


def lens_rays(calib_path):
    """Normalized undistorted image coords (x right, y down) of every W x H pixel of the calibrated lens."""
    cal = json.load(open(calib_path))
    sx, sy = W / cal["image_resolution"][0], H / cal["image_resolution"][1]
    K = np.array(cal["K"]) * np.array([[sx], [sy], [1]])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], -1)[:, None, :]
    xn = cv2.undistortPoints(pts, K, np.array(cal["D"]),
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6))[:, 0]
    return xn[:, 0].reshape(H, W).astype(np.float32), xn[:, 1].reshape(H, W).astype(np.float32)


def floor_coords(xn, yn, depth_m, pitch_deg, roll_deg):
    """Floor-plane coords (m): gx right, gz forward (camera yaw frame). Same conventions as the renderer's
    height_above_floor: Unity camera y up, roll about the optical axis, then pitch (positive looks down)."""
    x, y, z = xn * depth_m, -yn * depth_m, depth_m
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    x1 = x * math.cos(r) - y * math.sin(r)
    y1 = x * math.sin(r) + y * math.cos(r)
    return x1, z * math.cos(p) + y1 * math.sin(p)


def pyramid(tex):
    levels = [tex.astype(np.float32) / 255]
    for _ in range(LEVELS - 1):
        levels.append(cv2.pyrDown(levels[-1]))
    return levels


def procedural_tiles(rng):
    """Square or brick tiles: random base colour (incl. dark blue / terracotta), per-tile shade, speckle, grout.
    Returns (1024 px texture, its size in m)."""
    n = rng.choice([2, 3, 4, 5, 6, 8])              # tiles per texture side
    tile_m = rng.uniform(0.1, 0.6)
    px = 1024 // n
    base = np.array(rng.choice([(0.35, 0.2, 0.1), (0.15, 0.25, 0.55), (0.2, 0.35, 0.75), (0.85, 0.85, 0.82),
                                (0.55, 0.45, 0.35), (0.25, 0.25, 0.25), (0.75, 0.7, 0.6),
                                (rng.random(), rng.random(), rng.random())]), np.float32)  # BGR
    base *= rng.uniform(0.5, 1.2)
    grout = np.array([rng.uniform(0.3, 0.95)] * 3, np.float32) * rng.uniform(0.6, 1.1)
    tex = np.empty((n * px, n * px, 3), np.float32)
    brick = rng.random() < 0.3
    for i in range(n):
        for j in range(n):
            tex[i * px:(i + 1) * px, j * px:(j + 1) * px] = base * rng.uniform(0.85, 1.15)
    if brick:  # offset every other row by half a tile
        for i in range(1, n, 2):
            tex[i * px:(i + 1) * px] = np.roll(tex[i * px:(i + 1) * px], px // 2, axis=1)
    tex *= 1 + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0.02, 0.15), tex.shape[:2])[..., None]
    g = max(1, int(px * rng.uniform(0.01, 0.06)))
    for i in range(n):
        tex[i * px:i * px + g] = grout
        if brick and i % 2:
            for j in range(n):
                c = (j * px + px // 2) % (n * px)
                tex[i * px:(i + 1) * px, c:c + g] = grout
        else:
            tex[:, i * px:i * px + g] = grout
    tex = cv2.resize(np.clip(tex, 0, 1), (1024, 1024), interpolation=cv2.INTER_AREA)
    return (tex * 255).astype(np.uint8), tile_m * n


class FloorTex:
    def __init__(self, tex_root, calib_path):
        self.xn, self.yn = lens_rays(calib_path)
        self.tex = {g: sorted(glob.glob(os.path.join(tex_root, g, "*.jpg")))
                    for g in ("floor", "terrazzo", "marble", "stone", "concrete", "wood", "rug")}
        self.floor_groups = [g for g in self.tex if g != "rug" and self.tex[g]]

    def sample(self, levels, gx, gz, size_m, rng):
        """Texture colour at floor points, random rotation/offset, mip level from the local texel footprint."""
        th = rng.uniform(0, 2 * math.pi)
        s = 1024 / size_m
        a = (math.cos(th) * gx - math.sin(th) * gz) * s + rng.uniform(0, 1024)
        b = (math.sin(th) * gx + math.cos(th) * gz) * s + rng.uniform(0, 1024)
        foot = np.maximum(np.hypot(*np.gradient(a)), np.hypot(*np.gradient(b)))  # texels per pixel
        lev = np.clip(np.log2(np.maximum(foot, 1e-3)), 0, LEVELS - 1)
        out = np.zeros(gx.shape + (3,), np.float32)
        for k in range(LEVELS):
            w = np.clip(1 - np.abs(lev - k), 0, 1)
            if w.max() == 0:
                continue
            sc = 2.0 ** -k
            col = cv2.remap(levels[k], (a * sc).astype(np.float32), (b * sc).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            out += col * w[..., None]
        return out

    def load(self, group, rng):
        tex = cv2.resize(cv2.imread(rng.choice(self.tex[group])), (1024, 1024), interpolation=cv2.INTER_AREA)
        size = {"floor": (0.8, 3.0), "terrazzo": (0.6, 2.0), "marble": (0.8, 2.5), "stone": (1.0, 3.0),
                "concrete": (1.0, 3.0), "wood": (1.0, 3.0), "rug": (0.3, 1.2)}[group]
        return tex, rng.uniform(*size)

    def apply(self, bgr, floor, depth_mm, pitch, roll, rng):
        m = (floor > 0) & (depth_mm > 0)
        if m.mean() < 0.05:
            return bgr
        gx, gz = floor_coords(self.xn, self.yn, depth_mm.astype(np.float32) / 1000, pitch, roll)
        img = bgr.astype(np.float32) / 255
        mf = m.astype(np.float32)
        # floor shading: blurred floor luminance (masked normalized blur), relative to its median
        lum = img.mean(axis=2)
        lb = cv2.GaussianBlur(lum * mf, (0, 0), 4) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 4), 1e-3)
        shade = np.clip(lb / max(np.median(lb[m]), 1e-3), 0.15, 2.5)[..., None]
        out = img.copy()
        if rng.random() < P_RETEX:
            if rng.random() < P_PROCEDURAL:
                tex, size = procedural_tiles(rng)
            else:
                tex, size = self.load(rng.choice(self.floor_groups), rng)
            col = self.sample(pyramid(tex), gx, gz, size, rng)
            # tint + brightness; capped at 1.1 so bright textures do not clip away the object shadows
            col = col * np.array([rng.uniform(0.8, 1.2) for _ in range(3)], np.float32) * rng.uniform(0.5, 1.1)
            out[m] = (col * shade)[m]
        if self.tex["rug"] and rng.random() < P_RUG:
            for _ in range(rng.choice([1, 1, 2])):
                ys, xs = np.nonzero(m)
                i = rng.randrange(len(ys))
                cx, cz = gx[ys[i], xs[i]], gz[ys[i], xs[i]]    # rug centre on a visible floor point
                th = rng.uniform(0, math.pi)
                u = math.cos(th) * (gx - cx) - math.sin(th) * (gz - cz)
                v = math.sin(th) * (gx - cx) + math.cos(th) * (gz - cz)
                hw, hl = rng.uniform(0.3, 1.0), rng.uniform(0.4, 1.4)
                inside = m & (np.abs(u) < hw) & (np.abs(v) < hl)
                if not inside.any():
                    continue
                tex, size = self.load("rug", rng)
                col = self.sample(pyramid(tex), gx, gz, size, rng) * rng.uniform(0.6, 1.2)
                border = inside & ((hw - np.abs(u) < 0.04) | (hl - np.abs(v) < 0.04))
                col[border] *= rng.uniform(0.5, 1.3)          # hem / fringe colour
                out[inside] = (col * shade)[inside]
        if rng.random() < P_SUN:  # window light: sharp-edged parallelogram on the floor, warm, overexposed
            ys, xs = np.nonzero(m)
            i = rng.randrange(len(ys))
            cx, cz = gx[ys[i], xs[i]], gz[ys[i], xs[i]]
            th, shear = rng.uniform(0, math.pi), rng.uniform(-0.8, 0.8)
            u = math.cos(th) * (gx - cx) - math.sin(th) * (gz - cz)
            v = math.sin(th) * (gx - cx) + math.cos(th) * (gz - cz)
            sun = m & (np.abs(u + shear * v) < rng.uniform(0.2, 0.8)) & (np.abs(v) < rng.uniform(0.3, 1.5))
            gain = rng.uniform(1.6, 3.5) * np.array([rng.uniform(0.8, 0.95), rng.uniform(0.9, 1.0), 1.0], np.float32)
            soft = cv2.GaussianBlur(sun.astype(np.float32), (0, 0), rng.uniform(0.5, 2.0))[..., None] * mf[..., None]
            out = out * (1 + (gain - 1) * soft)
        if rng.random() < P_GLARE:  # soft specular glare streak, as on glossy tiles
            ys, xs = np.nonzero(m)
            i = rng.randrange(len(ys))
            blob = np.zeros((H, W), np.float32)
            cv2.ellipse(blob, (int(xs[i]), int(ys[i])), (rng.randint(10, 60), rng.randint(30, 140)),
                        rng.uniform(-30, 30), 0, 360, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(8, 25)) * mf * rng.uniform(0.3, 0.9)
            out = out + blob[..., None]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
