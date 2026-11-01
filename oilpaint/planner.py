"""
Stroke planner.

The painting is made in two separate steps:

  1. **plan**  - decide every stroke: its path, its colour, its size and a
     handful of random numbers.  This module does it, vectorised with numpy:
     all the strokes of a layer are traced *together*, one step at a time,
     instead of one stroke after the other in a Python loop.
  2. **paint** - rasterise the planned strokes.  ``engine.Brush`` does it on
     the CPU for the full-quality Run; ``static/js/live.js`` does it on the
     GPU for the live preview.  Both read the same plan, so the preview and
     the final painting show the same strokes.

What a stroke *looks like* - its bristles, softness, opacity, paint height,
colour correction and Abstract recolouring - is not baked into the plan.
``stroke_style()`` derives it from the plan plus the current settings (the
GPU shader does the same maths), which is why Intensity, Hair Thickness,
Brush Density, the Correction sliders and the Abstract stroke sliders can
change live without planning again.

Two planning modes
------------------
``exact``   used by Run.  Each finishing layer looks at the real canvas (the
            caller paints each layer before the next is planned), exactly as
            the original engine did.
``approx``  used by the live preview.  Nothing is rasterised; the canvas is
            estimated from the reference and the footprint of the strokes
            already planned.  Fast enough to re-plan while a slider moves.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import abstract as ab
from .params import hex_to_bgr

# --------------------------------------------------------------------------
# per-stroke record
# --------------------------------------------------------------------------

# column layout of Plan.strokes (float32)
C_B, C_G, C_R = 0, 1, 2          # base colour (BGR), before correction
C_JB, C_JG, C_JR = 3, 4, 5       # colour jitter added after correction
C_T = 6                          # brush width in px
C_TX = 7                         # how busy the photo is under the stroke 0..1
C_OP = 8                         # random opacity factor (edge: the opacity)
C_HA = 9                         # paint height laid down
C_KIND = 10                      # 0 layer, 1 last layer, 2 filling, 3 edge
C_FLAGS = 11                     # bit 0: correction applies, bit 1: Abstract recolour
C_R0 = 12                        # 6 random numbers for Abstract recolouring
C_SEED = 18                      # bristle seed 0..1
NCOL = 20

KIND_LAYER, KIND_LAST, KIND_FILL, KIND_EDGE = 0, 1, 2, 3
F_CORRECT, F_ABSTRACT = 1, 2


class Plan:
    """Every stroke of a painting, in image coordinates of ``size``.

    size     (w, h)
    strokes  (N, NCOL) float32
    pts      (M, 2) float32, all paths back to back
    offs     (N + 1,) int32 - path i is pts[offs[i]:offs[i+1]]
    ground   (h, w, 3) float32 BGR - the canvas before painting

    (A plain class rather than a dataclass, so the module compiles cleanly
    for the protected builds - see build/protect.py.)
    """

    __slots__ = ("size", "strokes", "pts", "offs", "ground", "meta")

    def __init__(self, size, strokes, pts, offs, ground, meta=None):
        self.size = size
        self.strokes = strokes
        self.pts = pts
        self.offs = offs
        self.ground = ground
        self.meta = meta if meta is not None else {}

    @property
    def count(self) -> int:
        return int(len(self.strokes))

    def path(self, i):
        return self.pts[self.offs[i]:self.offs[i + 1]]


def _empty_part():
    return {"strokes": np.zeros((0, NCOL), np.float32),
            "pts": np.zeros((0, 2), np.float32),
            "offs": np.zeros(1, np.int32)}


def concat(parts):
    """Join planned parts (each a dict of strokes/pts/offs) in order."""
    parts = [q for q in parts if len(q["strokes"])]
    if not parts:
        return _empty_part()
    strokes = np.concatenate([q["strokes"] for q in parts])
    pts = np.concatenate([q["pts"] for q in parts])
    offs = [np.zeros(1, np.int32)]
    base = 0
    for q in parts:
        offs.append(q["offs"][1:] + base)
        base += int(q["offs"][-1])
    return {"strokes": strokes, "pts": pts, "offs": np.concatenate(offs).astype(np.int32)}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def lerp(a, b, t):
    return a + (b - a) * t


def small_side(a) -> float:
    return float(min(a.shape[0], a.shape[1]))


def correction_active(p) -> bool:
    return bool(p.get("abstract_enabled")) and any(
        float(p.get(k, 0)) != 0 for k in ("cor_hue", "cor_saturation", "cor_brightness"))


def distortion_key(p):
    if p.get("abstract_enabled") and float(p.get("dist_strength", 0)) > 0:
        return (int(p["dist_strength"]), int(p["dist_curvature"]), int(p["dist_seed"]))
    return None


def _guides_key(guides):
    return tuple(tuple((round(x, 4), round(y, 4)) for x, y in line) for line in (guides or []))


# --------------------------------------------------------------------------
# image analysis (cached: it only depends on a few of the settings)
# --------------------------------------------------------------------------


class AnalysisCache:
    """Remembers the expensive image analysis between previews.

    Moving Stroke Thickness, Filling or Expression re-uses everything;
    Direction only redoes the cheap blending of the direction field;
    Simplicity redoes the simplification.
    """

    def __init__(self):
        self.bp = None          # brushplan.Cache (the planner since version 2)
        self.token = None
        self.base_key = None
        self.base = None
        self.dir_key = None
        self.dirs = None
        self.ref_key = None
        self.refs = None

    def reset(self):
        self.__init__()


def analyse(src: np.ndarray, p: dict, cache: AnalysisCache | None = None, token=None):
    """Everything the planner needs to know about the photograph."""
    from .engine import simplify, structure_tensor, direction_field

    cache = cache or AnalysisCache()
    if token is None or token != cache.token:
        cache.reset()
        cache.token = token

    h, w = src.shape[:2]
    bkey = (h, w, distortion_key(p))
    if cache.base_key != bkey:
        work = src.astype(np.float32)
        if distortion_key(p) is not None:
            work = ab.distort(work, p)
        cache.base = {"work": work}
        cache.base_key = bkey
        cache.ref_key = None
        cache.dir_key = None

    rkey = (bkey, int(p["simplicity"]))
    if cache.ref_key != rkey:
        work = cache.base["work"]
        # the two abstractions side by side (OpenCV lets go of Python while
        # it works): on a big photo each takes seconds
        if work.shape[0] * work.shape[1] > 1_500_000:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(2) as ex:
                f1 = ex.submit(simplify, work, int(p["simplicity"]))
                f2 = ex.submit(simplify, work, int(p["simplicity"]), 0.45)
                ref, ref_fine = f1.result(), f2.result()
        else:
            ref = simplify(work, int(p["simplicity"]))
            ref_fine = simplify(work, int(p["simplicity"]), scale_boost=0.45)
        gray = cv2.cvtColor(np.clip(ref_fine, 0, 255).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        tensor = structure_tensor(gray)
        grad = tensor["grad"]

        blur_g = ab.gblur(gray, max(2.0, small_side(gray) / 160.0))
        texture = cv2.GaussianBlur(np.abs(gray - blur_g), (0, 0), 6.0)
        tmax = float(np.percentile(texture, 92)) + 1e-6
        texture = np.clip(texture / tmax, 0.0, 1.0).astype(np.float32)

        open_sigma = max(4.0, small_side(gray) / 26.0)
        openness = ab.gblur(grad, open_sigma)
        omax = float(np.percentile(openness, 92)) + 1e-6
        openness = np.clip(openness / omax, 0.0, 1.0).astype(np.float32)

        cache.refs = {"ref": ref, "ref_fine": ref_fine, "gray": gray, "tensor": tensor,
                      "grad": grad, "coh": tensor["coherence"], "texture": texture,
                      "openness": openness, "_memo": {}}
        cache.ref_key = rkey
        cache.dir_key = None

    dkey = (rkey, int(p["direction"]), int(p["global_angle"]), _guides_key(p.get("guides")))
    if cache.dir_key != dkey:
        dirx, diry = direction_field(cache.refs["tensor"], p, p.get("guides"))
        cache.dirs = {"dirx": dirx, "diry": diry}
        cache.dir_key = dkey

    out = dict(cache.base)
    out.update(cache.refs)
    out.update(cache.dirs)
    return out


# --------------------------------------------------------------------------
# settings that shape the plan
# --------------------------------------------------------------------------


def layout(p: dict, h: int, w: int) -> dict:
    """Brush sizes and stroke rules derived from the settings (engine maths)."""
    small = float(min(h, w))
    thick = float(p["stroke_thickness"])
    base_t = 2.0 + (thick / 100.0) * small * 0.095
    base_t = float(np.clip(base_t, 2.4, small * 0.13))
    max_len = base_t * (1.4 + (float(p["max_stroke_length"]) / 100.0) * 9.0)
    expression = float(p["expression"])
    filling = float(p["filling"]) / 100.0

    radii = [base_t, base_t * 0.58, base_t * 0.34, base_t * 0.22]
    radii = [float(max(2.6, r)) for r in radii]
    dedup = []
    for r in radii:
        if not dedup or dedup[-1] - r > 0.45:
            dedup.append(r)

    return {
        "small": small, "base_t": base_t, "max_len": max_len,
        "min_len_frac": float(p["min_stroke_length"]) / 100.0,
        "filling": filling, "wide_bg": float(p["wide_bg_strokes"]),
        "wobble": expression / 100.0, "col_jit": 2.0 + expression * 0.30,
        "tol": 12.0 + expression * 0.52,
        "edge_stop": lerp(0.55, 1.12, expression / 100.0),
        "radii": dedup,
    }


def make_ground(ref: np.ndarray, p: dict, lay: dict) -> np.ndarray:
    """The canvas before painting: canvas colour with a thin imprimatura of
    the blurred photo (how much of it shows is what Filling controls)."""
    h, w = ref.shape[:2]
    canvas_rgb = np.array(hex_to_bgr(p.get("canvas_color", "#f3ece0")), np.float32)
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = canvas_rgb
    ground = ab.gblur(ref, max(1.5, lay["base_t"] * 0.85))
    canvas += (ground - canvas) * float(np.clip(lay["filling"], 0.0, 1.0)) * 0.92
    return canvas


# --------------------------------------------------------------------------
# vectorised stroke tracing
# --------------------------------------------------------------------------


def trace_many(ix, iy, dirx, diry, ref, colour, step, steps, min_steps, tol,
               wobble, rng, grad=None, coh=None, edge_stop=1.0, inertia=0.72):
    """Trace many strokes at once along the orientation field.

    Same rules as the original one-at-a-time tracer: both directions from the
    seed, curvature filter (inertia), random wobble, stop on leaving the
    picture, on a colour change larger than ``tol`` (once the stroke is past
    its minimum length) and on an incoherent strong edge.

    All per-stroke arguments are arrays of length N (scalars broadcast).
    Returns (pts, offs).
    """
    n = len(ix)
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros(1, np.int32)
    h, w = dirx.shape
    f32 = np.float32
    step = np.broadcast_to(np.asarray(step, f32), (n,)).copy()
    steps = np.broadcast_to(np.asarray(steps, np.int32), (n,)).copy()
    min_steps = np.broadcast_to(np.asarray(min_steps, np.int32), (n,)).copy()
    tol2 = np.broadcast_to(np.asarray(tol, f32) ** 2, (n,)).copy()
    wobble = np.broadcast_to(np.asarray(wobble, f32), (n,)).copy()
    colour = np.asarray(colour, f32)
    smax = int(steps.max()) if n else 0
    fc = 1.0 - inertia

    runs, counts = [], []
    for sign in (1.0, -1.0):
        x = ix.astype(f32).copy()
        y = iy.astype(f32).copy()
        pdx = np.zeros(n, f32)
        pdy = np.zeros(n, f32)
        alive = np.ones(n, bool)
        rec = np.zeros((n, max(1, smax), 2), f32)
        cnt = np.zeros(n, np.int32)
        for s in range(smax):
            alive &= steps > s
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            xs, ys = x[idx], y[idx]
            xi = xs.astype(np.int32)
            yi = ys.astype(np.int32)
            ok = (xi >= 1) & (yi >= 1) & (xi < w - 1) & (yi < h - 1)
            idx, xs, ys, xi, yi = idx[ok], xs[ok], ys[ok], xi[ok], yi[ok]
            alive[:] = False
            alive[idx] = True
            if not len(idx):
                break
            dx = dirx[yi, xi] * sign
            dy = diry[yi, xi] * sign
            if s > 0:
                px, py = pdx[idx], pdy[idx]
                flip = (dx * px + dy * py) < 0.0
                dx = np.where(flip, -dx, dx)
                dy = np.where(flip, -dy, dy)
                dx = fc * dx + inertia * px
                dy = fc * dy + inertia * py
                nn = np.hypot(dx, dy)
                good = nn >= 1e-6
                nn = np.maximum(nn, 1e-6)
                dx, dy = dx / nn, dy / nn
                if not good.all():
                    alive[idx[~good]] = False
                    idx, xs, ys, dx, dy = idx[good], xs[good], ys[good], dx[good], dy[good]
            wob = wobble[idx]
            a = rng.normal(0.0, 1.0, len(idx)).astype(f32) * (0.20 * wob)
            a = np.where(wob > 0.001, a, 0.0)
            ca, sa = np.cos(a), np.sin(a)
            dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
            pdx[idx], pdy[idx] = dx, dy
            st = step[idx]
            xs = xs + dx * st
            ys = ys + dy * st
            x[idx], y[idx] = xs, ys
            xi = xs.astype(np.int32)
            yi = ys.astype(np.int32)
            ok = (xs >= 0) & (ys >= 0) & (xi < w) & (yi < h)
            past = s >= min_steps[idx]
            if past.any():
                xc = np.clip(xi, 0, w - 1)
                yc = np.clip(yi, 0, h - 1)
                c = ref[yc, xc]
                d2 = np.sum((c - colour[idx]) ** 2, axis=1)
                ok &= ~(past & (d2 > tol2[idx]))
                if grad is not None:
                    later = s >= (min_steps[idx] + 1)
                    g = grad[yc, xc] > edge_stop
                    if coh is not None:
                        g &= coh[yc, xc] < 0.45
                    ok &= ~(later & g)
            dead = idx[~ok]
            alive[dead] = False
            keep = idx[ok]
            rec[keep, cnt[keep], 0] = xs[ok]
            rec[keep, cnt[keep], 1] = ys[ok]
            cnt[keep] += 1
        runs.append(rec)
        counts.append(cnt)

    (rp, rn), (cp, cn) = runs, counts
    total = cp + cn + 1
    dab = total < 2                         # nothing traced: a dab
    lens = np.where(dab, 2, total).astype(np.int32)
    offs = np.zeros(n + 1, np.int32)
    np.cumsum(lens, out=offs[1:])
    pts = np.zeros((int(offs[-1]), 2), f32)

    k = np.arange(max(1, smax), dtype=np.int32)[None, :]
    base = offs[:-1, None]
    # the backward run, reversed, then the seed, then the forward run
    m = (k < cn[:, None]) & ~dab[:, None]
    dest = base + cn[:, None] - 1 - k
    pts[dest[m]] = rn[m]
    seed_at = offs[:-1] + cn
    nd = ~dab
    pts[seed_at[nd], 0] = ix[nd]
    pts[seed_at[nd], 1] = iy[nd]
    m = (k < cp[:, None]) & ~dab[:, None]
    dest = base + cn[:, None] + 1 + k
    pts[dest[m]] = rp[m]

    if dab.any():
        di = np.flatnonzero(dab)
        a = rng.uniform(0, math.pi, len(di))
        r = step[di] * 0.6
        o = offs[di]
        pts[o, 0] = ix[di] - np.cos(a) * r
        pts[o, 1] = iy[di] - np.sin(a) * r
        pts[o + 1, 0] = ix[di] + np.cos(a) * r
        pts[o + 1, 1] = iy[di] + np.sin(a) * r
    return pts, offs


def path_colours(ref, pts, offs, seed, sigma=26.0):
    """The tone the loaded brush carries along each stroke (vectorised
    ``_path_colour``): up to 9 samples along the path, weighted by how close
    they are to the colour the brush was loaded with."""
    n = np.diff(offs).astype(np.int32)
    out = seed.astype(np.float32).copy()
    use = n > 2
    if not use.any():
        return out
    h, w = ref.shape[:2]
    nn = n[use]
    o = offs[:-1][use]
    m = np.minimum(nn, 9)
    k = np.arange(9)[None, :]
    stepk = (nn - 1) / np.maximum(m - 1, 1)
    idx = (k * stepk[:, None]).astype(np.int32)
    valid = k < m[:, None]
    idx = np.where(valid, idx, 0) + o[:, None]
    xs = np.clip(pts[idx, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(pts[idx, 1].astype(np.int32), 0, h - 1)
    cols = ref[ys, xs].astype(np.float32)                     # (K, 9, 3)
    d2 = np.sum((cols - seed[use][:, None, :]) ** 2, axis=2)
    wts = (np.exp(-d2 / (2.0 * sigma * sigma)) + 1e-4) * valid
    out[use] = (cols * wts[:, :, None]).sum(axis=1) / wts.sum(axis=1, keepdims=True)
    return out


# --------------------------------------------------------------------------
# planning one layer
# --------------------------------------------------------------------------


def _patch_pick(score, cx, cy, half):
    """Mean of the score in each cell's window and where its maximum is."""
    h, w = score.shape
    pad = np.pad(score, half, mode="constant", constant_values=np.nan)
    offs = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    win = pad[(cy[:, None] + half + dy[None, :]), (cx[:, None] + half + dx[None, :])]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(win, axis=1)
    k = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=1)
    return mean, cy + dy[k], cx + dx[k]


def layer_refs(an, lay, li):
    """The two references a layer paints from: ``ref_l`` (blurred to the
    brush size) and ``ref_s`` (sharper, used next to contours).  Cached in
    the analysis, since a preview re-plans with the same brush sizes often."""
    radii = lay["radii"]
    key = ("refs", li, len(radii), round(radii[li], 4))
    memo = an.get("_memo")
    if memo is not None and key in memo:
        return memo[key]
    ref, ref_fine = an["ref"], an["ref_fine"]
    sigma = max(0.7, radii[li] * 0.50)
    mix = 0.0 if len(radii) == 1 else (li / (len(radii) - 1.0)) ** 0.85
    ref_src = ref if mix <= 0.001 else (ref + (ref_fine - ref) * mix)
    ref_l = ab.gblur(ref_src, sigma)
    ref_s = ab.gblur(ref_src, max(0.5, sigma * 0.35))
    if memo is not None:
        if len(memo) > 24:
            memo.clear()
        memo[key] = (ref_l, ref_s)
    return ref_l, ref_s


def plan_layer(an, lay, li, score, rng, p):
    """Choose and trace the strokes of painting layer ``li``."""
    radii = lay["radii"]
    radius = radii[li]
    ref, ref_fine = an["ref"], an["ref_fine"]
    grad, coh = an["grad"], an["coh"]
    texture, openness = an["texture"], an["openness"]
    h, w = grad.shape
    filling = lay["filling"]

    ref_l, ref_s = layer_refs(an, lay, li)

    grid = max(2.0, radius * (1.05 if li == 0 else 0.85))
    ys = np.arange(grid * 0.5, h, grid)
    xs = np.arange(grid * 0.5, w, grid)
    gy_, gx_ = np.meshgrid(ys, xs, indexing="ij")
    cells = np.stack([gx_.ravel(), gy_.ravel()], axis=1)
    cells = cells + rng.uniform(-grid * 0.45, grid * 0.45, cells.shape)
    cells = cells[rng.permutation(len(cells))]
    n_cells = len(cells)

    ix = np.clip(cells[:, 0], 0, w - 1).astype(np.int32)
    iy = np.clip(cells[:, 1], 0, h - 1).astype(np.int32)

    if li == 0:
        keep = lerp(0.42, 1.0, filling)
        if keep < 1.0:
            sel = rng.random(len(ix)) <= keep
            ix, iy = ix[sel], iy[sel]
    else:
        thr_abs = lerp(70.0, 24.0, filling) * (1.0 + 0.18 * (li - 1))
        thr_pc = float(np.percentile(score, lerp(88.0, 45.0, filling)))
        threshold = max(thr_abs, thr_pc)
        half = int(grid * 0.5)
        mean, py, px = _patch_pick(score, ix, iy, half)
        sel = mean >= threshold
        ix, iy = px[sel].astype(np.int32), py[sel].astype(np.int32)

    n = len(ix)
    if n == 0:
        out = _empty_part()
        out["cells"] = n_cells
        return out

    last = li == len(radii) - 1
    layer_len = lay["max_len"] * (1.0 if li == 0 else lerp(0.62, 0.28, li / 3.0))
    layer_tol = lay["tol"] * (1.0 if li == 0 else lerp(0.80, 0.50, li / 3.0))
    inertia = 0.74 if li == 0 else 0.62

    g = grad[iy, ix]
    edge_shrink = 0.55 + 0.45 * (1.0 - g) ** 0.7
    op_n = openness[iy, ix]
    bg_boost = 1.0 + (lay["wide_bg"] - 1.0) / 9.0 * 1.85 * (1.0 - op_n) ** 1.5
    t_px = radius * edge_shrink * bg_boost * (1.0 + rng.uniform(-0.16, 0.20, n))
    t_px = np.clip(t_px, 1.3, lay["small"] * 0.18)

    sharp = g > 0.35
    seed_colour = np.where(sharp[:, None], ref_s[iy, ix], ref_l[iy, ix]).astype(np.float32)

    step = max(1.0, radius * 0.38)
    max_len = layer_len * rng.uniform(0.55, 1.15, n)
    steps = np.maximum(2, (max_len / step)).astype(np.int32)
    min_steps = np.maximum(1, (steps * lay["min_len_frac"]).astype(np.int32))
    tx = texture[iy, ix]
    pts, offs = trace_many(ix, iy, an["dirx"], an["diry"], ref_l, seed_colour,
                           step, steps, min_steps, layer_tol,
                           lay["wobble"] * (0.25 + 0.75 * tx), rng,
                           grad=grad, coh=coh, edge_stop=lay["edge_stop"],
                           inertia=inertia)

    col_l = path_colours(ref_l, pts, offs, seed_colour)
    col_s = path_colours(ref_s, pts, offs, seed_colour)
    colour = np.where(sharp[:, None], col_s, col_l)

    js = lay["col_jit"] * (0.22 + 0.78 * tx)
    jit = rng.normal(0.0, 1.0, n)[:, None] * js[:, None] \
        + rng.normal(0.0, 1.0, (n, 3)) * (js * 0.3)[:, None]

    st = np.zeros((n, NCOL), np.float32)
    st[:, C_B:C_B + 3] = colour
    st[:, C_JB:C_JB + 3] = jit
    st[:, C_T] = t_px
    st[:, C_TX] = tx
    st[:, C_OP] = rng.uniform(0.9, 1.06, n)
    st[:, C_HA] = 1.0 + 0.5 * rng.random(n)
    st[:, C_KIND] = KIND_LAST if last else KIND_LAYER
    st[:, C_FLAGS] = F_CORRECT | F_ABSTRACT
    st[:, C_R0:C_R0 + 6] = rng.random((n, 6))
    st[:, C_SEED] = rng.random(n)
    return {"strokes": st, "pts": pts, "offs": offs, "cells": n_cells,
            "ref_l": ref_l}


def plan_fill(an, lay, coverage, rng):
    """Small strokes wherever the canvas is still bare."""
    ref = an["ref_fine"]
    h, w = coverage.shape
    filling = lay["filling"]
    if filling <= 0.02:
        return _empty_part()
    thr = lerp(0.65, 0.16, filling)
    small = cv2.resize(coverage, (max(8, w // 6), max(8, h // 6)),
                       interpolation=cv2.INTER_AREA)
    ys, xs = np.where(small < thr)
    if len(ys) == 0:
        return _empty_part()
    budget = int(len(ys) * lerp(0.15, 1.0, filling))
    if budget <= 0:
        return _empty_part()
    pick = rng.permutation(len(ys))[:budget]
    n = len(pick)
    r = max(1.6, lay["base_t"] * 0.42)
    ref_l = ab.gblur(ref, max(0.8, r * 0.6))
    ix = np.clip(xs[pick] * 6 + rng.integers(0, 6, n), 0, w - 1).astype(np.int32)
    iy = np.clip(ys[pick] * 6 + rng.integers(0, 6, n), 0, h - 1).astype(np.int32)
    base = ref_l[iy, ix].astype(np.float32)
    jit = rng.normal(0, lay["col_jit"], (n, 3)).astype(np.float32)
    t_px = r * rng.uniform(0.85, 1.4, n)
    step = max(1.0, r * 0.5)
    steps = max(2, int(r * 5.0 / step))
    min_steps = max(1, int(steps * 0.3))
    pts, offs = trace_many(ix, iy, an["dirx"], an["diry"], ref_l,
                           np.clip(base + jit, 0, 255), step, steps, min_steps,
                           40.0, 0.35, rng)
    st = np.zeros((n, NCOL), np.float32)
    st[:, C_B:C_B + 3] = base
    st[:, C_JB:C_JB + 3] = jit
    st[:, C_T] = t_px
    st[:, C_TX] = 1.0
    st[:, C_OP] = 1.0
    st[:, C_HA] = 1.0
    st[:, C_KIND] = KIND_FILL
    st[:, C_FLAGS] = F_CORRECT | F_ABSTRACT
    st[:, C_R0:C_R0 + 6] = rng.random((n, 6))
    st[:, C_SEED] = rng.random(n)
    return {"strokes": st, "pts": pts, "offs": offs}


def plan_edges(an, p, px_scale=1.0):
    """Messy hand-painted strokes around the border (the Edge tab).

    ``px_scale`` is the size of this plan relative to the full painting, so
    a small preview plan walks its strokes in proportionally smaller steps.
    """
    if not p.get("edge_enabled"):
        return _empty_part()
    ref = an["ref_fine"]
    h, w = ref.shape[:2]
    rng = np.random.default_rng(int(p["edge_seed"]))
    n = int(p["edge_number"])
    small = float(min(h, w))
    thickness = edge_base_thickness(p, h, w)
    length = 6.0 + (float(p["edge_max_length"]) / 100.0) * small * 0.32
    use_colour = bool(p["edge_use_color"])
    fixed = np.array(hex_to_bgr(p["edge_color"]), np.float32)
    ellipse = bool(p["edge_ellipse"])
    cx, cy = w * 0.5, h * 0.5
    rx, ry = w * 0.5, h * 0.5
    walk = 3.0 * float(px_scale)

    rows, paths = [], []
    for i in range(n * 4):
        if ellipse:
            a = rng.uniform(0, 2 * math.pi)
            band = rng.uniform(0.86, 1.06)
            x = cx + math.cos(a) * rx * band
            y = cy + math.sin(a) * ry * band
            tx, ty = -math.sin(a), math.cos(a)
        else:
            side = i % 4
            band = rng.uniform(0.0, 0.085) * small
            if side == 0:
                x, y, tx, ty = rng.uniform(0, w), band, 1.0, 0.0
            elif side == 1:
                x, y, tx, ty = rng.uniform(0, w), h - band, 1.0, 0.0
            elif side == 2:
                x, y, tx, ty = band, rng.uniform(0, h), 0.0, 1.0
            else:
                x, y, tx, ty = w - band, rng.uniform(0, h), 0.0, 1.0
        a = rng.normal(0.0, 0.30)
        ca, sa = math.cos(a), math.sin(a)
        tx, ty = tx * ca - ty * sa, tx * sa + ty * ca
        ln = length * rng.uniform(0.35, 1.0)
        nsteps = max(2, int(ln / walk))
        wob = rng.normal(0.0, 0.12, nsteps)
        pts = np.empty((nsteps, 2), np.float32)
        px, py = x - tx * ln * 0.5, y - ty * ln * 0.5
        for s in range(nsteps):
            cw, sw = math.cos(wob[s]), math.sin(wob[s])
            tx, ty = tx * cw - ty * sw, tx * sw + ty * cw
            px += tx * walk
            py += ty * walk
            pts[s] = (px, py)
        if use_colour:
            base = fixed
            jit = rng.normal(0, 6, 3)
            flags = 0
        else:
            sx = int(np.clip(pts[nsteps // 2, 0], 0, w - 1))
            sy = int(np.clip(pts[nsteps // 2, 1], 0, h - 1))
            base = ref[sy, sx].astype(np.float32)
            jit = rng.normal(0, 10, 3)
            flags = F_CORRECT
        row = np.zeros(NCOL, np.float32)
        row[C_B:C_B + 3] = base
        row[C_JB:C_JB + 3] = jit
        row[C_T] = thickness * rng.uniform(0.6, 1.25)
        row[C_TX] = 1.0
        row[C_OP] = rng.uniform(0.75, 1.0)
        row[C_HA] = 1.0
        row[C_KIND] = KIND_EDGE
        row[C_FLAGS] = flags
        row[C_SEED] = rng.random()
        rows.append(row)
        paths.append(pts)
    if not rows:
        return _empty_part()
    lens = np.array([len(q) for q in paths], np.int32)
    offs = np.zeros(len(paths) + 1, np.int32)
    np.cumsum(lens, out=offs[1:])
    return {"strokes": np.stack(rows), "pts": np.concatenate(paths), "offs": offs}


def edge_base_thickness(p, h, w) -> float:
    small = float(min(h, w))
    return 2.0 + (float(p["edge_thickness"]) / 100.0) * small * 0.10


# --------------------------------------------------------------------------
# how each stroke looks with the current settings
# --------------------------------------------------------------------------


def stroke_style(strokes: np.ndarray, p: dict, size) -> dict:
    """Bristles, softness, opacity, height and final colour of every stroke.

    This is the CPU twin of the stroke shader in static/js/live.js: keep the
    two in step.
    """
    w, h = size
    n = len(strokes)
    kind = strokes[:, C_KIND].astype(np.int32)
    t = strokes[:, C_T]
    tx = strokes[:, C_TX]

    inten = float(p["intensity"]) / 100.0
    softness_k = lerp(0.85, 0.0, inten)
    opacity = lerp(0.62, 1.0, inten)
    shade_var = lerp(0.04, 0.20, inten)
    hair_frac = float(p["hair_thickness"]) / 100.0
    density = float(p["brush_density"]) / 100.0

    hair = np.clip(t * hair_frac * 0.9, 1.1, np.maximum(1.1, t))
    nb = np.clip(1 + density * (t / hair) * 1.6, 1, 14)
    soft = softness_k * np.maximum(0.25, hair * 0.30)
    op = np.minimum(1.0, opacity * np.where(kind == KIND_LAST, 1.0, 0.94)
                    * (0.74 + 0.26 * tx) * strokes[:, C_OP])
    sv = shade_var * (0.35 + 0.65 * tx)
    height = strokes[:, C_HA].copy()
    flat = np.full(n, hair_frac > 0.72)

    soft = np.where(kind == KIND_LAST, soft * 0.55, soft)

    fill = kind == KIND_FILL
    if fill.any():
        nb[fill] = np.clip(nb[fill], 1, 10)
        op[fill] = opacity
        sv[fill] = shade_var
        flat[fill] = False

    edge = kind == KIND_EDGE
    if edge.any():
        eb = edge_base_thickness(p, h, w)
        ehair = float(np.clip(eb * float(p["edge_hair_thickness"]) / 100.0, 1.0, max(1.0, eb)))
        hair[edge] = ehair
        nb[edge] = np.clip(eb / max(ehair, 0.9) * 1.6, 1, 14)
        soft[edge] = lerp(0.2, 4.5, float(p["edge_smoothness"]) / 100.0)
        op[edge] = strokes[edge, C_OP]
        height[edge] = 0.6 + 2.2 * float(p["edge_relief"]) / 100.0
        sv[edge] = 0.18
        flat[edge] = False

    return {"hair": hair.astype(np.float32), "nb": nb.astype(np.int32),
            "soft": soft.astype(np.float32), "opacity": op.astype(np.float32),
            "height": height.astype(np.float32), "shade_var": sv.astype(np.float32),
            "flat": flat, "colour": stroke_colours(strokes, p)}


def _hsv(c):
    return cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                        cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)


def _bgr(hsv):
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                        cv2.COLOR_HSV2BGR).reshape(-1, 3).astype(np.float32)


def correct_colours(c: np.ndarray, p: dict) -> np.ndarray:
    """Abstract Art > Correction applied to a list of colours (BGR)."""
    if not correction_active(p) or not len(c):
        return c
    return ab.correct(c.reshape(-1, 1, 3), p).reshape(-1, 3)


def stroke_colours(strokes: np.ndarray, p: dict) -> np.ndarray:
    """Final paint colour: correction, palette jitter, Abstract recolouring."""
    base = strokes[:, C_B:C_B + 3]
    flags = strokes[:, C_FLAGS].astype(np.int32)
    col = base.copy()
    corr = (flags & F_CORRECT) != 0
    if corr.any() and correction_active(p):
        col[corr] = correct_colours(base[corr], p)
    col = np.clip(col + strokes[:, C_JB:C_JB + 3], 0, 255)

    if not p.get("abstract_enabled"):
        return col
    cval = float(p["ab_color"])
    sval = float(p["ab_saturation"])
    kval = float(p["ab_contrast"])
    if cval <= 0 and sval <= 0 and kval <= 0:
        return col
    r = strokes[:, C_R0:C_R0 + 6]
    sel = (flags & F_ABSTRACT) != 0
    if cval > 0:
        m = sel & (r[:, 0] < cval / 90.0 * 0.85)
        if m.any():
            hsv = _hsv(col[m])
            hsv[:, 0] = np.mod(hsv[:, 0] + (r[m, 1] * 2 - 1) * (cval / 90.0) * 90.0, 180.0)
            col[m] = _bgr(hsv)
    if sval > 0:
        m = sel & (r[:, 2] < 0.6)
        if m.any():
            hsv = _hsv(col[m])
            hsv[:, 1] = np.clip(hsv[:, 1] * (1.0 + r[m, 3] * sval / 40.0 * 1.6), 0, 255)
            col[m] = _bgr(hsv)
    if kval > 0:
        m = sel & (r[:, 4] < 0.5)
        if m.any():
            f = 1.0 + (r[m, 5] * 2 - 1) * (kval / 100.0) * 0.75
            col[m] = 128.0 + (col[m] - 128.0) * f[:, None]
    return np.clip(col, 0, 255)


# --------------------------------------------------------------------------
# approximate footprint (live preview only)
# --------------------------------------------------------------------------


def footprint(pts, offs, widths, shape, scale=0.25):
    """Where the strokes put paint, drawn quickly at a reduced size.

    Strokes are grouped by (rounded) width so OpenCV can draw thousands of
    polylines in a single call.
    """
    h, w = shape
    sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
    cov = np.zeros((sh, sw), np.uint8)
    if len(widths) == 0:
        return np.zeros(shape, np.float32)
    shift = 2
    ip = np.round(pts * scale * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
    width = np.maximum(1, np.round(widths * scale)).astype(np.int32)
    lines = np.split(ip, offs[1:-1])
    order = np.argsort(width, kind="stable")
    bounds = np.flatnonzero(np.diff(width[order])) + 1
    for grp in np.split(order, bounds):
        if not len(grp):
            continue
        cv2.polylines(cov, [lines[i] for i in grp], False, 255, int(width[grp[0]]),
                      cv2.LINE_8, shift=shift)
    return cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# live preview plan
# --------------------------------------------------------------------------


def plan_preview(src: np.ndarray, p: dict, cache: AnalysisCache | None = None,
                 token=None, px_scale=1.0) -> Plan:
    """Plan the whole painting (BGR 0..255 picture, planned at its own size).

    Since version 2 this is brushplan.plan: the same steps the GPU runs
    (static/js/oilfx.js planBrush). The old planner below is kept for
    reference and tests (plan_preview_v1)."""
    from . import brushplan as BP
    bc = None
    if isinstance(cache, AnalysisCache):
        if cache.bp is None:
            cache.bp = BP.Cache()
        bc = cache.bp
    src = np.asarray(src, np.float32)
    if distortion_key(p) is not None:
        src = ab.distort(src, p)
    rgb = np.ascontiguousarray(src[..., ::-1]) / 255.0
    return BP.plan(rgb.astype(np.float32), p, bc, token, px_scale=px_scale)


def plan_preview_v1(src: np.ndarray, p: dict, cache: AnalysisCache | None = None,
                    token=None, px_scale=1.0) -> Plan:
    """Plan the whole painting without painting it (approximate mode)."""
    return _plan_preview_v1(src, p, cache, token, px_scale)


def _plan_preview_v1(src, p, cache=None, token=None, px_scale=1.0):
    """Plan the whole painting without painting it (approximate mode).

    The finishing layers need to know where the canvas still differs from
    the photo.  Instead of painting, the canvas is estimated: where the
    strokes of a layer land, the canvas takes on that layer's reference.
    """
    an = analyse(src, p, cache, token)
    h, w = an["grad"].shape
    lay = layout(p, h, w)
    rng = np.random.default_rng(int(p["random_seed"]))

    gkey = ("ground", round(lay["base_t"], 4), round(lay["filling"], 4), p.get("canvas_color"))
    memo = an.get("_memo", {})
    if gkey not in memo:
        memo[gkey] = make_ground(an["ref"], p, lay)
    ground = memo[gkey]
    canvas = ground.copy()
    coverage = np.zeros((h, w), np.float32)
    parts = []
    for li in range(len(lay["radii"])):
        score = None
        if li > 0:
            ref_l, _ = layer_refs(an, lay, li)
            d = canvas - ref_l
            # a real canvas also carries the brush texture of the strokes, so
            # it never matches the reference perfectly where the photo is
            # busy; without this the preview plans too few finishing strokes
            # (calibrated against Run on the sample photos)
            err = np.sqrt(np.einsum("ijk,ijk->ij", d, d)) + 14.0 * an["texture"]
            score = err * (0.40 + 1.80 * an["grad"])
        part = plan_layer(an, lay, li, score, rng, p)
        parts.append(part)
        if len(part["strokes"]):
            cov = footprint(part["pts"], part["offs"], part["strokes"][:, C_T], (h, w))
            # strokes are nearly opaque: the canvas there now holds the layer's
            # reference, softened a little by the brush
            a = np.clip(cov * 0.92, 0, 1)[:, :, None]
            canvas += (part["ref_l"] - canvas) * a
            np.maximum(coverage, cov, out=coverage)
    parts.append(plan_fill(an, lay, coverage, rng))
    parts.append(plan_edges(an, p, px_scale))
    allp = concat(parts)
    return Plan(size=(w, h), strokes=allp["strokes"], pts=allp["pts"], offs=allp["offs"],
                ground=ground, meta={"base_t": lay["base_t"],
                                     "edge_base": edge_base_thickness(p, h, w)})


# --------------------------------------------------------------------------
# binary encoding for the browser
# --------------------------------------------------------------------------


def decode(buf: bytes, size=None) -> Plan:
    """A plan sent by the page (static/js/oilfx.js encodePlan, the same
    layout as encode). ``size`` (w, h): the ground is enlarged to it."""
    import json
    import struct
    magic, hl = struct.unpack_from("<4sI", buf, 0)
    if magic != b"OSP1":
        raise ValueError("Not a plan")
    head = json.loads(buf[8:8 + hl].decode())
    at = 8 + hl
    n, m, ncol = int(head["n"]), int(head["m"]), int(head.get("ncol", NCOL))
    if ncol != NCOL or n < 0 or m < 0 or n > 5_000_000 or m > 200_000_000:
        raise ValueError("Bad plan")
    st = np.frombuffer(buf, "<f4", n * ncol, at).reshape(n, ncol).astype(np.float32); at += n * ncol * 4
    pts = np.frombuffer(buf, "<f4", m * 2, at).reshape(m, 2).astype(np.float32); at += m * 8
    offs = np.frombuffer(buf, "<u4", n + 1, at).astype(np.int32); at += (n + 1) * 4
    gw, gh = int(head["gw"]), int(head["gh"])
    g = np.frombuffer(buf, np.uint8, gw * gh * 3, at).reshape(gh, gw, 3)
    if offs[-1] != m or (n and offs.min() < 0):
        raise ValueError("Bad plan")
    w, h = size or (int(head["w"]), int(head["h"]))
    ground = cv2.resize(cv2.cvtColor(g, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return Plan(size=(w, h), strokes=st, pts=pts, offs=offs, ground=ground,
                meta={"base_t": float(head.get("base_t", 0)), "edge_base": float(head.get("edge_base", 0))})


def encode(plan: Plan, scale: float = 1.0, ground_side: int = 0) -> bytes:
    """Pack a plan for static/js/live.js.

    Layout (little endian):
      uint32  magic 'OSP1', header length L
      L bytes JSON header
      float32 strokes (N x NCOL)        - pixel values scaled by ``scale``
      float32 pts (M x 2)
      uint32  offs (N + 1)
      uint8   ground RGB (gh x gw x 3)
    each block padded to 4 bytes.
    """
    import json
    import struct

    w, h = plan.size
    st = plan.strokes.copy()
    st[:, C_T] *= scale
    pts = (plan.pts * scale).astype(np.float32)
    ow, oh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    g = plan.ground
    gw, gh = ow, oh
    if ground_side and max(gw, gh) > ground_side:
        k = ground_side / max(gw, gh)
        gw, gh = max(1, int(gw * k)), max(1, int(gh * k))
    g = cv2.resize(np.clip(g, 0, 255).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    header = {"w": ow, "h": oh, "n": int(len(st)), "m": int(len(pts)), "ncol": NCOL,
              "gw": gw, "gh": gh, "scale": scale,
              "edge_base": float(plan.meta.get("edge_base", 0.0)) * scale}
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 4)
    blocks = [struct.pack("<4sI", b"OSP1", len(hb)), hb,
              st.astype("<f4").tobytes(), pts.astype("<f4").tobytes(),
              plan.offs.astype("<u4").tobytes()]
    gbytes = g.tobytes()
    gbytes += b"\0" * (-len(gbytes) % 4)
    blocks.append(gbytes)
    return b"".join(blocks)
