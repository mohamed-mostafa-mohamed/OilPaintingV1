"""
Impression — a soft impressionist oil painting on canvas, built natively
(no Photoshop, no action, no pattern or brush files).

Pipeline (each step measured against a reference made in Photoshop):

  1. flow       a smooth stroke-direction field: along the edges of the
                photo where it has structure, gentle swirls elsewhere
  2. dabs       the photo smoothed along the flow (short strokes), with a
                fine ridged paint relief, sharpened twice and the shadows
                lifted (Unsharp Mask, Shadows/Highlights)
  3. light      long, swirling oil-paint strokes laid on with Lighten through
                a ragged stroke mask: light paint dragged over dark areas
  4. surface    embossed bristle grooves and a woven canvas, laid on with
                Overlay
  5. border     an even white border round the painting

Every texture is generated in code, so the output can be used and sold
freely. Distances are given for a 4032 px wide picture and scaled.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from actions import filters as F

REF_SIDE = 4032.0
MIN_SIDE = 1200


def _g(x, s):
    return cv2.GaussianBlur(x, (0, 0), s) if s > 0.05 else x


def _noise(shape, sigma, rng):
    h, w = shape
    f = 1.0
    if sigma > 6:
        f = sigma / 3.0
        h, w = max(4, int(h / f)), max(4, int(w / f))
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = _g(n, sigma / f)
    if f > 1:
        n = cv2.resize(n, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


# --------------------------------------------------------------------------
# 1. flow field
# --------------------------------------------------------------------------
def flow(lum, s, rng, swirl=0.5):
    """Unit stroke direction (dx, dy) per pixel."""
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    sig = 10 * s + 1
    jxx, jyy, jxy = _g(gx * gx, sig), _g(gy * gy, sig), _g(gx * gy, sig)
    # tangent (along edges) = minor eigenvector, as a doubled angle
    ang2 = np.arctan2(2 * jxy, jxx - jyy) + np.pi
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-6)
    strength = np.clip(coh * np.clip(tr / (np.percentile(tr, 90) + 1e-6), 0, 1) * 1.6, 0, 1)
    # swirling fallback field where the photo is flat
    sw = _noise(lum.shape, 90 * s + 2, rng) * math.pi * (0.6 + 0.8 * swirl) + 0.6
    c2 = strength * np.cos(ang2) + (1 - strength) * np.cos(2 * sw)
    s2 = strength * np.sin(ang2) + (1 - strength) * np.sin(2 * sw)
    c2, s2 = _g(c2, 3 * s + 0.5), _g(s2, 3 * s + 0.5)
    a = 0.5 * np.arctan2(s2, c2)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def lic(img, dx, dy, length, steps):
    """Average ``img`` along the flow lines (line integral convolution),
    ``length`` px each way in ``steps`` steps."""
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = img.astype(np.float32).copy()
    wsum = 1.0
    st = length / max(steps, 1)
    for sign in (1.0, -1.0):
        x, y = gx.copy(), gy.copy()
        pdx, pdy = dx * sign, dy * sign
        for i in range(steps):
            vx = cv2.remap(dx, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            vy = cv2.remap(dy, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flip = np.where(vx * pdx + vy * pdy < 0, -1.0, 1.0).astype(np.float32)
            vx, vy = vx * flip, vy * flip
            x += vx * st
            y += vy * st
            pdx, pdy = vx, vy
            wt = 1.0 - (i + 1) / (steps + 1) * 0.5
            acc += cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * wt
            wsum += wt
    return acc / wsum


def _emboss(hgt, angle=135, height=2.0):
    a = math.radians(angle)
    ox, oy = math.cos(a) * height, -math.sin(a) * height
    m = np.float32([[1, 0, ox], [0, 1, oy]])
    sh = cv2.warpAffine(hgt, m, (hgt.shape[1], hgt.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return hgt - sh


def _unsharp(x, amount, r):
    return x + (x - _g(x, r)) * amount


def bump_layer(h, w, s, rng, amount=1.0):
    """Embossed bristle grooves: many short, scratchy strokes, each a few
    parallel bristle lines, in locally similar directions (Overlay layer,
    0.5 = no change)."""
    hgt = np.zeros((h, w), np.float32)
    field = _noise((max(8, h // 16), max(8, w // 16)), 6, rng) * 1.8
    n = int(30000 * (h * w) / (s * s) / (3024 * 4032) * s * s)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    fi = field[np.clip((ys / 16).astype(int), 0, field.shape[0] - 1),
               np.clip((xs / 16).astype(int), 0, field.shape[1] - 1)]
    ang = fi + rng.normal(0, 0.35, n)
    ln = rng.uniform(25, 90, n) * s
    wd = rng.uniform(5, 16, n) * s
    val = rng.uniform(0.4, 1.0, n) * np.where(rng.random(n) < 0.5, 1, -1)
    for i in range(n):
        c, sn = math.cos(ang[i]), math.sin(ang[i])
        k = int(rng.integers(3, 8))
        for j in range(k):
            off = (j / max(k - 1, 1) - 0.5) * wd[i]
            ox, oy = -sn * off, c * off
            l2 = ln[i] * rng.uniform(0.6, 1.0) / 2
            p0 = (int((xs[i] + ox - c * l2) * 8), int((ys[i] + oy - sn * l2) * 8))
            p1 = (int((xs[i] + ox + c * l2) * 8), int((ys[i] + oy + sn * l2) * 8))
            cv2.line(hgt, p0, p1, float(val[i] * rng.uniform(0.5, 1.0)),
                     max(1, int(round(rng.uniform(1, 2.5) * s))), cv2.LINE_AA, shift=3)
    hgt += 0.35 * _noise((h, w), 0.6 * s + 0.25, rng) * hgt.std()
    hgt = _g(hgt, 0.4 * s + 0.2)
    hgt /= hgt.std() + 1e-6
    return 0.5 + _emboss(hgt, 135, 2.0 * s + 0.3) * 0.11 * amount


def canvas_layer(h, w, s, rng, amount=1.0):
    """Embossed woven canvas: small thread knots in slightly wavy diagonal
    rows (Overlay layer, 0.5 = no change)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p_ = 6.8 * s + 0.5
    wav = _noise((h, w), 80 * s + 2, rng) * 2.5 * s + np.sin(yy / (60 * s + 1)) * 1.5 * s
    a = math.radians(64.0)
    u = (xx * math.cos(a) + yy * math.sin(a) + wav) / p_
    v = (-xx * math.sin(a) + yy * math.cos(a)) / (p_ * 1.6)
    v = (-xx * math.sin(a) + yy * math.cos(a)) / (p_ * 1.9)
    knots = np.cos(2 * math.pi * u) * (0.55 + 0.45 * np.cos(2 * math.pi * v + math.pi * np.floor(u)))
    knots += 0.3 * _noise((h, w), 0.6 * s + 0.25, rng)
    knots = _g(knots.astype(np.float32), 0.35 * s + 0.15)
    knots /= knots.std() + 1e-6
    return 0.5 + _emboss(knots, 135, 3.0 * s) * 0.055 * amount


# --------------------------------------------------------------------------
# the whole look
# --------------------------------------------------------------------------
# fitted on the reference: photo brightness -> painting brightness (0..1)
_TONE_X = np.array([0, 16, 48, 80, 112, 144, 176, 208, 240, 255], np.float32) / 255
_TONE_Y = np.array([30, 45.5, 68.9, 94.0, 115.6, 142.9, 171.5, 203.9, 235.5, 250], np.float32) / 255


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    h0, w0 = image_bgr.shape[:2]
    if max(h0, w0) < min_side:
        k = min_side / max(h0, w0)
        big = cv2.resize(np.asarray(image_bgr, np.float32),
                         (int(round(w0 * k)), int(round(h0 * k))), interpolation=cv2.INTER_CUBIC)
        out = render(big, p, progress, min_side=0)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)

    def report(f, label):
        if progress:
            progress(f, label)

    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = rgb.shape[:2]
    s = max(h, w) / REF_SIDE
    seed = int(p.get("im_seed", 5))
    rng = np.random.default_rng(seed)
    dab = float(p.get("im_dab", 50)) / 50.0
    light_amt = float(p.get("im_light", 50)) / 50.0
    relief = float(p.get("im_relief", 50)) / 50.0
    canvas = float(p.get("im_canvas", 50)) / 50.0

    # ---- 1. flow (made at half size, it is smooth) --------------------------
    report(0.05, "Impression: stroke directions")
    hs, ws = max(8, h // 2), max(8, w // 2)
    small = cv2.resize(rgb, (ws, hs), interpolation=cv2.INTER_AREA)
    lum_s = F.lum(small)
    fdx, fdy = flow(lum_s, s / 2, rng, float(p.get("im_swirl", 50)) / 100.0)

    # ---- 2. dabs: median of several dab-shuffled copies (like a stack of
    #      Art History brush layers), then the sharpening and tone ------------
    report(0.15, "Impression: painting the dabs")
    pre = cv2.resize(_g(rgb, 3.0 * s), (ws, hs), interpolation=cv2.INTER_AREA)
    gy, gx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    amp = 7.0 * s * dab
    copies = []
    for _ in range(7):
        nx = lic(_noise((hs, ws), 1.5, rng), fdx, fdy, 5, 4)
        ny = lic(_noise((hs, ws), 1.5, rng), fdx, fdy, 5, 4)
        nx /= nx.std() + 1e-6
        ny /= ny.std() + 1e-6
        copies.append(cv2.remap(pre, gx + nx * amp, gy + ny * amp, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT))
    med = np.median(np.stack(copies, 0), axis=0).astype(np.float32)
    del copies
    base = cv2.resize(med, (w, h), interpolation=cv2.INTER_CUBIC)
    L = F.lum(np.clip(base, 0, 1))
    Lt = np.interp(np.clip(L, 0, 1), _TONE_X, _TONE_Y).astype(np.float32)
    base = np.clip(base * ((Lt + 0.02) / (L + 0.02))[..., None], 0, 1)
    report(0.35, "Impression: brush texture")
    # fine vertical weave the dabs pick up from the canvas
    xx = np.arange(w, dtype=np.float32)[None, :]
    mod = np.clip(0.5 + 0.6 * cv2.resize(_noise((max(4, h // 8), max(4, w // 2)), 1.0, rng), (w, h)), 0, 1.3)
    stripe = np.sin(2 * math.pi * xx / (5.75 * s + 0.4)) * mod
    base = base * (1 + stripe[..., None] * 0.012 * relief)
    # bristle ripples along the strokes, mostly in the light paint
    rid = lic(_noise((hs, ws), 0.6, rng), fdx, fdy, 6 * s + 1, 4)
    rid = cv2.resize(rid, (w, h), interpolation=cv2.INTER_CUBIC)
    rid /= rid.std() + 1e-6
    lg = F.lum(_g(rgb, 1.5 * s))
    det = np.sqrt(cv2.Sobel(lg, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lg, cv2.CV_32F, 0, 1) ** 2)
    det = np.clip(_g(det, 12 * s + 1) / (np.percentile(det, 80) + 1e-6), 0, 1)   # where the photo has detail
    base = base * (1 + (_emboss(rid, 135, 1.0) * det * (0.4 + 0.8 * L))[..., None] * 0.035 * relief)
    # Unsharp Mask x2 (181% r2.8, 101% r1.6)
    base = _unsharp(base, 1.81, 2.8 * s)
    effect = np.clip(_unsharp(base, 1.01, 1.6 * s), 0, 1)

    # ---- 3. light strokes (Lighten through a ragged stroke mask) -------------
    if light_amt > 0:
        report(0.5, "Impression: light strokes")
        longs = lic(small, fdx, fdy, length=28 * s + 2, steps=10)
        longs = cv2.resize(longs, (w, h), interpolation=cv2.INTER_CUBIC)
        longs = np.clip(longs * 0.93, 0, 1)            # this layer is a little darker
        m = _noise((hs, ws), 1.2, rng)
        m = lic(m, fdx, fdy, length=14 * s + 1, steps=6)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
        m = m / (m.std() + 1e-6)
        cover = np.clip((m + 0.85) * 2.2, 0, 1) * min(1.0, light_amt)
        lighter = np.maximum(effect, longs)
        effect = effect + (lighter - effect) * cover[..., None]

    # ---- 4. surface: bristle grooves + woven canvas (Overlay) ---------------
    report(0.75, "Impression: canvas")
    x = effect
    if relief > 0:
        bump = bump_layer(h, w, s, np.random.default_rng(seed + 77), relief)
        bump = 0.5 + (bump - 0.5) * (0.4 + 0.6 * F.lum(rgb))      # deeper in light paint
        x = F.blend("Ovrl", x, np.repeat(bump[..., None], 3, axis=2))
    if canvas > 0:
        cv = canvas_layer(h, w, s, np.random.default_rng(seed + 91), canvas)
        x = F.blend("Ovrl", x, np.repeat(cv[..., None], 3, axis=2))

    # ---- 5. border -----------------------------------------------------------
    out = np.clip(x, 0, 1)
    border = float(p.get("im_border", 50)) / 50.0
    if border > 0:
        b = int(round(125 * s * border))
        out = cv2.copyMakeBorder(out, b, b, b, b, cv2.BORDER_CONSTANT, value=(1, 1, 1))
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    report(0.95, "Impression: done")
    return (out[..., ::-1] * 255.0).astype(np.float32)
