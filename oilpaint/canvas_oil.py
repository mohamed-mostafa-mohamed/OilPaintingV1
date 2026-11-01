"""
Canvas Oil - the look of the Envato "Canvas Oil Painting" action (item
EXFF2KE by AB-TheDesigner), rebuilt so it runs on the graphics card in a
fraction of a second, like the other filter styles.

Every step is made of things a GPU does in one pass per pixel, and the SAME
steps run in static/js/oilfx.js (PIPELINES['Canvas Oil']); keep both in step:

  1. tone        Levels-style contrast and vibrance
  2. patches     flat, angular knife-cut patches: every pixel takes the colour
                 of the nearest patch centre on a jittered grid, unless the
                 photo's colour there says it belongs to a neighbour (patches
                 stop at the photo's edges); big patches in open areas, small
                 ones where there is detail, squashed so they run down
  3. dabs        small faceted dabs of a slightly different shade
  4. runs        faint vertical streaks of wet paint
  5. brushwork   the paint dragged along the photo's flow, with bristle
                 grooves and their relief
  6. rims        a lit / shadowed rim on every patch edge, pale outlines
  7. surface     mottled pigment, grain, white canvas opening up
  8. sketch      a shaky hand-drawn outline and hatching in the shadows
  9. scratches   cross-hatched knife scratches in clusters, loose ones
 10. speckles    small glowing paint spots
 11. weave       the canvas weave, and a light unsharp mask

The random parts use a hash of the pixel / cell position and the seed (no
random number generator), so the CPU and the GPU make the same picture.
Sizes are in pixels of a 1000 px picture and scale with the photo.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

REF_SIDE = 1000.0
MIN_SIDE = 1000          # previews are made at least this big, then shrunk
ASSET_NAMES = ()
WIDE_STEP = 4.0
CANVAS_WHITE = np.array([244.0, 242.0, 238.0], np.float32) / 255     # RGB, slightly warm
PALE = np.array([220.0, 218.0, 215.0], np.float32) / 255             # scratch colour
INK = np.array([42.0, 28.0, 20.0], np.float32) / 255                 # drawing brown


def settings(p: dict) -> dict:
    f = lambda k, d: float(p.get(k, d)) / 100.0
    return {"patch": f("co_patch", 45), "detail": f("co_detail", 55), "drag": f("co_drag", 40),
            "dabs": f("co_dabs", 55), "relief": f("co_relief", 45), "canvas": f("co_canvas", 45),
            "scratch": f("co_scratches", 20), "speck": 0.0,     # speckles (red / purple / white dots): taken out
            "brush": f("co_brush", 70), "size": f("co_stroke_size", 45), "loose": f("co_loose", 55),
            "sketch": f("co_sketch", 45), "contrast": f("co_contrast", 35), "vib": f("co_vibrance", 40),
            "seed": int(p.get("co_seed", 7)) % 100000}


def look_payload() -> dict:
    return {"ref_side": REF_SIDE, "min_side": MIN_SIDE, "wide_step": WIDE_STEP}


def asset_image(name, h, w):
    raise KeyError(name)


# --------------------------------------------------------------------------
# hash noise (the same integer maths as hsh() / rnd() / vn() in oilfx.js)
# --------------------------------------------------------------------------
_U = np.uint32


def _h(x):
    x = np.asarray(x).astype(np.uint32, copy=True)
    x ^= x >> _U(16)
    x *= _U(0x7FEB352D)
    x ^= x >> _U(15)
    x *= _U(0x846CA68B)
    x ^= x >> _U(16)
    return x


def _rnd(ix, iy, k):
    ix = np.asarray(ix).astype(np.int64).astype(np.uint32)
    iy = np.asarray(iy).astype(np.int64).astype(np.uint32)
    kk = _h(np.asarray(k, np.int64).astype(np.uint32))
    v = _h((ix * _U(0x8DA6B343)) ^ _h((iy * _U(0xD8163841)) ^ kk))
    return (v >> _U(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)


def rnd(ix, iy, k):
    """0..1 from integer cell coordinates and a key. Many pixels share a
    cell, so the hash is made once per cell and looked up (same numbers)."""
    ix, iy = np.asarray(ix), np.asarray(iy)
    if ix.size > 4096 and ix.shape == iy.shape:
        x0, x1 = int(ix.min()), int(ix.max())
        y0, y1 = int(iy.min()), int(iy.max())
        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        if nx * ny * 2 < ix.size:
            gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            tab = _rnd(gx, gy, k)
            return tab[(iy - y0).astype(np.intp), (ix - x0).astype(np.intp)]
    return _rnd(ix, iy, k)


def vn(x, y, k):
    """Smooth value noise 0..1 (one cell per unit)."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    fx, fy = np.floor(x), np.floor(y)
    tx, ty = x - fx, y - fy
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    ux, uy = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    a, b = rnd(ix, iy, k), rnd(ix + 1, iy, k)
    c, d = rnd(ix, iy + 1, k), rnd(ix + 1, iy + 1, k)
    top, bot = a + (b - a) * ux, c + (d - c) * ux
    return (top + (bot - top) * uy).astype(np.float32)


def fbm(x, y, sizes, weights, k):
    out = np.zeros(np.broadcast(x, y).shape, np.float32)
    for i, (sz, wt) in enumerate(zip(sizes, weights)):
        out += vn(x / sz, y / sz, k + i) * np.float32(wt)
    return out / np.float32(sum(weights))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# image helpers (each one a pass on the GPU)
# --------------------------------------------------------------------------
def lum(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def gauss(x, s):
    return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT) if s > 0.05 else x


def wide(x, sigma):
    """A wide Gaussian made on a copy shrunk by a whole factor (the GPU too)."""
    f = int(sigma // WIDE_STEP)
    if f < 2:
        return gauss(x, sigma)
    h, w = x.shape[:2]
    sw, sh = max(1, w // f), max(1, h // f)
    sm = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(gauss(sm, sigma / f), (w, h), interpolation=cv2.INTER_LINEAR)


def sobel(L):
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return gx, gy


def small(x):
    """The copy (at most 256 px long) a percentile is read from - the GPU
    reads the same small copy back."""
    h, w = x.shape[:2]
    k = min(1.0, 256.0 / max(h, w))
    if k < 1.0:
        x = cv2.resize(x, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_AREA)
    return x


def pct(x, q):
    return float(np.percentile(small(x), q))


def sample(img, sx, sy):
    """img at pixel positions (sx, sy) (pixel centres at whole numbers),
    bilinear, kept inside the picture."""
    h, w = img.shape[:2]
    return cv2.remap(img, np.clip(sx, 0, w - 1).astype(np.float32), np.clip(sy, 0, h - 1).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def lic_trace(img, dx, dy, length, steps):
    """Average along the flow, following it as it bends (licTrace on the GPU)."""
    if length < 0.3:
        return img
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = img.astype(np.float32).copy()
    st = length / steps
    for sign in (1.0, -1.0):
        x, y = gx.copy(), gy.copy()
        pdx, pdy = dx * sign, dy * sign
        for _ in range(steps):
            vx = cv2.remap(dx, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            vy = cv2.remap(dy, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flip = np.where(vx * pdx + vy * pdy < 0, -1.0, 1.0).astype(np.float32)
            vx, vy = vx * flip, vy * flip
            x += vx * st
            y += vy * st
            pdx, pdy = vx, vy
            acc += cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / (1 + 2 * steps)


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------
def tone(rgb, st):
    f = rgb
    c = st["contrast"]
    if c > 0:
        sm = small(f)
        lo = float(np.percentile(sm, 0.6)) * c
        hi = 1.0 - (1.0 - float(np.percentile(sm, 99.4))) * c
        f = np.clip((f - lo) / max(0.2, hi - lo), 0, 1)
        f = f + (f * f * (3 - 2 * f) - f) * (0.55 * c)
    v = st["vib"]
    if abs(v) > 1e-3:
        mx, mn = f.max(axis=2, keepdims=True), f.min(axis=2, keepdims=True)
        sat = (mx - mn) / np.maximum(mx, 1e-6)
        ns = sat + (1 - sat) * sat * v * 1.6 + sat * v * 0.25 if v > 0 else sat * (1 + v)
        ns = np.clip(ns, 0, 1)
        f = np.where(sat > 1e-6, mx - (mx - f) * (ns / np.maximum(sat, 1e-6)), f)
    return np.clip(f, 0, 1).astype(np.float32)


def detail_map(L, s):
    gx, gy = sobel(L)
    m = wide(np.sqrt(gx * gx + gy * gy), 10.0 * s)
    return np.clip(m / (pct(m, 90) + 1e-6), 0, 1)


def cells(src, px, py, size, sq, k, lam, warp=True):
    """Knife-cut patches: each pixel takes the colour (of src) at the centre
    of the best of the 3 x 3 nearest cells of a jittered grid; ``lam`` makes
    a pixel join the cell whose colour is closest to its own. Returns
    (colour, cell x, cell y)."""
    h, w = src.shape[:2]
    cw, ch = size, size * sq
    qx, qy = px, py
    if warp:
        amp = max(1.0, size * 0.18)
        qx = px + (vn(px / (1.2 * size), py / (1.2 * size), k) - 0.5) * amp
        qy = py + (vn(px / (1.2 * size), py / (1.2 * size), k + 1) - 0.5) * amp
    gx, gy = np.floor(qx / cw), np.floor(qy / ch)
    best = np.full((h, w), np.inf, np.float32)
    out = np.zeros_like(src)
    bcx, bcy = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            sx = (cx + 0.05 + 0.9 * rnd(cx, cy, k + 2)) * cw
            sy = (cy + 0.05 + 0.9 * rnd(cx, cy, k + 3)) * ch
            col = sample(src, sx - 0.5, sy - 0.5)
            dx, dy = (qx - sx) / cw, (qy - sy) / ch
            d = col - src
            cost = dx * dx + dy * dy + lam * (d * d).sum(axis=2)
            sel = cost < best
            best = np.where(sel, cost, best)
            out[sel] = col[sel]
            bcx, bcy = np.where(sel, cx, bcx), np.where(sel, cy, bcy)
    return out, bcx, bcy


def _cells_over(w, h, size):
    """Every cell of a grid of ``size`` that can touch the picture."""
    for cy in range(int(math.floor(0 / size)) - 1, int(math.floor(h / size)) + 2):
        for cx in range(int(math.floor(0 / size)) - 1, int(math.floor(w / size)) + 2):
            yield cx, cy


def _r(cx, cy, k):
    return float(_rnd(np.int64(cx), np.int64(cy), k))


def _window(w, h, x, y, r):
    x0, x1 = max(0, int(math.floor(x - r - 1))), min(w, int(math.ceil(x + r + 2)))
    y0, y1 = max(0, int(math.floor(y - r - 1))), min(h, int(math.ceil(y + r + 2)))
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return x0, x1, y0, y1, xx + 0.5, yy + 0.5


def scratches(paint, dm, a, s, K):
    """Cross-hatched knife scratches in clusters, and loose single ones,
    lighter than the paint (the GPU draws the same ones per pixel)."""
    h, w = paint.shape[:2]
    light = np.zeros((h, w), np.float32)
    dark = np.zeros((h, w), np.float32)
    Lp = lum(paint).astype(np.float32)
    lw = max(1.0, s / 1.4)
    R = 110.0 * s
    for cx, cy in _cells_over(w, h, R):
        ccx = (cx + _r(cx, cy, K + 130)) * R
        ccy = (cy + _r(cx, cy, K + 131)) * R
        pref = (0.25 + 0.75 * (1 - float(sample(Lp, np.float32([[ccx - 0.5]]), np.float32([[ccy - 0.5]]))[0, 0]))) * \
            (0.35 + 0.65 * float(sample(dm, np.float32([[ccx - 0.5]]), np.float32([[ccy - 0.5]]))[0, 0]))
        if not _r(cx, cy, K + 132) < min(1.0, a * 2.2) * pref:
            continue
        rad = (18.0 + 37.0 * _r(cx, cy, K + 133)) * s
        win = _window(w, h, ccx, ccy, rad)
        if win is None:
            continue
        x0, x1, y0, y1, px, py = win
        rx, ry = px - ccx, py - ccy
        inside = rx * rx + ry * ry < rad * rad
        cid = cx * 7 + cy
        for j, (a0, a1) in enumerate(((30.0, 60.0), (120.0, 150.0))):
            ang = (a0 + (a1 - a0) * _r(cx, cy, K + 134 + j)) * (math.pi / 180)
            gp = (6.0 + 7.0 * _r(cx, cy, K + 136 + j)) * s
            ca, sa = math.cos(ang), math.sin(ang)
            off = (-rx * sa + ry * ca) / gp
            li_ = np.floor(off)
            dline = np.abs(off - li_ - 0.5) * gp
            along = rx * ca + ry * sa
            cidv = np.full(li_.shape, cid, np.int64)
            keep = _rnd(li_, cidv, K + 138 + j) > 0.35
            oc = (li_ + 0.5) * gp
            half = np.sqrt(np.maximum(0.0, rad * rad - oc * oc)) * (0.4 + 0.6 * _rnd(li_, cidv, K + 140 + j))
            seg = inside & keep & (np.abs(along) < half) & (half >= 4.0 * s)
            cov = np.clip(lw - dline + 0.5, 0, 1) * seg
            v = 0.35 + 0.65 * _rnd(li_, cidv, K + 142 + j)
            is_light = _rnd(li_, cidv, K + 144 + j) < 0.8
            sl = (slice(y0, y1), slice(x0, x1))
            light[sl] = np.maximum(light[sl], np.where(is_light, cov * v, 0))
            dark[sl] = np.maximum(dark[sl], np.where(is_light, 0, cov * v))
    # loose single scratches
    Rs = 40.0 * s
    for cx, cy in _cells_over(w, h, Rs):
        if not _r(cx, cy, K + 150) < min(1.0, a * 0.5):
            continue
        sx = (cx + _r(cx, cy, K + 151)) * Rs
        sy = (cy + _r(cx, cy, K + 152)) * Rs
        deg = (45.0 if _r(cx, cy, K + 153) < 0.5 else 135.0) + (_r(cx, cy, K + 154) - 0.5) * 16.0
        ang = deg * (math.pi / 180)
        ln_ = (8.0 + 18.0 * _r(cx, cy, K + 155)) * s
        win = _window(w, h, sx, sy, ln_ + lw)
        if win is None:
            continue
        x0, x1, y0, y1, px, py = win
        ux, uy = math.cos(ang), math.sin(ang)
        rx, ry = px - sx, py - sy
        tq = np.clip(rx * ux + ry * uy, 0, ln_)
        ex, ey = rx - ux * tq, ry - uy * tq
        cov = np.clip(lw - np.sqrt(ex * ex + ey * ey) + 0.5, 0, 1)
        sl = (slice(y0, y1), slice(x0, x1))
        light[sl] = np.maximum(light[sl], cov * (0.3 + 0.5 * _r(cx, cy, K + 156)))
    light = np.clip(light, 0, 1) * (0.25 + 0.4 * a)
    dark = np.clip(dark, 0, 1) * 0.35
    pale = paint * 0.40 + PALE * 0.60
    paint = paint + (pale - paint) * light[..., None]
    return (paint * (1 - dark * 0.5)[..., None]).astype(np.float32)


def speckles(paint, toned, a, s, K):
    """Small glowing paint spots: a few vivid ones, many faint pale ones.
    Drawn in the order the GPU's per-pixel loop keeps them."""
    h, w = paint.shape[:2]
    dots = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    for faint, (count, k0) in enumerate(((90.0, K + 170), (700.0, K + 160))):
        C = math.sqrt(1e6 / (count * a)) * s
        for cx, cy in _cells_over(w, h, C):
            sx = (cx + _r(cx, cy, k0)) * C
            sy = (cy + _r(cx, cy, k0 + 1)) * C
            ix, iy = min(max(sx - 0.5, 0), w - 1), min(max(sy - 0.5, 0), h - 1)
            if faint:
                r = max(1.0, math.floor((1.0 + 1.6 * _r(cx, cy, k0 + 2)) * s + 0.5))
                base = sample(paint, np.float32([[ix]]), np.float32([[iy]]))[0, 0]
                col = np.clip(base * 1.6 + 60.0 / 255, 0, 1)
                av = (0.15 + 0.30 * _r(cx, cy, k0 + 3)) * (0.4 + 0.6 * (1 - float(lum(base))))
            else:
                r = max(1.0, math.floor((1.2 + 2.0 * _r(cx, cy, k0 + 2)) * s + 0.5))
                base = sample(toned, np.float32([[ix]]), np.float32([[iy]]))[0, 0]
                mx = base.max()
                viv = np.clip(mx - (mx - base) * 1.6, 0, 1) if _r(cx, cy, k0 + 4) < 0.7 else CANVAS_WHITE
                col = np.clip(viv * 1.1 + 25.0 / 255, 0, 1)
                av = 0.6 + 0.35 * _r(cx, cy, k0 + 3)
            win = _window(w, h, sx, sy, r)
            if win is None:
                continue
            x0, x1, y0, y1, px, py = win
            ex, ey = px - sx, py - sy
            cov = np.clip(r + 0.5 - np.sqrt(ex * ex + ey * ey), 0, 1)
            sel = cov > 0
            sl = (slice(y0, y1), slice(x0, x1))
            dots[sl][sel] = col
            alpha[sl] = np.where(sel, np.maximum(alpha[sl], cov * av), alpha[sl])
    ab = gauss(alpha, 0.5 * s)[..., None]
    prem = (dots * alpha[..., None]).astype(np.float32)
    colr = gauss(prem, 0.5 * s) / np.maximum(ab, 1e-4)
    paint = paint * (1 - ab) + colr * ab
    return (paint + gauss(prem, 2.5 * s) * 0.45).astype(np.float32)


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    """image_bgr uint8/float BGR 0..255 -> float32 BGR 0..255."""
    h0, w0 = image_bgr.shape[:2]
    if min_side and max(h0, w0) < min_side:
        k = min_side / max(h0, w0)
        big = cv2.resize(np.asarray(image_bgr, np.float32), (int(round(w0 * k)), int(round(h0 * k))),
                         interpolation=cv2.INTER_CUBIC)
        out = render(big, p, progress, min_side=0)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)

    def report(f, label):
        if progress:
            progress(f, f"Canvas Oil: {label}")

    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    out = apply(rgb, p, report)
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)


def apply(rgb, p, report=lambda f, x: None):
    """RGB 0..1 in, RGB 0..1 out - the steps the GPU runs too."""
    st = settings(p)
    h, w = rgb.shape[:2]
    s = max(h, w) / REF_SIDE
    K = st["seed"] * 101
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    px += 0.5
    py += 0.5                                   # pixel centres, like gl_FragCoord

    # 1. tone ---------------------------------------------------------------
    report(0.05, "levels and vibrance")
    toned = tone(rgb, st)
    cs = gauss(toned, 1.0 * s)                  # the colour the patches take
    dm = detail_map(lum(toned), s)

    # 2. patches ------------------------------------------------------------
    report(0.15, "paint patches")
    big_px = (10.0 + 50.0 * st["patch"]) * s
    paint = cells(cs, px, py, big_px, 1.0 + 0.9 * st["drag"], K + 10, 8.0)[0]
    if st["detail"] > 0.01:
        fine = cells(cs, px, py, max(4.0 * s, big_px * 0.32), 1.0 + 0.54 * st["drag"], K + 20, 8.0)[0]
        jit = (vn(px / (1.5 * big_px), py / (1.5 * big_px), K + 30) - 0.5) * 0.5
        d = st["detail"]
        m = smoothstep(0.62 - 0.5 * d, 0.78 - 0.5 * d, dm + jit)
        paint = paint + (fine - paint) * m[..., None]

    # 3. dabs ---------------------------------------------------------------
    if st["dabs"] > 0.01:
        report(0.3, "knife dabs")
        dsz = (6.0 + 10.0 * st["patch"]) * s
        col, cx, cy = cells(paint, px, py, dsz, 1.0 + 0.8 * st["drag"], K + 40, 0.0, warp=False)
        shade = (rnd(cx, cy, K + 41) + rnd(cx, cy, K + 42) + rnd(cx, cy, K + 43) - 1.5) * 0.11
        tint = np.stack([(rnd(cx, cy, K + 44 + i) - 0.5) * 0.0624 for i in range(3)], 2)
        cell_col = col * (1 + shade[..., None] + tint)
        shaded = paint * (1 + shade[..., None] * 0.8)
        drift = smoothstep(0.25, 0.75, fbm(px, py, (dsz * 14, dsz * 5), (0.6, 0.4), K + 50))
        f = ((1 - dm * 0.85) * st["dabs"] * (0.35 + 0.65 * drift))[..., None]
        shaded = paint + (shaded - paint) * f
        paint = shaded + (cell_col - shaded) * np.clip(f * 0.9, 0, 0.6)

    # 4. wet paint runs -----------------------------------------------------
    if st["drag"] > 0.01:
        streak = np.clip((vn(px / (3.0 * s), py / (40.0 * s), K + 60) - 0.5) * 3.0, -1, 1)
        paint = paint * (1 + (streak * 0.10 * st["drag"] * (1 - lum(paint)))[..., None])
    paint = paint.astype(np.float32)

    # 5. brushwork ----------------------------------------------------------
    relief = st["relief"]
    if st["brush"] > 0.01:
        report(0.45, "brushwork")
        L = lum(gauss(paint, 1.5 * s))
        gx, gy = sobel(L)
        fs = 4.0 * s + 1
        jxx, jyy, jxy = gauss(gx * gx, fs), gauss(gy * gy, fs), gauss(gx * gy, fs)
        ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
        ang = ang + (vn(px / (30.0 * s), py / (30.0 * s), K + 70) - 0.5) * st["loose"]
        dx, dy = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        ln = (8.0 + 22.0 * st["size"] + 10.0 * st["loose"]) * s
        sw = (3.0 + 8.0 * st["size"]) * s                      # stroke width
        # short strokes where there is detail, long sweeping ones in open areas
        stroked = lic_trace(paint, dx, dy, ln * 0.6, 6)
        sweep = lic_trace(paint, dx, dy, ln * 1.8, 8)
        stroked = stroked + (sweep - stroked) * (1 - dm)[..., None]
        # the strokes: noise the width of a stroke dragged along the flow
        sf = np.clip((lic_trace(vn(px / sw, py / sw, K + 72), dx, dy, ln, 6) - 0.5) * 6.0, -1, 1).astype(np.float32)
        # the bristles: fine noise dragged further
        noise = rnd(np.floor(px), np.floor(py), K + 71)
        bris = np.clip((lic_trace(noise, dx, dy, ln * 1.5, 6) - 0.5) * 7.0, -1.5, 1.5).astype(np.float32)
        hgt = sf * 0.6 + bris * 0.4
        emb = cv2.filter2D(hgt, -1, np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], np.float32),
                           borderType=cv2.BORDER_REFLECT)
        lit = sf * 0.10 * (0.5 + st["loose"]) + bris * 0.05 + emb * 0.35 * (0.48 + 0.2 * st["loose"])
        stroked = stroked + (stroked - gauss(stroked, 6.0 * s)) * 0.5      # crisp paint inside the strokes
        stroked = stroked * (1 + lit[..., None])
        paint = paint + (stroked - paint) * min(1.0, st["brush"] * 1.1)
        relief *= 1.0 - 0.55 * st["brush"]

    # 6. rims ---------------------------------------------------------------
    report(0.6, "paint rims")
    g = lum(paint)
    gx, gy = sobel(gauss(g, 0.9 * s))
    emb = -(gx + gy) * 0.7071
    emb = np.clip(emb / (pct(np.abs(emb), 98.5) + 1e-6), -1.6, 1.6)
    paint = paint * (1 + (emb * 0.34 * relief)[..., None])
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (pct(mag, 99.0) + 1e-6)
    seam = gauss(np.clip(mag, 0, 1), 0.6 * s)
    darker = (gauss(g, 2.0 * s) > g).astype(np.float32)
    paint = paint * (1 - (seam * darker * 0.35 * relief)[..., None])
    line = smoothstep(0.6, 1.0, mag) * (1 - darker) * 0.35 * relief
    paint = paint + (1 - paint) * line[..., None]

    # 7. mottled pigment, grain, the canvas opening up ----------------------
    mott = (fbm(px, py, (40 * s, 12 * s, 4 * s), (0.45, 0.35, 0.2), K + 80) - 0.5) * 1.6
    grain = (vn(px / (0.9 * s + 0.5), py / ((2.0 + 5.0 * st["drag"]) * 1.5 * s + 0.5), K + 84) - 0.5) * 4.0
    paint = paint * (1 + (mott * 0.14 + grain * 0.035)[..., None])
    if st["canvas"] > 0.01:
        Lp = lum(paint)
        blot = (fbm(px, py, (90 * s, 30 * s, 10 * s), (0.5, 0.3, 0.2), K + 90) - 0.5) * 1.6 + 0.5
        th = 0.88 - 0.20 * st["canvas"] + (blot - 0.5) * 0.28
        m = smoothstep(th, th + 0.10, Lp)
        rag = vn(px / (4 * s), py / (4 * s), K + 94)
        m = np.clip(m * (0.75 + 0.5 * rag), 0, 1)
        m = np.where(m > 0.55, np.minimum(1.0, m * 1.15), m * 0.6)
        paint = paint + (CANVAS_WHITE - paint) * (m * min(0.92, 0.30 + 0.8 * st["canvas"]))[..., None]
    paint = paint.astype(np.float32)

    # 8. sketch lines -------------------------------------------------------
    if st["sketch"] > 0.01:
        report(0.7, "sketch lines")
        a = st["sketch"]
        Ls = lum(cs)
        gx, gy = sobel(gauss(Ls, 1.6 * s))
        mag = np.sqrt(gx * gx + gy * gy)
        thr = pct(mag, 100.0 - (4.0 + 10.0 * a))           # the strongest few percent of edges
        e = (gauss(Ls, 1.0 * s) - gauss(Ls, 2.2 * s)).astype(np.float32)
        ex, ey = sobel(e)
        ge = np.sqrt(ex * ex + ey * ey) / 8.0                 # how fast e changes, per pixel
        emg = np.dstack([e, ge, mag]).astype(np.float32)
        lw = 0.9 * max(1.0, 0.8 * s)
        ink = np.zeros((h, w), np.float32)
        cell = 25.0 * s
        for i in range(2):                      # the outline drawn twice, never quite on top
            kk = K + 100 + 10 * i
            wx = px + (vn(px / (18 * s), py / (18 * s), kk) - 0.5) * 3.0 * s + (i - 0.5) * 1.2 * s
            wy = py + (vn(px / (18 * s), py / (18 * s), kk + 1) - 0.5) * 3.0 * s
            q = sample(emg, wx - 0.5, wy - 0.5)
            dist = np.abs(q[..., 0]) / (q[..., 1] + 1e-5)    # distance to the edge's middle
            li = (1 - smoothstep(0.35 * lw, lw + 0.35, dist)) * smoothstep(0.6 * thr, 1.0 * thr, q[..., 2])
            li = li * (vn(px / cell, py / cell, kk + 2) > 0.32)            # broken lines
            ink = np.maximum(ink, li * (0.55 + 0.45 * rnd(np.floor(px / cell), np.floor(py / cell), kk + 3)))
        # hatching in the shadows
        hc = 40.0 * s
        hx, hy = np.floor(px / hc), np.floor(py / hc)
        ha = (30.0 + 30.0 * rnd(hx, hy, K + 120)) * (math.pi / 180)
        gap = (3.0 + 2.5 * rnd(hx, hy, K + 121)) * s
        tt = (px * np.cos(ha) + py * np.sin(ha)) / gap
        lw = 0.65 * max(1.0, s)
        hl = 1 - smoothstep(0.3 * lw, lw, np.abs(tt - np.floor(tt) - 0.5) * gap)
        dk = 1 - lum(paint)
        hm = smoothstep(0.55, 0.8, vn(px / (22 * s), py / (22 * s), K + 122)) * smoothstep(0.35, 0.75, dk * dk * 1.6)
        ink = np.maximum(ink, hl * hm * (0.3 + 0.4 * rnd(hx, hy, K + 123)) * min(1.0, a * 1.6))
        dry = 0.45 + 0.55 * smoothstep(0.15, 0.5, vn(px / (2 * s), py / (2 * s), K + 124))
        ink = np.clip(ink, 0, 1) * dry * (0.55 + 0.4 * a)
        ink_col = paint * 0.22 + INK * 0.78
        paint = paint + (ink_col - paint) * ink[..., None]

    # 9. scratches ----------------------------------------------------------
    if st["scratch"] > 0.01:
        report(0.8, "knife scratches")
        paint = scratches(paint, dm, st["scratch"], s, K)

    # 10. speckles ----------------------------------------------------------
    if st["speck"] > 0.01:
        report(0.88, "speckles")
        paint = speckles(paint, toned, st["speck"], s, K)

    # 11. canvas weave, unsharp mask ----------------------------------------
    if st["canvas"] > 0.01:
        report(0.95, "canvas")
        per = max(2.5, 4.2 * s)
        jx = (vn(px / (60 * s), py / (60 * s), K + 180) - 0.5) * per * 1.2
        jy = (vn(px / (60 * s), py / (60 * s), K + 181) - 0.5) * per * 1.2
        wx = np.sin((px + jx) * (2 * math.pi / per))
        wy = np.sin((py + jy) * (2 * math.pi / (per * 1.12)))
        weave = np.where(wx * wy > 0, wx, wy) * 0.6
        gs = 0.9 * max(1.0, s)
        grain = (vn(px / gs, py / gs, K + 182) - 0.5) * 4.0
        fibre = (fbm(px, py, (25 * s, 8 * s, 3 * s), (0.4, 0.35, 0.25), K + 183) - 0.5) * 3.2
        tex = (weave * 0.55 + grain * 0.25 + fibre * 0.35).astype(np.float32)
        tex = tex + cv2.filter2D(tex, -1, np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], np.float32),
                                 borderType=cv2.BORDER_REFLECT) * 0.18
        tex = np.clip(tex * 1.333, -1, 1)
        kk = (0.04 + 0.12 * lum(paint)) * (0.4 + 0.9 * st["canvas"])
        paint = paint * (1 + (tex * kk)[..., None])
    paint = paint.astype(np.float32)
    paint = paint + (paint - gauss(paint, 1.2 * s)) * 0.35
    report(1.0, "done")
    return np.clip(paint, 0, 1).astype(np.float32)
