"""
Brush Strokes planner (version 2): the same steps on the CPU and the GPU.

Every step is something a graphics card can do in parallel, and every
random number is a hash of the stroke's grid cell (no random generator
that has to run in order), so this file and ``planBrush`` in
static/js/oilfx.js plan the same strokes.  Keep both in step.

  1. analysis   two edge-keeping simplifications of the photo (fast guided
                filter), the structure tensor (stroke directions, edges,
                how sure the direction is), how busy and how open each
                place is, the user's guiding lines
  2. ground     the canvas colour with a thin wash of the blurred photo
  3. layers     coarse to fine: one seed per grid cell (finer layers only
                where the canvas, estimated from the strokes so far, still
                differs from the photo), each seed traced both ways along
                the directions until the colour changes or an edge stops it
  4. filling    small strokes wherever the canvas is still bare
  5. edges      the messy border strokes of the Edge tab

The plan is made at a planning size (the caller shrinks the photo) and is
scaled to the painting by ``scaled``.  Colours here are RGB 0..1 like the
GPU's textures; the Plan carries BGR 0..255 like the rest of the app.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .canvas_oil import _rnd, gauss, wide, pct, lum
from .params import hex_to_bgr
from .planner import (Plan, NCOL, C_B, C_JB, C_T, C_TX, C_OP, C_HA, C_KIND, C_FLAGS, C_R0, C_SEED,
                      KIND_LAYER, KIND_LAST, KIND_FILL, KIND_EDGE, F_CORRECT, F_ABSTRACT,
                      layout, edge_base_thickness, lerp, concat, _empty_part)

RUN_SIDE = 1600          # a Run is planned at most this big (long side) and scaled up (GPU memory)
PATH_SIGMA = 26.0 / 255  # how strongly a stroke's colour keeps to its loaded colour
COH_EDGE = 0.45


# ---------------------------------------------------------------- randomness
def rnd(i, j, k):
    """0..1 from integer (i, j) and a key - the GPU's rnd()."""
    return _rnd(np.asarray(i, np.int64), np.asarray(j, np.int64), int(k))


def nrm(i, j, k):
    """A normal random number (Box-Muller of two hashes)."""
    u1 = np.maximum(rnd(i, j, k), np.float32(1e-7))
    u2 = rnd(i, j, k + 1)
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(6.2831853 * u2)).astype(np.float32)


# ---------------------------------------------------------------- image steps
def box(x, r):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT) if r > 0 else x


def down(x, f):
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)


def up(x, w, h):
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)


def guided(img, r, eps):
    """Self-guided filter, per channel; the coefficients are found on a copy
    shrunk by r // 4 (He's fast guided filter) - edges stay crisp because
    they are applied to the full-size picture."""
    h, w = img.shape[:2]
    f = max(1, r // 4)
    small = down(img, f) if f > 1 else img
    rs = max(1, int(round(r / f)))
    m = box(small, rs)
    # the spread around each pixel's own local mean (stable in half floats on
    # the GPU, where mean(x*x) - mean(x)^2 loses everything to rounding)
    var = box((small - m) ** 2, rs)
    a = var / (var + eps)
    b = m - a * m
    ma, mb = box(a, rs), box(b, rs)
    if f > 1:
        ma, mb = up(ma, w, h), up(mb, w, h)
    return (ma * img + mb).astype(np.float32)


def simplify_params(simplicity, boost, h, w):
    s = float(simplicity) * float(boost)
    scale = float(np.clip(max(h, w) / 900.0, 0.55, 1.8))
    # fitted against the old edge-keeping filter (cv2.edgePreservingFilter)
    r = int(round(float(np.clip(2.0 + s * 2.2, 2.0, 34.0)) * scale * 0.25))
    eps = float(np.clip(0.05 + s * 0.017, 0.04, 0.42)) ** 2 * 0.05
    return s, max(1, r), eps


def simplify(img, simplicity, boost=1.0):
    """Edge-keeping abstraction: flat colour masses, crisp contours."""
    h, w = img.shape[:2]
    s, r, eps = simplify_params(simplicity, boost, h, w)
    out = guided(guided(img, r, eps), r, eps)
    if s >= 9.5:                                        # a limited palette for the strongest settings
        levels = float(np.clip(34 - s, 19, 26))
        q = np.round(out * levels) / levels
        out = out + (q - out) * float(np.clip((s - 9.5) / 5.5, 0, 1))
    return np.clip(out, 0, 1).astype(np.float32)


def sobel(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return gx, gy


def guide_field(guides, h, w):
    """The user's guiding lines as directions: each segment a band (later
    segments over earlier ones), blurred. Returns (vx, vy, weight) or None."""
    segs = []
    for line in guides or []:
        pts = [(x * (w - 1), y * (h - 1)) for x, y in line]
        for a, b in zip(pts, pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) >= 1e-3:
                segs.append((a[0], a[1], b[0], b[1]))
    if not segs:
        return None
    radius = max(6, int(min(h, w) * 0.05))
    half = radius * 0.5
    vx = np.zeros((h, w), np.float32)
    vy = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    for ax, ay, bx, by in segs:
        x0, x1 = int(max(0, math.floor(min(ax, bx) - half - 1))), int(min(w, math.ceil(max(ax, bx) + half + 2)))
        y0, y1 = int(max(0, math.floor(min(ay, by) - half - 1))), int(min(h, math.ceil(max(ay, by) + half + 2)))
        if x1 <= x0 or y1 <= y0:
            continue
        py, px = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        px += 0.5
        py += 0.5
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / ll, 0, 1)
        d2 = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2
        m = d2 <= half * half
        n = math.sqrt(ll)
        vx[y0:y1, x0:x1][m] = dx / n
        vy[y0:y1, x0:x1][m] = dy / n
        wt[y0:y1, x0:x1][m] = 1.0
    sig = radius * 0.9
    vx, vy, wt = wide(vx, sig), wide(vy, sig), wide(wt, sig)
    n = np.sqrt(vx * vx + vy * vy) + 1e-9
    return vx / n, vy / n, np.clip(wt * 1.8, 0, 1)


class Cache:
    """The analysis kept between plans of the same photo (live previews)."""

    def __init__(self):
        self.keys = {}
        self.vals = {}

    def get(self, name, key, fn):
        if self.keys.get(name) != key:
            self.vals[name] = fn()
            self.keys[name] = key
            # later stages depend on this one
            order = ["refs", "tensor", "dirs"]
            if name in order:
                for later in order[order.index(name) + 1:]:
                    self.keys.pop(later, None)
        return self.vals[name]


def analyse(src, p, cache=None, token=None):
    """src: RGB 0..1 float32 at the planning size."""
    cache = cache if isinstance(cache, Cache) else Cache()
    h, w = src.shape[:2]
    base = (token, h, w)
    simp = int(p["simplicity"])

    def refs():
        ref = simplify(src, simp)
        ref_fine = simplify(src, simp, 0.45)
        return ref, ref_fine

    ref, ref_fine = cache.get("refs", (base, simp), refs)

    def tensor():
        small = float(min(h, w))
        gray = lum(ref_fine).astype(np.float32)
        gx, gy = sobel(gauss(gray, 1.2))
        T = np.dstack([gx * gx, gy * gy, gx * gy]).astype(np.float32)
        gm = gauss(np.sqrt(gx * gx + gy * gy), 2.0)
        grad = np.clip(gm / (pct(gm, 86) + 1e-6), 0, 1).astype(np.float32)
        Tb = wide(T, max(1.5, small / 90.0))
        j11, j22, j12 = Tb[..., 0], Tb[..., 1], Tb[..., 2]
        tmp = np.sqrt(np.maximum((j11 - j22) ** 2 + 4 * j12 * j12, 0))
        l1, l2 = 0.5 * (j11 + j22 + tmp), 0.5 * (j11 + j22 - tmp)
        coh = ((l1 - l2) / (l1 + l2 + 1e-9)).astype(np.float32)
        ex, ey = j12, l1 - j11
        n = np.sqrt(ex * ex + ey * ey)
        deg = n < 1e-12
        n = np.maximum(n, 1e-12)
        dx = np.where(deg, 1.0, -ey / n).astype(np.float32)
        dy = np.where(deg, 0.0, ex / n).astype(np.float32)
        wst = (np.clip(coh * 2.4, 0, 1) * np.clip(grad * 3.2, 0, 1)).astype(np.float32)
        blur_g = wide(gray, max(2.0, small / 160.0))
        tex = gauss(np.abs(gray - blur_g), 6.0)
        tex = np.clip(tex / (pct(tex, 92) + 1e-6), 0, 1).astype(np.float32)
        opn = wide(grad, max(4.0, small / 26.0))
        opn = np.clip(opn / (pct(opn, 92) + 1e-6), 0, 1).astype(np.float32)
        return {"dx": dx, "dy": dy, "coh": coh, "grad": grad, "wst": wst, "tex": tex, "open": opn}

    t = cache.get("tensor", (base, simp), tensor)
    gk = tuple(tuple((round(x, 4), round(y, 4)) for x, y in line) for line in (p.get("guides") or []))

    def dirs():
        k = float(p["direction"]) / 100.0
        ang = math.radians(float(p["global_angle"]))
        gxv, gyv = math.cos(ang), math.sin(ang)
        wl = (1.0 - k) * t["wst"]
        flip = np.where(t["dx"] * gxv + t["dy"] * gyv < 0, -1.0, 1.0).astype(np.float32)
        bx = t["dx"] * flip * wl + gxv * (1 - wl)
        by = t["dy"] * flip * wl + gyv * (1 - wl)
        g = guide_field(p.get("guides"), h, w)
        if g is not None:
            gvx, gvy, gw = g
            bx = bx * (1 - gw) + gvx * gw
            by = by * (1 - gw) + gvy * gw
        n = np.sqrt(bx * bx + by * by) + 1e-9
        return (bx / n).astype(np.float32), (by / n).astype(np.float32)

    dirx, diry = cache.get("dirs", (base, simp, int(p["direction"]), int(p["global_angle"]), gk), dirs)
    out = dict(t)
    out.update({"ref": ref, "ref_fine": ref_fine, "dirx": dirx, "diry": diry, "cache": cache})
    return out


def layer_refs(an, lay, li):
    radii = lay["radii"]
    mix = 0.0 if len(radii) == 1 else (li / (len(radii) - 1.0)) ** 0.85
    src = an["ref"] + (an["ref_fine"] - an["ref"]) * mix
    sig = max(0.7, radii[li] * 0.50)
    return wide(src, sig), wide(src, max(0.5, sig * 0.35))


def make_ground(ref, p, lay):
    bgr = hex_to_bgr(p.get("canvas_color", "#f3ece0"))
    canvas = np.array([bgr[2], bgr[1], bgr[0]], np.float32) / 255.0
    g = wide(ref, max(1.5, lay["base_t"] * 0.85))
    return (canvas + (g - canvas) * float(np.clip(lay["filling"], 0, 1)) * 0.92).astype(np.float32)


# ---------------------------------------------------------------- tracing
def trace(sx, sy, ci, cj, key, c0, steps, min_steps, wobble, an, ref, cref, step, tol,
          inertia, edge_stop):
    """Trace strokes both ways from their seeds along the directions.

    sx, sy: integer seeds; (ci, cj): the stroke's own numbers for the hash;
    c0: loaded colour (RGB 0..1); ref: what the colour is checked against;
    cref: (ref_a, ref_b, use_b) - where each stroke's colour is read along
    the way (ref_b where use_b), or None to keep c0;
    edge_stop: None for no edge stop.  Returns pts, offs, colours."""
    n = len(sx)
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros(1, np.int32), np.zeros((0, 3), np.float32)
    dirx, diry = an["dirx"], an["diry"]
    grad, coh = an["grad"], an["coh"]
    h, w = dirx.shape
    f32 = np.float32
    steps = steps.astype(np.int32)
    min_steps = min_steps.astype(np.int32)
    tol2 = f32(tol * tol)
    smax = int(steps.max())
    fc = f32(1.0 - inertia)
    inertia = f32(inertia)
    step = f32(step)
    s2 = f32(2.0 * PATH_SIGMA * PATH_SIGMA)

    acc = np.zeros((n, 3), f32)
    wsum = np.zeros(n, f32)
    if cref is not None:
        ra, rb, ub = cref
        pick = lambda idx, yy, xx: np.where(ub[idx][:, None], rb[yy, xx], ra[yy, xx])
        c = pick(np.arange(n), sy, sx)                    # the seed counts too
        wv = np.exp(-np.sum((c - c0) ** 2, axis=1) / s2) + 1e-4
        acc += c * wv[:, None]
        wsum += wv

    runs, counts = [], []
    for sgn, sb in ((1.0, 0), (-1.0, 1)):
        x = sx.astype(f32).copy()
        y = sy.astype(f32).copy()
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
            xi, yi = xs.astype(np.int32), ys.astype(np.int32)
            ok = (xi >= 1) & (yi >= 1) & (xi < w - 1) & (yi < h - 1)
            alive[idx[~ok]] = False
            idx, xs, ys, xi, yi = idx[ok], xs[ok], ys[ok], xi[ok], yi[ok]
            if not len(idx):
                break
            dx = dirx[yi, xi] * f32(sgn)
            dy = diry[yi, xi] * f32(sgn)
            if s > 0:
                px, py = pdx[idx], pdy[idx]
                fl = (dx * px + dy * py) < 0
                dx = np.where(fl, -dx, dx)
                dy = np.where(fl, -dy, dy)
                dx = fc * dx + inertia * px
                dy = fc * dy + inertia * py
                nn = np.sqrt(dx * dx + dy * dy)
                good = nn >= 1e-6
                nn = np.maximum(nn, f32(1e-6))
                dx, dy = dx / nn, dy / nn
                alive[idx[~good]] = False
                idx, xs, ys, dx, dy = idx[good], xs[good], ys[good], dx[good], dy[good]
            wob = wobble[idx]
            a = nrm(ci[idx], cj[idx] * 128 + s * 2 + sb, key) * (f32(0.20) * wob)
            a = np.where(wob > 0.001, a, f32(0)).astype(f32)
            ca, sa = np.cos(a), np.sin(a)
            dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
            pdx[idx], pdy[idx] = dx, dy
            xs = xs + dx * step
            ys = ys + dy * step
            x[idx], y[idx] = xs, ys
            xi, yi = xs.astype(np.int32), ys.astype(np.int32)
            ok = (xs >= 0) & (ys >= 0) & (xi < w) & (yi < h)
            xc, yc = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
            past = s >= min_steps[idx]
            c = ref[yc, xc]
            d2 = np.sum((c - c0[idx]) ** 2, axis=1)
            ok &= ~(past & (d2 > tol2))
            if edge_stop is not None:
                later = s >= (min_steps[idx] + 1)
                ok &= ~(later & (grad[yc, xc] > edge_stop) & (coh[yc, xc] < COH_EDGE))
            alive[idx[~ok]] = False
            keep = idx[ok]
            rec[keep, cnt[keep], 0] = xs[ok]
            rec[keep, cnt[keep], 1] = ys[ok]
            cnt[keep] += 1
            if cref is not None and len(keep):
                c = pick(keep, yc[ok], xc[ok])
                wv = np.exp(-np.sum((c - c0[keep]) ** 2, axis=1) / s2) + 1e-4
                acc[keep] += c * wv[:, None]
                wsum[keep] += wv
        runs.append(rec)
        counts.append(cnt)

    (rp, rn), (cp, cn) = runs, counts
    total = cp + cn + 1
    dab = total < 2
    lens = np.where(dab, 2, total).astype(np.int32)
    offs = np.zeros(n + 1, np.int32)
    np.cumsum(lens, out=offs[1:])
    pts = np.zeros((int(offs[-1]), 2), f32)
    k = np.arange(max(1, smax), dtype=np.int32)[None, :]
    base = offs[:-1, None]
    m = (k < cn[:, None]) & ~dab[:, None]
    pts[(base + cn[:, None] - 1 - k)[m]] = rn[m]
    nd = ~dab
    at = offs[:-1] + cn
    pts[at[nd], 0] = sx[nd]
    pts[at[nd], 1] = sy[nd]
    m = (k < cp[:, None]) & ~dab[:, None]
    pts[(base + cn[:, None] + 1 + k)[m]] = rp[m]
    if dab.any():
        di = np.flatnonzero(dab)
        a = rnd(ci[di], cj[di], key + 40) * f32(math.pi)
        r = step * f32(0.6)
        o = offs[di]
        pts[o, 0] = sx[di] - np.cos(a) * r
        pts[o, 1] = sy[di] - np.sin(a) * r
        pts[o + 1, 0] = sx[di] + np.cos(a) * r
        pts[o + 1, 1] = sy[di] + np.sin(a) * r
    if cref is None:
        col = c0.astype(f32)
    else:
        col = np.where((total > 2)[:, None], acc / np.maximum(wsum, 1e-9)[:, None], c0).astype(f32)
    return pts, offs, col


def footprint(pts, offs, widths, h, w):
    """Where the strokes put paint, drawn at a quarter size (the GPU draws
    the same round-ended bands), enlarged back."""
    sh, sw = max(8, int(h * 0.25)), max(8, int(w * 0.25))
    cov = np.zeros((sh, sw), np.uint8)
    if len(widths):
        shift = 2
        ip = np.round(pts * 0.25 * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        width = np.maximum(1, np.round(widths * 0.25)).astype(np.int32)
        lines = np.split(ip, offs[1:-1])
        order = np.argsort(width, kind="stable")
        bounds = np.flatnonzero(np.diff(width[order])) + 1
        for grp in np.split(order, bounds):
            if len(grp):
                cv2.polylines(cov, [lines[i] for i in grp], False, 255, int(width[grp[0]]),
                              cv2.LINE_8, shift=shift)
    return up(cov.astype(np.float32) / 255.0, w, h)


def _rows(n, colour_rgb, jit, t, tx, op, ha, kind, flags, rr, seed):
    st = np.zeros((n, NCOL), np.float32)
    st[:, C_B:C_B + 3] = colour_rgb[:, ::-1] * 255.0             # BGR 0..255
    st[:, C_JB:C_JB + 3] = jit[:, ::-1]
    st[:, C_T] = t
    st[:, C_TX] = tx
    st[:, C_OP] = op
    st[:, C_HA] = ha
    st[:, C_KIND] = kind
    st[:, C_FLAGS] = flags
    st[:, C_R0:C_R0 + 6] = rr
    st[:, C_SEED] = seed
    return st


# ---------------------------------------------------------------- the layers
def cells(h, w, grid):
    nx = int(math.ceil((w - grid * 0.5) / grid))
    ny = int(math.ceil((h - grid * 0.5) / grid))
    cj, ci = np.mgrid[0:ny, 0:nx]
    return ci.ravel().astype(np.int64), cj.ravel().astype(np.int64), nx, ny


def plan_layer(an, lay, li, score, K, p):
    radii = lay["radii"]
    radius = radii[li]
    grad, tex, opn = an["grad"], an["tex"], an["open"]
    h, w = grad.shape
    filling = lay["filling"]
    KL = K + 1000 * (li + 1)
    ref_l, ref_s = layer_refs(an, lay, li)
    grid = max(2.0, radius * (1.05 if li == 0 else 0.85))
    ci, cj, nx, ny = cells(h, w, grid)
    f32, g32 = np.float32, np.float32(grid)                 # float32 maths, like the GPU
    fx = (ci.astype(f32) + f32(0.5)) * g32 + (rnd(ci, cj, KL + 1) * f32(2) - f32(1)) * f32(0.45) * g32
    fy = (cj.astype(f32) + f32(0.5)) * g32 + (rnd(ci, cj, KL + 2) * f32(2) - f32(1)) * f32(0.45) * g32
    ix = np.clip(fx, 0, w - 1).astype(np.int32)
    iy = np.clip(fy, 0, h - 1).astype(np.int32)
    if li == 0:
        sel = rnd(ci, cj, KL + 4) <= lerp(0.42, 1.0, filling)
    else:
        thr = max(lerp(70.0, 24.0, filling) * (1.0 + 0.18 * (li - 1)), pct(score, lerp(88.0, 45.0, filling)))
        half = int(grid * 0.5)
        mean, py, px = patch_pick(score, ix, iy, half)
        sel = mean >= thr
        ix, iy = px.astype(np.int32), py.astype(np.int32)
    order = np.argsort(rnd(ci, cj, KL + 3)[sel], kind="stable")
    ci, cj, ix, iy = ci[sel][order], cj[sel][order], ix[sel][order], iy[sel][order]
    n = len(ix)
    if n == 0:
        return _empty_part(), ref_l
    last = li == len(radii) - 1
    layer_len = lay["max_len"] * (1.0 if li == 0 else lerp(0.62, 0.28, li / 3.0))
    layer_tol = lay["tol"] * (1.0 if li == 0 else lerp(0.80, 0.50, li / 3.0)) / 255.0
    inertia = 0.74 if li == 0 else 0.62
    g = grad[iy, ix]
    shrink = 0.55 + 0.45 * (1.0 - g) ** 0.7
    bg = 1.0 + (lay["wide_bg"] - 1.0) / 9.0 * 1.85 * (1.0 - opn[iy, ix]) ** 1.5
    t_px = radius * shrink * bg * (1.0 + (rnd(ci, cj, KL + 5) * 0.36 - 0.16))
    t_px = np.clip(t_px, 1.3, lay["small"] * 0.18).astype(np.float32)
    sharp = g > 0.35
    c0 = np.where(sharp[:, None], ref_s[iy, ix], ref_l[iy, ix]).astype(np.float32)
    step = max(1.0, radius * 0.38)
    max_len = layer_len * (0.55 + 0.6 * rnd(ci, cj, KL + 6))
    steps = np.maximum(2, (max_len / step).astype(np.int32))
    min_steps = np.maximum(1, (steps * lay["min_len_frac"]).astype(np.int32))
    tx = tex[iy, ix]
    wob = (lay["wobble"] * (0.25 + 0.75 * tx)).astype(np.float32)
    # the colour is read from ref_s next to contours, ref_l elsewhere
    pts, offs, col = trace(ix, iy, ci, cj, KL + 7, c0, steps, min_steps, wob, an, ref_l,
                           (ref_l, ref_s, sharp), step, layer_tol, inertia, lay["edge_stop"])
    js = (lay["col_jit"] * (0.22 + 0.78 * tx)).astype(np.float32)
    jit = (nrm(ci, cj, KL + 9)[:, None] * js[:, None]
           + np.stack([nrm(ci, cj, KL + 11 + 2 * c) for c in range(3)], 1) * (js * 0.3)[:, None])
    rr = np.stack([rnd(ci, cj, KL + 20 + c) for c in range(6)], 1)
    st = _rows(n, col, jit, t_px, tx, 0.9 + 0.16 * rnd(ci, cj, KL + 17), 1.0 + 0.5 * rnd(ci, cj, KL + 18),
               KIND_LAST if last else KIND_LAYER, F_CORRECT | F_ABSTRACT, rr, rnd(ci, cj, KL + 26))
    return {"strokes": st, "pts": pts, "offs": offs}, ref_l


def patch_pick(score, cx, cy, half):
    """Mean of the score in each cell's window (inside the picture) and the
    first place of its maximum (rows top to bottom, then left to right)."""
    h, w = score.shape
    pad = np.pad(score, half, mode="constant", constant_values=np.nan)
    o = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(o, o, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    mean = np.zeros(len(cx), np.float32)
    best = np.zeros(len(cx), np.int64)
    for a in range(0, len(cx), 20000):                  # in chunks: big windows use a lot of memory
        win = pad[cy[a:a + 20000, None] + half + dy[None, :], cx[a:a + 20000, None] + half + dx[None, :]]
        with np.errstate(invalid="ignore"):
            mean[a:a + 20000] = np.nanmean(win, axis=1)
        best[a:a + 20000] = np.argmax(np.where(np.isnan(win), -np.inf, win), axis=1)
    return mean, cy + dy[best], cx + dx[best]


def plan_fill(an, lay, coverage, K):
    filling = lay["filling"]
    if filling <= 0.02:
        return _empty_part()
    h, w = coverage.shape
    KF = K + 9000
    small = down(coverage, 6)
    sy_, sx_ = np.where(small < lerp(0.65, 0.16, filling))
    if not len(sy_):
        return _empty_part()
    sel = rnd(sx_, sy_, KF + 1) < lerp(0.15, 1.0, filling)
    sx_, sy_ = sx_[sel], sy_[sel]
    order = np.argsort(rnd(sx_, sy_, KF + 2), kind="stable")
    ci, cj = sx_[order].astype(np.int64), sy_[order].astype(np.int64)
    n = len(ci)
    if n == 0:
        return _empty_part()
    r = max(1.6, lay["base_t"] * 0.42)
    ref_f = gauss(an["ref_fine"], max(0.8, r * 0.6))
    ix = np.clip(ci * 6 + (rnd(ci, cj, KF + 3) * 6).astype(np.int64), 0, w - 1).astype(np.int32)
    iy = np.clip(cj * 6 + (rnd(ci, cj, KF + 4) * 6).astype(np.int64), 0, h - 1).astype(np.int32)
    base = ref_f[iy, ix].astype(np.float32)
    jit = np.stack([nrm(ci, cj, KF + 5 + 2 * c) for c in range(3)], 1) * lay["col_jit"]
    t_px = (r * (0.85 + 0.55 * rnd(ci, cj, KF + 11))).astype(np.float32)
    step = max(1.0, r * 0.5)
    steps = max(2, int(r * 5.0 / step))
    min_steps = max(1, int(steps * 0.3))
    pts, offs, _ = trace(ix, iy, ci, cj, KF + 12, np.clip(base + jit / 255.0, 0, 1),
                         np.full(n, steps, np.int32), np.full(n, min_steps, np.int32),
                         np.full(n, 0.35, np.float32), an, ref_f, None, step, 40.0 / 255.0, 0.72, None)
    rr = np.stack([rnd(ci, cj, KF + 20 + c) for c in range(6)], 1)
    st = _rows(n, base, jit, t_px, 1.0, 1.0, 1.0, KIND_FILL, F_CORRECT | F_ABSTRACT, rr, rnd(ci, cj, KF + 26))
    return {"strokes": st, "pts": pts, "offs": offs}


def edge_strokes(p, h, w, px_scale):
    """The Edge tab's border strokes: their paths, and where each one takes
    its colour (the colour itself is read by the caller)."""
    if not p.get("edge_enabled"):
        return None
    KE = int(p["edge_seed"]) * 7919 + 77
    n = int(p["edge_number"]) * 4
    small = float(min(h, w))
    thick = edge_base_thickness(p, h, w)
    length = 6.0 + (float(p["edge_max_length"]) / 100.0) * small * 0.32
    ell = bool(p["edge_ellipse"])
    walk = 3.0 * float(px_scale)
    R = lambda i, k: float(rnd(i, 0, KE + k))
    N = lambda i, k: float(nrm(i, 0, KE + k))
    paths, mids, rows = [], [], []
    for i in range(n):
        if ell:
            a = R(i, 1) * 2 * math.pi
            band = 0.86 + 0.2 * R(i, 2)
            x, y = w * 0.5 + math.cos(a) * w * 0.5 * band, h * 0.5 + math.sin(a) * h * 0.5 * band
            tx, ty = -math.sin(a), math.cos(a)
        else:
            side, band, u = i % 4, R(i, 2) * 0.085 * small, R(i, 3)
            if side == 0:
                x, y, tx, ty = u * w, band, 1.0, 0.0
            elif side == 1:
                x, y, tx, ty = u * w, h - band, 1.0, 0.0
            elif side == 2:
                x, y, tx, ty = band, u * h, 0.0, 1.0
            else:
                x, y, tx, ty = w - band, u * h, 0.0, 1.0
        a = N(i, 4) * 0.30
        tx, ty = tx * math.cos(a) - ty * math.sin(a), tx * math.sin(a) + ty * math.cos(a)
        ln = length * (0.35 + 0.65 * R(i, 6))
        ns = max(2, int(ln / walk))
        px, py = x - tx * ln * 0.5, y - ty * ln * 0.5
        pts = np.empty((ns, 2), np.float32)
        for s in range(ns):
            wb = float(nrm(i, s + 1, KE + 8)) * 0.12
            tx, ty = tx * math.cos(wb) - ty * math.sin(wb), tx * math.sin(wb) + ty * math.cos(wb)
            px += tx * walk
            py += ty * walk
            pts[s] = (px, py)
        paths.append(pts)
        mids.append(pts[ns // 2])
        rows.append((thick * (0.6 + 0.65 * R(i, 10)), 0.75 + 0.25 * R(i, 11), R(i, 12),
                     [N(i, 13 + 2 * c) for c in range(3)]))
    return paths, np.array(mids, np.float32), rows


def finish_edges(p, edges, mid_colours):
    paths, _, rows = edges
    use = bool(p["edge_use_color"])
    fixed = np.array(hex_to_bgr(p["edge_color"]), np.float32)          # BGR
    st = np.zeros((len(paths), NCOL), np.float32)
    for i, (t, op, sd, nz) in enumerate(rows):
        if use:
            st[i, C_B:C_B + 3] = fixed
            st[i, C_JB:C_JB + 3] = np.array(nz[::-1], np.float32) * 6
            st[i, C_FLAGS] = 0
        else:
            st[i, C_B:C_B + 3] = mid_colours[i][::-1] * 255.0
            st[i, C_JB:C_JB + 3] = np.array(nz[::-1], np.float32) * 10
            st[i, C_FLAGS] = F_CORRECT
        st[i, C_T] = t
        st[i, C_TX] = 1.0
        st[i, C_OP] = op
        st[i, C_HA] = 1.0
        st[i, C_KIND] = KIND_EDGE
        st[i, C_SEED] = sd
    lens = np.array([len(q) for q in paths], np.int32)
    offs = np.zeros(len(paths) + 1, np.int32)
    np.cumsum(lens, out=offs[1:])
    return {"strokes": st, "pts": np.concatenate(paths).astype(np.float32), "offs": offs}


# ---------------------------------------------------------------- whole plan
def plan(src_rgb, p, cache=None, token=None, px_scale=1.0, report=None):
    """src_rgb: float32 RGB 0..1 at the planning size. Returns a Plan in
    planning-size pixels (ground BGR 0..255)."""
    rep = report or (lambda f, s: None)
    rep(0.02, "Reading the photo")
    an = analyse(src_rgb, p, cache, token)
    h, w = an["grad"].shape
    lay = layout(p, h, w)
    K = int(p["random_seed"]) % 100000 * 1009
    ground = make_ground(an["ref"], p, lay)
    canvas = ground.copy()
    coverage = np.zeros((h, w), np.float32)
    parts = []
    L = len(lay["radii"])
    for li in range(L):
        rep(0.1 + 0.7 * li / L, f"Planning layer {li + 1}/{L}")
        score = None
        if li > 0:
            ref_l, _ = layer_refs(an, lay, li)
            d = canvas - ref_l
            err = np.sqrt(np.sum(d * d, axis=2)) * 255.0 + 14.0 * an["tex"]
            score = (err * (0.40 + 1.80 * an["grad"])).astype(np.float32)
        part, ref_l = plan_layer(an, lay, li, score, K, p)
        parts.append(part)
        if len(part["strokes"]):
            cov = footprint(part["pts"], part["offs"], part["strokes"][:, C_T], h, w)
            a = np.clip(cov * 0.92, 0, 1)[..., None]
            canvas += (ref_l - canvas) * a
            np.maximum(coverage, cov, out=coverage)
    rep(0.85, "Filling empty spaces")
    parts.append(plan_fill(an, lay, coverage, K))
    e = edge_strokes(p, h, w, px_scale)
    if e is not None:
        mids = e[1]
        xs = np.clip(mids[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(mids[:, 1].astype(np.int32), 0, h - 1)
        parts.append(finish_edges(p, e, an["ref_fine"][ys, xs]))
    allp = concat(parts)
    return Plan(size=(w, h), strokes=allp["strokes"], pts=allp["pts"], offs=allp["offs"],
                ground=(ground[..., ::-1] * 255.0).astype(np.float32),
                meta={"base_t": lay["base_t"], "edge_base": edge_base_thickness(p, h, w)})


def scaled(pl, w, h):
    """The plan at another size (the painting's)."""
    pw, ph = pl.size
    k = w / pw
    if abs(k - 1.0) < 1e-6 and (pw, ph) == (w, h):
        return pl
    st = pl.strokes.copy()
    st[:, C_T] *= k
    g = cv2.resize(pl.ground, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = dict(pl.meta)
    meta["base_t"] = meta.get("base_t", 0) * k
    meta["edge_base"] = meta.get("edge_base", 0) * k
    return Plan(size=(w, h), strokes=st, pts=(pl.pts * np.float32(k)).astype(np.float32), offs=pl.offs,
                ground=g, meta=meta)


def plan_bgr(image_bgr, p, side=None, cache=None, token=None, report=None):
    """A plan for a BGR 0..255 picture, made at ``side`` (long side, default
    RUN_SIDE) and scaled to the picture."""
    h, w = image_bgr.shape[:2]
    side = int(side or RUN_SIDE)
    src = np.asarray(image_bgr, np.float32)
    if max(h, w) > side:
        k = side / max(h, w)
        src = cv2.resize(src, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(src[..., ::-1]) / 255.0
    ph, pw = rgb.shape[:2]
    pl = plan(rgb.astype(np.float32), p, cache, token, px_scale=pw / w, report=report)
    return scaled(pl, w, h)
