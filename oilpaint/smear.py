"""
Oil Smear — a photo turned into flat oil paint that is dragged and smeared
through a thick impasto stroke texture, on a white painted ground. Built
natively (no Photoshop, no PSD, no texture files).

Pipeline (each step measured against a reference made in Photoshop):

  1. paint      edge-preserving smoothing (the "Oil Paint" clean-up),
                Posterize 10, then Cutout: flat patches by tone and colour
                family, and a 2 px blur
  2. texture    a thick oil-paint stroke texture is generated: upright,
                swirling strokes with bristle grooves, ridges and gloss
                over a dark canvas (grey, made at a long side of 1000 px)
  3. smear      Displace with that texture (30 % across, 60 % down, like
                the template), so the paint is dragged along the grooves
  4. tone       the fitted tone of the template's paint layer
  5. surface    the same texture laid on with Overlay 57 % and a lighter
                copy with Multiply 27 %
  6. finish     Vibrance -12 / Saturation +6, an S curve (85 %) and
                Levels 12 / gamma 1.51 (42 %)

Steps 5 and 6 match the template's layers to within one level.
Every texture is generated in code, so the output can be used and sold
freely. Distances are given for the short side of the picture (3000 px in the template) and scaled.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import cv2
import numpy as np

from actions import filters as F

from . import painted
from .impression import flow, lic

REF_SIDE = 3000.0            # short side of the template (3000 x 4000)
MIN_SIDE = 1800              # short side: smaller pictures are made this big, then shrunk
MAP_SIDE = 750               # the stroke texture is made at this short side

# quantiles (0, 5, ... 100 %) of the template's stroke texture brightness
_TEX_Q = np.array([0, 22, 29, 35, 41, 46, 52, 58, 64, 70, 77, 84, 93,
                   103, 114, 126, 139, 156, 176, 200, 255], np.float32)

# the lighter texture copy (Multiply layer) as a curve of the texture
_LIGHT_X = np.arange(0, 256, 8, dtype=np.float32)
_LIGHT_Y = np.array([2.6, 20.5, 38.7, 56.7, 74.8, 92.4, 109.0, 125.0, 140.2, 154.0, 166.8,
                     178.1, 188.6, 198.0, 207.0, 214.0, 220.9, 227.0, 232.2, 236.8, 240.8,
                     244.0, 246.6, 248.8, 250.8, 252.0, 253.0, 254.0, 254.2, 255.0, 255.0,
                     255.0], np.float32)

# tone of the template's paint layer after the cut-out (fitted per channel)
_TONE_X = np.linspace(0, 255, 18).astype(np.float32)
_TONE_RAW = np.array([
    [12.8, 12.8, 15.0, 27.0, 52.9, 66.4, 86.3, 106.3, 113.3, 130.8, 143.1, 164.2, 183.0,
     204.1, 217.5, 238.6, 244.7, 250.8],
    [16.4, 16.4, 18.5, 29.9, 53.7, 69.3, 82.0, 100.2, 116.8, 132.2, 149.1, 169.5, 179.4,
     202.4, 217.8, 247.0, 247.0, 249.1],
    [7.8, 14.1, 19.0, 29.1, 48.4, 67.7, 81.7, 102.4, 116.5, 136.6, 148.6, 171.7, 180.6,
     197.4, 219.7, 242.4, 242.4, 242.4]], np.float32)


def _smooth_curve(y):
    z = y.copy()
    z[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    return np.maximum.accumulate(z)


_TONE_Y = np.stack([_smooth_curve(c) for c in _TONE_RAW])


def _curve_lut(pts):
    """Photoshop-like smooth curve through (in, out) points, 256 entries."""
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    n = len(xs)
    # natural cubic spline
    hh = np.diff(xs)
    a = np.zeros((n, n))
    r = np.zeros(n)
    a[0, 0] = a[-1, -1] = 1
    for i in range(1, n - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = hh[i - 1], 2 * (hh[i - 1] + hh[i]), hh[i]
        r[i] = 3 * ((ys[i + 1] - ys[i]) / hh[i] - (ys[i] - ys[i - 1]) / hh[i - 1])
    c = np.linalg.solve(a, r)
    x = np.arange(256, dtype=np.float64)
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    dx = x - xs[i]
    b = (ys[i + 1] - ys[i]) / hh[i] - hh[i] * (2 * c[i] + c[i + 1]) / 3
    d = (c[i + 1] - c[i]) / (3 * hh[i])
    y = ys[i] + b * dx + c[i] * dx ** 2 + d * dx ** 3
    return (np.clip(y, 0, 255) / 255.0).astype(np.float32)


_S_CURVE = _curve_lut([(0, 0), (72, 68), (169, 196), (255, 255)])
_LUT_X = np.linspace(0, 1, 256, dtype=np.float32)


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
# 2. the impasto stroke texture
# --------------------------------------------------------------------------
def _strokes(h, w, rng, size=1.0, tilt=1.0, base=math.pi / 2):
    """Paint height and paint brightness at map scale."""
    sc = min(h, w) / float(MAP_SIDE)
    # stroke directions: mostly upright arcs that swing left and right
    ang = (base + tilt * (0.27 * _noise((h, w), 160 * sc, rng)
                                 + 0.32 * _noise((h, w), 55 * sc, rng)
                                 + 0.25 * _noise((h, w), 16 * sc, rng))).astype(np.float32)
    ca, sa = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    # bristle streaks along the strokes
    st1 = lic(_noise((h, w), 1.0 * sc, rng), ca, sa, 14 * sc, 8)
    st2 = lic(_noise((h, w), 2.4 * sc, rng), ca, sa, 30 * sc, 10)
    streak = (0.45 * st1 / (st1.std() + 1e-6) + 0.55 * st2 / (st2.std() + 1e-6)).astype(np.float32)
    # dark canvas underneath: weave and crumbs of dry paint
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = np.sin(2 * math.pi * xx / (3.2 * sc)) * np.sin(2 * math.pi * yy / (3.6 * sc))
    crumbs = np.clip(_noise((h, w), 1.4 * sc, rng) - 1.5, 0, None)
    alb = (0.16 + 0.04 * _noise((h, w), 20 * sc, rng) + crumbs * 0.3).astype(np.float32)
    hgt = (0.3 * weave + 0.6 * crumbs).astype(np.float32)

    n = int(300 * (h * w) / (750 * 1000) / size ** 2)
    for _ in range(n):
        x0, y0 = rng.uniform(-40, w + 40) , rng.uniform(-40, h + 40)
        length = rng.uniform(90, 330) * sc * size
        width = rng.uniform(22, 75) * sc * size
        bend = rng.normal(0, 0.002) * min(tilt, 1.0) / (sc * size)
        g = rng.beta(1.2, 2.0)                       # mostly dark paint, some white
        if rng.random() < 0.1:
            g = rng.uniform(0.85, 1.0)
        # follow the flow both ways from the start point
        pts = []
        stp = 3 * sc
        steps = int(length / 2 / stp) + 1
        for sign in (-1, 1):
            x, y, a_off = x0, y0, 0.0
            seg = []
            for _k in range(steps):
                xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
                c, s_ = float(ca[yi, xi]), float(sa[yi, xi])
                ca_, sa_ = math.cos(a_off), math.sin(a_off)
                c, s_ = c * ca_ - s_ * sa_, s_ * ca_ + c * sa_
                x += sign * c * stp
                y += sign * s_ * stp
                a_off += bend * stp * sign
                seg.append((x, y))
            pts = seg[::-1] + [(x0, y0)] if sign == -1 else pts + seg
        P = np.array(pts, np.float32)
        d = np.diff(P, axis=0)
        tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        tang = np.vstack([tang, tang[-1:]])
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        tpar = np.linspace(-1, 1, len(P))
        taper = np.sqrt(np.clip(1 - np.abs(tpar) ** 3, 0.08, 1))
        half = width / 2 * taper * (1 + 0.12 * np.sin(tpar * rng.uniform(2, 6) + rng.uniform(0, 6)))
        poly = np.vstack([P + nrm * half[:, None], (P - nrm * half[:, None])[::-1]])
        pad = int(width * 0.4) + 2
        bx0 = int(max(0, poly[:, 0].min() - pad))
        bx1 = int(min(w, poly[:, 0].max() + pad + 1))
        by0 = int(max(0, poly[:, 1].min() - pad))
        by1 = int(min(h, poly[:, 1].max() + pad + 1))
        if bx1 - bx0 < 3 or by1 - by0 < 3:
            continue
        mk = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        cv2.fillPoly(mk, [np.round((poly - [bx0, by0]) * 8).astype(np.int32)], 1.0,
                     cv2.LINE_AA, shift=3)
        soft = _g(mk, width * 0.12 + 0.5)
        sl = streak[by0:by1, bx0:bx1]
        cover = np.clip((soft - 0.5 + 0.22 * sl) * 5.0 + 0.5, 0, 1)     # frayed bristle ends
        prof = np.clip(soft * 1.4 - 0.2, 0, 1)
        ridge = np.clip(1 - np.abs(soft - 0.55) * 5, 0, 1)              # paint piled at the sides
        sub_h = hgt[by0:by1, bx0:bx1]
        sub_a = alb[by0:by1, bx0:bx1]
        newh = sub_h * 0.4 + 0.35 + 0.35 * prof + 0.6 * ridge + 0.5 * sl
        gv = g * (1 + 0.3 * np.tanh(sl * 1.3) * (0.6 + 0.4 * g)) + 0.08 * ridge + rng.normal(0, 0.02)
        sub_h[:] = sub_h * (1 - cover) + newh * cover
        sub_a[:] = sub_a * (1 - cover) + gv * cover
    return hgt, alb


def _shade(hgt, alb):
    """Light the paint from the top left, with a glossy highlight."""
    h, w = hgt.shape
    sc = min(h, w) / float(MAP_SIDE)
    hs = _g(hgt, 0.8 * sc)
    gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) / 8 * 2.5
    gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) / 8 * 2.5
    nl = np.sqrt(gx * gx + gy * gy + 1)
    lx, ly, lz = -0.6, -0.8, 0.9
    ll = math.sqrt(lx * lx + ly * ly + lz * lz)
    diff = (-gx * lx - gy * ly + lz) / (nl * ll)
    hx, hy, hz = lx / ll, ly / ll, lz / ll + 1
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    spec = np.clip((-gx * hx - gy * hy + hz) / (nl * hn), 0, 1) ** 40
    return alb * (0.35 + 0.85 * np.clip(diff, 0, 1)) + spec * 0.55


_TEX_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def texture(h, w, seed=1, size=1.0, tilt=1.0, base=math.pi / 2):
    """The stroke texture for an h x w picture, 0..255 float32 (cached)."""
    key = (h, w, int(seed), round(float(size), 3), round(float(tilt), 3), round(float(base), 3))
    hit = _TEX_CACHE.get(key)
    if hit is not None:
        _TEX_CACHE.move_to_end(key)
        return hit
    rng = np.random.default_rng(int(seed))
    k = MAP_SIDE / float(min(h, w))
    hm, wm = max(8, int(round(h * k))), max(8, int(round(w * k)))
    hgt, alb = _strokes(hm, wm, rng, size, tilt, base)
    t = _shade(hgt, alb)
    q = np.percentile(t, np.linspace(0, 100, len(_TEX_Q)))
    q = np.maximum.accumulate(q + np.arange(len(q)) * 1e-6)
    t = np.interp(t, q, _TEX_Q).astype(np.float32)
    t = np.clip(cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC), 0, 255)
    _TEX_CACHE[key] = t
    while len(_TEX_CACHE) > 4:
        _TEX_CACHE.popitem(last=False)
    return t


# --------------------------------------------------------------------------
# 3. smear
# --------------------------------------------------------------------------
def displace(img, tex, horiz=30.0, vert=60.0):
    """Photoshop Displace (stretch to fit, repeat edge pixels): a map value
    of 255 moves the picture by scale % of 128 px (on a 4000 px picture)."""
    h, w = img.shape[:2]
    k = 128.0 * min(h, w) / REF_SIDE
    off = (tex - 128.0) / 128.0
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + off * (horiz / 100.0 * k), yy + off * (vert / 100.0 * k),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# the whole look
# --------------------------------------------------------------------------
def _settings(p):
    direction = float(p.get("sm_direction", 0)) / 100.0
    return {
        "seed": int(p.get("sm_seed", 1)),
        "smear": float(p.get("sm_smear", 50)) / 50.0,
        "tex_amt": float(p.get("sm_texture", 50)) / 50.0,
        "tone_amt": float(p.get("sm_tone", 50)) / 50.0,
        "detail": float(p.get("sm_detail", 50)),
        "levels": int(p.get("sm_levels", 8)),
        "size": 2.0 ** ((float(p.get("sm_stroke", 50)) - 50.0) / 50.0),
        "relief": float(p.get("sm_relief", 50)) / 50.0,
        # Direction: 0 = strokes and smears run straight down, 100 = sideways
        "direction": direction,
        "base": math.pi / 2 * (1.0 - direction),
        # Curve: 0 = straight lines, 100 = strongly curving, swirling lines
        "tilt": 0.1 + 1.5 * float(p.get("sm_curve", 15)) / 100.0,
    }


_REL_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def relief_map(h, w, seed, base, tilt):
    """Oil Paint's lighting: a fine brush relief along the strokes (light
    from -60 deg), as the light/dark it adds (cached)."""
    key = (h, w, int(seed), round(float(base), 3), round(float(tilt), 3))
    hit = _REL_CACHE.get(key)
    if hit is not None:
        _REL_CACHE.move_to_end(key)
        return hit
    s = min(h, w) / REF_SIDE
    hs_, ws_ = max(8, h // 2), max(8, w // 2)
    # brush marks run the same way as the smear (keeps faces readable)
    ra = (base + tilt * 0.35 * _noise((hs_, ws_), 40 * s + 2,
                                      np.random.default_rng(seed + 3))).astype(np.float32)
    fdx, fdy = np.cos(ra), np.sin(ra)
    bh = lic(_noise((hs_, ws_), 1.2 * s + 0.3, np.random.default_rng(seed + 4)), fdx, fdy,
             10 * s + 2, 6)
    bh = cv2.resize(bh / (bh.std() + 1e-6), (w, h), interpolation=cv2.INTER_CUBIC)
    a_ = math.radians(-60.0)
    sh = cv2.warpAffine(bh, np.float32([[1, 0, math.cos(a_) * 2 * s + 0.5],
                                        [0, 1, -math.sin(a_) * 2 * s - 0.5]]), (w, h),
                        borderMode=cv2.BORDER_REFLECT)
    rel = (bh - sh).astype(np.float32)
    _REL_CACHE[key] = rel
    while len(_REL_CACHE) > 4:
        _REL_CACHE.popitem(last=False)
    return rel


def _poster(rgb, st, rel):
    """Oil Paint clean-up, its brush relief and Posterize 10 (the GPU does
    the same, static/js/oilfx.js 'Oil Smear')."""
    h, w = rgb.shape[:2]
    s = min(h, w) / REF_SIDE
    r = max(1.0, 6.0 * s * (1.5 - st["detail"] / 100.0))
    sm = F.guided(F.guided(rgb, r, 0.003), r, 0.003)
    # the cut-out below breaks the patch edges on the relief, which gives
    # the small paint fragments of the template
    sm = np.clip(sm * (1 + (rel * 0.06 * st["relief"])[..., None]), 0, 1)
    # Posterize 10 on the brightness only, so pale grounds do not break
    # into pink and green blotches
    L = F.lum(np.clip(sm, 0, 1))[..., None]
    return np.clip(sm * (np.round(L * 9) / 9 + 0.02) / (L + 0.02), 0, 1).astype(np.float32)


def _prepare(rgb, st, report, ana=None, tex=None):
    """Steps 1-2: the flat paint and the stroke texture (the slow part)."""
    h, w = rgb.shape[:2]
    s = min(h, w) / REF_SIDE
    seed, base, tilt = st["seed"], st["base"], st["tilt"]

    # ---- 1. paint: Oil Paint clean-up, Posterize 10, Cutout, blur 2 ---------
    report(0.05, "Oil Smear: oil paint")
    rel = tex["relief"] if tex else relief_map(h, w, seed, base, tilt)
    po = _poster(rgb, st, rel)
    report(0.25, "Oil Smear: cut-out")
    cut = painted.cutout(po, s, None, levels=st["levels"], detail=st["detail"], stats=ana)
    paint = _g(cut, 2.0 * s)

    # ---- 2. stroke texture ---------------------------------------------------
    report(0.45, "Oil Smear: paint texture")
    t = tex["tex"] * 255.0 if tex else texture(h, w, seed, st["size"], tilt, base)
    return paint, t


# --------------------------------------------------------------------------
# for the GPU (static/js/oilfx.js 'Oil Smear'): the random fields, the
# numbers it cannot find itself, and the same steps on the CPU
# --------------------------------------------------------------------------
ASSET_NAMES = ()
MIN_BY = "short"


def tex_key(p):
    st = _settings(p)
    return (st["seed"], round(st["size"], 3), round(st["tilt"], 3), round(st["base"], 3))


def textures(h, w, p):
    st = _settings(p)
    return {"relief": relief_map(h, w, st["seed"], st["base"], st["tilt"]),
            "tex": texture(h, w, st["seed"], st["size"], st["tilt"], st["base"]) / 255.0}


def analyse(rgb, p):
    """The cut-out's brightness range and colour families (painted.py)."""
    st = _settings(p)
    h, w = rgb.shape[:2]
    s = min(h, w) / REF_SIDE
    po = _poster(rgb, st, relief_map(h, w, st["seed"], st["base"], st["tilt"]))
    h2, w2, cs = painted.cutout_res(h, w, s)
    base = painted.cutout_base(painted._small(po, h2, w2), cs, st["detail"])
    return painted.cutout_stats(base, st["levels"])


def look_payload():
    x = np.arange(256, dtype=np.float32)
    tone = np.stack([np.interp(x, _TONE_X, _TONE_Y[c]) / 255.0 for c in range(3)], 1)
    light = np.interp(x, _LIGHT_X, _LIGHT_Y) / 255.0
    aux = np.stack([light, _S_CURVE, np.zeros(256)], 1)
    return {"ref_side": REF_SIDE, "min_side": MIN_SIDE, "min_by": MIN_BY, "textures": True,
            "analyse": True, "tone": np.round(tone, 6).tolist(), "aux": np.round(aux, 6).tolist(),
            "cut_sigma": painted.CUT_SIGMA, "wide_step": painted.WIDE_STEP}


def apply(rgb, tex, ana, p, report=lambda f, x: None):
    """The CPU twin of the GPU pipeline (for checks)."""
    st = _settings(p)
    paint, t = _prepare(rgb, st, report, ana, tex)
    return _finish(paint, t, st, st["smear"], report)


def _finish(paint, tex, st, smear, report):
    """Steps 3-6: smear, tone, surface, colour (fast)."""
    direction, tex_amt, tone_amt = st["direction"], st["tex_amt"], st["tone_amt"]
    # ---- 3. smear through the texture ----------------------------------------
    report(0.6, "Oil Smear: smearing")
    if smear > 0:
        # the drag follows the strokes: mostly down when vertical,
        # mostly sideways when horizontal (template: 30 % across, 60 % down)
        paint = displace(paint, tex, (30.0 + 30.0 * direction) * smear,
                         (60.0 - 30.0 * direction) * smear)

    # ---- 4. tone of the paint layer ------------------------------------------
    x = np.empty_like(paint)
    for c in range(3):
        x[..., c] = np.interp(paint[..., c] * 255.0, _TONE_X, _TONE_Y[c]) / 255.0

    # ---- 5. surface: texture with Overlay 57 % and Multiply 27 % ------------
    report(0.75, "Oil Smear: surface")
    if tex_amt > 0:
        t6 = np.repeat((tex / 255.0)[..., None], 3, axis=2)
        x = x + (F.blend("Ovrl", x, t6) - x) * min(1.0, 145 / 255 * tex_amt)
        t7 = np.interp(tex, _LIGHT_X, _LIGHT_Y).astype(np.float32) / 255.0
        x = x * (1 - min(1.0, 69 / 255 * tex_amt) * (1 - t7[..., None]))

    # ---- 6. finish: vibrance, S curve, levels --------------------------------
    report(0.9, "Oil Smear: finish")
    if tone_amt > 0:
        x = np.clip(x, 0, 1)
        v = F.vibrance(x, -12, 6)
        x = x + (v - x) * min(1.0, tone_amt)
        cur = np.interp(np.clip(x, 0, 1), _LUT_X, _S_CURVE).astype(np.float32)
        x = x + (cur - x) * min(1.0, 217 / 255 * tone_amt)
        lv = F.levels(np.clip(x, 0, 1), 12, 255, 1.51)
        x = x + (lv - x) * min(1.0, 107 / 255 * tone_amt)
    out = np.clip(x, 0, 1)
    return (out[..., ::-1] * 255.0).astype(np.float32)


def _to_work(image_bgr, min_side):
    h0, w0 = image_bgr.shape[:2]
    img = np.asarray(image_bgr, np.float32)
    if min(h0, w0) < min_side:
        k = min_side / min(h0, w0)
        img = cv2.resize(img, (int(round(w0 * k)), int(round(h0 * k))), interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(img[..., ::-1]) / 255.0
    back = (lambda out: cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)) \
        if img.shape[:2] != (h0, w0) else (lambda out: out)
    return rgb, back


_PREP_CACHE = {}


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    def report(f, label):
        if progress:
            progress(f, label)

    rgb, back = _to_work(image_bgr, min_side)
    st = _settings(p)
    # the slow part (paint + texture) is kept for the last picture, so live
    # previews of Smear / Texture / Contrast only redo the fast part
    key = (rgb.shape, float(rgb[::37, ::41].sum()), st["seed"], st["detail"], st["levels"],
           st["relief"], st["base"], st["tilt"], st["size"])
    hit = _PREP_CACHE.get("last")
    if hit is not None and hit[0] == key:
        paint, tex = hit[1]
    else:
        paint, tex = _prepare(rgb, st, report)
        _PREP_CACHE["last"] = (key, (paint, tex))
    out = _finish(paint, tex, st, st["smear"], report)
    report(0.97, "Oil Smear: done")
    return back(out)
