"""
Painted Photo — flat, cut-out paint patches on kraft paper with a warm film
grade, built natively (no Photoshop, no template files).

Pipeline (each step measured against a reference made in Photoshop):

  1. smooth    edge-preserving smoothing (the "Oil Paint" clean-up)
  2. cut out   the photo is split into flat patches by brightness level and
               by colour family; each patch takes its average colour
  3. brush     ragged patch edges, thin dark outlines on strong edges,
               palette-knife speckle and a light canvas relief
  4. paper     a kraft-paper grain laid on with Soft Light, and an old
               cream paper with stains laid on with Multiply
  5. grade     a warm film grade (orange skin, cool blue-grey shadows)

Every texture is generated in code, so the output can be used and sold
freely. Distances are given for a 3000 px wide picture and scaled.
"""
from __future__ import annotations

import cv2
import numpy as np

from actions import filters as F

REF_SIDE = 3000.0
MIN_SIDE = 1000            # previews are made at least this big, then shrunk

# Warm film grade, fitted against the reference: per output channel, the
# weights of [r, g, b, r², g², b², rg, rb, gb, r³, g³, b³, 1].
_GRADE = np.array([
    [-0.46363, -0.13196, 0.67563],
    [0.25193, -0.63449, -1.04260],
    [1.29169, 1.52646, 0.98443],
    [3.28805, 0.72469, -1.79698],
    [0.37956, 3.83014, 5.93925],
    [1.62226, 0.01343, -1.34996],
    [0.20077, 0.01673, -2.52395],
    [-0.71535, -0.47361, 0.69505],
    [-4.39733, -2.80400, 1.28643],
    [-2.19806, -0.34463, 1.97385],
    [1.40940, -1.59927, -3.95150],
    [0.12600, 0.66417, -0.12139],
    [0.12644, 0.12480, 0.07914],
], np.float32)


def _gauss(x, s):
    return cv2.GaussianBlur(x, (0, 0), s) if s > 0.05 else x


def _noise(shape, sigma, rng):
    """Smooth noise, std 1. Wide noise is made small and enlarged (fast)."""
    h, w = shape
    f = 1.0
    if sigma > 6:
        f = sigma / 3.0
        h, w = max(4, int(h / f)), max(4, int(w / f))
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = _gauss(n, sigma / f)
    if f > 1:
        n = cv2.resize(n, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


# --------------------------------------------------------------------------
# 1-3. the painted patches
# --------------------------------------------------------------------------
CUT_SIGMA = 32.0       # how far a patch gathers its colour (reference px)
FAMILIES = 6


WIDE_STEP = 4.0


def wide_blur(x, sigma):
    """A wide Gaussian blur made on a copy shrunk by a whole factor (fast);
    the GPU does exactly this too (static/js/oilfx.js, wideBlur)."""
    f = int(sigma // WIDE_STEP)
    if f < 2:
        return _gauss(x, sigma)
    h, w = x.shape[:2]
    sw, sh = max(1, w // f), max(1, h // f)
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = _gauss(small, sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def cutout_res(h, w, s):
    """Big pictures are cut out at half size (or less): the patches are large."""
    while s > 0.6:
        h, w, s = h // 2, w // 2, s / 2
    return h, w, s


def cutout_base(small, s, detail=50):
    r = max(2.0, (10.0 - 6.0 * detail / 100.0) * s)
    base = small
    for _ in range(2):
        base = F.guided(base, r, 0.003)
    return base


def cutout_stats(base, levels=8, seed=3):
    """The two numbers the GPU cannot find itself: the brightness range and
    the colour families (k-means). The server sends them to the GPU."""
    L = F.lum(base)
    lo, hi = np.percentile(L, 1), np.percentile(L, 99)
    lab = cv2.cvtColor(base.astype(np.float32), cv2.COLOR_RGB2Lab)
    ab = lab[..., 1:]
    sw = 160
    smp = cv2.resize(ab, (sw, max(8, int(sw * ab.shape[0] / ab.shape[1]))),
                     interpolation=cv2.INTER_AREA).reshape(-1, 2).astype(np.float32)
    cv2.setRNGSeed(seed)
    _, _, cen = cv2.kmeans(smp, FAMILIES, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-2),
                           2, cv2.KMEANS_PP_CENTERS)
    return {"lo": float(lo), "hi": float(hi), "cen": cen.astype(np.float32).tolist()}


def _small(rgb, h, w):
    return rgb if rgb.shape[:2] == (h, w) else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def cutout(rgb, s, rng=None, levels=8, detail=50, seed=3, stats=None):
    """Flat patches: brightness levels split by colour family; every pixel
    takes the mean colour of the nearby pixels of its own level and family
    (a wide, level-limited blur - the GPU can do the same, see oilfx.js)."""
    h0, w0 = rgb.shape[:2]
    h, w, cs = cutout_res(h0, w0, s)
    small = _small(rgb, h, w)
    base = cutout_base(small, cs, detail)
    st = stats or cutout_stats(base, levels, seed)
    L = F.lum(base)
    n = int(levels)
    lv = np.clip(((L - st["lo"]) / (st["hi"] - st["lo"] + 1e-6) * n).astype(np.int32), 0, n - 1).astype(np.uint8)
    ks = int((9 - 4 * detail / 100.0) * cs) | 1
    lv = cv2.medianBlur(lv, max(3, min(ks, 255)))
    # colour families, so a blue eye is not merged into the skin around it
    ab = cv2.cvtColor(base.astype(np.float32), cv2.COLOR_RGB2Lab)[..., 1:]
    cen = np.asarray(st["cen"], np.float32)
    d = np.stack([((ab - c) ** 2).sum(-1) for c in cen], 0)
    nearest = d.argmin(0)
    prob = np.stack([_gauss((nearest == i).astype(np.float32), 2.5 * cs) for i in range(len(cen))], 0)
    fam = prob.argmax(0).astype(np.int32)
    key = lv.astype(np.int32) * len(cen) + fam
    out = np.zeros_like(base)
    for v in np.unique(key):
        m = (key == v).astype(np.float32)
        acc = wide_blur(np.dstack([base * m[..., None], m]), CUT_SIGMA * cs)
        out += acc[..., :3] / np.maximum(acc[..., 3:], 1e-6) * m[..., None]
    if (h, w) != (h0, w0):
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return out


def brushwork(patches, rgb, s, rng, outline=50, texture=50, tex=None):
    # ragged patch edges (dry brush / palette knife)
    dx = tex["dx"] if tex else _noise(patches.shape[:2], 1.5 * s, rng)
    dy = tex["dy"] if tex else _noise(patches.shape[:2], 1.5 * s, rng)
    out = F.remap(patches, dx * 1.6 * s, dy * 1.6 * s)
    out = np.power(np.clip(out, 0, 1), 1.1)       # the paint filters darken a little
    # push neighbouring patches apart a little, like Poster Edges does
    out = np.clip(out + 0.3 * (out - _gauss(out, 12 * s)), 0, 1)
    # thin dark outlines on strong edges
    if outline > 0:
        Lb = F.lum(_gauss(rgb, 3.0 * s))           # only big, strong edges (not hair tips)
        gx = cv2.Sobel(Lb, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Lb, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        dark = (_gauss(Lb, 2.5 * s) - Lb) > 0
        thr = 0.2 * (50.0 / max(outline, 1)) ** 0.5
        edge = np.clip((mag - thr) / 0.18, 0, 1) * dark
        w = max(1, int(round(3.0 * s)))
        edge = cv2.dilate(edge.astype(np.float32), np.ones((w, w), np.uint8))
        edge = _gauss(edge, 0.6 * s)
        out = out * (1 - edge[..., None] * 0.85)
        # fine, very strong edges (eyes, lashes, lips) get a thinner line
        Lf = F.lum(_gauss(rgb, 1.0 * s))
        fx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=3)
        fmag = np.sqrt(fx * fx + fy * fy)
        fdark = (_gauss(Lf, 2.0 * s) - Lf) > 0.04
        fine = np.clip((fmag - 2.2 * thr) / 0.2, 0, 1) * fdark
        fine = _gauss(fine.astype(np.float32), 0.5 * s)
        out = out * (1 - fine[..., None] * 0.7)
    # palette-knife speckle and a light canvas relief
    t = texture / 50.0
    if t > 0:
        cells = np.tanh(2.5 * (tex["cells"] if tex else _noise(patches.shape[:2], 1.0 * s, rng)))
        out = out * (1 + cells[..., None] * 0.02 * t)
        shade = tex["shade"] if tex else canvas_shade(*patches.shape[:2], s)
        out = np.clip(np.clip(out, 0, 1) * (1 + shade[..., None] * (2 * t) / 50.0 * 2), 0, 1)
    return np.clip(out, 0, 1)


def canvas_shade(h, w, s):
    """The Texturizer's canvas relief (light from the top), -1..1 - the same
    numbers F.fg_texturizer uses, without the picture."""
    t = F.texture_map("TxCa", h, w, max(0.3, 200 / 100.0 * s))
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    shade = -(gx * 0.7 + gy * 0.7)
    return (shade / (np.abs(shade).max() + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------
# 4. paper
# --------------------------------------------------------------------------
def _fibres(h, w, s, rng, per_mpx, length, light, dark, horiz=0.6):
    """Short, slightly curved paper fibres: crisp 1 px strands, mostly lying
    one way like pressed paper, some lighter and some darker."""
    m = np.zeros((h, w), np.float32)
    n = int(per_mpx * h * w / 1e6 / max(s * s, 1e-3) * s * s)
    if n <= 0:
        return m
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    ang = rng.normal(0, 0.9 * (1 - horiz) + 0.35, n)
    ln = rng.uniform(length[0], length[1], n) * s
    bend = rng.normal(0, 0.25, n)
    val = np.where(rng.random(n) < light / (light + dark), 1.0, -1.0) * rng.uniform(0.5, 1.0, n)
    th = max(1, int(round(s)))
    for i in range(n):
        a = ang[i]
        pts = []
        x, y = xs[i], ys[i]
        for _ in range(4):
            pts.append((int(x * 8), int(y * 8)))
            x += np.cos(a) * ln[i] / 3
            y += np.sin(a) * ln[i] / 3
            a += bend[i]
        cv2.polylines(m, [np.array(pts, np.int32)], False, float(val[i]), th, cv2.LINE_AA, shift=3)
    return m


def _fractal(h, w, s, rng, sizes, gains):
    """Multi-scale (fractal) noise: irregular, paper-like, not round blobs."""
    out = np.zeros((h, w), np.float32)
    for size, gain in zip(sizes, gains):
        out += _noise((h, w), max(0.6, size * s), rng) * gain
    return out / (out.std() + 1e-6)


def kraft(h, w, s, rng, grain_amt=1.0):
    """Brown kraft paper: even, fine grain with light fibre hairs and a few
    dark specks; almost no large blotches (measured on the reference)."""
    base = np.array([178.3, 140.8, 95.2], np.float32) / 255.0
    g = rng.standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=0.55 * s + 0.15, sigmaY=0.45 * s + 0.12)
    g /= g.std() + 1e-6
    clump = _noise((h, w), 1.6 * s + 0.2, rng)
    g = 0.8 * g + 0.2 * clump
    g /= g.std() + 1e-6
    grain = g * 0.11 * grain_amt
    fib = _fibres(h, w, s, rng, per_mpx=700, length=(6, 24), light=3, dark=1)
    specks = np.zeros((h, w), np.float32)
    k = int(25 * h * w / 1e6 / max(s * s, 1e-3) * s * s)
    for x, y, r in zip(rng.uniform(0, w, k), rng.uniform(0, h, k), rng.uniform(1.0, 3.2, k)):
        cv2.circle(specks, (int(x * 8), int(y * 8)), int(r * s * 8) + 4, 1.0, -1, cv2.LINE_AA, shift=3)
    low = _fractal(h, w, s, rng, (40, 160), (0.5, 1.0)) * 0.006
    fine = grain + (fib * 0.14 - specks * 0.3) * min(1.0, grain_amt)   # neutral: light/dark only
    tint = np.array([1.0, 0.97, 0.88], np.float32)
    return np.clip(base + fine[..., None] * 0.94 + low[..., None] * tint, 0, 1)


def old_paper(h, w, s, rng, grain_amt=1.0):
    """Old cream paper: irregular yellowing that gets stronger towards the
    edges, faint creases, pressed fibres and tiny flecks (colours and
    strengths measured on the reference)."""
    base = np.array([241.9, 228.8, 197.4], np.float32) / 255.0
    # yellow-brown ageing: irregular at every scale, more of it near the edges
    age = _fractal(h, w, s, rng, (8, 30, 120, 450), (0.45, 0.75, 1.0, 0.6))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.maximum(np.abs(xx / w - 0.5), np.abs(yy / h - 0.5)) * 2      # 0 centre .. 1 edge
    age = age * 0.75 + edge ** 3 * 1.3
    age -= age.mean()
    # faint creases: long thin light/dark lines
    crease = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 7))):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        a = rng.uniform(0, np.pi)
        L = rng.uniform(0.3, 0.9) * max(h, w)
        x1, y1 = x0 + np.cos(a) * L, y0 + np.sin(a) * L
        cv2.line(crease, (int(x0 * 8), int(y0 * 8)), (int(x1 * 8), int(y1 * 8)),
                 float(rng.choice([-1, 1])), max(1, int(round(s))), cv2.LINE_AA, shift=3)
    crease = _gauss(crease, 0.8 * s)
    g = rng.standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2 * s + 0.2, sigmaY=0.5 * s + 0.15)
    g /= g.std() + 1e-6
    fib = _fibres(h, w, s, rng, per_mpx=900, length=(10, 40), light=4, dark=1)
    flecks = np.zeros((h, w), np.float32)
    k = int(40 * h * w / 1e6 / max(s * s, 1e-3) * s * s)
    for x, y, r in zip(rng.uniform(0, w, k), rng.uniform(0, h, k), rng.uniform(0.8, 2.2, k)):
        cv2.circle(flecks, (int(x * 8), int(y * 8)), int(r * s * 8) + 4, 1.0, -1, cv2.LINE_AA, shift=3)
    fine = (g * 0.085 + fib * 0.14 - flecks * 0.4) * grain_amt + crease * 0.05
    fine -= fine.mean()
    mid = _fractal(h, w, s, rng, (4, 12, 25), (0.6, 1.0, 0.8))
    # ageing darkens mostly the blue and green (yellow-brown), as measured
    age_tint = np.array([0.026, 0.037, 0.052], np.float32)
    mid_tint = np.array([0.010, 0.015, 0.024], np.float32)
    fleck_tint = np.array([0.45, 0.7, 1.0], np.float32)
    out = (base - age[..., None] * age_tint - mid[..., None] * mid_tint
           + fine[..., None] * fleck_tint)
    return np.clip(out, 0, 1)


def _soft_light(b, s):
    d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.clip(b, 0, 1)))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


# --------------------------------------------------------------------------
# 5. grade
# --------------------------------------------------------------------------
def grade(x, amount=100):
    if amount <= 0:
        return x
    v = x.reshape(-1, 3)
    r, g, b = v[:, 0], v[:, 1], v[:, 2]
    f = np.stack([r, g, b, r * r, g * g, b * b, r * g, r * b, g * b,
                  r ** 3, g ** 3, b ** 3, np.ones_like(r)], 1)
    y = np.clip(f @ _GRADE, 0, 1).reshape(x.shape)
    a = amount / 100.0
    return x * (1 - a) + y * a


# --------------------------------------------------------------------------
# the whole look
# --------------------------------------------------------------------------
ASSET_NAMES = ()


def tex_key(p):
    return (int(p.get("pp_seed", 3)), float(p.get("pp_grain", 50)),
            float(p.get("pp_paper", 100)) > 0, float(p.get("pp_texture", 50)) > 0)


def textures(h, w, p):
    """The random fields of the look (same generators and order as before)."""
    s = max(h, w) / REF_SIDE
    seed = int(p.get("pp_seed", 3))
    rng_brush = np.random.default_rng(seed)
    t = {"dx": _noise((h, w), 1.5 * s, rng_brush), "dy": _noise((h, w), 1.5 * s, rng_brush)}
    if float(p.get("pp_texture", 50)) > 0:
        t["cells"] = _noise((h, w), 1.0 * s, rng_brush)
        t["shade"] = canvas_shade(h, w, s)
    if float(p.get("pp_paper", 100)) > 0:
        rng_paper = np.random.default_rng(seed + 4099)
        ga = float(p.get("pp_grain", 50)) / 50.0
        t["kraft"] = kraft(h, w, s, rng_paper, ga)
        t["oldpaper"] = old_paper(h, w, s, rng_paper, ga)
    return t


def analyse(rgb, p):
    """Numbers from the photo the GPU needs (see cutout_stats)."""
    h0, w0 = rgb.shape[:2]
    s = max(h0, w0) / REF_SIDE
    h, w, cs = cutout_res(h0, w0, s)
    base = cutout_base(_small(rgb, h, w), cs, p.get("pp_detail", 50))
    return cutout_stats(base, p.get("pp_levels", 8), int(p.get("pp_seed", 3)))


def look_payload():
    return {"ref_side": REF_SIDE, "min_side": MIN_SIDE, "textures": True, "analyse": True,
            "grade": _GRADE.tolist(), "cut_sigma": CUT_SIGMA, "wide_step": WIDE_STEP}


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    """image_bgr float32/uint8 BGR 0..255 -> float32 BGR 0..255."""
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
    report(0.05, "Painted Photo: textures")
    return apply(rgb, textures(h, w, p), analyse(rgb, p), p, report)


def apply(rgb, tex, ana, p, report=lambda f, x: None):
    """The photo steps (the GPU runs the same ones, static/js/oilfx.js)."""
    h, w = rgb.shape[:2]
    s = max(h, w) / REF_SIDE
    seed = int(p.get("pp_seed", 3))
    report(0.1, "Painted Photo: cutting out the patches")
    patches = cutout(rgb, s, None, p.get("pp_levels", 8), p.get("pp_detail", 50), seed, stats=ana)
    report(0.5, "Painted Photo: brushwork")
    x = brushwork(patches, rgb, s, None, p.get("pp_outline", 50), p.get("pp_texture", 50), tex=tex)

    paper = float(p.get("pp_paper", 100)) / 100.0
    if paper > 0 and "kraft" in tex:
        report(0.7, "Painted Photo: paper")
        x = x + (_soft_light(x, tex["kraft"]) - x) * paper            # Soft Light 100%
        x = x + (x * tex["oldpaper"] - x) * 0.5 * paper               # Multiply 50%
    report(0.9, "Painted Photo: film grade")
    x = np.clip(x, 0, 1)
    # grade the smooth picture, then put the grain back as light/dark only:
    # grading the grain itself turns every speck orange or blue ("salt")
    amt = float(p.get("pp_grade", 100))
    smooth = _gauss(x, 1.6 * s)
    detail = (x - smooth).mean(-1, keepdims=True)
    graded = grade(smooth, amt)
    # how the grade stretches light/dark at each pixel (its slope along grey)
    e = 0.03
    slope = (grade(np.clip(smooth + e, 0, 1), amt) - grade(np.clip(smooth - e, 0, 1), amt)) / (2 * e)
    slope = np.clip(slope, 0.15, 3.0)
    ls = slope.mean(-1, keepdims=True)
    slope = ls + 0.3 * (slope - ls)                 # mostly light/dark, a touch of warmth
    x = graded + detail * slope
    return (np.clip(x, 0, 1)[..., ::-1] * 255.0).astype(np.float32)
