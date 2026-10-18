"""Image operations used by Photoshop actions, in numpy / OpenCV.

All images here are float32 RGB in 0..1, shape (h, w, 3), unless stated.
``s`` is the pixel scale: 1.0 when the image has the size the action was
made for (usually 3500 px long side), smaller when we work on a smaller
copy - every distance in pixels is multiplied by it.

Adjustments and blend modes follow Photoshop's formulas. The Filter
Gallery filters are our own versions: Adobe never published those, so they
are made to look like Photoshop's, not to match pixel for pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def gauss(img, sigma):
    if sigma <= 0.05:
        return img
    if sigma > 12:                     # wide blurs: same look, done on a smaller copy
        h, w = img.shape[:2]
        f = sigma / 6.0
        sw, sh = max(2, int(round(w / f))), max(2, int(round(h / f)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma * sw / w, borderType=cv2.BORDER_REFLECT)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return out.reshape(img.shape)
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def box(img, r):
    r = int(round(r))
    if r < 1:
        return img
    return cv2.blur(img, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def lum(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def rng_for(*key):
    return np.random.default_rng(abs(hash(key)) % (2 ** 32))


def noise(shape, sigma, seed):
    """Smooth noise in -1..1."""
    r = np.random.default_rng(seed)
    n = r.standard_normal(shape[:2]).astype(np.float32)
    if sigma > 0.3:
        n = gauss(n, sigma)
    n /= (n.std() + EPS) * 2.5
    return np.clip(n, -1, 1)


def remap(img, dx, dy):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx.astype(np.float32), gy + dy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def guided(img, r, eps):
    """Edge-preserving smoothing (He et al. guided filter, self-guided)."""
    r = max(1, int(round(r)))
    k = (2 * r + 1, 2 * r + 1)
    mean = cv2.blur(img, k)
    var = cv2.blur(img * img, k) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return cv2.blur(a, k) * img + cv2.blur(b, k)


def motion(img, length, angle_deg):
    length = max(1, int(round(length)))
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    k /= k.sum() + EPS
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------
# adjustments (Photoshop formulas)
# --------------------------------------------------------------------------
def levels(img, lo=0, hi=255, gamma=1.0, out_lo=0, out_hi=255):
    lo, hi = lo / 255.0, max(hi, lo + 1) / 255.0
    v = np.clip((img - lo) / (hi - lo), 0, 1)
    if abs(gamma - 1.0) > 1e-4:
        v = np.power(v, 1.0 / max(gamma, 0.01))
    return out_lo / 255.0 + v * (out_hi - out_lo) / 255.0


def desaturate(img):
    g = (img.max(axis=2) + img.min(axis=2)) * 0.5
    return np.repeat(g[..., None], 3, axis=2)


def invert(img):
    return 1.0 - img


def brightness_contrast(img, brightness=0, contrast=0, legacy=False):
    out = img
    if legacy:
        out = out + brightness / 255.0
        if contrast:
            f = (100 + contrast) / 100.0 if contrast < 0 else 100.0 / max(100 - contrast, 1)
            out = (out - 0.5) * f + 0.5
        return np.clip(out, 0, 1)
    if brightness:
        out = np.power(np.clip(out, 0, 1), 2.0 ** (-brightness / 100.0))
    if contrast:
        pivot = float(lum(out).mean())
        c = contrast / 100.0
        f = 1.0 + c * (1.0 if c > 0 else 0.7)
        out = pivot + (out - pivot) * f
    return np.clip(out, 0, 1)


def vibrance(img, vib=0, sat=0):
    out = img
    l = lum(out)[..., None]
    if vib:
        s = out.max(axis=2, keepdims=True) - out.min(axis=2, keepdims=True)
        amt = vib / 100.0 * (1.0 - s) ** 2 * (1.0 if vib > 0 else 1.0)
        out = l + (out - l) * (1.0 + amt)
    if sat:
        l = lum(out)[..., None]
        out = l + (out - l) * (1.0 + sat / 100.0 * 0.6)
    return np.clip(out, 0, 1)


def hue_saturation(img, hue=0, sat=0, light=0):
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + hue) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + sat / 100.0), 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if light > 0:
        out = out + (1 - out) * light / 100.0
    elif light < 0:
        out = out * (1 + light / 100.0)
    return out


def high_pass(img, r):
    return np.clip(img - gauss(img, r) + 0.5, 0, 1)


def surface_blur(img, r, threshold):
    eps = (max(threshold, 1) / 255.0) ** 2 * 1.5
    return np.clip(guided(img, r, eps), 0, 1)


def posterize(img, n):
    n = max(2, int(n))
    return np.round(img * (n - 1)) / (n - 1)


def threshold(img, level=128):
    t = (lum(img) >= level / 255.0).astype(np.float32)
    return np.repeat(t[..., None], 3, axis=2)


def unsharp(img, amount, r, thr=0):
    b = gauss(img, r)
    d = img - b
    if thr:
        d = np.where(np.abs(d) * 255 >= thr, d, 0)
    return np.clip(img + d * amount, 0, 1)


def add_noise(img, amount, mono=True, seed=7):
    r = np.random.default_rng(seed)
    if mono:
        n = r.standard_normal(img.shape[:2]).astype(np.float32)[..., None]
    else:
        n = r.standard_normal(img.shape).astype(np.float32)
    return np.clip(img + n * amount / 100.0 * 0.5, 0, 1)


def emboss(img, angle=135, height=3, amount=100):
    g = lum(img)
    a = np.deg2rad(angle)
    dx, dy = np.cos(a) * height, -np.sin(a) * height
    sh = remap(g, np.full_like(g, -dx), np.full_like(g, -dy))
    e = np.clip(0.5 + (g - sh) * amount / 100.0, 0, 1)
    return np.repeat(e[..., None], 3, axis=2)


def median(img, r):
    k = max(3, int(round(r)) * 2 + 1)
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    if k > 5:                                   # big kernels: uint8 only, per channel
        return np.dstack([cv2.medianBlur(u8[..., c], min(k, 255)) for c in range(3)]).astype(np.float32) / 255
    return cv2.medianBlur(u8, k).astype(np.float32) / 255


# --------------------------------------------------------------------------
# blend modes (b = backdrop, s = source), all in 0..1
# --------------------------------------------------------------------------
def _soft_light(b, s):
    d = np.sqrt(np.clip(b, 0, 1))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


def _vivid(b, s):
    burn = 1 - np.minimum(1, (1 - b) / np.maximum(2 * s, EPS))
    dodge = np.minimum(1, b / np.maximum(2 * (1 - s), EPS))
    return np.where(s <= 0.5, np.where(s <= EPS, 0.0, burn), np.where(s >= 1 - EPS, 1.0, dodge))


def _set_lum(c, l):
    d = l - lum(c)[..., None]
    c = c + d
    ll = lum(c)[..., None]
    n = c.min(axis=2, keepdims=True)
    x = c.max(axis=2, keepdims=True)
    c = np.where(n < 0, ll + (c - ll) * ll / np.maximum(ll - n, EPS), c)
    c = np.where(x > 1, ll + (c - ll) * (1 - ll) / np.maximum(x - ll, EPS), c)
    return c


def _sat(c):
    return c.max(axis=2, keepdims=True) - c.min(axis=2, keepdims=True)


def _set_sat(c, s):
    mx = c.max(axis=2, keepdims=True)
    mn = c.min(axis=2, keepdims=True)
    rng = mx - mn
    return np.where(rng > EPS, (c - mn) * s / np.maximum(rng, EPS), 0.0)


def blend(mode, b, s):
    if mode in ("Nrml", "normal", "Dslv", "passThrough"):
        return s
    if mode == "Mltp":
        return b * s
    if mode == "Scrn":
        return 1 - (1 - b) * (1 - s)
    if mode == "Ovrl":
        return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "HrdL":
        return np.where(s <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "SftL":
        return _soft_light(b, s)
    if mode == "linearLight":
        return np.clip(b + 2 * s - 1, 0, 1)
    if mode == "vividLight":
        return _vivid(b, s)
    if mode == "pinLight":
        return np.where(s <= 0.5, np.minimum(b, 2 * s), np.maximum(b, 2 * s - 1))
    if mode == "hardMix":
        return (_vivid(b, s) >= 0.5).astype(np.float32)
    if mode == "Drkn":
        return np.minimum(b, s)
    if mode == "Lghn":
        return np.maximum(b, s)
    if mode == "CBrn":
        return np.where(s <= EPS, 0.0, 1 - np.minimum(1, (1 - b) / np.maximum(s, EPS)))
    if mode == "CDdg":
        return np.where(s >= 1 - EPS, 1.0, np.minimum(1, b / np.maximum(1 - s, EPS)))
    if mode == "linearBurn":
        return np.clip(b + s - 1, 0, 1)
    if mode == "linearDodge":
        return np.clip(b + s, 0, 1)
    if mode == "Dfrn":
        return np.abs(b - s)
    if mode == "Xclu":
        return b + s - 2 * b * s
    if mode == "blendSubtraction" or mode == "Sbtr":
        return np.clip(b - s, 0, 1)
    if mode == "blendDivide" or mode == "divide":
        return np.clip(b / np.maximum(s, EPS), 0, 1)
    if mode == "darkerColor":
        return np.where(lum(s)[..., None] < lum(b)[..., None], s, b)
    if mode == "lighterColor":
        return np.where(lum(s)[..., None] > lum(b)[..., None], s, b)
    if mode == "H   ":
        return _set_lum(_set_sat(s, _sat(b)), lum(b)[..., None])
    if mode == "Strt":
        return _set_lum(_set_sat(b, _sat(s)), lum(b)[..., None])
    if mode == "Clr ":
        return _set_lum(s, lum(b)[..., None])
    if mode == "Lmns":
        return _set_lum(b, lum(s)[..., None])
    return s


SUPPORTED_MODES = {
    "Nrml", "normal", "Dslv", "passThrough", "Mltp", "Scrn", "Ovrl", "HrdL", "SftL",
    "linearLight", "vividLight", "pinLight", "hardMix", "Drkn", "Lghn", "CBrn",
    "CDdg", "linearBurn", "linearDodge", "Dfrn", "Xclu", "blendSubtraction",
    "Sbtr", "blendDivide", "divide", "darkerColor", "lighterColor", "H   ",
    "Strt", "Clr ", "Lmns",
}


def composite(dst_rgb, dst_a, src_rgb, src_a, mode="Nrml"):
    """Photoshop / W3C compositing of a layer onto what is below it.

    dst_a and src_a are (h, w, 1); dst_a None means "fully opaque" (the
    usual case once the Background is in), which is much faster.
    Returns (rgb, a) - a is None again when the result is opaque.
    """
    if mode in ("Nrml", "normal", "passThrough", "Dslv"):
        if dst_a is None:                              # opaque backdrop: a plain mix
            return dst_rgb + (src_rgb - dst_rgb) * src_a, None
        out_a = src_a + dst_a * (1 - src_a)
        rgb = (src_rgb * src_a + dst_rgb * dst_a * (1 - src_a)) / np.maximum(out_a, EPS)
        return rgb, out_a
    mixed = blend(mode, dst_rgb, src_rgb)
    if dst_a is None:
        return dst_rgb + (mixed - dst_rgb) * src_a, None
    # where nothing is below, the layer shows as it is
    src_eff = dst_a * mixed + (1 - dst_a) * src_rgb
    out_a = src_a + dst_a * (1 - src_a)
    rgb = (src_eff * src_a + dst_rgb * dst_a * (1 - src_a)) / np.maximum(out_a, EPS)
    return rgb, out_a


# --------------------------------------------------------------------------
# textures used by Filter Gallery
# --------------------------------------------------------------------------
def canvas_texture(h, w, scale=1.0, seed=3):
    """A woven canvas height map in 0..1, like Photoshop's "Canvas" texture."""
    p = max(3.0, 7.0 * scale)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / p, y / p
    warp = np.abs(np.sin(np.pi * u)) * (0.6 + 0.4 * np.cos(np.pi * (v + np.floor(u))))
    weft = np.abs(np.sin(np.pi * v)) * (0.6 + 0.4 * np.cos(np.pi * (u + np.floor(v))))
    over = ((np.floor(u) + np.floor(v)) % 2).astype(np.float32)
    t = over * warp + (1 - over) * weft
    t += noise((h, w), 0.8 * scale, seed) * 0.12
    t = (t - t.min()) / (t.max() - t.min() + EPS)
    return t.astype(np.float32)


def texture_map(kind, h, w, scale, seed=3):
    if kind in ("TxCa", "canvas"):
        return canvas_texture(h, w, scale, seed)
    if kind in ("TxBr", "brick"):
        p = 40 * scale
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        row = np.floor(y / (p / 2))
        xx = (x + (row % 2) * p / 2) % p
        yy = y % (p / 2)
        mortar = np.minimum(np.minimum(xx, p - xx), np.minimum(yy, p / 2 - yy))
        return np.clip(mortar / (2 * scale), 0, 1).astype(np.float32)
    if kind in ("TxBu", "burlap"):
        return canvas_texture(h, w, scale * 1.8, seed)
    # frosted / tiny lens: blurred noise
    return (noise((h, w), 2.0 * scale, seed) * 0.5 + 0.5).astype(np.float32)


# --------------------------------------------------------------------------
# Filter Gallery
# --------------------------------------------------------------------------
def smooth_weave(h, w, period, seed=3):
    """Soft woven height map (no hard steps), values about 0..1."""
    u = 2 * np.pi * np.arange(w, dtype=np.float32) / period
    v = 2 * np.pi * np.arange(h, dtype=np.float32) / period
    t = (0.5 + 0.22 * np.outer(np.sin(v * 0.5), np.sin(u))
         + 0.22 * np.outer(np.sin(v), np.sin(u * 0.5 + np.pi / 2)))
    t += noise((h, w), period * 0.25, seed) * 0.06
    return t.astype(np.float32)


def fg_glass(img, s, distortion=5, smoothness=3, texture="TxCa", scaling=100, invert_tex=False, seed=11):
    h, w = img.shape[:2]
    sc = max(0.3, scaling / 100.0 * s)
    if texture in ("TxCa", "canvas", "TxBu", "burlap"):
        period = (9.0 if texture in ("TxCa", "canvas") else 15.0) * sc
        t = smooth_weave(h, w, max(3.0, period), seed)
    else:
        period = 12.0 * sc
        t = texture_map(texture, h, w, sc, seed)
    if invert_tex:
        t = 1 - t
    t = gauss(t, max(0.4, smoothness * 0.12 * sc))
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gmax = 0.22 * 2 * np.pi / period                # steepest slope of the weave
    reach = distortion / 20.0 * period * 0.3        # 20 = about a third of a thread
    return np.clip(remap(img, gx / gmax * reach, gy / gmax * reach), 0, 1)


def fg_sponge(img, s, brush=2, definition=12, smooth=5, seed=12):
    h, w = img.shape[:2]
    base = guided(img, max(1, (brush + 1) * 1.5 * s), 0.004)
    n = noise((h, w), max(0.8, (1.5 + brush * 0.6) * s), seed)
    n = gauss(n, max(0.3, (smooth - 1) * 0.15 * s))
    pores = np.clip(-n, 0, 1) ** 1.5
    strength = 0.10 + definition / 25.0 * 0.45
    out = base * (1 - pores[..., None] * strength) + np.clip(n, 0, 1)[..., None] * strength * 0.25 * base
    return np.clip(out, 0, 1)


def fg_spatter(img, s, radius=10, smooth=5, seed=13):
    h, w = img.shape[:2]
    r = np.random.default_rng(seed)
    sig = max(0.0, (smooth - 1) * 0.12 * s)
    dx = r.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = r.uniform(-1, 1, (h, w)).astype(np.float32)
    if sig > 0.3:
        dx, dy = gauss(dx, sig), gauss(dy, sig)
        m = max(np.abs(dx).max(), np.abs(dy).max(), EPS)
        dx, dy = dx / m, dy / m
    k = radius * 0.5 * s
    out = remap(img, dx * k, dy * k)
    return np.clip(gauss(out, 0.3 + sig * 0.25), 0, 1)


def fg_palette_knife(img, s, size=25, detail=2, softness=0):
    h, w = img.shape[:2]
    f = max(1.0, size * s / 5.0)
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    u8 = np.clip(small * 255, 0, 255).astype(np.uint8)
    sr = 34 - detail * 8
    u8 = cv2.pyrMeanShiftFiltering(u8, sp=3 + detail, sr=sr)
    up = cv2.resize(u8, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
    blocky = median(up, max(1, f * 0.6))
    return np.clip(gauss(blocky, softness * 0.5 * s), 0, 1)


def fg_paint_daubs(img, s, size=8, sharpness=7, brush="BrSm"):
    out = median(img, max(1, size * s))
    if brush in ("BrDr", "dark"):
        out = out * out
    elif brush in ("BrWd", "wide"):
        out = gauss(out, size * 0.3 * s)
    if sharpness:
        out = unsharp(out, sharpness / 8.0, max(1.0, size * 0.5 * s))
    return np.clip(out, 0, 1)


def fg_glowing_edges(img, s, width=2, brightness=6, smooth=5):
    sm = gauss(img, max(0.5, (smooth - 1) * 0.25 * s + 0.6))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    wpx = max(1, int(round(width * s)))
    if wpx > 1:
        mag = cv2.dilate(mag, np.ones((wpx, wpx), np.uint8))
    return np.clip(mag * (0.25 + brightness * 0.22), 0, 1)


def fg_cutout(img, s, levels_n=4, simplicity=4, fidelity=2, seed=14):
    h, w = img.shape[:2]
    sm = guided(img, max(1, (simplicity + 1) * 2 * s), 0.01)
    sample = cv2.resize(sm, (min(w, 256), min(h, 256)), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    k = max(2, int(levels_n))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(sample.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)
    order = np.argsort(centers @ np.array([0.299, 0.587, 0.114], np.float32))
    centers = centers[order]
    # nearest colour through a 32x32x32 lookup table (fast at any size)
    g = (np.arange(32, dtype=np.float32) + 0.5) / 32
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lut = ((cube[:, None, :] - centers[None]) ** 2).sum(-1).argmin(-1).astype(np.uint8)
    q = np.clip((sm * 32).astype(np.int32), 0, 31)
    idx = lut[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]].astype(np.int32)
    ks = max(1, int(round((simplicity * 1.2 + (3 - fidelity) * 1.5) * s)))
    if ks > 1:
        idx = cv2.medianBlur(idx.astype(np.uint8), ks * 2 + 1 if ks * 2 + 1 <= 255 else 255).astype(np.int32)
    return centers[idx]


def fg_charcoal(img, s, thickness=1, detail=5, balance=50, fg=(0, 0, 0), bg=(1, 1, 1), seed=15):
    h, w = img.shape[:2]
    g = lum(img)
    dark = 1 - g
    n = noise((h, w), 0.6 * s, seed) * 0.5 + 0.5
    strokes = motion(n, max(3, (6 + thickness * 4) * s), 45)
    strokes = (strokes - strokes.mean()) / (strokes.std() + EPS)
    sm = gauss(g, max(0.6, (6 - detail) * 0.6 * s))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.clip(np.sqrt(gx * gx + gy * gy) * (1.5 + detail * 0.8), 0, 1)
    shift = (balance - 50) / 100.0 * 0.5
    cover = np.clip((dark - 0.55 + shift) * 2.0 + 0.5 + strokes * 0.22, 0, 1)
    cover = np.clip(np.maximum(cover, edges * 0.9), 0, 1)
    cover = motion(cover, max(2, thickness * 2 * s), 45)
    fg = np.asarray(fg, np.float32)
    bg = np.asarray(bg, np.float32)
    return bg + (fg - bg) * cover[..., None]


def fg_sprayed_strokes(img, s, length=12, radius=7, direction="SDRD", seed=16):
    ang = {"SDRD": 45, "SDLD": -45, "SDHz": 0, "SDVt": 90}.get(direction, 45)
    out = fg_spatter(img, s, radius, 3, seed) if radius else img
    if length:
        out = motion(out, max(2, (length * 1.5 + 1) * s), ang)
    return np.clip(out, 0, 1)


def fg_crosshatch(img, s, length=9, sharpness=6, strength=1, seed=17):
    h, w = img.shape[:2]
    out = img
    for i in range(max(1, int(strength))):
        n = noise((h, w), 0.5 * s, seed + i)[..., None] * 0.08
        a = motion(out + n, max(2, length * s), 45)
        b = motion(out + n, max(2, length * s), -45)
        g = lum(out)[..., None]
        out = np.where(g < 0.5, np.minimum(a, b), (a + b) * 0.5)
    if sharpness:
        out = unsharp(out, sharpness / 6.0, max(1.0, 1.5 * s))
    return np.clip(out, 0, 1)


def fg_dry_brush(img, s, size=2, detail=8, texture=1):
    out = fg_paint_daubs(img, s, size * 2 + 1, detail)
    if texture > 1:
        out = np.clip(out + noise(img.shape, 0.5 * s, 21)[..., None] * 0.03 * texture, 0, 1)
    return posterize(out, 18 - detail // 2)


def fg_watercolor(img, s, detail=9, shadow=1, texture=1):
    out = fg_paint_daubs(img, s, 16 - detail, 0)
    out = out * (1 - shadow * 0.06)
    e = fg_glowing_edges(img, s, 1, 3, 4)
    return np.clip(out - lum(e)[..., None] * 0.15 * texture, 0, 1)


def fg_accented_edges(img, s, width=2, brightness=38, smooth=5):
    e = lum(fg_glowing_edges(img, s, width, 6, smooth))[..., None]
    target = brightness / 50.0
    return np.clip(img * (1 - e) + e * target, 0, 1)


def fg_texturizer(img, s, texture="TxCa", scaling=100, relief=4, light="LgTp", invert_tex=False):
    h, w = img.shape[:2]
    t = texture_map(texture, h, w, max(0.3, scaling / 100.0 * s))
    if invert_tex:
        t = 1 - t
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    shade = -(gx * 0.7 + gy * 0.7)
    shade = shade / (np.abs(shade).max() + EPS)
    return np.clip(img * (1 + shade[..., None] * relief / 50.0 * 2), 0, 1)


FILTER_GALLERY = {
    # code: (name, function, {atn key: (argument, default)}, uses colours)
    "Gls ": ("Glass", fg_glass, {"Dstr": ("distortion", 5), "Smth": ("smoothness", 3),
                                 "TxtT": ("texture", "TxCa"), "Scln": ("scaling", 100),
                                 "InvT": ("invert_tex", False)}),
    "Spng": ("Sponge", fg_sponge, {"BrsS": ("brush", 2), "Dfnt": ("definition", 12), "Smth": ("smooth", 5)}),
    "Spt ": ("Spatter", fg_spatter, {"SprR": ("radius", 10), "Smth": ("smooth", 5)}),
    "PltK": ("Palette Knife", fg_palette_knife, {"StrS": ("size", 25), "StDt": ("detail", 2), "Sftn": ("softness", 0)}),
    "PntD": ("Paint Daubs", fg_paint_daubs, {"Sz  ": ("size", 8), "Shrp": ("sharpness", 7), "BrsT": ("brush", "BrSm")}),
    "GlwE": ("Glowing Edges", fg_glowing_edges, {"EdgW": ("width", 2), "EdgB": ("brightness", 6), "Smth": ("smooth", 5)}),
    "Ct  ": ("Cutout", fg_cutout, {"NmbL": ("levels_n", 4), "EdgS": ("simplicity", 4), "EdgF": ("fidelity", 2)}),
    "Chrc": ("Charcoal", fg_charcoal, {"ChAm": ("thickness", 1), "Dtl ": ("detail", 5), "LgDr": ("balance", 50)}),
    "SprS": ("Sprayed Strokes", fg_sprayed_strokes, {"StrL": ("length", 12), "SprR": ("radius", 7), "SDir": ("direction", "SDRD")}),
    "Crsh": ("Crosshatch", fg_crosshatch, {"StrL": ("length", 9), "Shrp": ("sharpness", 6), "Strg": ("strength", 1)}),
    "DryB": ("Dry Brush", fg_dry_brush, {"BrsS": ("size", 2), "BrsD": ("detail", 8), "TxtT": ("texture", 1)}),
    "Wtrc": ("Watercolor", fg_watercolor, {"BrsD": ("detail", 9), "ShdI": ("shadow", 1), "Txtr": ("texture", 1)}),
    "AccE": ("Accented Edges", fg_accented_edges, {"EdgW": ("width", 2), "EdgB": ("brightness", 38), "Smth": ("smooth", 5)}),
    "Txtz": ("Texturizer", fg_texturizer, {"TxtT": ("texture", "TxCa"), "Scln": ("scaling", 100),
                                           "Rlf ": ("relief", 4), "LghD": ("light", "LgTp"), "InvT": ("invert_tex", False)}),
}


def filter_gallery_name(code):
    f = FILTER_GALLERY.get(code)
    return f[0] if f else code.strip()
