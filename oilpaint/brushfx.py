"""
"Rough Brush" style - rebuilt from a Photoshop template (the "8TH" PSD).

The template, bottom to top:
  Background Color  black
  Image Fx          the photo with smart filters:
                      Oil Paint (stylization 10, cleanliness 10, scale 10),
                      Posterize 10, Cutout (8 levels), Displace 40 / 10
                      through a brush-stroke map, Gaussian Blur 2 px
  Texture           brush-stroke texture, Overlay 61 %
  Vibrance 8 / 2, Curves

Oil Paint, Posterize, Displace and the blur follow engine/psd_recipe.py
(the Oil Smear template had the same chain). Cutout is rebuilt so the GPU
can do it too: the brightness is cut into 8 tones and every pixel takes the
mean colour of the nearby pixels in its own tone (a wide, tone-limited
blur) instead of whole connected regions. The SAME steps run on the GPU in
static/js/oilfx.js ("Rough Brush" pipeline); keep both in step.

A per-channel tone curve fitted against the template's Image Fx layer
finishes the filters (the Oil Paint lighting and Cutout tones).

Distances are in template pixels (4000 x 2500, the photo covers it).
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from .oilfx import BASE, _g, lum, overlay, cover, apply_lut

ASSET_DIR = os.path.join(BASE, "recipes", "rough_brush")
ASSET_NAMES = ("displace", "texture")
DOC_W, DOC_H = 4000.0, 2500.0

# the template's settings (100 on every slider)
GUIDED_R, GUIDED_EPS = 3.75, 0.0012     # Oil Paint cleanliness 10
FLOW_BLUR, FLOW_SIGMA = 2.0, 4.0        # + 1 px, like psd_recipe
LIC_LEN, LIC_STEPS = 20.0, 6            # Oil Paint stylization 10
POSTER_LEVELS = 10
CUT_LEVELS, CUT_SIGMA = 8, 25.0
DISP_H, DISP_V = 40.0, 10.0             # Displace %, map stretched to fit
BLUR = 2.0
TEXTURE_OPACITY = 0.6118
VIB, SAT = 8.0, 2.0
CURVE = [(0, 0), (74, 61), (201, 212), (255, 255)]

_cache: dict = {}


# what a variation (rb_seed) nudges - same table in oilfx.js
VARY = (("paint", 0, 0.40), ("smooth", 1, 0.40), ("poster", 2, 0.40), ("warp", 3, 0.40),
        ("blur", 4, 0.40), ("texture", 5, 0.30))


def settings(p: dict) -> dict:
    from .variation import vary
    f = lambda k: float(p.get(k, 100)) / 100.0
    st = {"paint": f("rb_paint"), "smooth": f("rb_smooth"), "poster": f("rb_poster"),
          "cutout": f("rb_cutout"), "warp": f("rb_warp"), "blur": f("rb_blur"),
          "texture": f("rb_texture"), "adjust": f("rb_adjust")}
    return vary(st, VARY, int(p.get("rb_seed", 0)))


def doc_scale(h, w):
    return max(h / DOC_H, w / DOC_W)


def curve_lut():
    from engine import psd_recipe as R
    return R.curve_lut([{"in": a, "out": b} for a, b in CURVE])


def look() -> dict:
    """The tone curve fitted against the template's own Image Fx layer."""
    hit = _cache.get("look")
    if hit is None:
        with open(os.path.join(ASSET_DIR, "look.json"), "r", encoding="utf-8") as fh:
            hit = json.load(fh)
        hit["lut_np"] = np.asarray(hit["lut"], np.float32)
        _cache["look"] = hit
    return hit


def look_payload() -> dict:
    return {"lut": look()["lut"], "curve": [round(float(v), 6) for v in curve_lut()],
            "doc": [DOC_W, DOC_H]}


def _read(name, flags):
    img = cv2.imread(os.path.join(ASSET_DIR, name), flags)
    if img is None:
        raise OSError(f"Rough Brush: missing template file {name} in {ASSET_DIR}")
    return img


def assets(h: int, w: int) -> dict:
    key = ("assets", h, w)
    hit = _cache.get(key)
    if hit is None:
        mp = _read("displace.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ih, iw = mp.shape
        kk = max(h / ih, w / iw)
        mp = cover(mp, h, w) if kk < 1 else _cover_linear(mp, h, w)
        hit = {"displace": mp,                                         # 0..255 float
               "texture": cover(_read("texture.jpg", cv2.IMREAD_GRAYSCALE), h, w).astype(np.float32) / 255}
        for k in [k for k in _cache if isinstance(k, tuple)]:
            del _cache[k]
        _cache[key] = hit
    return hit


def _cover_linear(img, h, w):
    ih, iw = img.shape[:2]
    k = max(h / ih, w / iw)
    nh, nw = max(h, int(round(ih * k))), max(w, int(round(iw * k)))
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return np.ascontiguousarray(big[y0:y0 + h, x0:x0 + w])


def asset_image(name: str, h: int, w: int) -> np.ndarray:
    a = assets(h, w)
    if name == "displace":
        return np.clip(a["displace"] + 0.5, 0, 255).astype(np.uint8)
    if name == "texture":
        return (a["texture"] * 255 + 0.5).astype(np.uint8)
    raise KeyError(name)


# ---------------------------------------------------------------- operations
def box(x, r):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided(x, r, eps):
    """Edge-preserving smoothing, guide = the picture itself (per channel)."""
    r = max(1, int(round(r)))
    mI = box(x, r)
    var = box(x * x, r) - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return box(a, r) * x + box(b, r)


def flow(x, k):
    L = lum(_g(x, FLOW_BLUR * k))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    s = FLOW_SIGMA * k + 1
    jxx, jyy, jxy = _g(gx * gx, s), _g(gy * gy, s), _g(gx * gy, s)
    a = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def lic_trace(img, dx, dy, length, steps=LIC_STEPS):
    """Smoothing along the flow, following it as it bends (psd_recipe._lic)."""
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


def posterize(x, n):
    n = max(2, int(n))
    q = np.floor(np.clip(x, 0, 1) * 255.0 * n / 256.0)
    return (np.round(q * 255.0 / (n - 1)) / 255.0).astype(np.float32)


def tone_cutout(x, k, n=CUT_LEVELS, sigma=CUT_SIGMA):
    """Cutout rebuilt for the GPU: n brightness tones, each pixel takes the
    mean colour of the pixels of its own tone around it."""
    L = lum(x)
    idx = np.clip((L * n).astype(np.int32), 0, n - 1)
    out = np.zeros_like(x)
    for v in range(n):
        m = (idx == v).astype(np.float32)
        if not m.any():
            continue
        num = _g(x * m[..., None], sigma * k)
        den = _g(m, sigma * k)[..., None]
        out += num / np.maximum(den, 1e-6) * m[..., None]
    return out


def displace(x, mp, k, amt):
    h, w = x.shape[:2]
    off = (mp - 128.0) / 128.0
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ax, ay = DISP_H / 100 * 128 * k * amt, DISP_V / 100 * 128 * k * amt
    return cv2.remap(x, xx + off * ax, yy + off * ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def adjustments(x):
    mx, mn = x.max(axis=2, keepdims=True), x.min(axis=2, keepdims=True)
    l = lum(x)[..., None]
    y = l + (x - l) * (1.0 + VIB / 100.0 * (1.0 - (mx - mn)) ** 2)
    l = lum(y)[..., None]
    y = np.clip(l + (y - l) * (1.0 + SAT / 100.0 * 0.6), 0, 1)
    lut = curve_lut()
    t = y * 255.0
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, 255)
    f = t - i0
    return lut[i0] * (1 - f) + lut[i1] * f


def filters(rgb, st, a, k, report=lambda f, t: None):
    x = rgb.astype(np.float32)
    report(0.10, "oil paint")
    if st["smooth"] > 0.01:
        x = guided(x, GUIDED_R * k, GUIDED_EPS * st["smooth"])
    dx, dy = flow(x, k)
    x = np.clip(lic_trace(x, dx, dy, LIC_LEN * k * st["paint"]), 0, 1)
    report(0.40, "posterize, cutout")
    lv = int(round(POSTER_LEVELS / max(st["poster"], 0.1)))
    if st["poster"] > 0.01:
        x = posterize(x, lv)
    if st["cutout"] > 0.01:
        x = x + (tone_cutout(x, k) - x) * min(1.0, st["cutout"])
    report(0.65, "brush strokes")
    if st["warp"] > 0.01:
        x = displace(x, a["displace"], k, st["warp"])
    x = _g(x, BLUR * k * st["blur"])
    return apply_lut(np.clip(x, 0, 1), look()["lut_np"])


def composite(x, st, a):
    t = TEXTURE_OPACITY * st["texture"]
    x = x + (overlay(x, a["texture"][..., None]) - x) * t
    x = np.clip(x, 0, 1)
    if st["adjust"] > 0.001:
        x = x + (adjustments(x) - x) * st["adjust"]
    return np.clip(x, 0, 1).astype(np.float32)


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = 0) -> np.ndarray:
    def report(f, label):
        if progress:
            progress(f, f"Rough Brush: {label}")
    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = rgb.shape[:2]
    k = doc_scale(h, w)
    st = settings(p)
    a = assets(h, w)
    x = filters(rgb, st, a, k, report)
    report(0.85, "texture and colour")
    out = composite(x, st, a)
    report(0.98, "done")
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)
