"""
"Oil Paint FX" style - rebuilt from a Photoshop template (the "6TH" PSD).

The template, bottom to top:
  BG          solid fill (227, 216, 128)
  Your Image  the photo with smart filters:
                Surface Blur 12 px / 10, Oil Paint (stylization 5, scale 4.6),
                High Pass 2 px + 3 px in Overlay, Camera Raw (contrast,
                texture, dehaze), Smart Sharpen 193 % / 2.5 px, Poster Edges,
                Cutout 20 %, Smart Sharpen 200 % / 1 px
              + a grunge layer mask (the yellow BG shows through at the edges)
  Color Tone  a soft colour cloud, clipped to the photo
  Texture     a cracked-paint texture in Overlay, clipped to the photo
  Vibrance 20 / 15, Curves, Color Balance (midtones +10 / 0 / -12)

The filters are rebuilt from plain operations - a bilateral blur, smoothing
along the image's own structure (line integral convolution), high-pass
overlays and unsharp masks - and a per-channel tone curve fitted against
the template's own filtered layer (the Camera Raw tone part). The SAME
steps run on the graphics card in static/js/oilfx.js, so a GPU Run and a
CPU Run give the same picture; keep both files in step.

Sizes are in template pixels (short side 2000) and scale with the photo.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(BASE, "recipes", "oil_fx")
REF_SHORT = 2000.0
BG = (227.0, 216.0, 128.0)          # RGB of the template's background fill

# the template's settings (100 on every slider)
SB_SIGMA_COLOR = 0.08
SB_RADIUS = 12.0
FLOW_SIGMA = 4.0
LIC_LEN = 6.0
LIC_STEPS = 4
LOCAL_AMT, LOCAL_SIGMA = 0.4, 12.0
SHARP1_AMT, SHARP1_SIGMA = 1.93, 1.5
SHARP2_AMT, SHARP2_SIGMA = 2.0, 0.6
VIB = 0.5
CB_AMOUNT = 0.35                    # Color Balance strength (fitted)

_cache: dict = {}


# ---------------------------------------------------------------- settings
# what a variation (fx_seed) nudges: (setting, slot, +- amount) - same table in oilfx.js
VARY = (("smooth", 0, 0.35), ("brush", 1, 0.45), ("detail", 2, 0.30), ("color", 3, 0.20),
        ("texture", 4, 0.35), ("tone", 5, 0.50))


def settings(p: dict) -> dict:
    from .variation import vary
    f = lambda k, d=100: float(p.get(k, d)) / 100.0
    st = {"smooth": f("fx_smooth"), "brush": f("fx_brush"), "detail": f("fx_detail"),
          "color": f("fx_color"), "texture": f("fx_texture"), "tone": f("fx_tone"),
          "edges": f("fx_edges"), "adjust": f("fx_adjust")}
    return vary(st, VARY, int(p.get("fx_seed", 0)))


# ---------------------------------------------------------------- assets
def _read(name, flags):
    path = os.path.join(ASSET_DIR, name)
    img = cv2.imread(path, flags)
    if img is None:
        raise OSError(f"Oil Paint FX: missing template file {path}")
    return img


def look() -> dict:
    hit = _cache.get("look")
    if hit is None:
        with open(os.path.join(ASSET_DIR, "look.json"), "r", encoding="utf-8") as fh:
            hit = json.load(fh)
        hit["lut_np"] = np.asarray(hit["lut"], np.float32)        # 256 x 3, RGB
        _cache["look"] = hit
    return hit


def cover(img, h, w):
    """Scale to cover (h, w) and crop the middle - Photoshop's Place / fill."""
    ih, iw = img.shape[:2]
    k = max(h / ih, w / iw)
    nh, nw = max(h, int(round(ih * k))), max(w, int(round(iw * k)))
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return np.ascontiguousarray(big[y0:y0 + h, x0:x0 + w])


def assets(h: int, w: int) -> dict:
    """The template's pictures at the photo's size (0..1 floats, RGB)."""
    key = ("assets", h, w)
    hit = _cache.get(key)
    if hit is None:
        mask = cover(_read("mask.jpg", cv2.IMREAD_GRAYSCALE), h, w).astype(np.float32) / 255
        tone = cover(_read("tone.png", cv2.IMREAD_UNCHANGED), h, w).astype(np.float32) / 255
        tex = cover(_read("texture.jpg", cv2.IMREAD_GRAYSCALE), h, w).astype(np.float32) / 255
        hit = {"mask": mask, "tone": np.ascontiguousarray(tone[..., 2::-1]),
               "tone_a": tone[..., 3], "texture": tex}
        for k in [k for k in _cache if isinstance(k, tuple)]:
            del _cache[k]                   # keep one size only
        _cache[key] = hit
    return hit


ASSET_NAMES = ("mask", "tone", "texture")


def look_payload() -> dict:
    """What the GPU needs: the fitted tone curve and the Curves layer."""
    return {"lut": look()["lut"], "curve": [round(float(v), 6) for v in curve_lut()]}


def asset_image(name: str, h: int, w: int) -> np.ndarray:
    """A template picture at w x h as 8-bit (BGRA for the tone, gray else)."""
    a = assets(h, w)
    if name == "mask":
        return (a["mask"] * 255 + 0.5).astype(np.uint8)
    if name == "texture":
        return (a["texture"] * 255 + 0.5).astype(np.uint8)
    if name == "tone":
        rgba = np.dstack([a["tone"][..., ::-1], a["tone_a"]])
        return (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    raise KeyError(name)


# ---------------------------------------------------------------- operations
def _g(x, s):
    return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT) if s > 0.05 else x


def lum(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def overlay(b, s):
    return np.where(b < 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))


def surface_blur(x, k, amt):
    if amt <= 0.01:
        return x
    r = max(1, int(round(SB_RADIUS * k * amt)))
    return cv2.bilateralFilter(x, 2 * r + 1, SB_SIGMA_COLOR, r / 2.0)


def flow(x, k):
    L = lum(_g(x, 1.5 * k))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    s = FLOW_SIGMA * k
    jxx, jyy, jxy = _g(gx * gx, s), _g(gy * gy, s), _g(gx * gy, s)
    a = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2      # along the edges
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def lic(x, dx, dy, length):
    """Average along a short straight line in the local stroke direction."""
    if length < 0.3:
        return x
    h, w = x.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = x.copy()
    st = length / LIC_STEPS
    for i in range(1, LIC_STEPS + 1):
        for sgn in (1.0, -1.0):
            acc += cv2.remap(x, gx + dx * (st * i * sgn), gy + dy * (st * i * sgn),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / (1 + 2 * LIC_STEPS)


def filters(rgb, st, k, report=lambda f, t: None):
    """The smart filters of 'Your Image' (RGB 0..1 in, RGB 0..1 out)."""
    x = rgb.astype(np.float32)
    report(0.10, "surface blur")
    x = surface_blur(x, k, st["smooth"])
    report(0.35, "oil paint")
    dx, dy = flow(x, k)
    x = lic(x, dx, dy, LIC_LEN * k * st["brush"])
    report(0.60, "high pass, sharpen")
    for r in (2.0, 3.0):
        x = overlay(x, np.clip(x - _g(x, r * k) + 0.5, 0, 1))
    d = st["detail"]
    x = np.clip(x + LOCAL_AMT * d * (x - _g(x, LOCAL_SIGMA * k)), 0, 1)
    x = np.clip(x + SHARP1_AMT * d * (x - _g(x, SHARP1_SIGMA * k)), 0, 1)
    x = np.clip(x + SHARP2_AMT * d * (x - _g(x, SHARP2_SIGMA * k)), 0, 1)
    report(0.75, "colour")
    l = lum(x)[..., None]
    s = x.max(axis=2, keepdims=True) - x.min(axis=2, keepdims=True)
    x = np.clip(l + (x - l) * (1 + VIB * st["color"] * (1 - s) ** 2), 0, 1)
    return apply_lut(x, look()["lut_np"], st["color"])


def apply_lut(x, lut, amount=1.0):
    """Per-channel tone curve (256 entries per channel, linear between)."""
    t = np.clip(x, 0, 1) * 255.0
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, 255)
    f = t - i0
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = lut[i0[..., c], c] * (1 - f[..., c]) + lut[i1[..., c], c] * f[..., c]
    if amount < 0.999:
        out = x + (out - x) * amount
    return out.astype(np.float32)


# ---------------------------------------------------------------- adjustments
def curve_lut():
    """The template's Curves layer (0..255 -> 0..1)."""
    pts = [(0, 0), (110, 123), (191, 214), (255, 255)]
    xs = np.array([p[0] for p in pts], np.float64) / 255
    ys = np.array([p[1] for p in pts], np.float64) / 255
    # Photoshop draws curves as a natural cubic spline through the points
    n = len(xs)
    hh = np.diff(xs)
    A = np.zeros((n, n)); B = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = hh[i - 1], 2 * (hh[i - 1] + hh[i]), hh[i]
        B[i] = 3 * ((ys[i + 1] - ys[i]) / hh[i] - (ys[i] - ys[i - 1]) / hh[i - 1])
    c = np.linalg.solve(A, B)
    t = np.linspace(0, 1, 256)
    j = np.clip(np.searchsorted(xs, t) - 1, 0, n - 2)
    dx = t - xs[j]
    b = (ys[j + 1] - ys[j]) / hh[j] - hh[j] * (2 * c[j] + c[j + 1]) / 3
    d = (c[j + 1] - c[j]) / (3 * hh[j])
    return np.clip(ys[j] + b * dx + c[j] * dx ** 2 + d * dx ** 3, 0, 1).astype(np.float32)


def adjustments(x):
    # Vibrance 20 / Saturation 15
    mx, mn = x.max(axis=2, keepdims=True), x.min(axis=2, keepdims=True)
    l = lum(x)[..., None]
    y = l + (x - l) * (1.0 + 0.20 * (1.0 - (mx - mn)) ** 2)
    l = lum(y)[..., None]
    y = np.clip(l + (y - l) * (1.0 + 0.15 * 0.6), 0, 1)
    # Curves
    y = apply_lut(y, np.repeat(curve_lut()[:, None], 3, axis=1))
    # Color Balance, midtones +10 / 0 / -12, preserve luminosity
    l = lum(y)[..., None]
    w = np.clip(1 - np.abs(2 * l - 1), 0, 1)
    y = np.clip(y + np.array([0.10, 0.0, -0.12], np.float32) * CB_AMOUNT * w, 0, 1)
    return np.clip(y + (l - lum(y)[..., None]), 0, 1)


def composite(x, st, a):
    """The layers over and under the filtered photo."""
    ta = (a["tone_a"] * st["tone"])[..., None]
    x = x * (1 - ta) + a["tone"] * ta
    t = st["texture"]
    if t > 0.001:
        x = x + (overlay(x, a["texture"][..., None]) - x) * t
    m = (1 - st["edges"] * (1 - a["mask"]))[..., None]
    bg = np.array(BG, np.float32) / 255
    x = x * m + bg * (1 - m)
    ad = st["adjust"]
    if ad > 0.001:
        x = x + (adjustments(x) - x) * ad
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- render
def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = 0) -> np.ndarray:
    """BGR 0..255 float in, BGR 0..255 float out (the style's bare painting)."""
    def report(f, label):
        if progress:
            progress(f, f"Oil Paint FX: {label}")
    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = rgb.shape[:2]
    k = min(h, w) / REF_SHORT
    st = settings(p)
    x = filters(rgb, st, k, report)
    report(0.85, "texture and colour")
    out = composite(x, st, assets(h, w))
    report(0.98, "done")
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)
