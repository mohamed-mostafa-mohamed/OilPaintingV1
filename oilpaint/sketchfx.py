"""
"Sketch Paint" style - rebuilt from a Photoshop template (the "7TH" PSD).

The template, bottom to top:
  Color        white fill
  Back Paper   paper texture, Multiply 83 % (hidden under the photo)
  Photo Effect the photo with smart filters: Surface Blur 6 px / 11,
               Reduce Noise, Oil Paint (lighting), Unsharp Mask 121 % / 9 px,
               Diffuse Glow, Dry Brush x3, Poster Edges (2 / 1 / 3)
    Sketch     "pencil sketch" pattern, Overlay 70 %, clipped
  Levels       28 / 0.94 / 211, clipped
  Photo Outline the photo through the same filters + Black & White,
               Unsharp Mask and Photocopy (detail 9, darkness 13),
               Multiply 71 % (its mask only fits the template's own photo,
               so it is left out)

The filters are rebuilt from plain steps (bilateral blur, smoothing along
the image's structure, unsharp masks, Kuwahara blotches for Dry Brush,
brightness posterize + dark edges for Poster Edges, a local-mean threshold
for Photocopy) and a tone curve fitted against the template's own Photo
Effect layer. The SAME steps run on the graphics card in
static/js/oilfx.js ("Sketch Paint" pipeline); keep both in step.

Sizes are in template pixels (short side 2000) and scale with the photo.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .oilfx import BASE, REF_SHORT, _g, lum, overlay, flow, lic, apply_lut, cover

ASSET_DIR = os.path.join(BASE, "recipes", "sketch_paint")
ASSET_NAMES = ("sketch",)

# the template's settings (100 on every slider)
SB_RADIUS, SB_SIGMA_COLOR = 6.0, 0.09
LIC_LEN = 8.0
SHARP_AMT, SHARP_SIGMA = 3.0, 4.5
KW_RADIUS, KW_PASSES = 3.0, 3
POSTER_LEVELS, POSTER_SOFT, POSTER_MIX = 6, 6.0, 0.5
EDGE_AMT, EDGE_SIGMA = 10.0, 1.5
OUT_SHARP, OUT_SIGMA = 1.21, 4.5
PC_SIGMA, PC_GAIN, PC_OFFSET = 8.0, 12.0, 0.1
SKETCH_OPACITY = 0.702
OUTLINE_OPACITY = 0.71
LEVELS = (28.0, 211.0, 0.94)             # in black, in white, gamma

_cache: dict = {}


# what a variation (sk_seed) nudges - same table in oilfx.js
VARY = (("smooth", 0, 0.35), ("brush", 1, 0.45), ("blotch", 2, 0.40), ("poster", 3, 0.40),
        ("edges", 4, 0.35), ("outline", 5, 0.30))


def settings(p: dict) -> dict:
    from .variation import vary
    f = lambda k: float(p.get(k, 100)) / 100.0
    st = {"smooth": f("sk_smooth"), "brush": f("sk_brush"), "detail": f("sk_detail"),
          "blotch": f("sk_blotch"), "poster": f("sk_poster"), "edges": f("sk_edges"),
          "outline": f("sk_outline"), "sketch": f("sk_sketch"), "contrast": f("sk_contrast")}
    return vary(st, VARY, int(p.get("sk_seed", 0)))


def look() -> dict:
    hit = _cache.get("look")
    if hit is None:
        with open(os.path.join(ASSET_DIR, "look.json"), "r", encoding="utf-8") as fh:
            hit = json.load(fh)
        hit["lut_np"] = np.asarray(hit["lut"], np.float32)
        _cache["look"] = hit
    return hit


def look_payload() -> dict:
    return {"lut": look()["lut"]}


def assets(h: int, w: int) -> dict:
    key = ("assets", h, w)
    hit = _cache.get(key)
    if hit is None:
        img = cv2.imread(os.path.join(ASSET_DIR, "sketch.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f"Sketch Paint: missing template file in {ASSET_DIR}")
        hit = {"sketch": cover(img, h, w).astype(np.float32) / 255}
        for k in [k for k in _cache if isinstance(k, tuple)]:
            del _cache[k]
        _cache[key] = hit
    return hit


def asset_image(name: str, h: int, w: int) -> np.ndarray:
    """A template picture at w x h as 8-bit (for the GPU)."""
    if name not in ASSET_NAMES:
        raise KeyError(name)
    return (assets(h, w)[name] * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- operations
def _shift(a, ox, oy):
    """value at p = a[p + (ox, oy)], reflected at the border."""
    h, w = a.shape[:2]
    m = np.float32([[1, 0, -ox], [0, 1, -oy]])
    return cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def kuwahara(x, r):
    """4-quadrant Kuwahara: each pixel takes the mean of its calmest quadrant
    ((r+1) x (r+1) boxes with the pixel in a corner) - Dry Brush blotches."""
    r = max(1, int(round(r)))
    k = (r + 1, r + 1)
    m = cv2.blur(x, k, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.blur(x * x, k, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
    best, bv = None, None
    for ox, oy in ((0, 0), (-r, 0), (0, -r), (-r, -r)):
        mm, mm2 = _shift(m, ox, oy), _shift(m2, ox, oy)
        v = (mm2 - mm * mm).sum(axis=2)
        if best is None:
            best, bv = mm.copy(), v
        else:
            sel = v < bv
            best[sel] = mm[sel]
            bv = np.where(sel, v, bv)
    return best


def pre(rgb, st, k):
    """Shared by both layers: Surface Blur, then Oil Paint smoothing."""
    x = rgb.astype(np.float32)
    if st["smooth"] > 0.01:
        r = max(1, int(round(SB_RADIUS * k * st["smooth"])))
        x = cv2.bilateralFilter(x, 2 * r + 1, SB_SIGMA_COLOR, r / 2.0)
    dx, dy = flow(x, k)
    return lic(x, dx, dy, LIC_LEN * k * st["brush"])


def effect(x, st, k):
    """The Photo Effect layer."""
    x = np.clip(x + SHARP_AMT * st["detail"] * (x - _g(x, SHARP_SIGMA * k)), 0, 1)
    x = apply_lut(x, look()["lut_np"])
    if st["blotch"] > 0.01:
        for _ in range(KW_PASSES):
            x = kuwahara(x, KW_RADIUS * k * st["blotch"])
    mix = POSTER_MIX * st["poster"]
    if mix > 0.001:
        L = lum(x)
        t = L * POSTER_LEVELS
        f = t - np.floor(t)
        q = (np.floor(t) + np.clip((f - 0.5) * POSTER_SOFT + 0.5, 0, 1)) / POSTER_LEVELS
        ratio = (q + 0.02) / (L + 0.02)
        x = np.clip(x + (x * ratio[..., None] - x) * mix, 0, 1)
    if st["edges"] > 0.001:
        L = lum(_g(x, 0.7 * k))
        e = np.clip(_g(L, EDGE_SIGMA * k) - L, 0, 1)
        x = np.clip(x * (1 - EDGE_AMT * st["edges"] * e[..., None]), 0, 1)
    return x.astype(np.float32)


def outline(x, k):
    """The Photo Outline layer: dark lines where the picture is darker than
    around it (Photocopy), 0..1 gray."""
    L = lum(x)
    L = np.clip(L + OUT_SHARP * (L - _g(L, OUT_SIGMA * k)), 0, 1)
    d = _g(L, PC_SIGMA * k) - L
    return np.clip(1 - d * PC_GAIN + PC_OFFSET, 0, 1).astype(np.float32)


def levels(x):
    lo, hi, gamma = LEVELS
    v = np.clip((x - lo / 255.0) / ((hi - lo) / 255.0), 0, 1)
    return np.power(v, 1.0 / gamma)


def composite(e, line, st, a):
    s = SKETCH_OPACITY * st["sketch"]
    x = e + (overlay(e, a["sketch"][..., None]) - e) * s
    c = st["contrast"]
    if c > 0.001:
        x = x + (levels(x) - x) * c
    o = OUTLINE_OPACITY * st["outline"]
    x = x + (x * line[..., None] - x) * o
    return np.clip(x, 0, 1).astype(np.float32)


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = 0) -> np.ndarray:
    def report(f, label):
        if progress:
            progress(f, f"Sketch Paint: {label}")
    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = rgb.shape[:2]
    k = min(h, w) / REF_SHORT
    st = settings(p)
    report(0.10, "surface blur, oil paint")
    x = pre(rgb, st, k)
    report(0.45, "dry brush, poster edges")
    e = effect(x, st, k)
    report(0.75, "outline")
    line = outline(x, k) if st["outline"] > 0.001 else np.ones(x.shape[:2], np.float32)
    out = composite(e, line, st, assets(h, w))
    report(0.98, "done")
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)
