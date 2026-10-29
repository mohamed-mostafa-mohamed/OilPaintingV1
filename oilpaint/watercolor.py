"""
"Watercolor" style - rebuilt from a Photoshop template (the "5TH" PSD).

The template, bottom to top:
  Color Fill    cream paper (240, 234, 227)
  Effect Image  eight copies of the photo, each with its own smart filters
                and its own brush-shaped layer mask (the ragged paper edge):
                  FX 2        Gaussian Blur 132, Vibrance, Brightness/Contrast, Dry Brush
                  FX 2 copy   same with Blur 92, 82 %
                  FX 1        Cutout, Sponge, Palette Knife, Dry Brush, Unsharp Mask
                  FX 3        Dry Brush, Contrast, Crosshatch          Screen 72 %
                  FX 4        Palette Knife, Levels                    Soft Light 62 %
                  FX 5        Poster Edges
                  FX 6        Contrast, Crosshatch
                  FX 7        Dry Brush + Watercolor                   Darken 92 %
  Adjustments   Levels, Curves, Hue/Saturation +18, Vibrance 42 / -4,
                Color Lookup 52 % (its table cannot be read: fitted)
  Textures      paper 01 Linear Burn 52 %, 02 Overlay 22 %, 03 Overlay ~51 %,
                04 Soft Light 22 %

Each layer is rebuilt from a plain step (wide blur + saturation, Kuwahara
blotches, brightness posterize, diagonal crosshatch smoothing) followed by
a per-channel tone curve fitted against that layer as Photoshop rendered it.
The masks, blend modes, adjustments and paper textures are the template's.
The SAME steps run on the GPU in static/js/oilfx.js ("Watercolor"); keep
both in step.  Distances are in template pixels (3000 x 3000, covered).
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from engine import psd_recipe as R
from .oilfx import BASE, _g, lum, cover, apply_lut
from .sketchfx import kuwahara

ASSET_DIR = os.path.join(BASE, "recipes", "watercolor")
DOC = 3000.0
BG = (240.0, 234.26, 226.82)
MASKS = ("m_l2", "m_l2c", "m_l1", "m_l3", "m_l4")
PAPERS = ("paper1", "paper2", "paper3", "paper4")
ASSET_NAMES = MASKS + PAPERS

# the eight layers, bottom to top: (name, step, mask, blend mode, opacity)
LAYERS = (
    ("L2", {"blur": 132, "sat": 2.4}, "m_l2", "normal", 1.0),
    ("L2c", {"blur": 92, "sat": 2.4}, "m_l2c", "normal", 0.82),
    ("L1", {"kw": 8, "us": 0.92, "us_r": 4.2}, "m_l1", "normal", 1.0),
    ("L3", {"kw": 7}, "m_l3", "screen", 0.722),
    ("L4", {"kw": 6}, "m_l4", "soft light", 0.62),
    ("L5", {"poster": 8}, "m_l4", "normal", 1.0),
    ("L6", {"hatch": 10, "hatch_mix": 0.5}, "m_l4", "normal", 1.0),
    ("L7", {"kw": 4, "kwn": 2}, "m_l3", "darken", 0.922),
)
PAPER_BLEND = (("paper1", "linear burn", 0.522), ("paper2", "overlay", 0.22),
               ("paper3", "overlay", 0.506), ("paper4", "soft light", 0.22))
LEVELS = (12.0, 235.0, 1.22, 22.0, 255.0)          # in black, in white, gamma, out black, out white
CURVE = [(0, 28), (130, 112), (197, 195), (255, 255)]
HUESAT = 18.0
VIB, VSAT = 42.0, -4.0
LOOKUP_OPACITY = 1.0        # the fitted table already holds the 52 % mix
BIG_BLUR_STEP = 8.0         # blurs wider than this many px run on a smaller copy

_cache: dict = {}


# what a variation (wc_seed) nudges - same table in oilfx.js
VARY = (("wash", 0, 0.40), ("blotch", 1, 0.45), ("detail", 2, 0.40), ("color", 3, 0.25),
        ("paper", 4, 0.35))


def settings(p: dict) -> dict:
    from .variation import vary
    f = lambda k: float(p.get(k, 100)) / 100.0
    st = {"frame": f("wc_frame"), "wash": f("wc_wash"), "blotch": f("wc_blotch"),
          "detail": f("wc_detail"), "color": f("wc_color"), "paper": f("wc_paper"),
          "adjust": f("wc_adjust")}
    return vary(st, VARY, int(p.get("wc_seed", 0)))


def doc_scale(h, w):
    return max(h / DOC, w / DOC)


def look() -> dict:
    hit = _cache.get("look")
    if hit is None:
        with open(os.path.join(ASSET_DIR, "look.json"), "r", encoding="utf-8") as fh:
            hit = json.load(fh)
        hit["np"] = {k: np.asarray(v["lut"], np.float32) for k, v in hit["layers"].items()}
        hit["lookup_np"] = np.asarray(hit["lookup"], np.float32)
        _cache["look"] = hit
    return hit


def curve_lut():
    return R.curve_lut([{"in": a, "out": b} for a, b in CURVE])


def look_payload() -> dict:
    lk = look()
    return {"layers": {k: v["lut"] for k, v in lk["layers"].items()}, "lookup": lk["lookup"],
            "curve": [round(float(v), 6) for v in curve_lut()], "doc": DOC}


def assets(h: int, w: int) -> dict:
    key = ("assets", h, w)
    hit = _cache.get(key)
    if hit is None:
        hit = {}
        for n in MASKS:
            m = cv2.imread(os.path.join(ASSET_DIR, n + ".jpg"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise OSError(f"Watercolor: missing {n}.jpg in {ASSET_DIR}")
            hit[n] = cover(m, h, w).astype(np.float32) / 255
        p1 = cv2.imread(os.path.join(ASSET_DIR, "paper1.jpg"), cv2.IMREAD_COLOR)
        if p1 is None:
            raise OSError(f"Watercolor: missing paper1.jpg in {ASSET_DIR}")
        hit["paper1"] = np.ascontiguousarray(cover(p1, h, w)[..., ::-1]).astype(np.float32) / 255
        for n in PAPERS[1:]:
            g = cv2.imread(os.path.join(ASSET_DIR, n + ".jpg"), cv2.IMREAD_GRAYSCALE)
            hit[n] = cover(g, h, w).astype(np.float32) / 255
        for k in [k for k in _cache if isinstance(k, tuple)]:
            del _cache[k]
        _cache[key] = hit
    return hit


def asset_image(name: str, h: int, w: int) -> np.ndarray:
    """One template picture at w x h, 8-bit (for the GPU; masks are asked
    for at half size). Made on its own so it does not disturb the cache."""
    if name not in ASSET_NAMES:
        raise KeyError(name)
    flags = cv2.IMREAD_COLOR if name == "paper1" else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(os.path.join(ASSET_DIR, name + ".jpg"), flags)
    if img is None:
        raise OSError(f"Watercolor: missing {name}.jpg in {ASSET_DIR}")
    return cover(img, h, w)


# ---------------------------------------------------------------- operations
def big_blur(x, sigma):
    """Gaussian blur; a wide one runs on a copy shrunk by a whole factor
    (the GPU does the same, so both match)."""
    if sigma <= BIG_BLUR_STEP * 2:
        return _g(x, sigma)
    f = int(sigma // BIG_BLUR_STEP)
    h, w = x.shape[:2]
    sw, sh = max(1, w // f), max(1, h // f)
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = _g(small, sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def diag(x, length):
    """Crosshatch: mean along short lines at +45 and -45 degrees."""
    if length < 1:
        return x
    h, w = x.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros_like(x)
    for ang in (45.0, -45.0):
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        acc = x.copy()
        for i in range(1, 5):
            for s in (1, -1):
                o = length / 2 * i / 4 * s
                acc += cv2.remap(x, gx + dx * o, gy + dy * o, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out += acc / 9
    return out / 2


def layer(src, name, P, st, k):
    x = src
    if P.get("blur"):
        x = big_blur(x, P["blur"] * k * st["wash"])
    if P.get("kw") and st["blotch"] > 0.01:
        for _ in range(P.get("kwn", 1)):
            x = kuwahara(x, P["kw"] * k * st["blotch"])
    if P.get("hatch"):
        h = diag(x, P["hatch"] * k)
        x = x + (h - x) * P["hatch_mix"]
    if P.get("poster"):
        n = P["poster"]
        L = lum(x)
        t = L * n
        f = t - np.floor(t)
        q = (np.floor(t) + np.clip((f - 0.5) * 6 + 0.5, 0, 1)) / n
        x = np.clip(x * ((q + 0.02) / (L + 0.02))[..., None], 0, 1)
    if P.get("us"):
        x = np.clip(x + P["us"] * st["detail"] * (x - _g(x, P["us_r"] * k)), 0, 1)
    sat = P.get("sat", 1.0)
    if sat != 1.0:
        sat = 1 + (sat - 1) * st["color"]
        l = lum(x)[..., None]
        x = np.clip(l + (x - l) * sat, 0, 1)
    return apply_lut(np.clip(x, 0, 1), look()["np"][name])


def adjustments(x):
    lo, hi, gamma, olo, ohi = LEVELS
    v = np.clip((x - lo / 255) / ((hi - lo) / 255), 0, 1) ** (1 / gamma)
    x = olo / 255 + v * (ohi - olo) / 255
    x = apply_lut(np.clip(x, 0, 1), np.repeat(curve_lut()[:, None], 3, axis=1))
    l = lum(x)[..., None]
    x = np.clip(l + (x - l) * (1 + HUESAT / 100), 0, 1)
    mx, mn = x.max(axis=2, keepdims=True), x.min(axis=2, keepdims=True)
    l = lum(x)[..., None]
    x = l + (x - l) * (1 + VIB / 100 * (1 - (mx - mn)) ** 2)
    l = lum(x)[..., None]
    x = np.clip(l + (x - l) * (1 + VSAT / 100 * 0.6), 0, 1)
    return apply_lut(x, look()["lookup_np"])


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = 0) -> np.ndarray:
    def report(f, label):
        if progress:
            progress(f, f"Watercolor: {label}")
    src = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = src.shape[:2]
    k = doc_scale(h, w)
    st = settings(p)
    a = assets(h, w)
    x = np.broadcast_to(np.array(BG, np.float32) / 255, src.shape).copy()
    for i, (name, P, mask, mode, op) in enumerate(LAYERS):
        report(0.05 + 0.8 * i / len(LAYERS), f"layer {i + 1} of {len(LAYERS)}")
        y = layer(src, name, P, st, k)
        m = 1 - st["frame"] * (1 - a[mask])
        x = x + (R.blend(mode, x, y) - x) * (op * m)[..., None]
    x = np.clip(x, 0, 1)
    report(0.88, "colour and paper")
    if st["adjust"] > 0.001:
        x = x + (adjustments(x) - x) * st["adjust"]
    for name, mode, op in PAPER_BLEND:
        t = a[name] if a[name].ndim == 3 else a[name][..., None]
        x = x + (R.blend(mode, x, t) - x) * (op * st["paper"])
    out = np.clip(x, 0, 1)
    report(0.98, "done")
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)
