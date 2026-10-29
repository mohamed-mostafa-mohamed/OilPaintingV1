"""
"Silk Oil" style - rebuilt from a Photoshop template (the "9th" PSD).

The template is one smart object with seven smart filters:
  1. Liquify 50 %            (a hand-made warp of that photo only: left out)
  2. Filter Gallery 30 %     Underpainting, Palette Knife, Paint Daubs, Sponge,
                             Dry Brush, Smudge Stick, Angled / Sprayed Strokes
  3. Filter Gallery          Glass + Texturizer with an impasto texture
  4. Oil Paint               light (stylization 0.8)
  5. Filter Gallery 30 %     Graphic Pen, Soft Light
  6. Oil Paint, Darken       everything at 10: long smears with fine grooves
  7. Filter Gallery          Texturizer, the impasto texture again

Rebuilt from plain steps (Kuwahara blotches, a displacement and relief from
the impasto texture, diagonal pen strokes, smoothing along the photo's flow
with bristle streaks, Darken) and a saturation + tone curve fitted against
the template's own result. The SAME steps run on the GPU in
static/js/oilfx.js ("Silk Oil"); keep both in step.

Sizes are in template pixels (short side 2160) and scale with the photo.
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from .oilfx import BASE, cover
from .canvas_oil import rnd, gauss, sobel, lic_trace, lum
from .sketchfx import kuwahara

ASSET_DIR = os.path.join(BASE, "recipes", "silk_oil")
ASSET_NAMES = ("texture",)
REF_SHORT = 2160.0
MIN_SIDE = 1600          # previews are made at least this big (long side), then shrunk

KW_R, KW_MIX = 3.0, 0.30          # gallery: Kuwahara blotches, 30 %
GLASS, TZ1, TZ2 = 3.0, 0.15, 0.15  # glass displacement, texturizer reliefs
PEN_LEN, PEN_T, PEN_MIX = 15.0, 0.15, 0.30
FLOW_BLUR, FLOW_SIGMA = 2.0, 40.0
SMEAR, STREAK_LEN, GROOVE = 45.0, 24.0, 0.07

_cache: dict = {}

# what a variation (sl_seed) nudges - same table in oilfx.js
VARY = (("smooth", 0, 0.35), ("smear", 1, 0.40), ("streaks", 2, 0.45), ("texture", 3, 0.40),
        ("light", 4, 0.40))


def settings(p: dict) -> dict:
    from .variation import vary
    f = lambda k: float(p.get(k, 100)) / 100.0
    st = {"smooth": f("sl_smooth"), "smear": f("sl_smear"), "streaks": f("sl_streaks"),
          "texture": f("sl_texture"), "light": f("sl_light"), "color": f("sl_color")}
    return vary(st, VARY, int(p.get("sl_seed", 0)))


def look() -> dict:
    hit = _cache.get("look")
    if hit is None:
        with open(os.path.join(ASSET_DIR, "look.json"), "r", encoding="utf-8") as fh:
            hit = json.load(fh)
        hit["lut_np"] = np.asarray(hit["lut"], np.float32)
        _cache["look"] = hit
    return hit


def look_payload() -> dict:
    lk = look()
    return {"lut": lk["lut"], "sat": lk["sat"], "ref_short": REF_SHORT, "min_side": MIN_SIDE}


def _texture_img():
    img = cv2.imread(os.path.join(ASSET_DIR, "texture.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Silk Oil: missing texture.jpg in {ASSET_DIR}")
    return img


def assets(h: int, w: int) -> dict:
    key = ("assets", h, w)
    hit = _cache.get(key)
    if hit is None:
        hit = {"texture": cover(_texture_img(), h, w).astype(np.float32) / 255}
        for k in [k for k in _cache if isinstance(k, tuple)]:
            del _cache[k]
        _cache[key] = hit
    return hit


def asset_image(name: str, h: int, w: int) -> np.ndarray:
    """A template picture at w x h as 8-bit (for the GPU)."""
    if name not in ASSET_NAMES:
        raise KeyError(name)
    return cover(_texture_img(), h, w)


# ---------------------------------------------------------------- operations
def softlight(b, s):
    d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.clip(b, 0, 1)))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


def emboss(hgt, deg):
    """Relief of a height map lit from ``deg`` degrees (0 = right, 90 = top)."""
    a = math.radians(deg)
    lx, ly = math.cos(a), -math.sin(a)
    gx, gy = sobel(hgt)
    return -(gx * lx + gy * ly) / 8.0


def apply_lut(x, lut):
    i = np.clip(x, 0, 1) * 255
    i0 = np.floor(i).astype(np.int32)
    i1 = np.minimum(i0 + 1, 255)
    f = (i - i0).astype(np.float32)
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = lut[i0[..., c], c] * (1 - f[..., c]) + lut[i1[..., c], c] * f[..., c]
    return out


def apply(rgb, p, tex, report=lambda f, t: None):
    """RGB 0..1 in, RGB 0..1 out (tex: the impasto texture at the same size)."""
    st = settings(p)
    h, w = rgb.shape[:2]
    k = min(h, w) / REF_SHORT
    K = int(p.get("sl_seed", 0)) % 100000 * 101
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    px += 0.5
    py += 0.5
    x = rgb.astype(np.float32)

    # 1. Filter Gallery (30 %): blotchy underpainting
    report(0.1, "underpainting")
    if st["smooth"] > 0.01:
        a = kuwahara(x, KW_R * k)
        x = x + (a - x) * (KW_MIX * st["smooth"])

    # 2. Glass + Texturizer with the impasto texture
    t = st["texture"]
    Ht = gauss(tex, 1.0 * k)
    if t > 0.01:
        gx, gy = sobel(Ht)
        amp = GLASS * k * t
        x = cv2.remap(x, (px - 0.5 + gx * amp).astype(np.float32), (py - 0.5 + gy * amp).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = x * (1 + emboss(Ht, 225)[..., None] * (TZ1 * t))

    # 3. Graphic Pen (Soft Light 30 %): diagonal light strokes
    report(0.3, "pen strokes")
    if st["light"] > 0.01:
        d = np.float32(math.sqrt(0.5))
        dd = np.full((h, w), d, np.float32)
        n = rnd(np.floor(px), np.floor(py), K + 7)
        pin = lic_trace((lum(x) + (n - 0.5) * 0.5).astype(np.float32), dd, dd, PEN_LEN * k, 6)
        pen = np.clip((pin - PEN_T) / 0.08 + 0.5, 0, 1)
        x = x + (softlight(x, pen[..., None]) - x) * (PEN_MIX * st["light"])

    # 4. Oil Paint (Darken): long smears along the flow, fine bristle streaks
    report(0.5, "oil paint")
    if st["smear"] > 0.01:
        L = lum(gauss(x, FLOW_BLUR * k))
        gx, gy = sobel(L)
        s = FLOW_SIGMA * k + 1
        jxx, jyy, jxy = gauss(gx * gx, s), gauss(gy * gy, s), gauss(gx * gy, s)
        ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
        fx, fy = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        o = lic_trace(x, fx, fy, SMEAR * k * st["smear"], 8)
        nz = rnd(np.floor(px), np.floor(py), K + 11)
        g = lic_trace(nz, fx, fy, STREAK_LEN * k, 10)
        z = np.clip((g - 0.5) * 14.0, -2.5, 2.5)
        o = o * (1 + z[..., None] * (GROOVE * st["streaks"]))
        x = np.minimum(x, o)

    # 5. Texturizer again
    if t > 0.01:
        x = x * (1 + emboss(Ht, 90)[..., None] * (TZ2 * t))

    # 6. colour: saturation and tone fitted to the template
    report(0.85, "colour")
    c = st["color"]
    if c > 0.001:
        lk = look()
        l = lum(x)[..., None]
        y = np.clip(l + (x - l) * lk["sat"], 0, 1)
        y = apply_lut(y, lk["lut_np"])
        x = x + (y - x) * c
    return np.clip(x, 0, 1).astype(np.float32)


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    """image_bgr uint8/float BGR 0..255 -> float32 BGR 0..255."""
    h0, w0 = image_bgr.shape[:2]
    if min_side and max(h0, w0) < min_side:
        kk = min_side / max(h0, w0)
        big = cv2.resize(np.asarray(image_bgr, np.float32), (int(round(w0 * kk)), int(round(h0 * kk))),
                         interpolation=cv2.INTER_CUBIC)
        out = render(big, p, progress, min_side=0)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)

    def report(f, label):
        if progress:
            progress(f, f"Silk Oil: {label}")

    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    h, w = rgb.shape[:2]
    out = apply(rgb, p, assets(h, w)["texture"], report)
    report(0.98, "done")
    return np.ascontiguousarray(out[..., ::-1] * 255.0).astype(np.float32)
