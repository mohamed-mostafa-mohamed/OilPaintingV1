"""
Decoration: Canvas, Frame and Text / Watermark.

The three effects are applied in the order the user has arranged the tabs,
exactly as in the desktop program.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .params import hex_to_bgr

# --------------------------------------------------------------------------
# canvas surfaces
# --------------------------------------------------------------------------


def _weave(h, w, pitch, sharpness=1.0, angle=0.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if angle:
        a = math.radians(angle)
        xs, ys = xs * math.cos(a) - ys * math.sin(a), xs * math.sin(a) + ys * math.cos(a)
    # each thread wanders a little, the way a real woven thread does
    warp_x = np.sin(ys / (pitch * 2.7)) * pitch * 0.18
    warp_y = np.sin(xs / (pitch * 3.1)) * pitch * 0.18
    u = np.sin((xs + warp_x) * (2 * np.pi / pitch))
    v = np.sin((ys + warp_y) * (2 * np.pi / pitch))
    t = (np.abs(u) ** sharpness) * 0.5 + (np.abs(v) ** sharpness) * 0.5
    return t.astype(np.float32)


def _value_noise(h, w, cells, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cells * (2 ** o)))
        g = rng.random((c, max(2, int(c * w / max(h, 1))))).astype(np.float32)
        out += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= 0.5
    return out / max(tot, 1e-6)


def surface_height(name: str, h: int, w: int, scale: float) -> np.ndarray:
    """A height map in 0..1 for the chosen surface."""
    k = max(2.0, 9.0 * scale / 100.0 * (min(h, w) / 600.0) * 100.0 / 100.0)
    pitch = max(2.5, 7.0 * (scale / 100.0) * max(0.6, min(h, w) / 700.0))

    if name == "Canvas Rough":
        t = _weave(h, w, pitch * 1.7, 0.75)
        t = t * 0.8 + _value_noise(h, w, 14, 5) * 0.2
    elif name == "Linen":
        t = _weave(h, w, pitch * 1.25, 1.4)
        t = t * 0.75 + _value_noise(h, w, 30, 11) * 0.25
    elif name == "Corrugated Paper":
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (np.sin(xs * (2 * np.pi / (pitch * 3.2))) * 0.5 + 0.5)
        t = t * 0.85 + _value_noise(h, w, 26, 3) * 0.15
    elif name == "Brick Wall":
        t = _brick(h, w, pitch * 9.0)
    elif name == "Ceramic":
        t = _value_noise(h, w, 7, 21, octaves=5)
        t = cv2.GaussianBlur(t, (0, 0), 1.1)
        t = np.clip((t - 0.35) * 2.2 + 0.5, 0, 1)
    elif name == "Watercolor Paper":
        t = _value_noise(h, w, 40, 33, octaves=3)
        t = cv2.GaussianBlur(t, (0, 0), 0.7)
    else:  # Canvas Fine
        t = _weave(h, w, pitch, 1.1)
        t = t * 0.82 + _value_noise(h, w, 22, 7) * 0.18

    t = t - t.min()
    return (t / max(t.max(), 1e-6)).astype(np.float32)


def _brick(h, w, unit):
    unit = max(8.0, unit)
    bh = unit * 0.45
    t = np.ones((h, w), np.float32)
    mortar = max(1, int(unit * 0.07))
    rng = np.random.default_rng(99)
    y = 0.0
    row = 0
    while y < h:
        cv2.rectangle(t, (0, int(y)), (w, int(y + mortar)), 0.25, -1)
        off = 0.0 if row % 2 == 0 else unit * 0.5
        x = -off
        while x < w:
            cv2.rectangle(t, (int(x), int(y)), (int(x + mortar), int(y + bh)), 0.25, -1)
            x += unit
        row += 1
        y += bh
    t = t * (0.86 + 0.14 * _value_noise(h, w, 30, int(rng.integers(0, 1000))))
    return cv2.GaussianBlur(t, (0, 0), 0.8)


def canvas_surface(h: int, w: int, p: dict, custom: np.ndarray | None = None) -> np.ndarray:
    """The height map (0..1) of the chosen surface at this size."""
    if custom is not None:
        g = cv2.cvtColor(custom, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        s = max(0.05, float(p["canvas_scale"]) / 100.0)
        th = max(2, int(round(g.shape[0] * s)))
        tw = max(2, int(round(g.shape[1] * s)))
        g = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA)
        # tile the sample across the picture rather than stretching it
        g = np.tile(g, (int(np.ceil(h / th)), int(np.ceil(w / tw))))[:h, :w]
        g = g - g.min()
        return (g / max(g.max(), 1e-6)).astype(np.float32)
    return surface_height(p["canvas_texture"], h, w, float(p["canvas_scale"]))


def canvas_light_angle(p: dict) -> float:
    return math.radians(45.0 + 270.0 * float(p["canvas_light"]) / 100.0)


def apply_canvas(img: np.ndarray, p: dict, custom: np.ndarray | None = None,
                 height: np.ndarray | None = None):
    """Imitate painting on canvas or any other uneven surface."""
    if not p["canvas_enabled"]:
        return img
    h, w = img.shape[:2]
    if height is None:
        height = canvas_surface(h, w, p, custom)

    strength = float(p["canvas_strength"]) / 100.0
    relief = float(p["canvas_relief"]) / 100.0

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    a = canvas_light_angle(p)
    emb = gx * math.cos(a) + gy * math.sin(a)
    sc = float(np.percentile(np.abs(emb), 98)) + 1e-6
    emb = np.clip(emb / sc, -1.5, 1.5)

    shade = 1.0 + emb * (0.55 * relief * strength)
    # the weave also holds back the paint in its hollows
    tone = 1.0 - (1.0 - height) * (0.22 * strength)
    out = img * (shade * tone)[:, :, None]
    return np.clip(out, 0, 255)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------


def frame_size(h: int, w: int, p: dict):
    """(border width, output height, output width) of the framed picture."""
    bw = max(4, int(min(h, w) * float(p["frame_width"]) / 100.0))
    if p["frame_style"] == "Raw Canvas Edge":
        return bw, h, w
    return bw, h + 2 * bw, w + 2 * bw


def frame_layer(h: int, w: int, p: dict) -> np.ndarray:
    """The frame as an overlay: float32 (H, W, 4) BGR + alpha (0..1).

    The picture goes underneath, centred; ``apply_frame`` and the live
    preview both composite it the same way.
    """
    bw, H, W = frame_size(h, w, p)
    style = p["frame_style"]
    base = np.array(hex_to_bgr(p["frame_color"]), np.float32)
    rng = np.random.default_rng(int(p["frame_seed"]))

    if style == "Raw Canvas Edge":
        m = np.zeros((h, w), np.float32)
        cv2.rectangle(m, (bw, bw), (w - bw, h - bw), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), bw * 0.35)
        noise = _value_noise(h, w, 40, 5)
        m = np.clip(m * (0.75 + 0.5 * noise), 0, 1)
        out = np.empty((h, w, 4), np.float32)
        out[:, :, :3] = (236.0 * (0.9 + 0.1 * noise))[:, :, None]
        out[:, :, 3] = 1.0 - m
        return out

    out = np.zeros((H, W, 3), np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
    t = np.clip(d / max(bw, 1), 0, 1)

    if style == "Gilded":
        base = np.array([40.0, 170.0, 225.0], np.float32)
        prof = 0.55 + 0.75 * np.sin(np.clip(t, 0, 1) * math.pi * 2.0) ** 2
        grain = 1.0 + 0.10 * (_value_noise(H, W, 60, 3) - 0.5)
    elif style == "Museum Black":
        base = np.array([26.0, 24.0, 22.0], np.float32)
        prof = 0.70 + 0.55 * np.clip(1.0 - np.abs(t - 0.75) * 4.0, 0, 1)
        grain = 1.0 + 0.04 * (_value_noise(H, W, 80, 9) - 0.5)
    elif style == "Scratched Metal":
        base = np.array([150.0, 150.0, 155.0], np.float32)
        prof = 0.6 + 0.6 * np.abs(np.sin(t * math.pi * 1.5))
        grain = 1.0 + 0.22 * (_value_noise(H, W, 5, 4, octaves=2) - 0.5)
    elif style == "Passe-Partout":
        base = np.array([238.0, 240.0, 244.0], np.float32)
        prof = np.where(t > 0.80, 0.80, 1.0).astype(np.float32)
        grain = np.ones((H, W), np.float32)
    else:  # Classic Wood
        prof = 0.62 + 0.62 * np.abs(np.sin(np.clip(t, 0, 1) * math.pi)) ** 1.3
        grain = 1.0 + 0.16 * (_value_noise(H, W, 8, int(rng.integers(0, 999))) - 0.5)

    out[:] = base[None, None, :] * (prof * grain)[:, :, None]

    # a mitre line from each corner, and a shadow where the picture sits
    for (ax, ay, bx, by) in ((0, 0, bw, bw), (W, 0, W - bw, bw),
                             (0, H, bw, H - bw), (W, H, W - bw, H - bw)):
        cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)),
                 (float(base[0] * .45), float(base[1] * .45), float(base[2] * .45)), 1,
                 cv2.LINE_AA)

    inner = np.zeros((H, W), np.float32)
    cv2.rectangle(inner, (bw, bw), (W - bw, H - bw), 1.0, -1)
    shadow = cv2.GaussianBlur(inner, (0, 0), max(1.5, bw * 0.22))
    out *= (1.0 - 0.45 * np.clip(shadow - inner, 0, 1))[:, :, None]

    alpha = np.ones((H, W, 1), np.float32)
    alpha[bw:bw + h, bw:bw + w] = 0.0
    return np.concatenate([np.clip(out, 0, 255), alpha], axis=2)


def apply_frame(img: np.ndarray, p: dict, layer: np.ndarray | None = None) -> np.ndarray:
    """Put the artwork in a frame."""
    if not p["frame_enabled"]:
        return img
    h, w = img.shape[:2]
    bw, H, W = frame_size(h, w, p)
    if layer is None:
        layer = frame_layer(h, w, p)
    a = layer[:, :, 3:4]
    if (H, W) == (h, w):
        return np.clip(img * (1 - a) + layer[:, :, :3] * a, 0, 255)
    out = layer[:, :, :3].copy()
    out[bw:bw + h, bw:bw + w] = img
    return np.clip(out, 0, 255)


# --------------------------------------------------------------------------
# text / watermark
# --------------------------------------------------------------------------

_FONT_FILES = {
    "serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    "script": ["/usr/share/fonts/truetype/google-fonts/Poppins-Italic.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"],
}


def _load_font(kind: str, size: int):
    import os
    for path in _FONT_FILES.get(kind, []) + _FONT_FILES["sans"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_layer(h: int, w: int, p: dict) -> np.ndarray:
    """The inscription as an overlay: float32 (h, w, 4) BGR + alpha 0..1,
    before Opacity (which the preview applies live)."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = max(8, int(min(h, w) * float(p["text_size"]) / 100.0))
    font = _load_font(p["text_font"], size)
    text = p["text_content"] or ""
    b, g, r = hex_to_bgr(p["text_color"])
    fill = (int(r), int(g), int(b), 255)

    if p["text_tile"]:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
        step_x, step_y = tw + size, th + size * 2
        for yy in range(-th, h + step_y, step_y):
            for xx in range(-tw, w + step_x, step_x):
                draw.text((xx, yy), text, font=font, fill=fill)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = int(min(h, w) * 0.035)
        pos = {
            "Bottom Right": (w - tw - pad, h - th - pad * 2),
            "Bottom Left": (pad, h - th - pad * 2),
            "Top Right": (w - tw - pad, pad),
            "Top Left": (pad, pad),
            "Center": ((w - tw) // 2, (h - th) // 2),
        }[p["text_position"]]
        draw.text(pos, text, font=font, fill=fill)

    angle = float(p["text_angle"])
    if angle:
        layer = layer.rotate(angle, resample=Image.BICUBIC, center=(w // 2, h // 2))

    rgba = np.asarray(layer).astype(np.float32)
    out = np.empty((h, w, 4), np.float32)
    out[:, :, :3] = rgba[:, :, [2, 1, 0]]
    out[:, :, 3] = rgba[:, :, 3] / 255.0
    return out


def logo_layer(h: int, w: int, p: dict, logo: np.ndarray | None) -> np.ndarray | None:
    """The logo as an overlay (opaque where the logo is), or None."""
    if logo is None:
        return None
    lh = max(8, int(min(h, w) * float(p["text_size"]) / 100.0 * 2.0))
    sc = lh / max(1, logo.shape[0])
    lg = cv2.resize(logo, (max(1, int(logo.shape[1] * sc)), lh))
    pad = int(min(h, w) * 0.035)
    y0 = h - lg.shape[0] - pad
    x0 = w - lg.shape[1] - pad
    out = np.zeros((h, w, 4), np.float32)
    if y0 >= 0 and x0 >= 0:
        out[y0:y0 + lg.shape[0], x0:x0 + lg.shape[1], :3] = lg
        out[y0:y0 + lg.shape[0], x0:x0 + lg.shape[1], 3] = 1.0
    return out


def apply_text(img: np.ndarray, p: dict, logo: np.ndarray | None = None,
               layers=None):
    """Add an inscription, a signature or a watermark."""
    if not p["text_enabled"]:
        return img
    h, w = img.shape[:2]
    opacity = float(p["text_opacity"]) / 100.0
    if opacity <= 0:
        return img
    txt, lg = layers if layers is not None else (text_layer(h, w, p),
                                                  logo_layer(h, w, p, logo))
    alpha = txt[:, :, 3:4] * opacity
    out = img * (1.0 - alpha) + txt[:, :, :3] * alpha
    if lg is not None:
        a = lg[:, :, 3:4] * opacity
        out = out * (1.0 - a) + lg[:, :, :3] * a
    return np.clip(out, 0, 255)


# --------------------------------------------------------------------------
# studio lighting
# --------------------------------------------------------------------------


def apply_light(img: np.ndarray, p: dict) -> np.ndarray:
    """Light the finished painting the way a still life is lit in a studio.

    A key light falling from one corner, the rest of the picture sinking
    into shadow, a warm bloom around the brightest paint and a warm/cool
    split between the lit and shadowed halves.  This is what separates an
    old-master still life from a photograph of a bunch of flowers, and no
    amount of brushwork substitutes for it.
    """
    if not p["light_enabled"]:
        return img
    h, w = img.shape[:2]
    vig = float(p["light_vignette"]) / 100.0
    glow = float(p["light_glow"]) / 100.0
    warm = float(p["light_warmth"]) / 100.0
    spread = max(0.12, float(p["light_spread"]) / 100.0)

    out = np.clip(img, 0, 255).astype(np.float32)

    # --- key light falloff ------------------------------------------------
    lx = float(p["light_x"]) / 100.0 * (w - 1)
    ly = float(p["light_y"]) / 100.0 * (h - 1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    diag = math.hypot(w, h)
    d = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2) / (diag * spread)
    key = np.exp(-d * d * 0.5).astype(np.float32)

    if vig > 0.001:
        # never crush to black: shadowed paint keeps a little of its colour
        out *= (1.0 - vig * 0.92 * (1.0 - key))[:, :, None]

    # --- bloom around the brightest paint ---------------------------------
    if glow > 0.001:
        lum = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hot = np.clip((lum - 0.55) / 0.45, 0, 1) ** 1.5
        bloom = cv2.GaussianBlur(out * hot[:, :, None], (0, 0),
                                 max(3.0, min(h, w) * 0.022))
        out = 255.0 - (255.0 - out) * (255.0 - bloom * glow * 0.85) / 255.0

    # --- warm light, cool shadow ------------------------------------------
    if abs(warm) > 0.001:
        tint = np.array([-0.16, 0.05, 0.22], np.float32) * warm
        out *= (1.0 + tint[None, None, :] * key[:, :, None])
        out *= (1.0 - tint[None, None, :] * 0.45 * (1.0 - key)[:, :, None])

    return np.clip(out, 0, 255)


def decorate(img: np.ndarray, p: dict, order=("Light", "Canvas", "Frame", "Text"),
             canvas_custom=None, logo=None) -> np.ndarray:
    fn = {"Light": lambda a: apply_light(a, p),
          "Canvas": lambda a: apply_canvas(a, p, canvas_custom),
          "Frame": lambda a: apply_frame(a, p),
          "Text": lambda a: apply_text(a, p, logo)}
    out = img
    for name in order:
        if name in fn:
            out = fn[name](out)
    return out
