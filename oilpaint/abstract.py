"""
Abstract Art: recolours random strokes, corrects the overall colour of the
image and warps the shapes of objects into fantastic, wavy forms.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def gblur(img: np.ndarray, sigma: float) -> np.ndarray:
    """cv2.GaussianBlur(img, (0, 0), sigma), fast for a big sigma.

    A wide blur of a big photo is slow (sigma 115 on 4000 x 3000: 8 s); the
    same blur made on a smaller copy and enlarged looks the same (a blur
    that wide has no fine detail to lose) and takes a few ms. Small sigmas
    are blurred as they are."""
    sigma = float(sigma)
    if sigma < 8.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = int(sigma // 4)                       # still >= 4 px of blur on the small copy
    h, w = img.shape[:2]
    sw, sh = max(1, round(w / f)), max(1, round(h / f))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    # shrinking and enlarging blur a little by themselves (about f/2 px)
    s_small = math.sqrt(max(sigma * sigma - (0.5 * f) ** 2, 1.0)) / f
    small = cv2.GaussianBlur(small, (0, 0), s_small)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def correct(img: np.ndarray, p: dict) -> np.ndarray:
    """Correction group - Hue / Saturation / Brightness of the whole image."""
    hue = float(p["cor_hue"])
    sat = float(p["cor_saturation"])
    bri = float(p["cor_brightness"])
    if hue == 0 and sat == 0 and bri == 0:
        return img

    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    if hue:
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue / 2.0, 180.0)
    if sat:
        f = 1.0 + sat / 100.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * f, 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                       cv2.COLOR_HSV2BGR).astype(np.float32)
    if bri:
        if bri > 0:
            out = out + (255.0 - out) * (bri / 100.0) * 0.85
        else:
            out = out * (1.0 + bri / 100.0 * 0.85)
    return np.clip(out, 0, 255)


def _noise_field(h, w, cells, seed):
    """Smooth value noise in [-1, 1] at the given cell resolution."""
    rng = np.random.default_rng(seed)
    gh = max(2, int(cells))
    gw = max(2, int(cells * w / max(h, 1)))
    gw = max(2, gw)
    grid = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(field, (0, 0), max(1.0, min(h, w) / (cells * 6.0)))


def distort(img: np.ndarray, p: dict) -> np.ndarray:
    """Distortion group - shifts and stretches objects into wavy shapes."""
    strength = float(p["dist_strength"])
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    curvature = float(p["dist_curvature"])      # 2..25 -> waviness
    seed = int(p["dist_seed"])

    amp = (strength / 100.0) * min(h, w) * 0.075
    nx = _noise_field(h, w, curvature, seed)
    ny = _noise_field(h, w, curvature, seed + 977)
    # a second octave gives the contours their fine ripple
    nx = nx + 0.45 * _noise_field(h, w, curvature * 2.2, seed + 3121)
    ny = ny + 0.45 * _noise_field(h, w, curvature * 2.2, seed + 5417)

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = np.clip(xs + nx * amp, 0, w - 1)
    map_y = np.clip(ys + ny * amp, 0, h - 1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def stroke_colour(colour: np.ndarray, p: dict, rng) -> np.ndarray:
    """Strokes group - recolour / brighten / contrast random strokes."""
    col = float(p["ab_color"])          # 0..90
    sat = float(p["ab_saturation"])     # 0..40
    con = float(p["ab_contrast"])       # 0..100
    if col <= 0 and sat <= 0 and con <= 0:
        return colour

    out = colour.astype(np.float32).reshape(1, 1, 3)

    if col > 0 and rng.random() < col / 90.0 * 0.85:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        shift = rng.uniform(-1.0, 1.0) * (col / 90.0) * 90.0
        hsv[0, 0, 0] = np.mod(hsv[0, 0, 0] + shift, 180.0)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                           cv2.COLOR_HSV2BGR).astype(np.float32)

    if sat > 0 and rng.random() < 0.6:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * (1.0 + rng.random() * sat / 40.0 * 1.6),
                               0, 255)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                           cv2.COLOR_HSV2BGR).astype(np.float32)

    if con > 0 and rng.random() < 0.5:
        f = 1.0 + rng.uniform(-1.0, 1.0) * (con / 100.0) * 0.75
        mid = 128.0
        out = mid + (out - mid) * f

    return np.clip(out.reshape(3), 0, 255)
