"""The "brush" layer most Envato actions ask for: the focus area of the photo.

In Photoshop the user paints it by hand (or uses Select Subject). Here:
    subject - found automatically (saliency + a centre bias)
    full    - the whole picture
    none    - nothing painted
A mask painted by the user in OilStudio can be passed in instead.
"""
from __future__ import annotations

import cv2
import numpy as np


def subject_mask(rgb: np.ndarray) -> np.ndarray:
    """uint8 mask (h, w), 255 where the main subject is."""
    h, w = rgb.shape[:2]
    side = 256
    k = side / max(h, w)
    small = cv2.resize(rgb, (max(16, int(w * k)), max(16, int(h * k))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    sh, sw = lab.shape[:2]

    # spectral residual saliency on each Lab channel
    sal = np.zeros((sh, sw), np.float32)
    for c in range(3):
        f = np.fft.fft2(lab[..., c])
        amp = np.log(np.abs(f) + 1e-6)
        ph = np.angle(f)
        res = amp - cv2.blur(amp, (3, 3))
        s = np.abs(np.fft.ifft2(np.exp(res + 1j * ph))) ** 2
        sal += cv2.GaussianBlur(s.astype(np.float32), (0, 0), 3)
    sal /= sal.max() + 1e-6

    # colour contrast against the image border (background)
    border = np.concatenate([lab[:4].reshape(-1, 3), lab[-4:].reshape(-1, 3),
                             lab[:, :4].reshape(-1, 3), lab[:, -4:].reshape(-1, 3)])
    bmean = border.mean(0)
    dist = np.sqrt(((lab - bmean) ** 2).sum(-1))
    dist = cv2.GaussianBlur(dist, (0, 0), 4)
    dist /= dist.max() + 1e-6

    y, x = np.mgrid[0:sh, 0:sw].astype(np.float32)
    centre = np.exp(-(((x - sw / 2) / (sw * 0.38)) ** 2 + ((y - sh / 2) / (sh * 0.42)) ** 2))

    score = (0.45 * sal + 0.55 * dist) * (0.35 + 0.65 * centre)
    score = cv2.GaussianBlur(score, (0, 0), 3)
    t = np.percentile(score, 62)
    m = (score > t).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_id, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        keep = np.zeros(n, bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        keep[1:] = areas >= big * 0.15
        m = keep[lab_id].astype(np.uint8)
    if m.sum() < 0.05 * sh * sw:                      # nothing clear: use the centre
        m = (centre > 0.45).astype(np.uint8)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), max(2.0, max(h, w) / 300))
    return np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8)


def make(rgb: np.ndarray, mode: str, painted: np.ndarray | None = None) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    if painted is not None:
        if painted.shape[:2] != (h, w):
            painted = cv2.resize(painted, (w, h), interpolation=cv2.INTER_LINEAR)
        return painted.astype(np.uint8)
    if mode == "full":
        return np.full((h, w), 255, np.uint8)
    if mode == "none":
        return None
    return subject_mask(rgb)
